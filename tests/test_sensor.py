"""Tests for Sensor device and drivers.

Tests the high-level async Sensor abstraction and DigitalTwin driver.
Arduino driver tests require hardware and are integration tests.

The Sensor class provides an async interface wrapping synchronous drivers:
- async connect() / disconnect() for lifecycle
- async read(samples=N) for single or averaged readings
- async read_for(duration_ms) for time-based sampling
- sample_rate_hz property queried from device on connect
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from telescope_mcp.devices.sensor import DeviceSensorInfo, Sensor
from telescope_mcp.drivers.sensors import (
    DigitalTwinSensorConfig,
    DigitalTwinSensorDriver,
    SensorReading,
)


class TestDigitalTwinSensorDriver:
    """Test suite for DigitalTwinSensorDriver functionality.

    These are synchronous driver tests - the driver layer remains sync.
    The Sensor device class wraps these in async operations.

    Categories:
    1. Discovery Tests - Sensor enumeration, availability
    2. Lifecycle Tests - Open, close, instance management
    3. Configuration Tests - Custom config, behavior modification

    Total: 5 tests.
    """

    def test_get_available_sensors(self) -> None:
        """Verifies driver discovers exactly one simulated sensor.

        Tests sensor enumeration by querying the driver for available sensors.

        Business context:
        Before connecting, applications need to discover what sensors exist.
        The digital twin always provides exactly one simulated sensor for testing.

        Arrangement:
            Create DigitalTwinSensorDriver instance.

        Action:
            Call get_available_sensors() to enumerate sensors.

        Assertion Strategy:
            Verify exactly one sensor returned with type="digital_twin" and name field.

        Testing Principle:
            Tests sensor discovery mechanism for system initialization.
        """
        driver = DigitalTwinSensorDriver()
        sensors = driver.get_available_sensors()

        assert len(sensors) == 1
        assert sensors[0]["type"] == "digital_twin"
        assert "name" in sensors[0]

    def test_open_creates_instance(self) -> None:
        """Verifies open() creates a usable sensor instance.

        Tests the factory pattern where driver.open() creates instances.

        Business context:
        Drivers manage sensor lifecycle. Opening a driver creates an active
        instance that can read sensor data.

        Arrangement:
            Create DigitalTwinSensorDriver instance.

        Action:
            Call driver.open() to create sensor instance.

        Assertion Strategy:
            Verify non-null instance returned with type="digital_twin" in info.

        Testing Principle:
            Tests factory pattern for sensor instance creation.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        assert instance is not None
        info = instance.get_info()
        assert info["type"] == "digital_twin"

        driver.close()

    def test_open_twice_raises(self) -> None:
        """Verifies opening an already-open driver raises RuntimeError.

        Tests single-instance enforcement to prevent resource conflicts.

        Business context:
        Sensors are physical resources. Opening twice would cause conflicts
        in serial port access or thread management.

        Arrangement:
            Create and open DigitalTwinSensorDriver instance.

        Action:
            Attempt to open the same driver again.

        Assertion Strategy:
            Verify RuntimeError raised with message containing "already open".

        Testing Principle:
            Tests resource exclusivity to prevent hardware conflicts.
        """
        driver = DigitalTwinSensorDriver()
        driver.open()

        with pytest.raises(RuntimeError, match="already open"):
            driver.open()

        driver.close()

    def test_close_without_open_is_safe(self) -> None:
        """Verifies close() is safe to call when no instance is open.

        Tests defensive close behavior for robust lifecycle management.

        Business context:
        Cleanup code often calls close() unconditionally in finally blocks.
        Drivers must handle close() gracefully when no instance was ever opened.

        Arrangement:
            Create DigitalTwinSensorDriver without opening.

        Action:
            Call close() on unopened driver.

        Assertion Strategy:
            Verify no exception raised and driver remains usable afterward.

        Testing Principle:
            Tests idempotent cleanup for robust error handling.
        """
        driver = DigitalTwinSensorDriver()

        # Should not raise - close() when _instance is None
        driver.close()

        # Driver should still be usable after safe close
        instance = driver.open()
        assert instance is not None
        driver.close()

    def test_custom_config(self) -> None:
        """Verifies custom configuration affects simulated sensor readings.

        Tests configuration injection for deterministic test scenarios.

        Business context:
        Tests need predictable sensor values. Custom config sets initial
        position and environmental values, enabling reproducible tests.

        Arrangement:
            Create DigitalTwinSensorConfig with custom values.

        Action:
            Open driver with custom config and read values.

        Assertion Strategy:
            Verify readings match configured initial values.

        Testing Principle:
            Tests configuration injection for deterministic testing.
        """
        config = DigitalTwinSensorConfig(
            initial_altitude=30.0,
            initial_azimuth=90.0,
            temperature=15.0,
            humidity=60.0,
        )
        driver = DigitalTwinSensorDriver(config)
        instance = driver.open()

        reading = instance.read()
        # Values should be close to config (with noise)
        assert 25.0 <= reading.altitude <= 35.0
        assert 85.0 <= reading.azimuth <= 95.0
        assert 14.0 <= reading.temperature <= 16.0
        assert 58.0 <= reading.humidity <= 62.0

        driver.close()


class TestDigitalTwinSensorInstance:
    """Test suite for DigitalTwinSensorInstance functionality.

    Categories:
    1. Reading Tests - Data retrieval, format validation
    2. Position Tests - Position control for testing
    3. Calibration Tests - Offset calibration, magnetometer
    4. Status Tests - State reporting, lifecycle

    Total: 7 tests.
    """

    def test_read_returns_sensor_reading(self) -> None:
        """Verifies read() returns complete SensorReading with all fields.

        Tests data contract for sensor reading structure.

        Business context:
        Telescope pointing requires accelerometer (tilt), magnetometer
        (heading), and environmental data. All fields must be present.

        Arrangement:
            Open DigitalTwinSensorDriver instance.

        Action:
            Call instance.read() to get sensor data.

        Assertion Strategy:
            Verify SensorReading contains all expected fields with correct types.

        Testing Principle:
            Tests data contract completeness for sensor interface.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        reading = instance.read()

        assert isinstance(reading.accelerometer, dict)
        assert "aX" in reading.accelerometer
        assert "aY" in reading.accelerometer
        assert "aZ" in reading.accelerometer

        assert isinstance(reading.magnetometer, dict)
        assert "mX" in reading.magnetometer
        assert "mY" in reading.magnetometer
        assert "mZ" in reading.magnetometer

        assert isinstance(reading.altitude, float)
        assert isinstance(reading.azimuth, float)
        assert isinstance(reading.temperature, float)
        assert isinstance(reading.humidity, float)
        assert isinstance(reading.timestamp, datetime)
        assert reading.raw_values  # Non-empty string

        driver.close()

    def test_set_position(self) -> None:
        """Verifies set_position updates simulated telescope position.

        Tests internal position control for deterministic testing.

        Business context:
        Integration tests need to simulate telescope movement. This
        method sets the "true" position that the digital twin reports.

        Arrangement:
            Open DigitalTwinSensorDriver instance.

        Action:
            Call set_position(altitude=60.0, azimuth=270.0) and read back.

        Assertion Strategy:
            Verify reading values are close to set position (within noise tolerance).

        Testing Principle:
            Tests position injection for integration test scenarios.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        instance.set_position(altitude=60.0, azimuth=270.0)
        reading = instance.read()

        # Should be close to set position (with noise)
        assert 55.0 <= reading.altitude <= 65.0
        assert 265.0 <= reading.azimuth <= 275.0

        driver.close()

    def test_calibrate_adjusts_readings(self) -> None:
        """Verifies calibration adjusts readings to match true position.

        Tests offset calibration model: calibrated = raw + offset.

        Business context:
        Sensor mounting introduces systematic errors. Calibration against
        a known reference computes offsets so readings match actual position.

        Arrangement:
            Create driver with known position and zero noise.

        Action:
            Calibrate with different true position and read back.

        Assertion Strategy:
            Verify readings match true position within tolerance.

        Testing Principle:
            Tests offset calibration for measurement accuracy.
        """
        config = DigitalTwinSensorConfig(
            initial_altitude=40.0,
            initial_azimuth=180.0,
            noise_std_alt=0.0,  # No noise for predictable test
            noise_std_az=0.0,
        )
        driver = DigitalTwinSensorDriver(config)
        instance = driver.open()

        # Calibrate to true position
        instance.calibrate(true_altitude=45.0, true_azimuth=200.0)

        reading = instance.read()
        assert abs(reading.altitude - 45.0) < 1.0
        assert abs(reading.azimuth - 200.0) < 1.0

        driver.close()

    def test_reset_clears_calibration(self) -> None:
        """Verifies reset() clears calibration offsets on driver instance.

        Tests state reset for sensor re-initialization at driver level.

        Business context:
        When re-mounting sensors, old calibration becomes invalid. Reset
        clears offsets so fresh calibration can be applied.

        Note: At device level, disconnect/reconnect serves as reset.

        Arrangement:
            Create driver with zero noise and calibrate with offsets.

        Action:
            Call reset() and read back values.

        Assertion Strategy:
            Verify readings return to uncalibrated initial values.

        Testing Principle:
            Tests state reset for sensor recalibration scenarios.
        """
        config = DigitalTwinSensorConfig(
            initial_altitude=45.0,
            initial_azimuth=180.0,
            noise_std_alt=0.0,
            noise_std_az=0.0,
        )
        driver = DigitalTwinSensorDriver(config)
        instance = driver.open()

        instance.calibrate(true_altitude=90.0, true_azimuth=0.0)
        instance.reset()

        reading = instance.read()
        # Should be back to initial position
        assert abs(reading.altitude - 45.0) < 1.0
        assert abs(reading.azimuth - 180.0) < 1.0

        driver.close()

    def test_get_status(self) -> None:
        """Verifies get_status() returns connection and calibration state.

        Tests status reporting for monitoring and diagnostics.

        Business context:
        Observatory software needs sensor health data. Status includes
        connection state and calibration state.

        Arrangement:
            Open DigitalTwinSensorDriver instance (uncalibrated).

        Action:
            Call get_status() to retrieve status dict.

        Assertion Strategy:
            Verify status contains connected=True, calibrated=False, is_open=True.

        Testing Principle:
            Tests status reporting for system health monitoring.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        status = instance.get_status()

        assert status["connected"] is True
        assert status["calibrated"] is False
        assert status["is_open"] is True

        driver.close()

    def test_read_after_close_raises(self) -> None:
        """Verifies reading after close() raises RuntimeError.

        Tests lifecycle enforcement to prevent use-after-close bugs.

        Business context:
        Closed sensors have released resources. Reading from closed
        instance would access invalid state.

        Arrangement:
            Open and then close DigitalTwinSensorDriver instance.

        Action:
            Attempt to read() from closed instance.

        Assertion Strategy:
            Verify RuntimeError raised with message containing "closed".

        Testing Principle:
            Tests lifecycle enforcement to prevent use-after-close errors.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()
        instance.close()

        with pytest.raises(RuntimeError, match="closed"):
            instance.read()

    def test_calibrate_magnetometer(self) -> None:
        """Verifies magnetometer calibration returns offset values.

        Tests hard-iron calibration simulation.

        Business context:
        Magnetometers suffer hard-iron distortion from nearby ferrous
        materials. Calibration computes offsets to center magnetic readings.

        Arrangement:
            Open DigitalTwinSensorDriver instance.

        Action:
            Call calibrate_magnetometer() to compute offsets.

        Assertion Strategy:
            Verify result contains offset_x, offset_y, offset_z fields.

        Testing Principle:
            Tests magnetometer calibration for heading accuracy.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        result = instance.calibrate_magnetometer()

        assert "offset_x" in result
        assert "offset_y" in result
        assert "offset_z" in result

        driver.close()


class TestSensorDevice:
    """Test suite for high-level async Sensor device wrapper.

    All tests use pytest.mark.asyncio for async method testing.

    Categories:
    1. Lifecycle Tests - Connect, disconnect, auto-connect
    2. Reading Tests - Single read, averaged read, duration read
    3. Sample Rate Tests - Query from device, fallback
    4. Calibration Tests - Input validation, state management
    5. Interface Tests - Context manager, repr, discovery

    Total: 18 tests.
    """

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self) -> None:
        """Verifies basic async connect and disconnect lifecycle.

        Tests high-level Sensor device connection management.

        Business context:
        The Sensor class wraps low-level drivers for application use.
        Connect/disconnect manage the underlying driver instance.

        Arrangement:
            Create Sensor with DigitalTwinSensorDriver.

        Action:
            Connect, verify state, disconnect, verify state.

        Assertion Strategy:
            Verify connected flag and info presence match lifecycle state.

        Testing Principle:
            Tests async device lifecycle management.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert not sensor.connected

        await sensor.connect()
        assert sensor.connected
        assert sensor.info is not None

        await sensor.disconnect()
        assert not sensor.connected
        assert sensor.info is None

    @pytest.mark.asyncio
    async def test_double_disconnect_is_safe(self) -> None:
        """Verifies disconnect() can be called multiple times safely.

        Tests idempotent disconnect behavior.

        Business context:
        Cleanup code may call disconnect() multiple times.
        Should be safe without raising errors.

        Arrangement:
            Create and connect Sensor.

        Action:
            Call disconnect() twice in succession.

        Assertion Strategy:
            Verify second disconnect does not raise and connected remains False.

        Testing Principle:
            Tests idempotent cleanup for robust resource management.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        await sensor.disconnect()
        # Second disconnect should not raise
        await sensor.disconnect()
        assert not sensor.connected

    @pytest.mark.asyncio
    async def test_connect_twice_raises(self) -> None:
        """Verifies connecting when already connected raises RuntimeError.

        Tests idempotency protection at the Sensor wrapper level.

        Business context:
        Double-connect could cause resource leaks or state corruption.
        The Sensor class enforces single connection.

        Arrangement:
            Create and connect Sensor.

        Action:
            Attempt second connect() call.

        Assertion Strategy:
            Verify RuntimeError raised with "already connected" message.

        Testing Principle:
            Tests connection idempotency to prevent resource leaks.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        with pytest.raises(RuntimeError, match="already connected"):
            await sensor.connect()

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_read_without_connect_raises(self) -> None:
        """Verifies reading without connect() raises RuntimeError.

        Tests precondition enforcement for read operations.

        Business context:
        Reading requires an active connection. Attempting to read
        without connecting would access null instance.

        Arrangement:
            Create Sensor without connecting.

        Action:
            Attempt to call read().

        Assertion Strategy:
            Verify RuntimeError raised with "not connected" message.

        Testing Principle:
            Tests precondition validation for operation safety.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with pytest.raises(RuntimeError, match="not connected"):
            await sensor.read()

    @pytest.mark.asyncio
    async def test_read_single_sample(self) -> None:
        """Verifies read() returns single SensorReading by default.

        Tests basic read with samples=1 (default).

        Business context:
        Most reads want the current sensor value without averaging.
        Default samples=1 provides simple single-read behavior.

        Arrangement:
            Connect Sensor instance.

        Action:
            Call read() with default parameters.

        Assertion Strategy:
            Verify complete SensorReading returned with all expected fields.

        Testing Principle:
            Tests default read behavior for basic sensor data retrieval.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        reading = await sensor.read()

        assert isinstance(reading.altitude, float)
        assert isinstance(reading.azimuth, float)
        assert reading.accelerometer
        assert reading.magnetometer

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_read_with_averaging(self) -> None:
        """Verifies read(samples=N) averages multiple readings.

        Tests averaged read for noise reduction.

        Business context:
        Sensor noise can be reduced by averaging multiple samples.
        read(samples=5) takes 5 readings and returns the average.

        Arrangement:
            Create driver with noise, connect Sensor.

        Action:
            Call read(samples=5) to get averaged reading.

        Assertion Strategy:
            Verify reading returned (averaging logic tested separately).

        Testing Principle:
            Tests multi-sample averaging for noise reduction.
        """
        config = DigitalTwinSensorConfig(
            initial_altitude=45.0,
            initial_azimuth=180.0,
            noise_std_alt=2.0,  # Some noise to test averaging
            noise_std_az=2.0,
        )
        driver = DigitalTwinSensorDriver(config)
        sensor = Sensor(driver)
        await sensor.connect()

        reading = await sensor.read(samples=5)

        # Averaged reading should be close to true position
        # With 5 samples, noise averages out somewhat
        assert 40.0 <= reading.altitude <= 50.0
        assert 175.0 <= reading.azimuth <= 185.0

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_read_for_duration(self) -> None:
        """Verifies read_for(duration_ms) reads for specified time.

        Tests time-based sampling that converts to sample count.

        Business context:
        Sometimes you want to sample for a known duration rather than
        a specific count. read_for() calculates samples from sample_rate.

        Arrangement:
            Connect Sensor with known sample rate (10 Hz).

        Action:
            Call read_for(duration_ms=200) to read for 200ms.

        Assertion Strategy:
            Verify valid reading returned (2 samples at 10 Hz).

        Testing Principle:
            Tests duration-based sampling for time-constrained observations.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        # Digital twin has 10 Hz sample rate, so 200ms = 2 samples
        reading = await sensor.read_for(duration_ms=200)

        assert isinstance(reading.altitude, float)
        assert isinstance(reading.azimuth, float)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_read_for_minimum_one_sample(self) -> None:
        """Verifies read_for() always reads at least one sample.

        Tests floor behavior for very short durations.

        Business context:
        Even with duration_ms=1, we should get at least one reading.
        Prevents edge case of zero samples.

        Arrangement:
            Connect Sensor instance.

        Action:
            Call read_for(duration_ms=1) with very short duration.

        Assertion Strategy:
            Verify valid reading returned (minimum 1 sample).

        Testing Principle:
            Tests minimum sample count enforcement for edge cases.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        # Very short duration should still produce a reading
        reading = await sensor.read_for(duration_ms=1)

        assert isinstance(reading.altitude, float)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_sample_rate_queried_on_connect(self) -> None:
        """Verifies sample_rate_hz is populated from device on connect.

        Tests sample rate query during connection.

        Business context:
        Different hardware has different sample rates. Querying on connect
        ensures accurate duration-to-samples conversion.

        Arrangement:
            Create Sensor with default sample rate.

        Action:
            Connect and verify sample_rate_hz populated from driver.

        Assertion Strategy:
            Verify sample_rate_hz is 10.0 before and after connect
            (DigitalTwin default).

        Testing Principle:
            Tests sample rate discovery during device initialization.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        # Before connect, sample rate is default
        assert sensor.sample_rate_hz == 10.0  # Default value

        await sensor.connect()

        # After connect, sample rate should still be 10 Hz for digital twin
        assert sensor.sample_rate_hz == 10.0

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_calibrate(self) -> None:
        """Verifies calibrate() sets transform to match true position.

        Tests calibration delegation from Sensor to driver instance.

        Business context:
        Calibration is critical for telescope pointing accuracy.
        The Sensor class validates inputs and delegates to driver.

        Arrangement:
            Connect Sensor instance.

        Action:
            Call calibrate(true_altitude=50.0, true_azimuth=120.0).

        Assertion Strategy:
            Verify calibrated flag set in status after calibration.

        Testing Principle:
            Tests calibration delegation for positioning accuracy.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        sensor.calibrate(true_altitude=50.0, true_azimuth=120.0)

        # Verify calibration was applied
        status = sensor.get_status()
        assert status.get("calibrated", False)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_calibrate_validates_altitude(self) -> None:
        """Verifies calibrate() validates altitude range 0-90 degrees.

        Tests input validation for physical constraints.

        Business context:
        Altitude must be 0-90° (horizon to zenith). Values outside
        this range are physically impossible.

        Arrangement:
            Connect Sensor instance.

        Action:
            Call calibrate with out-of-range altitude (100.0).

        Assertion Strategy:
            Verify ValueError raised with range validation message.

        Testing Principle:
            Tests input validation for physical measurement constraints.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        with pytest.raises(ValueError, match=r"Altitude must be in range \[0, 90\]"):
            sensor.calibrate(true_altitude=100.0, true_azimuth=180.0)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_calibrate_validates_azimuth(self) -> None:
        """Verifies calibrate() validates azimuth range 0-360 degrees.

        Tests input validation for compass heading constraints.

        Business context:
        Azimuth must be 0-360° (full compass circle). Values outside
        this range indicate user error.

        Arrangement:
            Connect Sensor instance.

        Action:
            Call calibrate with out-of-range azimuth (400.0).

        Assertion Strategy:
            Verify ValueError raised with range validation message.

        Testing Principle:
            Tests input validation for compass heading constraints.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        with pytest.raises(ValueError, match=r"Azimuth must be in range \[0, 360\)"):
            sensor.calibrate(true_altitude=45.0, true_azimuth=400.0)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_calibrate_negative_values_raise(self) -> None:
        """Verifies calibrate() rejects negative altitude and azimuth.

        Tests lower bound validation for both parameters.

        Business context:
        Negative values are physically impossible for telescope coordinates.

        Arrangement:
        1. Create DigitalTwinSensorDriver for controlled testing.
        2. Initialize Sensor and establish connection.
        3. Prepare invalid calibration values (negative altitude/azimuth).

        Action:
        Call sensor.calibrate() twice with negative values:
        - First with altitude=-10.0 (below 0° minimum)
        - Second with azimuth=-45.0 (below 0° minimum)

        Assertion Strategy:
        - Both calls raise ValueError with specific error messages.
        - Error messages specify valid ranges [0,90] and [0,360).
        - Validates input bounds checked before applying calibration.

        Testing Principle:
        Validates precondition enforcement prevents invalid calibration,
        ensuring sensor maintains physically meaningful coordinate system.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        with pytest.raises(ValueError, match=r"Altitude must be in range \[0, 90\]"):
            sensor.calibrate(true_altitude=-10.0, true_azimuth=180.0)

        with pytest.raises(ValueError, match=r"Azimuth must be in range \[0, 360\)"):
            sensor.calibrate(true_altitude=45.0, true_azimuth=-45.0)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_calibrate_not_connected_raises(self) -> None:
        """Verifies calibrate() raises RuntimeError when not connected.

        Tests precondition enforcement for calibration.

        Business context:
        Calibration requires active sensor to read current position.

        Arrangement:
        1. Create DigitalTwinSensorDriver for controlled testing.
        2. Initialize Sensor but do NOT connect.
        3. Sensor in disconnected state (no active instance).

        Action:
        Call sensor.calibrate(45.0, 180.0) without prior connection.
        Attempts calibration when sensor cannot read current position.

        Assertion Strategy:
        - Raises RuntimeError with "not connected" message.
        - Validates connection state checked before calibration.
        - Prevents calibration attempts on inactive sensor.

        Testing Principle:
        Validates state precondition enforcement ensures calibration
        only occurs when sensor can provide current readings.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with pytest.raises(RuntimeError, match="not connected"):
            sensor.calibrate(45.0, 180.0)

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        """Verifies get_status() returns comprehensive sensor info.

        Tests status aggregation from driver and wrapper layers.

        Business context:
        Observatory dashboards need unified sensor status combining
        driver status with device-level information.

        Arrangement:
            Connect Sensor instance.

        Action:
            Call get_status() to retrieve complete status dict.

        Assertion Strategy:
            Verify status contains connected, type, and sample_rate_hz fields.

        Testing Principle:
            Tests status aggregation for monitoring dashboards.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        status = sensor.get_status()

        assert status["connected"] is True
        assert status["type"] == "digital_twin"
        assert "sample_rate_hz" in status

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Verifies async context manager connects and disconnects.

        Tests async with-statement support for automatic resource cleanup.

        Business context:
        RAII pattern ensures sensors are properly disconnected even
        on exceptions.

        Arrangement:
            Create Sensor instance.

        Action:
            Use async with statement to auto-connect and disconnect.

        Assertion Strategy:
            Verify connected inside context, disconnected after exit.

        Testing Principle:
            Tests context manager protocol for automatic resource cleanup.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        async with sensor:
            assert sensor.connected
            reading = await sensor.read()
            assert reading is not None

        assert not sensor.connected

    def test_get_available_sensors(self) -> None:
        """Verifies get_available_sensors() delegates to driver.

        Tests sensor enumeration passthrough from wrapper.
        This is a sync method - doesn't need asyncio marker.

        Business context:
        Applications discover sensors through the Sensor class.

        Arrangement:
            Create Sensor with DigitalTwinSensorDriver.

        Action:
            Call get_available_sensors() synchronously.

        Assertion Strategy:
            Verify exactly one sensor returned (DigitalTwin).

        Testing Principle:
            Tests sensor discovery delegation to driver layer.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        sensors = sensor.get_available_sensors()
        assert len(sensors) == 1

    @pytest.mark.asyncio
    async def test_info_property(self) -> None:
        """Verifies info property returns DeviceSensorInfo when connected.

        Tests sensor metadata access through typed dataclass.

        Business context:
        Applications need sensor metadata (type, name) for UI display.

        Arrangement:
        1. Create DigitalTwinSensorDriver providing sensor metadata.
        2. Initialize Sensor in disconnected state.
        3. info property initially None (no metadata available).

        Action:
        Check info property before connection (should be None).
        Connect sensor to activate instance and populate metadata.
        Check info property after connection (should have data).

        Assertion Strategy:
        - info is None when disconnected (no instance available).
        - info is DeviceSensorInfo instance when connected.
        - info.type matches driver type ("digital_twin").
        - Validates metadata correctly extracted from driver.

        Testing Principle:
        Validates metadata lifecycle tied to connection state,
        ensuring applications access sensor info only when available.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert sensor.info is None

        await sensor.connect()
        assert sensor.info is not None
        assert isinstance(sensor.info, DeviceSensorInfo)
        assert sensor.info.type == "digital_twin"

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_repr(self) -> None:
        """Verifies repr shows connection state and sensor type.

        Tests string representation for debugging.

        Business context:
        Debugging and logging benefit from informative repr.

        Arrangement:
        1. Create DigitalTwinSensorDriver for sensor instance.
        2. Initialize Sensor in disconnected state.
        3. repr() should show connection=False initially.

        Action:
        Check repr() before connection (disconnected state).
        Connect sensor to establish active instance.
        Check repr() after connection (connected state with type).

        Assertion Strategy:
        - repr() contains "connected=False" when disconnected.
        - repr() contains "connected=True" after connection.
        - repr() includes sensor type ("digital_twin") when connected.
        - Provides human-readable debugging information.

        Testing Principle:
        Validates string representation provides useful debugging info,
        showing both connection state and sensor type at a glance.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert "connected=False" in repr(sensor)

        await sensor.connect()
        assert "connected=True" in repr(sensor)
        assert "digital_twin" in repr(sensor)

        await sensor.disconnect()


class TestSensorEdgeCases:
    """Test suite for Sensor edge cases and error paths.

    Covers error handling, disconnect behavior, and unusual states.

    Total: 10 tests.
    """

    @pytest.mark.asyncio
    async def test_connect_failure_wraps_exception(self) -> None:
        """Verifies connect() wraps driver exceptions in RuntimeError.

        Tests exception handling in connect() when driver.open() fails.

        Business context:
        Hardware failures (port busy, device not found) must be wrapped
        in a consistent exception type for application error handling.

        Arrangement:
            Create mock driver that raises exception on open().

        Action:
            Attempt to connect sensor.

        Assertion Strategy:
            Verify RuntimeError raised wrapping original exception.

        Testing Principle:
            Tests exception wrapping for consistent error handling.
        """
        mock_driver = Mock()
        mock_driver.open.side_effect = RuntimeError("Port /dev/ttyACM0 not found")
        mock_driver.get_available_sensors.return_value = []

        sensor = Sensor(mock_driver)

        with pytest.raises(RuntimeError, match="Failed to connect"):
            await sensor.connect()

        assert not sensor.connected
        assert sensor._instance is None

    @pytest.mark.asyncio
    async def test_disconnect_suppresses_close_errors(self) -> None:
        """Verifies disconnect() suppresses errors from instance.close().

        Tests error suppression during cleanup to ensure cleanup completes.

        Business context:
        Serial port cleanup may fail (already closed, hardware removed).
        disconnect() must complete cleanup regardless.

        Arrangement:
        1. Create mock instance that raises RuntimeError on close().
        2. Mock driver returns this error-prone instance.
        3. Connect sensor to establish instance reference.

        Action:
        Call disconnect() which triggers instance.close() internally.
        close() raises RuntimeError but disconnect() suppresses it.
        Cleanup continues despite the error.

        Assertion Strategy:
        - disconnect() does not raise (error suppressed).
        - sensor.connected becomes False (cleanup completed).
        - sensor._instance set to None (resources released).
        - mock_instance.close() was called (cleanup attempted).

        Testing Principle:
        Validates error resilience during cleanup ensures resources
        released even when underlying hardware cleanup fails.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock Sensor"}
        mock_instance.get_status.return_value = {"sample_rate_hz": 10.0}
        mock_instance.close.side_effect = RuntimeError("Close failed")

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        await sensor.connect()

        # Should not raise - close error is suppressed
        await sensor.disconnect()

        assert not sensor.connected
        assert sensor._instance is None
        mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_is_safe(self) -> None:
        """Verifies disconnect() is safe when not connected.

        Tests idempotent disconnect behavior.

        Business context:
        Cleanup code may call disconnect() unconditionally.
        Should be safe when already disconnected.

        Arrangement:
            Create Sensor without connecting.

        Action:
            Call disconnect() on disconnected sensor.

        Assertion Strategy:
            Verify no exception raised and connected remains False.

        Testing Principle:
            Tests idempotent cleanup for defensive programming.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        # Should not raise
        await sensor.disconnect()
        assert not sensor.connected

    @pytest.mark.asyncio
    async def test_read_error_propagates(self) -> None:
        """Verifies read errors propagate to caller.

        Tests that driver read failures are surfaced to application.

        Business context:
        Applications need to know when reads fail so they can
        handle the error (retry, alert user, etc.).

        Arrangement:
            Create mock driver with instance that raises on read().

        Action:
            Connect and attempt read().

        Assertion Strategy:
            Verify RuntimeError propagates with original message.

        Testing Principle:
            Tests error propagation for application error handling.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.return_value = {"sample_rate_hz": 10.0}
        mock_instance.read.side_effect = RuntimeError("Sensor disconnected")

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        await sensor.connect()

        with pytest.raises(RuntimeError, match="Sensor disconnected"):
            await sensor.read()

    @pytest.mark.asyncio
    async def test_get_status_handles_driver_error(self) -> None:
        """Verifies get_status() captures driver status errors gracefully.

        Tests error handling when driver get_status() fails.

        Business context:
        Status queries should never crash monitoring systems.
        If driver status fails, error is captured in response.

        Arrangement:
            Create mock driver with instance that raises on get_status().

        Action:
            Manually set connected state and call get_status().

        Assertion Strategy:
            Verify status dict contains connected=True and status_error field.

        Testing Principle:
            Tests defensive error handling in status queries.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.side_effect = RuntimeError("Status unavailable")

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        # Manually set connected state since get_status during connect will fail
        sensor._instance = mock_instance
        sensor._connected = True
        sensor._info = DeviceSensorInfo(type="mock", name="Mock")

        status = sensor.get_status()

        assert status["connected"] is True
        assert "status_error" in status
        assert "Status unavailable" in status["status_error"]

    @pytest.mark.asyncio
    async def test_get_status_when_not_connected(self) -> None:
        """Verifies get_status() returns base status when not connected.

        Tests status without driver status merge (disconnected path).

        Business context:
        Status queries should work even when disconnected.

        Arrangement:
        1. Create DigitalTwinSensorDriver for sensor.
        2. Initialize Sensor without connecting.
        3. No instance available (disconnected state).

        Action:
        Call get_status() while sensor disconnected.
        Status constructed from device layer only (no driver status).

        Assertion Strategy:
        - Returns dict with connected=False.
        - type and name are None (no instance metadata).
        - Validates status accessible regardless of connection state.

        Testing Principle:
        Validates status query robustness ensures monitoring
        systems can check sensor state even when disconnected.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        status = sensor.get_status()

        assert status["connected"] is False
        assert status["type"] is None
        assert status["name"] is None

    @pytest.mark.asyncio
    async def test_read_negative_samples_raises(self) -> None:
        """Verifies read(samples=-1) raises ValueError.

        Tests input validation for negative sample count.

        Business context:
        Negative samples is nonsensical.

        Arrangement:
        1. Create DigitalTwinSensorDriver for sensor testing.
        2. Connect sensor to enable reading.
        3. Prepare invalid samples parameter (negative value).

        Action:
        Call sensor.read(samples=-1) with invalid parameter.
        Validation should reject before attempting driver read.

        Assertion Strategy:
        - Raises ValueError with "positive" in error message.
        - Validates parameter constraint enforcement.
        - Prevents invalid read attempts to driver.

        Testing Principle:
        Validates input validation prevents nonsensical operations,
        ensuring API contract enforcement at device layer.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        with pytest.raises(ValueError, match="samples must be >= 1"):
            await sensor.read(samples=-1)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_context_manager_handles_connect_error(self) -> None:
        """Verifies context manager handles connect errors gracefully.

        Tests __aenter__ exception propagation.

        Business context:
        If connection fails during context entry, error should propagate
        and context should not be entered.

        Arrangement:
            Create mock driver that raises on open().

        Action:
            Attempt to use sensor in async with context.

        Assertion Strategy:
            Verify RuntimeError propagates and connected remains False.

        Testing Principle:
            Tests context manager exception handling during entry.
        """
        mock_driver = Mock()
        mock_driver.open.side_effect = RuntimeError("Connection failed")
        mock_driver.get_available_sensors.return_value = []

        sensor = Sensor(mock_driver)

        with pytest.raises(RuntimeError, match="Failed to connect"):
            async with sensor:
                pass  # Should not reach here

        assert not sensor.connected

    @pytest.mark.asyncio
    async def test_read_samples_zero_raises(self) -> None:
        """Verifies read(samples=0) raises ValueError.

        Tests input validation for sample count.

        Business context:
        Zero samples makes no sense - must read at least one.

        Arrangement:
        1. Create and connect DigitalTwinSensorDriver.
        2. Sensor ready to accept read commands.
        3. Prepare invalid samples=0 parameter.

        Action:
        Call sensor.read(samples=0) with invalid parameter.
        Validation rejects before attempting driver read.

        Assertion Strategy:
        - Raises ValueError with "samples must be >= 1" message.
        - Validates precondition enforced at API boundary.

        Testing Principle:
        Validates input validation prevents nonsensical read attempts.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        with pytest.raises(ValueError, match="samples must be >= 1"):
            await sensor.read(samples=0)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_read_for_zero_duration_raises(self) -> None:
        """Verifies read_for(duration_ms=0) raises ValueError.

        Tests input validation for duration.

        Business context:
        Zero duration makes no sense - must sample for some time.

        Arrangement:
        1. Create and connect DigitalTwinSensorDriver.
        2. Sensor ready to accept timed read commands.
        3. Prepare invalid duration_ms=0 parameter.

        Action:
        Call sensor.read_for(duration_ms=0) with invalid duration.
        Validation rejects before calculating sample count.

        Assertion Strategy:
        - Raises ValueError with "duration_ms must be >= 1" message.
        - Validates duration constraint at API boundary.

        Testing Principle:
        Validates input validation prevents impossible time-based reads.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        with pytest.raises(ValueError, match="duration_ms must be >= 1"):
            await sensor.read_for(duration_ms=0)

        await sensor.disconnect()


class TestSensorReading:
    """Test suite for SensorReading dataclass.

    Categories:
    1. Structure Tests - Field presence, types

    Total: 1 test.
    """

    def test_sensor_reading_fields(self) -> None:
        """Verifies SensorReading has all required fields with correct types.

        Tests dataclass field definition and initialization.

        Business context:
        SensorReading is the core data contract for sensor data. All
        consuming code depends on these fields existing with correct types.

        Arrangement:
            Create SensorReading with all required fields.

        Action:
            Access fields to verify presence and values.

        Assertion Strategy:
            Verify altitude, azimuth, and accelerometer fields accessible.

        Testing Principle:
            Tests dataclass structure for API contract compliance.
        """
        reading = SensorReading(
            accelerometer={"aX": 0.0, "aY": 0.0, "aZ": 1.0},
            magnetometer={"mX": 30.0, "mY": 0.0, "mZ": 40.0},
            altitude=45.0,
            azimuth=180.0,
            temperature=20.0,
            humidity=50.0,
            timestamp=datetime.now(UTC),
            raw_values="0.0\t0.0\t1.0\t30.0\t0.0\t40.0\t20.0\t50.0",
        )

        assert reading.altitude == 45.0
        assert reading.azimuth == 180.0
        assert reading.accelerometer["aZ"] == 1.0

    def test_sensor_reading_str(self) -> None:
        """Verifies SensorReading.__str__ returns human-readable format.

        Tests F3 code review fix: __str__ returns string representation.

        Business context:
        Human-readable output for logging and debugging.
        Format: "ALT 45.00° AZ 180.00° | T=20.0°C H=50.0%"

        Arrangement:
        Create SensorReading with known ALT/AZ and environmental values.

        Action:
        Call __str__() to get formatted string representation.

        Assertion Strategy:
        - String contains altitude (45.00).
        - String contains azimuth (180.00).
        - String contains temperature and humidity.
        - Format is human-readable for logs.

        Testing Principle:
        Validates string representation provides useful debugging output.
        """
        reading = SensorReading(
            accelerometer={"aX": 0.0, "aY": 0.0, "aZ": 1.0},
            magnetometer={"mX": 30.0, "mY": 0.0, "mZ": 40.0},
            altitude=45.0,
            azimuth=180.0,
            temperature=20.5,
            humidity=45.0,
            timestamp=datetime.now(UTC),
            raw_values="test",
        )

        result = str(reading)

        assert isinstance(result, str)
        assert "45.00°" in result
        assert "180.00°" in result
        assert "T=20.5°C" in result
        assert "H=45.0%" in result

    @pytest.mark.asyncio
    async def test_averaging_preserves_timestamp(self) -> None:
        """Verifies averaged reading uses timestamp from last sample.

        Tests timestamp behavior during averaging.

        Business context:
        When averaging, the timestamp should reflect when the
        averaging completed (last sample timestamp).

        Arrangement:
            Create driver with zero noise, connect Sensor.

        Action:
            Read with samples=3 to trigger averaging.

        Assertion Strategy:
            Verify reading.timestamp is valid datetime instance.

        Testing Principle:
            Tests timestamp preservation during multi-sample averaging.
        """
        config = DigitalTwinSensorConfig(
            initial_altitude=45.0,
            initial_azimuth=180.0,
            noise_std_alt=0.0,
            noise_std_az=0.0,
        )
        driver = DigitalTwinSensorDriver(config)
        sensor = Sensor(driver)
        await sensor.connect()

        reading = await sensor.read(samples=3)

        # Timestamp should be a valid datetime
        assert isinstance(reading.timestamp, datetime)

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_averaging_combines_raw_values(self) -> None:
        """Verifies averaged reading has combined raw_values string.

        Tests raw_values concatenation during averaging.

        Business context:
        For debugging, raw_values should indicate multiple samples
        were taken (though exact format may vary).

        Arrangement:
            Connect Sensor instance.

        Action:
            Read with samples=2 to trigger averaging.

        Assertion Strategy:
            Verify reading.raw_values is non-empty string.

        Testing Principle:
            Tests raw value preservation during averaging for debugging.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        await sensor.connect()

        reading = await sensor.read(samples=2)

        # Raw values should exist
        assert reading.raw_values

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_single_sample_no_averaging(self) -> None:
        """Verifies samples=1 returns reading directly without averaging.

        Tests optimization path for single-sample reads.

        Business context:
        Single sample should be returned as-is without overhead
        of averaging calculation.

        Arrangement:
            Create driver with zero noise, connect Sensor.

        Action:
            Read with samples=1 (no averaging).

        Assertion Strategy:
            Verify reading matches exact configured value (no noise).

        Testing Principle:
            Tests single-sample optimization path bypasses averaging.
        """
        config = DigitalTwinSensorConfig(
            initial_altitude=45.0,
            initial_azimuth=180.0,
            noise_std_alt=0.0,
            noise_std_az=0.0,
        )
        driver = DigitalTwinSensorDriver(config)
        sensor = Sensor(driver)
        await sensor.connect()

        reading = await sensor.read(samples=1)

        # With no noise, should be exact
        assert abs(reading.altitude - 45.0) < 0.01
        assert abs(reading.azimuth - 180.0) < 0.01

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_azimuth_averaging_handles_wraparound(self) -> None:
        """Verifies azimuth averaging handles 0°/360° wraparound correctly.

        Tests F2 fix: circular mean for azimuth averaging.

        Business context:
        Averaging 350° and 10° should give ~0° (north), not 180° (south).
        This is critical for accurate pointing near north.

        Arrangement:
            Create mock readings straddling 0°/360° boundary (350° and 10°).

        Action:
            Call _average_readings() to compute circular mean.

        Assertion Strategy:
            Verify averaged azimuth near 0° (circular mean), not 180° (arithmetic).

        Testing Principle:
            Tests circular averaging for compass heading accuracy.
        """
        from telescope_mcp.drivers.sensors.types import SensorReading

        # Create mock readings that straddle 0°/360°
        mock_readings = [
            SensorReading(
                accelerometer={"aX": 0.0, "aY": 0.0, "aZ": 1.0},
                magnetometer={"mX": 30.0, "mY": 0.0, "mZ": 40.0},
                altitude=45.0,
                azimuth=350.0,  # 10° west of north
                temperature=20.0,
                humidity=50.0,
                timestamp=datetime.now(UTC),
                raw_values="test",
            ),
            SensorReading(
                accelerometer={"aX": 0.0, "aY": 0.0, "aZ": 1.0},
                magnetometer={"mX": 30.0, "mY": 0.0, "mZ": 40.0},
                altitude=45.0,
                azimuth=10.0,  # 10° east of north
                temperature=20.0,
                humidity=50.0,
                timestamp=datetime.now(UTC),
                raw_values="test",
            ),
        ]

        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        # Directly call _average_readings
        result = sensor._average_readings(mock_readings)

        # Average of 350° and 10° should be ~0° (north), not 180°
        # Allow small tolerance for floating point
        assert (
            result.azimuth < 5.0 or result.azimuth > 355.0
        ), f"Expected azimuth near 0°, got {result.azimuth}°"


class TestSensorSampleRate:
    """Test suite for sample rate functionality.

    Tests sample rate query and fallback behavior.

    Total: 4 tests.
    """

    @pytest.mark.asyncio
    async def test_get_sample_rate_protocol_method(self) -> None:
        """Verifies get_sample_rate() protocol method works correctly.

        Tests F1 code review fix: SensorInstance protocol get_sample_rate().

        Business context:
        Device layer can query sample rate from driver instance using
        the protocol method instead of accessing internal attributes.

        Arrangement:
            Create driver with custom sample rate (15 Hz), open instance.

        Action:
            Call instance.get_sample_rate() protocol method.

        Assertion Strategy:
            Verify returned rate matches configured value (15.0).

        Testing Principle:
            Tests protocol method for sample rate discovery.
        """
        config = DigitalTwinSensorConfig(sample_rate_hz=15.0)
        driver = DigitalTwinSensorDriver(config)
        instance = driver.open()

        # Protocol method should return configured rate
        rate = instance.get_sample_rate()

        assert rate == 15.0
        driver.close()

    @pytest.mark.asyncio
    async def test_sample_rate_from_driver_config(self) -> None:
        """Verifies sample rate is extracted from driver config.

        Tests sample rate query during connect.

        Business context:
        DigitalTwin drivers expose sample rate via _config.sample_rate_hz.
        This is used for read_for() duration calculation.

        Arrangement:
            Create driver with custom sample rate (20 Hz).

        Action:
            Connect sensor and check sample_rate_hz property.

        Assertion Strategy:
            Verify sensor.sample_rate_hz equals configured 20.0.

        Testing Principle:
            Tests sample rate extraction from driver configuration.
        """
        config = DigitalTwinSensorConfig(sample_rate_hz=20.0)
        driver = DigitalTwinSensorDriver(config)
        sensor = Sensor(driver)
        await sensor.connect()

        assert sensor.sample_rate_hz == 20.0

        await sensor.disconnect()

    @pytest.mark.asyncio
    async def test_sample_rate_fallback_default(self) -> None:
        """Verifies sample rate falls back to default when not in status.

        Tests fallback behavior when driver doesn't report rate.

        Business context:
        Some drivers may not report sample rate. Default to 10 Hz
        (common Arduino rate) as reasonable fallback.

        Arrangement:
            Create mock driver that doesn't provide get_sample_rate().

        Action:
            Connect sensor and check sample_rate_hz property.

        Assertion Strategy:
            Verify sensor.sample_rate_hz falls back to default 10.0.

        Testing Principle:
            Tests default fallback for missing sample rate info.
        """
        mock_instance = Mock(spec=[])  # Empty spec = no methods by default
        mock_instance.get_info = Mock(return_value={"type": "mock", "name": "Mock"})
        mock_instance.get_status = Mock(return_value={"connected": True})
        # Don't add get_sample_rate or _send_command

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        await sensor.connect()

        # Should fall back to default (10 Hz)
        assert sensor.sample_rate_hz == 10.0

    @pytest.mark.asyncio
    async def test_read_for_uses_sample_rate(self) -> None:
        """Verifies read_for() converts duration using sample_rate_hz.

        Tests duration to sample count conversion.

        Business context:
        read_for(500) with 10 Hz rate should read 5 samples.

        Arrangement:
        1. Mock instance with sample_rate_hz=10.0.
        2. Track read() call count with nested function.
        3. Connect sensor to establish sample rate.

        Action:
        Call read_for(duration_ms=500).
        Should convert to 5 samples (500ms / 100ms per sample).

        Assertion Strategy:
        - read() called exactly 5 times.
        - Validates duration → sample count math.

        Testing Principle:
        Validates time-based reading uses correct sample rate
        for accurate duration conversion.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.return_value = {"sample_rate_hz": 10.0}
        mock_instance.get_sample_rate.return_value = 10.0  # Protocol method

        # Track read calls to verify sample count
        read_count = [0]

        def mock_read() -> SensorReading:
            """Mock read function that increments counter and returns sample reading.

            Tracks read() call count while providing valid SensorReading data.
            Used to verify read_for() calls read() correct number of times.

            Args:
                None - closure accesses read_count from enclosing scope.

            Returns:
                SensorReading: Complete sensor reading with default test values.
                    altitude=45.0, azimuth=180.0, temperature=20.0, humidity=50.0.
                    Accelerometer and magnetometer populated with test vectors.

            Raises:
                None - always returns successfully with mock data.

            Example:
                >>> mock_instance.read = mock_read
                >>> await sensor.read_for(duration_ms=500)
                >>> assert read_count[0] == 5  # Verified 5 calls

            Business Context:
                Duration-based reads (read_for) must convert milliseconds to sample
                count using sample rate. This mock verifies the calculation is
                correct by counting actual read() invocations.

            Implementation Details:
                Uses closure over read_count list to maintain call counter across
                invocations. Returns fixed SensorReading to simplify assertions.
            """
            read_count[0] += 1
            return SensorReading(
                accelerometer={"aX": 0.0, "aY": 0.0, "aZ": 1.0},
                magnetometer={"mX": 30.0, "mY": 0.0, "mZ": 40.0},
                altitude=45.0,
                azimuth=180.0,
                temperature=20.0,
                humidity=50.0,
                timestamp=datetime.now(UTC),
                raw_values="test",
            )

        mock_instance.read = mock_read

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        await sensor.connect()

        # 500ms at 10 Hz = 5 samples
        await sensor.read_for(duration_ms=500)

        assert read_count[0] == 5


class TestSensorCoverageGaps:
    """Tests for coverage gaps in sensor.py.

    Covers edge cases and rarely-executed paths.
    """

    @pytest.mark.asyncio
    async def test_query_sample_rate_with_none_instance(self) -> None:
        """Verifies _query_sample_rate returns early when instance is None.

        Tests the guard clause at start of _query_sample_rate.

        Arrangement:
        1. Create sensor without connecting.
        2. Instance is None (no driver instance created).

        Action:
        Call _query_sample_rate() on disconnected sensor.
        Guard clause should return early.

        Assertion Strategy:
        - No error raised (safe guard clause).
        - Validates None check prevents attribute access errors.

        Testing Principle:
        Validates defensive programming pattern prevents errors
        when called in unexpected state.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        # Instance is None before connect
        assert sensor._instance is None
        await sensor._query_sample_rate()  # Should return early without error

    @pytest.mark.asyncio
    async def test_query_sample_rate_no_config_attribute(self) -> None:
        """Verifies fallback when instance has neither _send_command nor _config.

        Tests the else branch in _query_sample_rate.

        Arrangement:
        1. Mock instance without _send_command or get_sample_rate methods.
        2. Manually set connected state (bypass normal connect).
        3. No sample rate discovery methods available.

        Action:
        Call _query_sample_rate() with minimal instance.
        Falls back to default when no query methods exist.

        Assertion Strategy:
        - sample_rate_hz set to 10.0 (default).
        - Validates fallback path for simple drivers.

        Testing Principle:
        Validates graceful degradation when driver lacks
        sample rate reporting capabilities.
        """
        # Create mock excluding _send_command and get_sample_rate
        mock_instance = Mock(spec=[])  # Empty spec = no methods by default
        mock_instance.get_info = Mock(return_value={"type": "mock", "name": "Mock"})
        # Don't add _send_command or get_sample_rate

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor._instance = mock_instance
        sensor._connected = True

        await sensor._query_sample_rate()

        # Should fall back to default (no _send_command or get_sample_rate)
        assert sensor._sample_rate_hz == 10.0

    @pytest.mark.asyncio
    async def test_get_status_with_is_open_field(self) -> None:
        """Verifies get_status includes is_open from driver status.

        Tests the is_open branch in get_status.

        Arrangement:
        1. Mock instance returning status with is_open field.
        2. Connect sensor to populate status.

        Action:
        Call get_status() to retrieve merged status.
        is_open field should pass through from driver.

        Assertion Strategy:
        - Status contains is_open=True.
        - Validates driver field passthrough.

        Testing Principle:
        Validates status merging preserves driver-specific fields.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.return_value = {
            "calibrated": False,
            "is_open": True,
        }

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        await sensor.connect()

        status = sensor.get_status()
        assert status["is_open"] is True

    @pytest.mark.asyncio
    async def test_get_status_with_error_field(self) -> None:
        """Verifies get_status includes error from driver status.

        Tests the error branch in get_status.

        Arrangement:
        1. Mock instance returning status with error field.
        2. Connect sensor to populate status.

        Action:
        Call get_status() to retrieve merged status.
        Error field should pass through from driver.

        Assertion Strategy:
        - Status contains error message.
        - Validates error reporting mechanism.

        Testing Principle:
        Validates status merging preserves driver error information.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.return_value = {
            "calibrated": False,
            "error": "Sensor overheating",
        }

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        await sensor.connect()

        status = sensor.get_status()
        assert status["error"] == "Sensor overheating"

    @pytest.mark.asyncio
    async def test_context_manager_when_already_connected(self) -> None:
        """Verifies __aenter__ skips connect when already connected.

        Tests the 'if not self._connected' branch in __aenter__.

        Arrangement:
        1. Create and connect sensor before context manager.
        2. Sensor already in connected state.

        Action:
        Enter context manager with pre-connected sensor.
        __aenter__ should detect existing connection and skip connect.
        __aexit__ should still disconnect.

        Assertion Strategy:
        - Sensor remains connected during context.
        - Sensor disconnected after context exit.
        - Validates idempotent connect behavior.

        Testing Principle:
        Validates context manager handles already-connected state
        gracefully without double-connect.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        # Pre-connect before context manager
        await sensor.connect()
        assert sensor.connected

        # Context manager should not reconnect
        async with sensor:
            assert sensor.connected

        assert not sensor.connected

    @pytest.mark.asyncio
    async def test_average_readings_empty_list_raises(self) -> None:
        """Verifies _average_readings raises on empty list.

        Tests the n == 0 branch.

        Arrangement:
        Create sensor instance.

        Action:
        Call _average_readings([]) with empty list.

        Assertion Strategy:
        - Raises ValueError with "No readings" message.
        - Validates precondition check.

        Testing Principle:
        Validates input validation prevents division by zero.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with pytest.raises(ValueError, match="No readings to average"):
            sensor._average_readings([])

    @pytest.mark.asyncio
    async def test_average_readings_single_item_returns_directly(self) -> None:
        """Verifies _average_readings returns single item directly.

        Tests the n == 1 optimization path.

        Arrangement:
        1. Create sensor instance.
        2. Create single SensorReading.

        Action:
        Call _average_readings([reading]) with one-item list.
        Should return input directly (no averaging needed).

        Assertion Strategy:
        - Returns same object instance (no copy).
        - Validates optimization path.

        Testing Principle:
        Validates performance optimization avoids unnecessary
        averaging computation for single reading.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        reading = SensorReading(
            accelerometer={"aX": 1.0, "aY": 2.0, "aZ": 3.0},
            magnetometer={"mX": 10.0, "mY": 20.0, "mZ": 30.0},
            altitude=45.0,
            azimuth=180.0,
            temperature=20.0,
            humidity=50.0,
            timestamp=datetime.now(UTC),
            raw_values="test",
        )

        result = sensor._average_readings([reading])

        # Should return the same object
        assert result is reading

    @pytest.mark.asyncio
    async def test_query_sample_rate_with_send_command(self) -> None:
        """Verifies sample rate query via _send_command (Arduino path).

        Tests the Arduino driver code path that uses STATUS command.
        Only triggered when get_sample_rate() is not available.

        Arrangement:
            Create mock instance with _send_command but no get_sample_rate().

        Action:
            Call _query_sample_rate() to trigger STATUS command path.

        Assertion Strategy:
            Verify sample_rate_hz parsed from STATUS response (15.0).

        Testing Principle:
            Tests Arduino-specific sample rate discovery via serial command.
        """
        mock_instance = Mock(spec=[])  # Empty spec = no methods by default
        mock_instance.get_info = Mock(
            return_value={"type": "arduino", "name": "Arduino"}
        )
        mock_instance._send_command = Mock(
            return_value="Status: OK\nSample Rate: 15 Hz\nCalibrated: No"
        )
        # Don't add get_sample_rate - testing _send_command fallback path

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor._instance = mock_instance
        sensor._connected = True

        await sensor._query_sample_rate()

        # Should parse 15 Hz from STATUS response
        assert sensor._sample_rate_hz == 15.0
        mock_instance._send_command.assert_any_call("STOP", wait_response=False)
        mock_instance._send_command.assert_any_call("STATUS", timeout=3.0)

    @pytest.mark.asyncio
    async def test_query_sample_rate_no_match_in_status(self) -> None:
        """Verifies fallback when STATUS response doesn't contain sample rate.

        Tests when regex doesn't match in _send_command path.

        Arrangement:
            Create mock instance with _send_command returning STATUS without rate.

        Action:
            Call _query_sample_rate() with non-matching response.

        Assertion Strategy:
            Verify sample_rate_hz remains unchanged (no match, no update).

        Testing Principle:
            Tests graceful handling of missing sample rate in STATUS response.
        """
        mock_instance = Mock(spec=[])  # Empty spec = no methods by default
        mock_instance.get_info = Mock(
            return_value={"type": "arduino", "name": "Arduino"}
        )
        mock_instance._send_command = Mock(
            return_value="Status: OK\nCalibrated: No"
        )  # No sample rate
        # Don't add get_sample_rate - testing _send_command fallback path

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor._instance = mock_instance
        sensor._connected = True
        sensor._sample_rate_hz = 99.0  # Set to non-default to verify it stays unchanged

        await sensor._query_sample_rate()

        # Sample rate should remain unchanged (no match found)
        assert sensor._sample_rate_hz == 99.0

    @pytest.mark.asyncio
    async def test_query_sample_rate_else_branch_no_methods(self) -> None:
        """Verifies default sample rate when instance lacks both methods.

        Tests lines 260-261: the else branch when instance has neither
        get_sample_rate nor _send_command methods.

        Arrangement:
        1. Create mock instance without get_sample_rate or _send_command.
        2. Use spec_set to ensure hasattr returns False.
        3. Inject mock directly into sensor.

        Action:
        Call _query_sample_rate().

        Assertion Strategy:
        Validates fallback to DEFAULT_SAMPLE_RATE_HZ (10.0) when no
        sample rate query method is available.
        """

        class BareInstance:
            """Instance with minimal interface - no sample rate methods."""

            def get_info(self) -> dict[str, str]:
                """Return minimal sensor info for bare instance testing.

                Provides type and name fields without sample rate capabilities.

                Args:
                    None - no parameters required for static info.

                Returns:
                    dict[str, str]: Sensor metadata with 'type' and 'name' keys.
                        type='bare', name='Bare' for minimal test instance.

                Raises:
                    None - always returns successfully with hardcoded data.

                Example:
                    >>> instance = BareInstance()
                    >>> info = instance.get_info()
                    >>> assert info['type'] == 'bare'

                Business Context:
                    Sensors must provide identification even when sample rate
                    query methods are unavailable. Bare instance simulates
                    minimal sensor implementation.

                Implementation Details:
                    Returns hardcoded dict. No dependencies on configuration
                    or state. Used to test fallback behavior when instance
                    lacks get_sample_rate() and _send_command() methods.
                """
                return {"type": "bare", "name": "Bare"}

            def read(self) -> SensorReading:
                """Return mock sensor reading with default/zero values.

                Provides minimal valid reading for testing without actual sensor.

                Args:
                    None - no parameters for stateless mock reading.

                Returns:
                    SensorReading: Complete reading with default/zero values.
                        altitude=0.0, azimuth=0.0 for horizon north position.
                        temperature=20.0°C, humidity=50.0% for standard conditions.
                        Accelerometer shows 1g vertical (aZ=1.0).
                        Magnetometer all zeros (uncalibrated simulation).

                Raises:
                    None - always returns successfully with mock data.

                Example:
                    >>> instance = BareInstance()
                    >>> reading = instance.read()
                    >>> assert reading.altitude == 0.0
                    >>> assert reading.azimuth == 0.0

                Business Context:
                    Testing sensor fallback behavior requires instance that
                    can provide readings without sample rate capabilities.
                    Simulates minimal sensor implementation.

                Implementation Details:
                    Returns new SensorReading on each call with current timestamp.
                    No state maintained between reads. Raw values empty string
                    since no actual hardware communication occurred.
                """
                return SensorReading(
                    accelerometer={"aX": 0.0, "aY": 0.0, "aZ": 1.0},
                    magnetometer={"mX": 0.0, "mY": 0.0, "mZ": 0.0},
                    altitude=0.0,
                    azimuth=0.0,
                    temperature=20.0,
                    humidity=50.0,
                    timestamp=datetime.now(UTC),
                    raw_values="",
                )

            def get_status(self) -> dict[str, object]:
                """Return minimal status without sample rate info.

                Returns calibration status only, no sample rate fields.

                Args:
                    None - no parameters for stateless status query.

                Returns:
                    dict[str, object]: Status dict with calibrated=False.
                        Deliberately omits sample_rate_hz to test fallback.

                Raises:
                    None - always returns successfully with minimal status.

                Example:
                    >>> instance = BareInstance()
                    >>> status = instance.get_status()
                    >>> assert 'sample_rate_hz' not in status
                    >>> assert status['calibrated'] is False

                Business Context:
                    Not all sensor drivers expose sample rate via status dict.
                    Tests must verify graceful fallback to default when
                    sample rate information unavailable.

                Implementation Details:
                    Returns minimal status to trigger fallback logic in
                    _query_sample_rate(). Used alongside BareInstance
                    lacking get_sample_rate() method to test else branch.
                """
                return {"calibrated": False}

            def close(self) -> None:
                """No-op close for mock instance cleanup.

                Bare instance has no resources to release.

                Args:
                    None - no parameters for cleanup operation.

                Returns:
                    None - cleanup operation has no return value.

                Raises:
                    None - safe no-op always succeeds.

                Example:
                    >>> instance = BareInstance()
                    >>> instance.close()  # Safe to call, no effect

                Business Context:
                    Sensor instances must implement close() for resource cleanup
                    protocol compliance. Bare instance simulates minimal sensor
                    without hardware resources.

                Implementation Details:
                    Pass statement - no cleanup needed. Satisfies SensorInstance
                    protocol requirement for close() method. Used in test to
                    verify sample rate query works with minimal implementations.
                """
                pass

        bare_instance = BareInstance()
        mock_driver = Mock()
        mock_driver.open.return_value = bare_instance

        sensor = Sensor(mock_driver)
        sensor._instance = bare_instance
        sensor._connected = True
        sensor._sample_rate_hz = 99.0  # Non-default to verify it gets changed

        # Verify methods are NOT present
        assert not hasattr(bare_instance, "get_sample_rate")
        assert not hasattr(bare_instance, "_send_command")

        await sensor._query_sample_rate()

        # Should fall back to DEFAULT_SAMPLE_RATE_HZ
        from telescope_mcp.devices.sensor import DEFAULT_SAMPLE_RATE_HZ

        assert sensor._sample_rate_hz == DEFAULT_SAMPLE_RATE_HZ

    @pytest.mark.asyncio
    async def test_query_sample_rate_exception_handler(self) -> None:
        """Verifies exception handling in _query_sample_rate.

        Tests lines 262-264: the except branch when get_sample_rate raises.

        Arrangement:
        1. Create mock instance where get_sample_rate raises exception.
        2. Inject mock directly into sensor.

        Action:
        Call _query_sample_rate().

        Assertion Strategy:
        Validates fallback to DEFAULT_SAMPLE_RATE_HZ when exception occurs
        during sample rate query.
        """
        mock_instance = Mock()
        mock_instance.get_sample_rate.side_effect = RuntimeError("Hardware error")

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor._instance = mock_instance
        sensor._connected = True
        sensor._sample_rate_hz = 99.0  # Non-default to verify it gets changed

        await sensor._query_sample_rate()

        # Should fall back to DEFAULT_SAMPLE_RATE_HZ after exception
        from telescope_mcp.devices.sensor import DEFAULT_SAMPLE_RATE_HZ

        assert sensor._sample_rate_hz == DEFAULT_SAMPLE_RATE_HZ

    @pytest.mark.asyncio
    async def test_get_status_exception_sets_status_error(self) -> None:
        """Verifies get_status handles driver exception gracefully.

        Tests branch 452->454: when get_status() raises an exception,
        the error is captured in status_error field.

        Arrangement:
        1. Create mock instance where get_status raises exception.
        2. Connect sensor with mock.

        Action:
        Call sensor.get_status().

        Assertion Strategy:
        Validates status_error field contains exception message when
        driver's get_status raises.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.side_effect = RuntimeError(
            "Hardware communication error"
        )

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        await sensor.connect()

        status = sensor.get_status()

        # Should have status_error field with exception message
        assert "status_error" in status
        assert "Hardware communication error" in status["status_error"]
        # Should still have basic fields
        assert status["connected"] is True
        assert status["type"] == "mock"

    @pytest.mark.asyncio
    async def test_get_status_calibrated_none_skipped(self) -> None:
        """Verifies get_status skips calibrated when None in driver status.

        Tests branch 452->454: when calibrated is None, skip to is_open check.

        Arrangement:
        1. Create mock instance where get_status returns no calibrated key.
        2. Include is_open to verify execution continues.

        Action:
        Call sensor.get_status().

        Assertion Strategy:
        Validates calibrated is NOT in status when driver returns None,
        but is_open is still included.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        # Return dict WITHOUT calibrated but WITH is_open
        mock_instance.get_status.return_value = {
            "is_open": True,
            "error": None,  # Explicitly None, not missing
        }

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        await sensor.connect()

        status = sensor.get_status()

        # calibrated should NOT be in status (was None from driver)
        assert "calibrated" not in status
        # is_open should be included
        assert status["is_open"] is True
        # error was None, should NOT be included
        assert "error" not in status
