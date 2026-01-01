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
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.return_value = {"sample_rate_hz": 10.0}
        mock_instance.get_sample_rate.return_value = 10.0  # Protocol method

        # Track read calls to verify sample count
        read_count = [0]

        def mock_read() -> SensorReading:
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
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with pytest.raises(ValueError, match="No readings to average"):
            sensor._average_readings([])

    @pytest.mark.asyncio
    async def test_average_readings_single_item_returns_directly(self) -> None:
        """Verifies _average_readings returns single item directly.

        Tests the n == 1 optimization path.
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
                return {"type": "bare", "name": "Bare"}

            def read(self) -> SensorReading:
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
                return {"calibrated": False}

            def close(self) -> None:
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
