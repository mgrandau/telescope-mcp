"""Tests for Sensor device and drivers.

Tests the high-level Sensor abstraction and DigitalTwin driver.
Arduino driver tests require hardware and are integration tests.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from telescope_mcp.devices.sensor import DeviceSensorInfo, Sensor, SensorConfig
from telescope_mcp.drivers.sensors import (
    DigitalTwinSensorConfig,
    DigitalTwinSensorDriver,
    SensorReading,
)


class TestDigitalTwinSensorDriver:
    """Test suite for DigitalTwinSensorDriver functionality.

    Categories:
    1. Discovery Tests - Sensor enumeration, availability (1 test)
    2. Lifecycle Tests - Open, close, instance management (2 tests)
    3. Configuration Tests - Custom config, behavior modification (1 test)

    Total: 4 tests.
    """

    def test_get_available_sensors(self) -> None:
        """Verifies driver discovers exactly one simulated sensor.

        Tests sensor enumeration by querying the driver for available sensors.

        Business context:
        Before connecting, applications need to discover what sensors exist.
        The digital twin always provides exactly one simulated sensor for testing.

        Arrangement:
        1. Create DigitalTwinSensorDriver with default configuration.

        Action:
        Call get_available_sensors() to enumerate discoverable sensors.

        Assertion Strategy:
        Validates discovery by confirming:
        - Exactly one sensor returned (digital twin is single-instance).
        - Sensor type is 'digital_twin' for proper identification.
        - Name field exists for UI display purposes.

        Testing Principle:
        Validates sensor discovery contract, ensuring applications can
        enumerate and identify available sensors before connection.
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
        instance that can read sensor data. Essential for telescope pointing.

        Arrangement:
        1. Create DigitalTwinSensorDriver with default configuration.

        Action:
        Call open() to create and return a sensor instance.

        Assertion Strategy:
        Validates instance creation by confirming:
        - Instance is not None (creation succeeded).
        - Instance info reports correct type for identification.

        Testing Principle:
        Validates factory pattern contract, ensuring drivers produce
        properly typed, functional sensor instances.
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
        in serial port access or thread management. Drivers enforce single
        active instance to prevent undefined behavior.

        Arrangement:
        1. Create DigitalTwinSensorDriver.
        2. Open driver to create first instance.

        Action:
        Attempt to open driver again while first instance is active.

        Assertion Strategy:
        Validates enforcement by confirming:
        - RuntimeError raised with 'already open' message.
        - Pattern matching ensures correct error type.

        Testing Principle:
        Validates resource protection, ensuring drivers prevent
        accidental double-open that could corrupt state.
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
        Cleanup code often calls close() unconditionally in finally blocks
        or teardown methods. Drivers must handle close() gracefully when
        no instance was ever opened, avoiding exceptions during cleanup.

        Arrangement:
        1. Create DigitalTwinSensorDriver without opening.

        Action:
        Call close() on driver that was never opened.

        Assertion Strategy:
        Validates safe close by confirming:
        - No exception raised.
        - Driver remains in valid state for potential future open().

        Testing Principle:
        Validates defensive programming, ensuring drivers handle
        edge cases gracefully without requiring caller state tracking.
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
        position and environmental values, enabling reproducible tests
        for calibration algorithms and pointing verification.

        Arrangement:
        1. Create config with specific values: alt=30°, az=90°, temp=15°C.
        2. Create driver with custom configuration.
        3. Open driver to create configured instance.

        Action:
        Read sensor data to verify configuration was applied.

        Assertion Strategy:
        Validates configuration by confirming readings within tolerance:
        - Altitude: 30° ±5° (accounts for default noise).
        - Azimuth: 90° ±5° (accounts for default noise).
        - Temperature: 15°C ±1°C (environmental readings).
        - Humidity: 60% ±2% (environmental readings).

        Testing Principle:
        Validates dependency injection, ensuring configuration
        propagates correctly through driver to instance behavior.
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
    1. Reading Tests - Data retrieval, format validation (1 test)
    2. Position Tests - Position control for testing (1 test)
    3. Calibration Tests - Offset calibration, magnetometer (3 tests)
    4. Status Tests - State reporting, lifecycle (2 tests)

    Total: 7 tests.
    """

    def test_read_returns_sensor_reading(self) -> None:
        """Verifies read() returns complete SensorReading with all fields.

        Tests data contract for sensor reading structure.

        Business context:
        Telescope pointing requires accelerometer (tilt), magnetometer
        (heading), and environmental data. All fields must be present
        for accurate position calculation and calibration.

        Arrangement:
        1. Create and open DigitalTwinSensorDriver.

        Action:
        Call read() to get current sensor data.

        Assertion Strategy:
        Validates data completeness by confirming:
        - Accelerometer dict with aX, aY, aZ axes.
        - Magnetometer dict with mX, mY, mZ axes.
        - Altitude and azimuth as floats for position.
        - Temperature and humidity as floats for environment.
        - Timestamp as datetime for temporal tracking.
        - Raw values string for debugging/logging.

        Testing Principle:
        Validates data contract, ensuring all consumers receive
        complete sensor data for position and environment tracking.
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
        method sets the "true" position that the digital twin
        reports, enabling tests for calibration and tracking.

        Arrangement:
        1. Create and open DigitalTwinSensorDriver.

        Action:
        Set position to alt=60°, az=270° (pointing west, 60° elevation).
        Then read to verify position update.

        Assertion Strategy:
        Validates position update by confirming readings within noise:
        - Altitude: 60° ±5° (default noise tolerance).
        - Azimuth: 270° ±5° (default noise tolerance).

        Testing Principle:
        Validates test controllability, ensuring simulated position
        can be set for reproducible integration test scenarios.
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
        a known reference (plate-solved star) computes offsets so readings
        match actual telescope position. Critical for Go-To accuracy.

        Arrangement:
        1. Configure twin with known position (40°, 180°) and no noise.
        2. Zero noise ensures deterministic offset calculation.

        Action:
        Calibrate to "true" position (45°, 200°), then read.

        Assertion Strategy:
        Validates calibration by confirming readings match true position:
        - Altitude within 1° of calibration target.
        - Azimuth within 1° of calibration target.

        Testing Principle:
        Validates calibration mathematics, ensuring offset model
        correctly transforms raw readings to calibrated values.
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
        """Verifies reset() clears calibration offsets.

        Tests state reset for sensor re-initialization.

        Business context:
        When re-mounting sensors or starting new observation sessions,
        old calibration becomes invalid. Reset clears offsets so fresh
        calibration can be applied against new reference position.

        Arrangement:
        1. Configure twin at (45°, 180°) with no noise.
        2. Apply calibration to (90°, 0°) creating large offsets.

        Action:
        Call reset() then read to verify offsets cleared.

        Assertion Strategy:
        Validates reset by confirming readings return to initial values:
        - Altitude within 1° of initial 45°.
        - Azimuth within 1° of initial 180°.

        Testing Principle:
        Validates state management, ensuring reset provides clean
        slate for recalibration after hardware changes.
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
        connection state, calibration state, and open state
        for monitoring dashboards and health checks.

        Arrangement:
        1. Create and open DigitalTwinSensorDriver.

        Action:
        Call get_status() to retrieve current sensor state.

        Assertion Strategy:
        Validates status per SensorStatus protocol by confirming:
        - Connected is True (instance is open).
        - Calibrated is False (no calibration applied yet).
        - is_open is True.

        Testing Principle:
        Validates observability, ensuring sensor state is
        inspectable for monitoring and debugging.
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
        instance would access invalid state. Error prevents undefined
        behavior and clarifies the closed state to callers.

        Arrangement:
        1. Create and open DigitalTwinSensorDriver.
        2. Close the instance to release resources.

        Action:
        Attempt to read() from the closed instance.

        Assertion Strategy:
        Validates enforcement by confirming:
        - RuntimeError raised with 'closed' in message.
        - Pattern matching ensures correct error context.

        Testing Principle:
        Validates lifecycle management, ensuring closed resources
        cannot be used and provide clear error messages.
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
        materials. Calibration computes offsets to center magnetic
        readings. The digital twin simulates this for testing workflows.

        Arrangement:
        1. Create and open DigitalTwinSensorDriver.

        Action:
        Call calibrate_magnetometer() to run simulated calibration.

        Assertion Strategy:
        Validates calibration output by confirming:
        - Result contains offset_x for X-axis correction.
        - Result contains offset_y for Y-axis correction.
        - Result contains offset_z for Z-axis correction.

        Testing Principle:
        Validates calibration interface, ensuring magnetometer
        calibration returns expected offset structure.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        result = instance.calibrate_magnetometer()

        assert "offset_x" in result
        assert "offset_y" in result
        assert "offset_z" in result

        driver.close()


class TestSensorDevice:
    """Test suite for high-level Sensor device wrapper.

    Categories:
    1. Lifecycle Tests - Connect, disconnect, auto-connect (4 tests)
    2. Reading Tests - Data retrieval, caching (3 tests)
    3. Calibration Tests - Input validation, state management (4 tests)
    4. Status Tests - Statistics, info, status reporting (3 tests)
    5. Interface Tests - Context manager, repr, discovery (3 tests)

    Total: 17 tests.
    """

    def test_connect_and_disconnect(self) -> None:
        """Verifies basic connect and disconnect lifecycle workflow.

        Tests high-level Sensor device connection management.

        Business context:
        The Sensor class wraps low-level drivers for application use.
        Connect/disconnect manage the underlying driver instance and
        track connection state for the application layer.

        Arrangement:
        1. Create DigitalTwinSensorDriver.
        2. Create Sensor wrapper around driver.

        Action:
        Connect sensor, verify state, then disconnect and verify cleanup.

        Assertion Strategy:
        Validates lifecycle by confirming:
        - Initially not connected.
        - After connect(): connected=True, info populated.
        - After disconnect(): connected=False, info=None.

        Testing Principle:
        Validates state machine, ensuring connect/disconnect
        transitions are correct and reversible.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert not sensor.connected

        sensor.connect()
        assert sensor.connected
        assert sensor.info is not None

        sensor.disconnect()
        assert not sensor.connected
        assert sensor.info is None

    def test_auto_connect(self) -> None:
        """Verifies auto_connect config option connects on initialization.

        Tests configuration-driven automatic connection.

        Business context:
        Some applications want sensors connected immediately on creation.
        Auto-connect simplifies setup for single-sensor scenarios where
        manual connect() call is unnecessary boilerplate.

        Arrangement:
        1. Create DigitalTwinSensorDriver.
        2. Create SensorConfig with auto_connect=True.

        Action:
        Create Sensor with auto_connect config.

        Assertion Strategy:
        Validates auto-connection by confirming:
        - Sensor is connected immediately after construction.
        - No explicit connect() call required.

        Testing Principle:
        Validates configuration behavior, ensuring auto_connect
        option triggers connection during initialization.
        """
        driver = DigitalTwinSensorDriver()
        config = SensorConfig(auto_connect=True)
        sensor = Sensor(driver, config)

        assert sensor.connected

        sensor.disconnect()

    def test_connect_twice_raises(self) -> None:
        """Verifies connecting when already connected raises RuntimeError.

        Tests idempotency protection at the Sensor wrapper level.

        Business context:
        Double-connect could cause resource leaks or state corruption.
        The Sensor class enforces single connection to match driver
        behavior and prevent application bugs.

        Arrangement:
        1. Create and connect Sensor.

        Action:
        Attempt to connect() again while already connected.

        Assertion Strategy:
        Validates protection by confirming:
        - RuntimeError raised with 'already connected' message.
        - Pattern matching ensures clear error context.

        Testing Principle:
        Validates defensive programming, ensuring redundant
        connect() calls fail explicitly rather than silently.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        with pytest.raises(RuntimeError, match="already connected"):
            sensor.connect()

        sensor.disconnect()

    def test_read_without_connect_raises(self) -> None:
        """Verifies reading without connect() raises RuntimeError.

        Tests precondition enforcement for read operations.

        Business context:
        Reading requires an active connection. Attempting to read
        without connecting would access null instance. Error message
        guides user to call connect() first.

        Arrangement:
        1. Create Sensor but do not connect.

        Action:
        Attempt to read() from unconnected sensor.

        Assertion Strategy:
        Validates precondition by confirming:
        - RuntimeError raised with 'not connected' message.
        - Error guides user to required connect() call.

        Testing Principle:
        Validates fail-fast behavior, ensuring operations
        fail immediately with helpful error messages.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with pytest.raises(RuntimeError, match="not connected"):
            sensor.read()

    def test_read_returns_sensor_reading(self) -> None:
        """Verifies Sensor.read() returns SensorReading from driver.

        Tests data passthrough from driver instance to Sensor wrapper.

        Business context:
        The Sensor class delegates to driver instance while adding
        statistics tracking. Read data must pass through unchanged
        for accurate telescope pointing.

        Arrangement:
        1. Create and connect Sensor.

        Action:
        Call read() to get sensor data through wrapper.

        Assertion Strategy:
        Validates passthrough by confirming:
        - Altitude and azimuth are floats for position.
        - Accelerometer dict is populated for tilt.
        - Magnetometer dict is populated for heading.

        Testing Principle:
        Validates delegation pattern, ensuring wrapper provides
        data exactly as received from underlying driver.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        reading = sensor.read()

        assert isinstance(reading.altitude, float)
        assert isinstance(reading.azimuth, float)
        assert reading.accelerometer
        assert reading.magnetometer

        sensor.disconnect()

    def test_calibrate(self) -> None:
        """Verifies calibrate() sets transform to match true position.

        Tests calibration delegation from Sensor to driver instance.

        Business context:
        Calibration is critical for telescope pointing accuracy.
        The Sensor class validates inputs and delegates to driver,
        then tracks calibrated state in status.

        Arrangement:
        1. Create and connect Sensor.

        Action:
        Calibrate to true position (50°, 120°) pointing southeast.

        Assertion Strategy:
        Validates calibration by confirming:
        - Status shows calibrated=True after calibration.
        - Calibration was applied to underlying instance.

        Testing Principle:
        Validates delegation with side effects, ensuring
        calibration propagates and updates state correctly.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        sensor.calibrate(true_altitude=50.0, true_azimuth=120.0)

        # Verify calibration was applied
        status = sensor.get_status()
        assert status.get("calibrated", False)

        sensor.disconnect()

    def test_calibrate_validates_altitude(self) -> None:
        """Verifies calibrate() validates altitude range 0-90 degrees.

        Tests input validation for physical constraints.

        Business context:
        Altitude must be 0-90° (horizon to zenith). Values outside
        this range are physically impossible and indicate user error.
        Early validation prevents corrupted calibration state.

        Arrangement:
        1. Create and connect Sensor.

        Action:
        Attempt to calibrate with invalid altitude (100° > 90° max).

        Assertion Strategy:
        Validates range check by confirming:
        - ValueError raised with 'Altitude must be 0-90' message.
        - Clear error guides user to valid range.

        Testing Principle:
        Validates input validation, ensuring invalid inputs
        fail early with actionable error messages.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        with pytest.raises(ValueError, match=r"Altitude must be in range \[0, 90\]"):
            sensor.calibrate(true_altitude=100.0, true_azimuth=180.0)

        sensor.disconnect()

    def test_calibrate_validates_azimuth(self) -> None:
        """Verifies calibrate() validates azimuth range 0-360 degrees.

        Tests input validation for compass heading constraints.

        Business context:
        Azimuth must be 0-360° (full compass circle). Values outside
        this range indicate user error. Unlike altitude, azimuth
        could wrap, but explicit range enforces clear intent.

        Arrangement:
        1. Create and connect Sensor.

        Action:
        Attempt to calibrate with invalid azimuth (400° > 360° max).

        Assertion Strategy:
        Validates range check by confirming:
        - ValueError raised with 'Azimuth must be 0-360' message.
        - Clear error guides user to valid range.

        Testing Principle:
        Validates input validation, ensuring azimuth boundaries
        are enforced with helpful error messages.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        with pytest.raises(ValueError, match=r"Azimuth must be in range \[0, 360\)"):
            sensor.calibrate(true_altitude=45.0, true_azimuth=400.0)

        sensor.disconnect()

    def test_reset(self) -> None:
        """Verifies reset() clears calibration and sensor state.

        Tests state reset for recalibration workflows.

        Business context:
        After hardware changes or observation session transitions,
        calibration becomes invalid. Reset clears state so fresh
        calibration can be applied against new reference.

        Arrangement:
        1. Create and connect Sensor.
        2. Apply calibration to establish non-default state.

        Action:
        Call reset() then check status to verify state cleared.

        Assertion Strategy:
        Validates reset by confirming:
        - Status shows calibrated=False after reset.
        - Calibration offsets have been cleared.

        Testing Principle:
        Validates state management, ensuring reset provides
        clean slate for recalibration workflows.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        sensor.calibrate(true_altitude=80.0, true_azimuth=300.0)
        sensor.reset()

        status = sensor.get_status()
        assert not status.get("calibrated", True)

        sensor.disconnect()

    def test_get_status(self) -> None:
        """Verifies get_status() returns comprehensive sensor info.

        Tests status aggregation from driver and wrapper layers.

        Business context:
        Observatory dashboards need unified sensor status. The Sensor
        class merges driver status with wrapper statistics (connect
        time, read count) for comprehensive monitoring.

        Arrangement:
        1. Create and connect Sensor.

        Action:
        Call get_status() to retrieve merged status.

        Assertion Strategy:
        Validates completeness by confirming:
        - Connection state from wrapper layer.
        - Sensor type from driver layer.
        - Connect_time for session tracking.
        - Read_count for usage statistics.

        Testing Principle:
        Validates status aggregation, ensuring unified view
        combines driver and wrapper information.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        status = sensor.get_status()

        assert status["connected"] is True
        assert status["type"] == "digital_twin"
        assert "connect_time" in status
        assert "read_count" in status

        sensor.disconnect()

    def test_statistics(self) -> None:
        """Verifies statistics property tracks reads and errors.

        Tests usage metrics collection for monitoring.

        Business context:
        Production systems need usage metrics. Statistics track
        read count and error rate for health monitoring, capacity
        planning, and debugging sensor issues.

        Arrangement:
        1. Create and connect Sensor.

        Action:
        Perform 5 reads then check statistics.

        Assertion Strategy:
        Validates tracking by confirming:
        - Read count equals 5 (all reads tracked).
        - Error count equals 0 (no failures).
        - Uptime_seconds populated for runtime tracking.

        Testing Principle:
        Validates metrics collection, ensuring accurate
        statistics for operational monitoring.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        for _ in range(5):
            sensor.read()

        stats = sensor.statistics
        assert stats["read_count"] == 5
        assert stats["error_count"] == 0
        assert stats["uptime_seconds"] is not None

        sensor.disconnect()

    def test_last_reading(self) -> None:
        """Verifies last_reading property returns most recent reading.

        Tests reading cache for quick access without re-read.

        Business context:
        Applications often need the last reading multiple times
        (display, logging, calculations). Caching avoids redundant
        sensor queries and provides consistent data reference.

        Arrangement:
        1. Create and connect Sensor.

        Action:
        Check last_reading before and after read().

        Assertion Strategy:
        Validates caching by confirming:
        - Initially None (no reads yet).
        - After read(), equals the returned reading object.
        - Same object reference (identity check).

        Testing Principle:
        Validates caching behavior, ensuring last_reading
        provides efficient access to most recent data.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        assert sensor.last_reading is None

        reading = sensor.read()
        assert sensor.last_reading is reading

        sensor.disconnect()

    def test_context_manager(self) -> None:
        """Verifies context manager connects on enter and disconnects on exit.

        Tests with-statement support for automatic resource cleanup.

        Business context:
        RAII pattern ensures sensors are properly disconnected even
        on exceptions. Context manager simplifies code and prevents
        resource leaks in error scenarios.

        Arrangement:
        1. Create Sensor (not connected).

        Action:
        Use sensor in with-statement, read data, then exit.

        Assertion Strategy:
        Validates lifecycle by confirming:
        - Inside with: connected=True, read() succeeds.
        - After with: connected=False (auto-disconnected).

        Testing Principle:
        Validates RAII pattern, ensuring automatic cleanup
        on scope exit for reliable resource management.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with sensor:
            assert sensor.connected
            reading = sensor.read()
            assert reading is not None

        assert not sensor.connected

    def test_get_available_sensors(self) -> None:
        """Verifies get_available_sensors() delegates to driver.

        Tests sensor enumeration passthrough from wrapper.

        Business context:
        Applications discover sensors through the Sensor class.
        The wrapper delegates to driver for enumeration while
        providing consistent interface across driver types.

        Arrangement:
        1. Create Sensor with DigitalTwinSensorDriver.

        Action:
        Call get_available_sensors() to enumerate.

        Assertion Strategy:
        Validates delegation by confirming:
        - Returns exactly one sensor (digital twin).
        - Result matches driver's direct enumeration.

        Testing Principle:
        Validates delegation pattern, ensuring wrapper
        provides access to driver discovery capabilities.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        sensors = sensor.get_available_sensors()
        assert len(sensors) == 1

    def test_info_property(self) -> None:
        """Verifies info property returns SensorInfo when connected.

        Tests sensor metadata access through typed dataclass.

        Business context:
        Applications need sensor metadata (type, name, capabilities)
        for UI display and capability checks. SensorInfo provides
        typed access rather than raw dict.

        Arrangement:
        1. Create Sensor (not connected).

        Action:
        Check info before and after connect().

        Assertion Strategy:
        Validates info state by confirming:
        - Before connect: info is None.
        - After connect: info is SensorInfo instance.
        - Type field matches expected 'digital_twin'.

        Testing Principle:
        Validates property state management, ensuring info
        reflects connection state accurately.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert sensor.info is None

        sensor.connect()
        assert sensor.info is not None
        assert isinstance(sensor.info, DeviceSensorInfo)
        assert sensor.info.type == "digital_twin"

        sensor.disconnect()

    def test_repr(self) -> None:
        """Verifies repr shows connection state and sensor type.

        Tests string representation for debugging.

        Business context:
        Debugging and logging benefit from informative repr.
        Shows connection state and sensor type at a glance
        without needing to call multiple methods.

        Arrangement:
        1. Create Sensor (not connected).

        Action:
        Check repr before and after connect().

        Assertion Strategy:
        Validates repr content by confirming:
        - Before connect: shows 'connected=False'.
        - After connect: shows 'connected=True' and type.

        Testing Principle:
        Validates repr usability, ensuring informative
        output for debugging and logging contexts.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert "connected=False" in repr(sensor)

        sensor.connect()
        assert "connected=True" in repr(sensor)
        assert "digital_twin" in repr(sensor)

        sensor.disconnect()


class TestSensorReading:
    """Test suite for SensorReading dataclass.

    Categories:
    1. Structure Tests - Field presence, types (1 test)

    Total: 1 test.
    """

    def test_sensor_reading_fields(self) -> None:
        """Verifies SensorReading has all required fields with correct types.

        Tests dataclass field definition and initialization.

        Business context:
        SensorReading is the core data contract for sensor data. All
        consuming code depends on these fields existing with correct
        types. Field validation ensures contract stability.

        Arrangement:
        1. Prepare valid values for all fields.

        Action:
        Construct SensorReading with all required fields.

        Assertion Strategy:
        Validates structure by confirming:
        - Altitude and azimuth accessible with correct values.
        - Accelerometer dict accessible with expected structure.
        - All fields accept and store provided values.

        Testing Principle:
        Validates data contract, ensuring SensorReading fields
        are defined and accessible as expected by consumers.
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


class TestSensorEdgeCases:
    """Test suite for Sensor edge cases and error paths.

    Covers uncovered lines for 100% coverage:
    - Connection failures
    - Disconnect error suppression
    - Read error recovery
    - Precondition validation
    - Status error handling
    - Reconnection logic

    Total: 13 tests.
    """

    def test_connect_failure_wraps_exception(self) -> None:
        """Verifies connect() wraps driver exceptions in RuntimeError.

        Tests exception handling in connect() when driver.open() fails.

        Business context:
            Hardware failures (port busy, device not found) must be wrapped
            in a consistent exception type for application error handling.

        Arrangement:
            1. Create mock driver that raises on open().

        Action:
            Attempt to connect() with failing driver.

        Assertion Strategy:
            Validates exception wrapping by confirming:
            - RuntimeError raised with "Failed to connect" message.
            - Sensor remains disconnected after failure.

        Testing Principle:
            Validates defensive programming, ensuring driver failures
            propagate as consistent exception types.
        """
        mock_driver = Mock()
        mock_driver.open.side_effect = RuntimeError("Port /dev/ttyACM0 not found")
        mock_driver.get_available_sensors.return_value = []

        sensor = Sensor(mock_driver)

        with pytest.raises(RuntimeError, match="Failed to connect"):
            sensor.connect()

        assert not sensor.connected
        assert sensor._instance is None

    def test_disconnect_suppresses_close_errors(self) -> None:
        """Verifies disconnect() suppresses errors from instance.close().

        Tests error suppression during cleanup to ensure cleanup completes.

        Business context:
            Serial port cleanup may fail (already closed, hardware removed).
            disconnect() must complete cleanup regardless, ensuring sensor
            state is consistent for potential reconnection.

        Arrangement:
            1. Create mock driver with mock instance.
            2. Make instance.close() raise exception.
            3. Connect sensor to establish state.

        Action:
            Call disconnect() where close() will fail.

        Assertion Strategy:
            Validates suppression by confirming:
            - disconnect() completes without raising.
            - Sensor marked disconnected despite close error.
            - Instance cleared for garbage collection.

        Testing Principle:
            Validates defensive cleanup, ensuring disconnect always
            leaves sensor in consistent disconnected state.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock Sensor"}
        mock_instance.close.side_effect = RuntimeError("Close failed")

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor.connect()

        # Should not raise - close error is suppressed
        sensor.disconnect()

        assert not sensor.connected
        assert sensor._instance is None
        mock_instance.close.assert_called_once()

    def test_read_error_triggers_reconnect(self) -> None:
        """Verifies read() triggers reconnect on error when configured.

        Tests automatic recovery flow for transient read failures.

        Business context:
            USB sensors may disconnect temporarily during long sessions.
            With reconnect_on_error=True, sensor automatically attempts
            recovery, maintaining data flow without manual intervention.

        Arrangement:
            1. Create mock driver returning instance that fails first read.
            2. Configure sensor with reconnect_on_error=True (default).
            3. Track reconnect attempts via open() calls.

        Action:
            Call read() which fails, triggers reconnect, then retries.

        Assertion Strategy:
            Validates recovery by confirming:
            - Second read succeeds after reconnect.
            - driver.open() called twice (initial + reconnect).
            - Error count incremented.

        Testing Principle:
            Validates automatic recovery, ensuring transient failures
            don't require manual intervention.
        """
        # First instance fails on read, second succeeds
        call_count = [0]

        def create_mock_instance(sensor_id: int | str = 0) -> Mock:
            """Create mock sensor instance with conditional read behavior.

            Factory function for mock sensors where first instance fails
            on read() but second instance succeeds, simulating recovery.

            Args:
                sensor_id: Sensor ID (unused, matches driver signature).

            Returns:
                Mock sensor with get_info(), close(), and conditional read().
                First call raises RuntimeError, subsequent return valid reading.

            Raises:
                RuntimeError: First instance's read() raises to trigger recovery.

            Business Context:
                Simulates hardware failure requiring reconnection. First
                sensor instance becomes unresponsive, forcing driver to
                create new connection.
            """
            call_count[0] += 1
            mock = Mock()
            mock.get_info.return_value = {"type": "mock", "name": "Mock Sensor"}
            mock.close.return_value = None

            if call_count[0] == 1:
                mock.read.side_effect = RuntimeError("Read failed")
            else:
                mock.read.return_value = SensorReading(
                    accelerometer={"aX": 0.0, "aY": 0.0, "aZ": 1.0},
                    magnetometer={"mX": 30.0, "mY": 0.0, "mZ": 40.0},
                    altitude=45.0,
                    azimuth=180.0,
                    temperature=20.0,
                    humidity=50.0,
                    timestamp=datetime.now(UTC),
                    raw_values="test",
                )
            return mock

        mock_driver = Mock()
        mock_driver.open.side_effect = create_mock_instance

        config = SensorConfig(reconnect_on_error=True, max_reconnect_attempts=3)
        sensor = Sensor(mock_driver, config)
        sensor.connect()

        # First read fails, triggers reconnect, retry succeeds
        reading = sensor.read()

        assert reading.altitude == 45.0
        assert sensor._error_count == 1
        assert mock_driver.open.call_count == 2

        sensor.disconnect()

    def test_read_error_raises_when_reconnect_disabled(self) -> None:
        """Verifies read() raises when reconnect_on_error=False.

        Tests that error propagates without recovery attempt.

        Business context:
            Some applications prefer manual error handling over automatic
            reconnection. Disabling reconnect lets caller control recovery.

        Arrangement:
            1. Create mock instance that fails on read.
            2. Configure sensor with reconnect_on_error=False.

        Action:
            Call read() which fails.

        Assertion Strategy:
            Validates error propagation by confirming:
            - Original exception raised (not wrapped).
            - Error count incremented.
            - No reconnect attempt made.

        Testing Principle:
            Validates configurable behavior, ensuring reconnect_on_error
            flag controls recovery behavior.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.read.side_effect = RuntimeError("Sensor disconnected")

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        config = SensorConfig(reconnect_on_error=False)
        sensor = Sensor(mock_driver, config)
        sensor.connect()

        with pytest.raises(RuntimeError, match="Sensor disconnected"):
            sensor.read()

        assert sensor._error_count == 1
        # Only initial open, no reconnect
        assert mock_driver.open.call_count == 1

    def test_calibrate_not_connected_raises(self) -> None:
        """Verifies calibrate() raises RuntimeError when not connected.

        Tests precondition enforcement for calibration.

        Business context:
            Calibration requires active sensor to read current position.
            Attempting calibration without connection is a programming error.

        Arrangement:
            1. Create Sensor but do not connect.

        Action:
            Attempt to calibrate() without connecting.

        Assertion Strategy:
            Validates precondition by confirming:
            - RuntimeError raised with "not connected" message.
            - Clear guidance to call connect() first.

        Testing Principle:
            Validates fail-fast behavior, ensuring invalid operations
            fail early with helpful error messages.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with pytest.raises(RuntimeError, match="not connected"):
            sensor.calibrate(45.0, 180.0)

    def test_reset_not_connected_raises(self) -> None:
        """Verifies reset() raises RuntimeError when not connected.

        Tests precondition enforcement for reset.

        Business context:
            Reset clears calibration on the sensor instance. Without
            connection, there's no instance to reset.

        Arrangement:
            1. Create Sensor but do not connect.

        Action:
            Attempt to reset() without connecting.

        Assertion Strategy:
            Validates precondition by confirming:
            - RuntimeError raised with "not connected" message.

        Testing Principle:
            Validates fail-fast behavior for invalid state operations.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with pytest.raises(RuntimeError, match="not connected"):
            sensor.reset()

    def test_get_status_handles_driver_error(self) -> None:
        """Verifies get_status() captures driver status errors gracefully.

        Tests error handling when driver get_status() fails.

        Business context:
            Status queries should never crash monitoring systems.
            If driver status fails, error is captured in response
            rather than propagated as exception.

        Arrangement:
            1. Create mock instance where get_status() raises.
            2. Connect sensor with mocked driver.

        Action:
            Call get_status() where driver status fails.

        Assertion Strategy:
            Validates error capture by confirming:
            - Status returned (no exception raised).
            - status_error field contains error message.
            - Base status fields still populated.

        Testing Principle:
            Validates graceful degradation, ensuring partial status
            available even when driver fails.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.side_effect = RuntimeError("Status unavailable")

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor.connect()

        status = sensor.get_status()

        assert status["connected"] is True
        assert "status_error" in status
        assert "Status unavailable" in status["status_error"]

        sensor.disconnect()

    def test_reconnect_all_attempts_fail(self) -> None:
        """Verifies _attempt_reconnect raises after all attempts fail.

        Tests reconnection loop exhaustion.

        Business context:
            If hardware is truly gone (unplugged, broken), reconnection
            will fail repeatedly. After max_reconnect_attempts, give up
            and raise to let caller handle.

        Arrangement:
            1. Create mock driver that always fails to reconnect.
            2. Configure max_reconnect_attempts=2.

        Action:
            Call _attempt_reconnect() which will fail twice.

        Assertion Strategy:
            Validates exhaustion by confirming:
            - RuntimeError raised with "Failed to reconnect" message.
            - All attempts were made (open called max times).

        Testing Principle:
            Validates bounded retry, ensuring reconnection gives up
            after configured number of attempts.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.close.return_value = None

        mock_driver = Mock()
        # First open succeeds, subsequent opens fail
        call_count = [0]

        def open_failing(sensor_id: int | str = 0) -> Mock:
            """Factory function where first open succeeds but reconnects fail.

            Simulates scenario where initial connection works but hardware
            becomes unavailable for reconnection attempts.

            Args:
                sensor_id: Sensor ID (unused, matches driver signature).

            Returns:
                Mock instance on first call only.

            Raises:
                RuntimeError: On all calls after the first.

            Business Context:
                Models hardware failure where device is physically
                disconnected or powered off after initial connection.
            """
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_instance
            raise RuntimeError("Reconnect failed")

        mock_driver.open.side_effect = open_failing

        config = SensorConfig(max_reconnect_attempts=2)
        sensor = Sensor(mock_driver, config)
        sensor.connect()

        with pytest.raises(RuntimeError, match="Failed to reconnect after 2 attempts"):
            sensor._attempt_reconnect()

        # Initial open + 2 reconnect attempts
        assert mock_driver.open.call_count == 3

    def test_reconnect_succeeds_after_partial_failure(self) -> None:
        """Verifies _attempt_reconnect succeeds after initial failures.

        Tests reconnection succeeding on non-first attempt.

        Business context:
            Transient issues (USB reset, power glitch) may cause initial
            reconnects to fail while later attempts succeed. The loop
            should continue trying until success or exhaustion.

        Arrangement:
            1. Create mock driver that fails first reconnect but succeeds second.
            2. Configure max_reconnect_attempts=3.

        Action:
            Call _attempt_reconnect() which fails once then succeeds.

        Assertion Strategy:
            Validates partial recovery by confirming:
            - No exception raised (reconnect succeeded).
            - Sensor is connected after recovery.

        Testing Principle:
            Validates retry persistence, ensuring temporary failures
            don't prevent eventual recovery.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.close.return_value = None

        call_count = [0]

        def open_with_partial_failure(sensor_id: int | str = 0) -> Mock:
            """Factory function simulating transient reconnection failure.

            Initial open and third open succeed; second open (first reconnect)
            fails. Simulates USB bus reset or temporary hardware unavailability.

            Args:
                sensor_id: Sensor ID (unused, matches driver signature).

            Returns:
                Mock instance on calls 1 and 3+.

            Raises:
                RuntimeError: On second call only (first reconnect attempt).

            Business Context:
                Models transient hardware issues like USB bus resets
                where first reconnect fails but subsequent succeed.
            """
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_instance  # Initial connect
            elif call_count[0] == 2:
                raise RuntimeError("Transient failure")  # First reconnect fails
            else:
                return mock_instance  # Second reconnect succeeds

        mock_driver = Mock()
        mock_driver.open.side_effect = open_with_partial_failure

        config = SensorConfig(max_reconnect_attempts=3)
        sensor = Sensor(mock_driver, config)
        sensor.connect()

        # Should succeed after one failed attempt
        sensor._attempt_reconnect()

        assert sensor.connected
        # Initial + 2 reconnect attempts (fail then success)
        assert mock_driver.open.call_count == 3

    def test_context_manager_when_already_connected(self) -> None:
        """Verifies context manager skips connect when already connected.

        Tests __enter__ optimization for pre-connected sensors.

        Business context:
            Applications may connect sensors before using context manager.
            __enter__ should not attempt redundant connection which would
            raise "already connected" error.

        Arrangement:
            1. Create and connect Sensor.

        Action:
            Use already-connected sensor in context manager.

        Assertion Strategy:
            Validates skip behavior by confirming:
            - No exception raised entering context.
            - Sensor remains connected throughout.
            - Sensor disconnected after context exit.

        Testing Principle:
            Validates idempotency, ensuring context manager handles
            pre-connected state gracefully.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        # Enter context when already connected - should not fail
        with sensor:
            assert sensor.connected
            reading = sensor.read()
            assert reading is not None

        assert not sensor.connected

    def test_calibrate_negative_altitude_raises(self) -> None:
        """Verifies calibrate() rejects negative altitude.

        Tests lower bound validation for altitude.

        Business context:
            Altitude below horizon (negative) is physically impossible
            for telescope pointing. Validates input before corrupting
            calibration state.

        Arrangement:
            1. Create and connect Sensor.

        Action:
            Attempt calibrate with altitude=-10 (below horizon).

        Assertion Strategy:
            Validates bound check by confirming:
            - ValueError raised with "0-90" range in message.

        Testing Principle:
            Validates input validation, ensuring physically impossible
            values are rejected early.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        with pytest.raises(ValueError, match=r"Altitude must be in range \[0, 90\]"):
            sensor.calibrate(true_altitude=-10.0, true_azimuth=180.0)

        sensor.disconnect()

    def test_calibrate_negative_azimuth_raises(self) -> None:
        """Verifies calibrate() rejects negative azimuth.

        Tests lower bound validation for azimuth.

        Business context:
            Azimuth must be 0-360° representing compass direction.
            Negative values indicate user error in coordinate conversion.

        Arrangement:
            1. Create and connect Sensor.

        Action:
            Attempt calibrate with azimuth=-45 (invalid).

        Assertion Strategy:
            Validates bound check by confirming:
            - ValueError raised with "0-360" range in message.

        Testing Principle:
            Validates input validation for azimuth lower bound.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        with pytest.raises(ValueError, match=r"Azimuth must be in range \[0, 360\)"):
            sensor.calibrate(true_altitude=45.0, true_azimuth=-45.0)

        sensor.disconnect()

    def test_auto_connect_failure_propagates(self) -> None:
        """Verifies auto_connect failure raises during __init__.

        Tests exception propagation from auto_connect.

        Business context:
            If auto_connect is enabled but connection fails, the exception
            should propagate so caller knows sensor creation failed.

        Arrangement:
            1. Create mock driver that fails on open.
            2. Configure auto_connect=True.

        Action:
            Create Sensor with failing auto_connect.

        Assertion Strategy:
            Validates propagation by confirming:
            - RuntimeError raised during Sensor.__init__.
            - Contains "Failed to connect" message.

        Testing Principle:
            Validates fail-fast for auto_connect, ensuring callers
            know immediately when sensor creation fails.
        """
        mock_driver = Mock()
        mock_driver.open.side_effect = RuntimeError("No sensors available")
        mock_driver.get_available_sensors.return_value = []

        config = SensorConfig(auto_connect=True)

        with pytest.raises(RuntimeError, match="Failed to connect"):
            Sensor(mock_driver, config)

    def test_get_status_when_not_connected(self) -> None:
        """Verifies get_status() returns base status when not connected.

        Tests status without driver status merge (disconnected path).

        Business context:
            Status queries should work even when disconnected. Returns
            base status fields without driver-specific data.

        Arrangement:
            1. Create Sensor but do not connect.

        Action:
            Call get_status() on disconnected sensor.

        Assertion Strategy:
            Validates base status by confirming:
            - connected is False.
            - type and name are None.
            - No driver status fields merged.

        Testing Principle:
            Validates graceful handling of disconnected state in status.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        status = sensor.get_status()

        assert status["connected"] is False
        assert status["type"] is None
        assert status["name"] is None
        assert status["connect_time"] is None
        assert "calibrated" not in status  # No driver status merged

    def test_statistics_when_never_connected(self) -> None:
        """Verifies statistics property works when never connected.

        Tests statistics with no connect_time (uptime=None path).

        Business context:
            Statistics may be queried before first connection. Should
            return valid data with uptime=None indicating no session.

        Arrangement:
            1. Create Sensor but do not connect.

        Action:
            Access statistics property.

        Assertion Strategy:
            Validates default state by confirming:
            - read_count is 0.
            - error_count is 0.
            - uptime_seconds is None (never connected).
            - error_rate is 0.0.

        Testing Principle:
            Validates safe defaults when accessing stats before connection.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        stats = sensor.statistics

        assert stats["read_count"] == 0
        assert stats["error_count"] == 0
        assert stats["uptime_seconds"] is None
        assert stats["error_rate"] == 0.0

    def test_get_status_merges_all_driver_status_fields(self) -> None:
        """Verifies get_status() merges all optional driver status fields.

        Tests that all driver status fields are properly merged into response.

        Business context:
            Driver status may include is_open, error, last_reading_age_ms,
            and reading_rate_hz. All should be included in device status.

        Arrangement:
            1. Create mock driver returning full status.

        Action:
            Call get_status() and verify all fields present.

        Assertion Strategy:
            Validates field merging by confirming all optional fields appear.

        Testing Principle:
            Validates complete data passthrough from driver to device status.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.return_value = {
            "calibrated": True,
            "is_open": True,
            "error": None,
            "last_reading_age_ms": 50.0,
            "reading_rate_hz": 10.0,
        }

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor.connect()

        status = sensor.get_status()

        assert status["calibrated"] is True
        assert status["is_open"] is True
        assert status["error"] is None
        assert status["last_reading_age_ms"] == 50.0
        assert status["reading_rate_hz"] == 10.0

        sensor.disconnect()

    def test_repr_shows_uptime_when_connected(self) -> None:
        """Verifies __repr__ includes uptime when connected.

        Tests enhanced repr for debugging connected sensors.

        Business context:
            When debugging, seeing uptime in repr helps identify
            connection age and potential issues.

        Arrangement:
            1. Create and connect Sensor.

        Action:
            Call repr() on connected sensor.

        Assertion Strategy:
            Validates repr format by confirming uptime appears in string.

        Testing Principle:
            Validates debugging information in repr for connected state.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        repr_str = repr(sensor)

        assert "type='digital_twin'" in repr_str
        assert "connected=True" in repr_str
        assert "uptime=" in repr_str

        sensor.disconnect()

    def test_get_status_handles_partial_driver_status(self) -> None:
        """Verifies get_status() handles driver status without all fields.

        Tests graceful handling when driver returns sparse status.

        Business context:
            Some drivers may not report all status fields. Device layer
            should only include fields that are present.

        Arrangement:
            1. Create mock driver returning partial status (calibrated only).

        Action:
            Call get_status() and verify only present fields are merged.

        Assertion Strategy:
            Validates partial merge by confirming:
            - calibrated field is present.
            - is_open, error, etc. are not in status (not provided by driver).

        Testing Principle:
            Validates robust handling of optional driver status fields.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.return_value = {
            "calibrated": False,
            # Intentionally omitting is_open, error, etc.
        }

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor.connect()

        status = sensor.get_status()

        # calibrated should be merged
        assert status["calibrated"] is False
        # These should NOT be present since driver didn't provide them
        assert "is_open" not in status
        assert "error" not in status
        assert "last_reading_age_ms" not in status
        assert "reading_rate_hz" not in status

        sensor.disconnect()

    def test_get_status_handles_empty_driver_status(self) -> None:
        """Verifies get_status() handles driver returning empty status.

        Tests edge case where driver status contains no optional fields.

        Business context:
            A minimal driver implementation might return empty status dict.
            Device layer should handle this gracefully without KeyError.

        Arrangement:
            1. Create mock driver returning empty status dict.

        Action:
            Call get_status() and verify no optional fields present.

        Assertion Strategy:
            Validates empty handling by confirming:
            - Base status fields are present (connected, read_count, etc.).
            - No driver-specific fields merged.

        Testing Principle:
            Validates defensive handling of minimal driver responses.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"type": "mock", "name": "Mock"}
        mock_instance.get_status.return_value = {}  # Empty status dict

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        sensor = Sensor(mock_driver)
        sensor.connect()

        status = sensor.get_status()

        # Base status should be present
        assert status["connected"] is True
        assert status["type"] == "mock"
        assert status["name"] == "Mock"
        assert status["read_count"] == 0
        assert status["error_count"] == 0

        # No driver-specific fields should be present
        assert "calibrated" not in status
        assert "is_open" not in status
        assert "error" not in status
        assert "last_reading_age_ms" not in status
        assert "reading_rate_hz" not in status

        sensor.disconnect()

    def test_repr_with_info_but_no_connect_time(self) -> None:
        """Verifies __repr__ handles info exists but connect_time is None.

        Tests defensive repr handling for unusual state.

        Business context:
            While normally _info and _connect_time are set together during
            connect(), defensive code should handle edge cases gracefully.

        Arrangement:
            1. Create sensor and manually set _info without _connect_time.

        Action:
            Call repr() on sensor with partial state.

        Assertion Strategy:
            Validates repr handles missing connect_time:
            - Shows type from info.
            - No uptime displayed (since connect_time is None).

        Testing Principle:
            Validates defensive programming in __repr__ for edge cases.
        """
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        # Manually set _info without connect_time to test edge case
        sensor._info = DeviceSensorInfo(type="test_type", name="Test Sensor")
        sensor._connected = True
        sensor._connect_time = None  # Edge case: info exists but no connect_time

        repr_str = repr(sensor)

        assert "type='test_type'" in repr_str
        assert "connected=True" in repr_str
        # Should NOT have uptime since connect_time is None
        assert "uptime=" not in repr_str
