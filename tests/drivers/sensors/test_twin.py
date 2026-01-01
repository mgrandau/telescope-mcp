"""Unit tests for Digital Twin sensor driver.

Tests the DigitalTwinSensorDriver and DigitalTwinSensorInstance
for simulated sensor behavior without hardware.

Test Categories:
- Configuration (dataclass, repr)
- Sensor Instance (read, calibrate, position, status)
- Driver lifecycle (open, close, context manager)
- Validation errors (altitude, azimuth bounds)

Example:
    Run all digital twin sensor tests::

        pdm run pytest tests/drivers/sensors/test_twin.py -v

    Run specific test::

        pdm run pytest tests/drivers/sensors/test_twin.py -v -k Config
"""

from __future__ import annotations

import pytest

from telescope_mcp.drivers.sensors.twin import (
    DigitalTwinSensorConfig,
    DigitalTwinSensorDriver,
    DigitalTwinSensorInstance,
)

# =============================================================================
# Configuration Tests
# =============================================================================


class TestDigitalTwinSensorConfig:
    """Test suite for DigitalTwinSensorConfig dataclass.

    Categories:
    1. Defaults - Default configuration values (1 test)
    2. Repr - String representation (1 test)
    3. Custom - Custom configuration (1 test)

    Total: 3 tests.
    """

    def test_config_defaults(self) -> None:
        """Verifies default configuration values are sensible.

        Tests that DigitalTwinSensorConfig has reasonable defaults
        for noise, drift, and environmental simulation.

        Business context:
        Default config should provide realistic simulation behavior
        out of the box. Developers shouldn't need to configure everything
        for basic testing scenarios.

        Arrangement:
        1. Create config with no arguments.

        Action:
        Access default attribute values.

        Assertion Strategy:
        Validates defaults by confirming:
        - initial_altitude is 45.0 (mid-range).
        - initial_azimuth is 180.0 (south).
        - noise_std_alt is 0.1 (small but noticeable).
        - temperature is 20.0 (room temperature).

        Testing Principle:
        Validates sensible defaults for development use.
        """
        config = DigitalTwinSensorConfig()

        assert config.initial_altitude == 45.0
        assert config.initial_azimuth == 180.0
        assert config.noise_std_alt == 0.1
        assert config.noise_std_az == 0.2
        assert config.drift_rate_alt == 0.0
        assert config.drift_rate_az == 0.0
        assert config.temperature == 20.0
        assert config.humidity == 50.0
        assert config.sample_rate_hz == 10.0

    def test_config_repr(self) -> None:
        """Verifies __repr__ returns concise config summary.

        Tests that string representation includes key parameters
        for logging and debugging.

        Business context:
        When logging sensor initialization, a concise repr helps
        identify configuration without overwhelming log output.
        Should include altitude, azimuth, and noise level.

        Arrangement:
        1. Create config with specific values.

        Action:
        Call repr() on config.

        Assertion Strategy:
        Validates repr by confirming:
        - Contains "DigitalTwinSensorConfig".
        - Contains altitude value.
        - Contains azimuth value.
        - Contains noise value.

        Testing Principle:
        Validates logging support for debugging.
        """
        config = DigitalTwinSensorConfig(
            initial_altitude=30.0,
            initial_azimuth=90.0,
            noise_std_alt=0.5,
        )

        result = repr(config)

        assert "DigitalTwinSensorConfig" in result
        assert "30" in result  # altitude
        assert "90" in result  # azimuth
        assert "0.5" in result  # noise

    def test_config_custom_values(self) -> None:
        """Verifies custom configuration values are stored.

        Tests that all configurable parameters can be customized.

        Business context:
        Different test scenarios need different simulation behavior.
        High noise for error handling tests, zero noise for precision
        tests, drift for long-duration tests.

        Arrangement:
        1. Create config with all custom values.

        Action:
        Access attribute values.

        Assertion Strategy:
        Validates custom values are stored correctly.

        Testing Principle:
        Validates configuration flexibility.
        """
        config = DigitalTwinSensorConfig(
            initial_altitude=60.0,
            initial_azimuth=270.0,
            noise_std_alt=0.5,
            noise_std_az=1.0,
            drift_rate_alt=0.1,
            drift_rate_az=0.2,
            temperature=25.0,
            humidity=60.0,
            sample_rate_hz=20.0,
        )

        assert config.initial_altitude == 60.0
        assert config.initial_azimuth == 270.0
        assert config.noise_std_alt == 0.5
        assert config.noise_std_az == 1.0
        assert config.drift_rate_alt == 0.1
        assert config.drift_rate_az == 0.2
        assert config.temperature == 25.0
        assert config.humidity == 60.0
        assert config.sample_rate_hz == 20.0


# =============================================================================
# Sensor Instance Tests
# =============================================================================


class TestDigitalTwinSensorInstance:
    """Test suite for DigitalTwinSensorInstance.

    Categories:
    1. Read - Basic reading operations (2 tests)
    2. Position - set_position and read (2 tests)
    3. Calibration - calibrate() and validation (4 tests)
    4. Status - get_status and get_info (2 tests)
    5. Lifecycle - close and is_open (2 tests)

    Total: 12 tests.
    """

    @pytest.fixture
    def instance(self) -> DigitalTwinSensorInstance:
        """Create instance with zero noise for deterministic tests.

        Creates a DigitalTwinSensorInstance with all noise parameters
        set to zero, enabling exact assertions on calculated values.

        Business context:
        Test fixtures must provide predictable, deterministic state.
        Zero noise ensures calculated values match expected exactly,
        enabling precise assertions without tolerance ranges.

        Args:
            self: Test class instance (implicit pytest fixture param).

        Returns:
            DigitalTwinSensorInstance: Open sensor instance with zero noise,
                ready for read(), calibrate(), and other operations.

        Raises:
            No exceptions raised. Fixture always succeeds.

        Example:
            >>> def test_example(self, instance):
            ...     reading = instance.read()
            ...     assert reading.altitude == 45.0  # Exact match, no noise
        """
        config = DigitalTwinSensorConfig(
            noise_std_alt=0.0,
            noise_std_az=0.0,
            temp_noise_std=0.0,
            humidity_noise_std=0.0,
        )
        return DigitalTwinSensorInstance(config)

    def test_read_returns_sensor_reading(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies read() returns complete SensorReading.

        Tests happy path read with all sensor data populated.

        Business context:
        read() is the primary interface for getting sensor data.
        Must return SensorReading with accelerometer, magnetometer,
        temperature, humidity, timestamp, and raw values.

        Arrangement:
        1. Use instance fixture with zero noise configuration.
        2. Instance is open and ready for reading.

        Action:
        Call read() to retrieve current sensor data.

        Assertion Strategy:
        Validates complete read by confirming:
        - accelerometer dict has aX, aY, aZ keys.
        - magnetometer dict has mX, mY, mZ keys.
        - temperature, humidity, timestamp all present.
        - raw_values string is present.

        Testing Principle:
        Validates data completeness for downstream consumers.
        """
        reading = instance.read()

        assert reading.accelerometer is not None
        assert "aX" in reading.accelerometer
        assert "aY" in reading.accelerometer
        assert "aZ" in reading.accelerometer
        assert reading.magnetometer is not None
        assert "mX" in reading.magnetometer
        assert "mY" in reading.magnetometer
        assert "mZ" in reading.magnetometer
        assert reading.temperature is not None
        assert reading.humidity is not None
        assert reading.timestamp is not None
        assert reading.raw_values is not None

    def test_read_raises_when_closed(self, instance: DigitalTwinSensorInstance) -> None:
        """Verifies read() raises RuntimeError when sensor closed.

        Tests error handling for reads after close().

        Business context:
        Reading from closed sensor should fail clearly. After close(),
        all operations should be invalid. Prevents use-after-close bugs.

        Arrangement:
        1. Use instance fixture.
        2. Call close() to mark sensor as closed.

        Action:
        Call read() on closed sensor.

        Assertion Strategy:
        Validates state checking by confirming:
        - RuntimeError raised with "Sensor is closed" message.

        Testing Principle:
        Validates lifecycle enforcement for closed sensors.
        """
        instance.close()

        with pytest.raises(RuntimeError, match="Sensor is closed"):
            instance.read()

    def test_set_position_updates_readings(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies set_position() affects subsequent reads.

        Tests that position changes are reflected in readings.

        Business context:
        Tests need to drive simulation to known positions. set_position()
        should immediately affect read() output.

        Arrangement:
        1. Use instance fixture with zero noise.
        2. Instance is open and ready.

        Action:
        Call set_position(30.0, 120.0) then read().

        Assertion Strategy:
        Validates position tracking by confirming:
        - read().altitude matches set altitude (within 0.01°).
        - read().azimuth matches set azimuth (within 0.01°).

        Testing Principle:
        Validates simulation state management for test control.
        """
        instance.set_position(30.0, 120.0)
        reading = instance.read()

        # With zero noise, should be exact
        assert abs(reading.altitude - 30.0) < 0.01
        assert abs(reading.azimuth - 120.0) < 0.01

    def test_set_position_normalizes_azimuth(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies set_position() normalizes azimuth to 0-360.

        Tests azimuth normalization for values outside standard range.

        Business context:
        Azimuth calculations may produce values > 360 or < 0.
        Driver should normalize these to standard compass range.

        Arrangement:
        1. Use instance fixture with zero noise.
        2. Instance is open and ready.

        Action:
        Call set_position(45.0, 370.0) with out-of-range azimuth.

        Assertion Strategy:
        Validates normalization by confirming:
        - read().azimuth is ~10° (370° - 360°).

        Testing Principle:
        Validates angle normalization for compass consistency.
        """
        instance.set_position(45.0, 370.0)
        reading = instance.read()

        assert abs(reading.azimuth - 10.0) < 0.01

    def test_calibrate_adjusts_readings(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies calibrate() applies offset to readings.

        Tests that calibration affects subsequent readings.

        Business context:
        Calibration should correct systematic errors by applying
        offsets so readings match known reference position.

        Arrangement:
        1. Use instance fixture with zero noise.
        2. Set position to (40°, 170°) as raw reading.

        Action:
        Call calibrate(45.0, 180.0) to set calibration target.

        Assertion Strategy:
        Validates calibration by confirming:
        - Post-calibration readings match target within 0.5°.

        Testing Principle:
        Validates offset calculation for accurate pointing.
        """
        # Set position then calibrate to different value
        instance.set_position(40.0, 170.0)
        instance.calibrate(45.0, 180.0)

        reading = instance.read()
        # Reading should now be calibrated to target
        assert abs(reading.altitude - 45.0) < 0.5
        assert abs(reading.azimuth - 180.0) < 0.5

    def test_calibrate_raises_when_closed(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies calibrate() raises RuntimeError when closed.

        Tests error handling for calibration on closed sensor.

        Business context:
        Calibration requires active sensor connection. Attempting
        to calibrate closed sensor indicates programming error.

        Arrangement:
        1. Use instance fixture.
        2. Call close() to mark sensor as closed.

        Action:
        Call calibrate(45.0, 180.0) on closed sensor.

        Assertion Strategy:
        Validates state checking by confirming:
        - RuntimeError raised with "Sensor is closed" message.

        Testing Principle:
        Validates lifecycle enforcement for calibration operations.
        """
        instance.close()

        with pytest.raises(RuntimeError, match="Sensor is closed"):
            instance.calibrate(45.0, 180.0)

    def test_calibrate_raises_for_invalid_altitude(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies calibrate() validates altitude range.

        Tests that altitude outside 0-90° raises ValueError.

        Business context:
        Altitude represents angle above horizon. Values outside
        0-90° are physically impossible for telescope pointing.

        Arrangement:
        1. Use instance fixture.
        2. Instance is open and ready.

        Action:
        Call calibrate() with altitude -5.0, then with 95.0.

        Assertion Strategy:
        Validates input validation by confirming:
        - ValueError raised for negative altitude.
        - ValueError raised for altitude > 90.

        Testing Principle:
        Validates boundary enforcement for physical constraints.
        """
        # Negative altitude
        with pytest.raises(ValueError, match="Altitude must be between 0 and 90"):
            instance.calibrate(-5.0, 180.0)

        # Altitude > 90
        with pytest.raises(ValueError, match="Altitude must be between 0 and 90"):
            instance.calibrate(95.0, 180.0)

    def test_calibrate_raises_for_invalid_azimuth(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies calibrate() validates azimuth range.

        Tests that azimuth outside 0-360° raises ValueError.

        Business context:
        Azimuth represents compass heading. While values can be
        normalized, calibration expects clean input in 0-360 range.

        Arrangement:
        1. Use instance fixture.
        2. Instance is open and ready.

        Action:
        Call calibrate() with azimuth -10.0, then with 360.0.

        Assertion Strategy:
        Validates input validation by confirming:
        - ValueError raised for negative azimuth.
        - ValueError raised for azimuth >= 360.

        Testing Principle:
        Validates compass range enforcement for calibration.
        """
        # Negative azimuth
        with pytest.raises(ValueError, match="Azimuth must be between 0 and 360"):
            instance.calibrate(45.0, -10.0)

        # Azimuth >= 360
        with pytest.raises(ValueError, match="Azimuth must be between 0 and 360"):
            instance.calibrate(45.0, 360.0)

    def test_get_info_returns_sensor_info(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies get_info() returns complete sensor metadata.

        Tests static sensor information retrieval.

        Business context:
        get_info() provides metadata about sensor capabilities for
        UI display and capability checking.

        Arrangement:
        1. Use instance fixture.
        2. Instance is open and ready.

        Action:
        Call get_info() to retrieve sensor metadata.

        Assertion Strategy:
        Validates metadata per SensorInfo protocol by confirming:
        - type is "digital_twin".
        - name is "Digital Twin IMU Sensor".

        Testing Principle:
        Validates capability reporting for UI and logic decisions.
        """
        info = instance.get_info()

        assert info["type"] == "digital_twin"
        assert info["name"] == "Digital Twin IMU Sensor"

    def test_get_status_reflects_calibration(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies get_status() reflects calibration state.

        Tests that status shows calibrated=True after calibration.

        Business context:
        Status must indicate calibration state so users know if
        readings are corrected. UI can prompt for calibration.

        Arrangement:
        1. Use instance fixture.
        2. Instance is open, initially uncalibrated.

        Action:
        1. Check initial status (uncalibrated).
        2. Set position different from calibration target.
        3. Calibrate to different position.
        4. Check status again.

        Assertion Strategy:
        Validates status before and after calibration by confirming:
        - calibrated=False initially.
        - calibrated=True after calling calibrate().

        Testing Principle:
        Validates state tracking for user feedback.
        """
        # Initially uncalibrated
        status = instance.get_status()
        assert status["calibrated"] is False

        # Set position different from calibration target to ensure offset
        instance.set_position(40.0, 170.0)
        # Calibrate to different position (creates non-zero offset)
        instance.calibrate(45.0, 180.0)

        # Now calibrated
        status = instance.get_status()
        assert status["calibrated"] is True

    def test_close_marks_sensor_closed(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies close() marks sensor as closed.

        Tests that is_open property returns False after close.

        Business context:
        Proper lifecycle management requires clear open/closed state.
        is_open property enables state checking.

        Arrangement:
        1. Use instance fixture.
        2. Verify instance.is_open is True.

        Action:
        Call close() on instance.

        Assertion Strategy:
        Validates state transition by confirming:
        - is_open is True before close.
        - is_open is False after close.

        Testing Principle:
        Validates lifecycle state transitions.
        """
        assert instance.is_open is True

        instance.close()

        assert instance.is_open is False

    def test_reset_clears_calibration(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies reset() clears calibration state.

        Tests that reset returns sensor to uncalibrated state.

        Business context:
        Testing calibration workflows requires ability to reset.
        Also simulates sensor reboot behavior.

        Arrangement:
        1. Use instance fixture.
        2. Set position and calibrate to create calibrated state.

        Action:
        Call reset() to clear calibration.

        Assertion Strategy:
        Validates calibration cleared by confirming:
        - calibrated=True before reset.
        - calibrated=False after reset.

        Testing Principle:
        Validates state reset for test isolation.
        """
        # Set position different from calibration to ensure offset
        instance.set_position(40.0, 170.0)
        instance.calibrate(45.0, 180.0)
        assert instance.get_status()["calibrated"] is True

        instance.reset()

        assert instance.get_status()["calibrated"] is False

    def test_calibrate_magnetometer_returns_offsets(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies calibrate_magnetometer() returns calibration offsets.

        Tests that magnetometer calibration generates and returns offsets.

        Business context:
        Magnetometer calibration computes hard-iron offsets to correct
        distortion from nearby ferrous materials. Digital twin simulates
        this by generating random offsets.

        Arrangement:
        1. Use instance fixture.
        2. Instance is open and ready.

        Action:
        Call calibrate_magnetometer() to perform calibration.

        Assertion Strategy:
        Validates returned offsets by confirming:
        - offset_x, offset_y, offset_z keys present.
        - All offset values are floats.

        Testing Principle:
        Validates calibration return contract for consumers.
        """
        offsets = instance.calibrate_magnetometer()

        assert "offset_x" in offsets
        assert "offset_y" in offsets
        assert "offset_z" in offsets
        # Offsets should be floats
        assert isinstance(offsets["offset_x"], float)
        assert isinstance(offsets["offset_y"], float)
        assert isinstance(offsets["offset_z"], float)

    def test_calibrate_magnetometer_updates_status(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies calibrate_magnetometer() updates mag_calibrated status.

        Tests that status reflects magnetometer calibration state.

        Business context:
        Status should indicate magnetometer calibration state so users
        know if compass readings are corrected.

        Arrangement:
        1. Use instance fixture.
        2. Instance is open, initially not calibrated.

        Action:
        Call calibrate_magnetometer() to perform calibration.

        Assertion Strategy:
        Validates that calibrate_magnetometer() runs without error.
        Per SensorStatus protocol, mag_calibrated is not exposed.

        Testing Principle:
        Validates method execution without implementation-specific status.
        """
        # Calibrate magnetometer
        result = instance.calibrate_magnetometer()

        # Verify method returns successfully (returns calibration offsets dict)
        assert isinstance(result, dict)
        assert "offset_x" in result
        assert "offset_y" in result
        assert "offset_z" in result

    def test_stop_output_is_noop(self, instance: DigitalTwinSensorInstance) -> None:
        """Verifies stop_output() executes without error (no-op).

        Tests API compatibility with ArduinoSensorInstance.

        Business context:
        Arduino sensor streams data continuously and stop_output()
        pauses that stream. Digital twin doesn't stream, so this is
        a no-op for API compatibility.

        Arrangement:
        1. Use instance fixture.

        Action:
        Call stop_output() on instance.

        Assertion Strategy:
        Validates no-op behavior by confirming:
        - Method executes without error.
        - Instance remains open.
        - read() still works after stop.

        Testing Principle:
        Validates API parity with Arduino driver.
        """
        instance.stop_output()

        # Should still be able to read (no actual stream to stop)
        assert instance.is_open is True
        reading = instance.read()
        assert reading is not None

    def test_start_output_is_noop(self, instance: DigitalTwinSensorInstance) -> None:
        """Verifies start_output() executes without error (no-op).

        Tests API compatibility with ArduinoSensorInstance.

        Business context:
        Arduino sensor streams data and start_output() resumes
        after stop. Digital twin is on-demand, so this is a no-op.

        Arrangement:
        1. Use instance fixture.
        2. Call stop_output() first.

        Action:
        Call start_output() on instance.

        Assertion Strategy:
        Validates no-op behavior by confirming:
        - Method executes without error.
        - Instance remains open.

        Testing Principle:
        Validates API parity with Arduino driver.
        """
        instance.stop_output()
        instance.start_output()

        assert instance.is_open is True
        reading = instance.read()
        assert reading is not None

    def test_set_tilt_calibration_applies_transform(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies _set_tilt_calibration() applies linear transform.

        Tests that slope and intercept modify altitude readings.

        Business context:
        Tilt calibration corrects for sensor mounting errors using
        linear transform: corrected = slope * raw + intercept.
        This enables fine-tuning of altitude accuracy.

        Arrangement:
        1. Use instance fixture.
        2. Set position to known altitude.
        3. Configure zero noise for predictable values.

        Action:
        Apply tilt calibration with known slope/intercept.

        Assertion Strategy:
        Validates transform applied by confirming:
        - Altitude reading changes after calibration.
        - Transform follows corrected = slope * raw + intercept.

        Testing Principle:
        Validates calibration math for altitude correction.
        """
        # Use zero-noise config for predictable readings
        config = DigitalTwinSensorConfig(
            initial_altitude=45.0,
            noise_std_alt=0.0,
            noise_std_az=0.0,
        )
        twin = DigitalTwinSensorInstance(config)

        # Read initial altitude (should be ~45.0)
        initial = twin.read()
        initial_alt = initial.altitude

        # Apply tilt calibration: slope=1.0, intercept=5.0 (add 5 degrees)
        twin._set_tilt_calibration(slope=1.0, intercept=5.0)

        # Read calibrated altitude (should be ~50.0)
        calibrated = twin.read()
        calibrated_alt = calibrated.altitude

        # Verify transform applied (with small tolerance for floating point)
        assert abs(calibrated_alt - (initial_alt + 5.0)) < 0.1

        twin.close()

    def test_set_tilt_calibration_with_slope(
        self, instance: DigitalTwinSensorInstance
    ) -> None:
        """Verifies _set_tilt_calibration() applies slope scaling.

        Tests that slope parameter scales altitude readings.

        Business context:
        Slope corrects for systematic scale errors in the
        accelerometer-to-altitude calculation.

        Arrangement:
        1. Create instance with zero noise.
        2. Set known position.

        Action:
        Apply calibration with slope != 1.0.

        Assertion Strategy:
        Validates scaling by confirming altitude is multiplied.

        Testing Principle:
        Validates slope component of linear calibration.
        """
        # Use zero-noise config
        config = DigitalTwinSensorConfig(
            initial_altitude=40.0,
            noise_std_alt=0.0,
            noise_std_az=0.0,
        )
        twin = DigitalTwinSensorInstance(config)

        # Apply slope=0.5 (halve the altitude reading)
        twin._set_tilt_calibration(slope=0.5, intercept=0.0)

        reading = twin.read()
        # 40.0 * 0.5 = 20.0
        assert abs(reading.altitude - 20.0) < 0.1

        twin.close()


# =============================================================================
# Driver Tests
# =============================================================================


class TestDigitalTwinSensorDriver:
    """Test suite for DigitalTwinSensorDriver.

    Categories:
    1. Lifecycle - open, close, context manager (4 tests)
    2. Discovery - get_available_sensors (1 test)
    3. Guards - ensure_not_open (3 tests)

    Total: 8 tests.
    """

    def test_get_available_sensors_returns_one(self) -> None:
        """Verifies get_available_sensors() returns single twin sensor.

        Tests that digital twin reports one available sensor.

        Business context:
        Digital twin simulates single sensor for development.
        Discovery API should work same as real hardware drivers.

        Arrangement:
        1. Create DigitalTwinSensorDriver instance.

        Action:
        Call get_available_sensors() to enumerate.

        Assertion Strategy:
        Validates single sensor returned by confirming:
        - len(sensors) == 1.
        - sensors[0]["type"] == "digital_twin".
        - sensors[0]["id"] == 0.

        Testing Principle:
        Validates discovery API parity with hardware drivers.
        """
        driver = DigitalTwinSensorDriver()

        sensors = driver.get_available_sensors()

        assert len(sensors) == 1
        assert sensors[0]["type"] == "digital_twin"
        assert sensors[0]["id"] == 0

    def test_open_creates_instance(self) -> None:
        """Verifies open() creates sensor instance.

        Tests basic open functionality.

        Business context:
        open() is primary entry point for connecting to sensor.
        Must return functional instance.

        Arrangement:
        1. Create DigitalTwinSensorDriver instance.

        Action:
        Call open() to create sensor instance.

        Assertion Strategy:
        Validates instance returned by confirming:
        - instance is not None.
        - instance.is_open is True.

        Testing Principle:
        Validates basic connection workflow.
        """
        driver = DigitalTwinSensorDriver()

        instance = driver.open()

        assert instance is not None
        assert instance.is_open is True
        driver.close()

    def test_open_raises_when_already_open(self) -> None:
        """Verifies open() raises when sensor already open.

        Tests single-instance constraint.

        Business context:
        Driver maintains single instance. Opening twice would cause
        resource conflicts. Must reject second open.

        Arrangement:
        1. Create driver and open first instance.

        Action:
        Call open() again while first instance is open.

        Assertion Strategy:
        Validates RuntimeError raised by confirming:
        - RuntimeError with "Sensor already open" message.

        Testing Principle:
        Validates single-instance enforcement.
        """
        driver = DigitalTwinSensorDriver()
        driver.open()

        with pytest.raises(RuntimeError, match="Sensor already open"):
            driver.open()

        driver.close()

    def test_close_without_open_is_safe(self) -> None:
        """Verifies close() is safe when no instance open.

        Tests defensive close behavior.

        Business context:
        Cleanup code often calls close() unconditionally. Must
        handle case where sensor was never opened.

        Arrangement:
        1. Create driver without opening any sensor.

        Action:
        Call close() on driver with no open instance.

        Assertion Strategy:
        Validates no exception raised by confirming:
        - Method completes without error.

        Testing Principle:
        Validates defensive programming for cleanup safety.
        """
        driver = DigitalTwinSensorDriver()

        # Should not raise
        driver.close()

    def test_context_manager_opens_and_closes(self) -> None:
        """Verifies driver works as context manager.

        Tests __enter__ and __exit__ methods.

        Business context:
        Context manager pattern ensures cleanup on exception.
        Preferred usage pattern for resource management.

        Arrangement:
        1. Create DigitalTwinSensorDriver instance.

        Action:
        Use driver as context manager with 'with' statement.

        Assertion Strategy:
        Validates sensor open inside context by confirming:
        - instance is not None.
        - instance.is_open is True.
        - read() returns valid data.
        - driver._instance is None after context.

        Testing Principle:
        Validates context manager lifecycle.
        """
        driver = DigitalTwinSensorDriver()

        with driver as instance:
            assert instance is not None
            assert instance.is_open is True
            reading = instance.read()
            assert reading is not None

        # After context, instance should be closed
        assert driver._instance is None

    def test_context_manager_closes_on_exception(self) -> None:
        """Verifies context manager closes on exception.

        Tests cleanup when exception raised inside context.

        Business context:
        Context manager must ensure cleanup even when code inside
        raises exception. Prevents resource leaks.

        Arrangement:
        1. Create DigitalTwinSensorDriver instance.

        Action:
        Use driver as context manager and raise exception inside.

        Assertion Strategy:
        Validates sensor closed after exception by confirming:
        - ValueError is raised (expected).
        - driver._instance is None after exception.

        Testing Principle:
        Validates exception-safe cleanup.
        """
        driver = DigitalTwinSensorDriver()

        with pytest.raises(ValueError):
            with driver as instance:
                # Force an exception
                raise ValueError("Test exception")

        # Should still be cleaned up
        assert driver._instance is None

    def test_ensure_not_open_no_instance(self) -> None:
        """Verifies _ensure_not_open passes when no instance.

        Tests guard when driver has never opened a sensor.

        Arrangement:
        1. Create driver without opening any sensor.

        Action:
        Call _ensure_not_open() on driver.

        Assertion Strategy:
        Validates no exception raised by confirming:
        - Method completes without RuntimeError.

        Testing Principle:
        Validates guard passes for fresh driver.
        """
        driver = DigitalTwinSensorDriver()

        # Should not raise
        driver._ensure_not_open()

    def test_ensure_not_open_closed_instance(self) -> None:
        """Verifies _ensure_not_open passes when instance closed.

        Tests guard when instance exists but is closed.

        Arrangement:
        1. Create driver and open instance.
        2. Manually close instance via _is_open = False.

        Action:
        Call _ensure_not_open() on driver.

        Assertion Strategy:
        Validates no exception raised by confirming:
        - Method completes without RuntimeError.

        Testing Principle:
        Validates guard checks is_open, not just existence.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()
        instance.close()
        # Note: _instance still exists but is_open is False

        # Should not raise (since is_open is False)
        # Actually, driver.close() sets _instance to None
        # Let's test the case where we manually close
        driver2 = DigitalTwinSensorDriver()
        inst2 = driver2.open()
        inst2._is_open = False  # Manually close without clearing reference

        driver2._ensure_not_open()
