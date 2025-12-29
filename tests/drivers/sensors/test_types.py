"""Tests for sensor type definitions and validation functions.

This module tests the shared types and validation helpers in types.py.
Tests focus on the validate_position() function and SensorReading dataclass.

Example:
    pdm run pytest tests/drivers/sensors/test_types.py -v

Test Organization:
- TestValidatePosition: Tests for validate_position() helper
- TestSensorReading: Tests for SensorReading dataclass

Coverage target: 100% for validate_position() function.
"""

from datetime import UTC, datetime

import pytest

from telescope_mcp.drivers.sensors.types import (
    AccelerometerData,
    MagnetometerData,
    SensorReading,
    validate_position,
)


class TestValidatePosition:
    """Tests for validate_position() helper function.

    Business context: Position validation is centralized to ensure
    consistent error handling across all sensor implementations.
    """

    def test_valid_altitude_azimuth(self) -> None:
        """Verifies valid coordinates pass validation.

        Tests that typical coordinate values don't raise errors.

        Business context:
        Standard telescope pointings should always pass.

        Arrangement:
        No setup required - function is pure.

        Action:
        Call validate_position() with various valid coordinates.

        Assertion Strategy:
        No exception raised for valid inputs.

        Testing Principle:
        Validates happy path for normal telescope operations.
        """
        # Standard pointing
        validate_position(45.0, 180.0)

        # Edge cases
        validate_position(0.0, 0.0)  # Horizon north
        validate_position(90.0, 0.0)  # Zenith
        validate_position(45.0, 359.9)  # Near north wrap

    def test_altitude_at_boundaries(self) -> None:
        """Verifies boundary values for altitude.

        Tests 0° and 90° are accepted as valid boundary values.

        Business context:
        Horizon (0°) and zenith (90°) are valid extreme pointings.

        Arrangement:
        No setup required - function is pure.

        Action:
        Call validate_position() with altitude 0° and 90°.

        Assertion Strategy:
        No exception raised at boundaries.

        Testing Principle:
        Validates inclusive boundary handling.
        """
        validate_position(0.0, 180.0)  # Horizon
        validate_position(90.0, 180.0)  # Zenith

    def test_azimuth_at_boundaries(self) -> None:
        """Verifies boundary values for azimuth.

        Tests 0° is valid but 360° is not (wraps to 0).

        Business context:
        Azimuth 0° is north, 360° should be rejected as it equals 0°.

        Arrangement:
        No setup required - function is pure.

        Action:
        Call validate_position() with azimuth 0° and 360°.

        Assertion Strategy:
        0° passes, 360° raises ValueError.

        Testing Principle:
        Validates exclusive upper bound [0, 360).
        """
        validate_position(45.0, 0.0)  # North - valid

        # 360° is invalid (should use 0°)
        with pytest.raises(ValueError, match="Azimuth must be between 0 and 360"):
            validate_position(45.0, 360.0)

    def test_negative_altitude_raises(self) -> None:
        """Verifies negative altitude raises ValueError.

        Tests that altitude below horizon is rejected.

        Business context:
        Negative altitude means below horizon - invalid for
        telescope pointing.

        Arrangement:
        No setup required - function is pure.

        Action:
        Call validate_position() with altitude -1.0 and -90.0.

        Assertion Strategy:
        ValueError raised with descriptive message.

        Testing Principle:
        Validates lower bound enforcement.
        """
        with pytest.raises(ValueError, match="Altitude must be between 0 and 90"):
            validate_position(-1.0, 180.0)

        with pytest.raises(ValueError, match="Altitude must be between 0 and 90"):
            validate_position(-90.0, 180.0)

    def test_altitude_over_90_raises(self) -> None:
        """Verifies altitude > 90° raises ValueError.

        Tests that altitude beyond zenith is rejected.

        Business context:
        Altitude > 90° would be "behind" the telescope - invalid.

        Arrangement:
        No setup required - function is pure.

        Action:
        Call validate_position() with altitude 91.0 and 180.0.

        Assertion Strategy:
        ValueError raised with descriptive message.

        Testing Principle:
        Validates upper bound enforcement.
        """
        with pytest.raises(ValueError, match="Altitude must be between 0 and 90"):
            validate_position(91.0, 180.0)

        with pytest.raises(ValueError, match="Altitude must be between 0 and 90"):
            validate_position(180.0, 180.0)

    def test_negative_azimuth_raises(self) -> None:
        """Verifies negative azimuth raises ValueError.

        Tests that azimuth < 0° is rejected.

        Business context:
        Azimuth should be normalized to 0-360 range by caller.

        Arrangement:
        No setup required - function is pure.

        Action:
        Call validate_position() with azimuth -1.0 and -180.0.

        Assertion Strategy:
        ValueError raised with descriptive message.

        Testing Principle:
        Validates lower bound enforcement for azimuth.
        """
        with pytest.raises(ValueError, match="Azimuth must be between 0 and 360"):
            validate_position(45.0, -1.0)

        with pytest.raises(ValueError, match="Azimuth must be between 0 and 360"):
            validate_position(45.0, -180.0)

    def test_azimuth_at_360_raises(self) -> None:
        """Verifies azimuth at exactly 360° raises ValueError.

        Tests the exclusive upper bound (0 <= az < 360).

        Business context:
        360° equals 0° (north), so should use 0° instead.

        Arrangement:
        No setup required - function is pure.

        Action:
        Call validate_position() with azimuth exactly 360.0.

        Assertion Strategy:
        ValueError raised at exactly 360°.

        Testing Principle:
        Validates exclusive upper bound.
        """
        with pytest.raises(ValueError, match="Azimuth must be between 0 and 360"):
            validate_position(45.0, 360.0)

    def test_error_message_includes_value(self) -> None:
        """Verifies error messages include the invalid value.

        Tests that error messages are descriptive.

        Business context:
        Helpful error messages aid debugging calibration issues.

        Arrangement:
        No setup required - function is pure.

        Action:
        Call validate_position() with invalid values and capture error.

        Assertion Strategy:
        Error message contains the actual invalid value.

        Testing Principle:
        Validates error message quality for debugging.
        """
        with pytest.raises(ValueError, match="got -5.0"):
            validate_position(-5.0, 180.0)

        with pytest.raises(ValueError, match="got 400.0"):
            validate_position(45.0, 400.0)


class TestSensorReading:
    """Tests for SensorReading dataclass.

    Business context: SensorReading is the core data contract
    between sensor drivers and consumers.
    """

    def test_sensor_reading_creation(self) -> None:
        """Verifies SensorReading can be created with all fields.

        Tests dataclass instantiation with typical values.

        Business context:
        SensorReading is created by every read() call.

        Arrangement:
        1. Create AccelerometerData dict with aX, aY, aZ.
        2. Create MagnetometerData dict with mX, mY, mZ.
        3. Get current UTC timestamp.

        Action:
        Construct SensorReading with all required fields.

        Assertion Strategy:
        All fields accessible after construction.

        Testing Principle:
        Validates dataclass construction contract.
        """
        accel: AccelerometerData = {"aX": 0.0, "aY": 0.0, "aZ": 1.0}
        mag: MagnetometerData = {"mX": 30.0, "mY": 0.0, "mZ": 40.0}
        timestamp = datetime.now(UTC)

        reading = SensorReading(
            accelerometer=accel,
            magnetometer=mag,
            altitude=45.0,
            azimuth=180.0,
            temperature=22.5,
            humidity=55.0,
            timestamp=timestamp,
            raw_values="0.0\t0.0\t1.0\t30.0\t0.0\t40.0\t22.5\t55.0",
        )

        assert reading.accelerometer == accel
        assert reading.magnetometer == mag
        assert reading.altitude == 45.0
        assert reading.azimuth == 180.0
        assert reading.temperature == 22.5
        assert reading.humidity == 55.0
        assert reading.timestamp == timestamp
        assert "22.5" in reading.raw_values

    def test_sensor_reading_default_raw_values(self) -> None:
        """Verifies raw_values has default empty string.

        Tests that raw_values is optional.

        Business context:
        Digital twin sensors may not have raw serial data.

        Arrangement:
        1. Create AccelerometerData and MagnetometerData dicts.
        2. Prepare all required fields except raw_values.

        Action:
        Construct SensorReading without raw_values parameter.

        Assertion Strategy:
        Empty string default for raw_values.

        Testing Principle:
        Validates default value for optional field.
        """
        accel: AccelerometerData = {"aX": 0.0, "aY": 0.0, "aZ": 1.0}
        mag: MagnetometerData = {"mX": 30.0, "mY": 0.0, "mZ": 40.0}

        reading = SensorReading(
            accelerometer=accel,
            magnetometer=mag,
            altitude=45.0,
            azimuth=180.0,
            temperature=22.5,
            humidity=55.0,
            timestamp=datetime.now(UTC),
        )

        assert reading.raw_values == ""
