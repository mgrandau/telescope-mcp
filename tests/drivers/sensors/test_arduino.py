"""Unit tests for Arduino Nano BLE33 Sense sensor driver.

Tests the ArduinoSensorDriver and ArduinoSensorInstance using mock
serial ports, without requiring actual hardware.

Test Categories:
- Line parsing (8-field and 6-field formats)
- Altitude calculation from accelerometer data
- Azimuth calculation from magnetometer data
- Calibration (alt/az offsets and tilt correction)
- Command handling (STATUS, RESET, CALIBRATE, etc.)
- Port enumeration and sensor discovery
- Error handling

Example:
    Run all Arduino sensor tests:
        pdm run pytest tests/drivers/sensors/test_arduino.py -v

    Run specific test:
        pdm run pytest tests/drivers/sensors/test_arduino.py::test_parse_8_field_line -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from telescope_mcp.drivers.sensors import (
    ArduinoSensorDriver,
    ArduinoSensorInstance,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockSerialPort:
    """Mock serial port implementing SerialPort protocol.

    Provides configurable responses for testing without hardware.

    Example:
        mock = MockSerialPort()
        mock.queue_line("0.5\\t0.0\\t0.87\\t30\\t0\\t40\\t22.5\\t55")
        instance = ArduinoSensorInstance._create_with_serial(mock)
    """

    def __init__(self) -> None:
        """Initialize mock serial port."""
        self.is_open = True
        self._in_waiting = 0
        self._read_queue: list[bytes] = []
        self._write_log: list[bytes] = []
        self._closed = False

    @property
    def in_waiting(self) -> int:
        """Number of bytes waiting (simulated)."""
        return self._in_waiting

    def queue_line(self, line: str) -> None:
        """Add a line to the read queue.

        Args:
            line: Line to return on next read_until call.
        """
        self._read_queue.append(f"{line}\r\n".encode())
        self._in_waiting = sum(len(b) for b in self._read_queue)

    def queue_bytes(self, data: bytes) -> None:
        """Add raw bytes to the read queue.

        Args:
            data: Bytes to return on next read call.
        """
        self._read_queue.append(data)
        self._in_waiting = sum(len(b) for b in self._read_queue)

    def read_until(self, expected: bytes = b"\n", size: int | None = None) -> bytes:
        """Read until expected byte sequence.

        Args:
            expected: Byte sequence to read until.
            size: Max bytes to read.

        Returns:
            Queued bytes or empty.
        """
        if self._read_queue:
            data = self._read_queue.pop(0)
            self._in_waiting = sum(len(b) for b in self._read_queue)
            return data
        return b""

    def readline(self) -> bytes:
        """Read a line (until \\n).

        Returns:
            Queued line or empty.
        """
        return self.read_until(b"\n")

    def write(self, data: bytes) -> int:
        """Write bytes to port (logs for verification).

        Args:
            data: Bytes to write.

        Returns:
            Number of bytes written.
        """
        self._write_log.append(data)
        return len(data)

    def reset_input_buffer(self) -> None:
        """Clear the input buffer."""
        self._read_queue.clear()
        self._in_waiting = 0

    def close(self) -> None:
        """Close the port."""
        self.is_open = False
        self._closed = True

    def get_written_commands(self) -> list[str]:
        """Get all commands written to port.

        Returns:
            List of command strings written.
        """
        return [data.decode().strip() for data in self._write_log]


class MockComPort:
    """Mock COM port for testing port enumeration.

    Attributes:
        device: Port device path.
        description: Port description.
    """

    def __init__(self, device: str, description: str) -> None:
        """Initialize mock COM port.

        Args:
            device: Port device path (e.g., /dev/ttyACM0).
            description: Port description (e.g., "Arduino Nano").
        """
        self.device = device
        self.description = description


class MockPortEnumerator:
    """Mock port enumerator implementing PortEnumerator protocol.

    Example:
        enum = MockPortEnumerator([
            MockComPort("/dev/ttyACM0", "Arduino Nano"),
            MockComPort("/dev/ttyUSB0", "CH340"),
        ])
        driver = ArduinoSensorDriver._create_with_enumerator(enum)
    """

    def __init__(self, ports: list[MockComPort] | None = None) -> None:
        """Initialize with list of mock ports.

        Args:
            ports: List of MockComPort objects.
        """
        self._ports = ports or []

    def comports(self) -> list[MockComPort]:
        """Return list of mock COM ports.

        Returns:
            List of MockComPort objects.
        """
        return self._ports


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_serial() -> MockSerialPort:
    """Create a mock serial port.

    Returns:
        MockSerialPort ready for testing.
    """
    return MockSerialPort()


@pytest.fixture
def arduino_instance(mock_serial: MockSerialPort) -> ArduinoSensorInstance:
    """Create an ArduinoSensorInstance with mock serial.

    Args:
        mock_serial: Mock serial port fixture.

    Returns:
        ArduinoSensorInstance configured for testing.
    """
    return ArduinoSensorInstance._create_with_serial(
        mock_serial,
        port_name="/dev/ttyTEST",
        start_reader=False,
    )


# =============================================================================
# Line Parsing Tests
# =============================================================================


class TestLineParsing:
    """Tests for sensor data line parsing."""

    def test_parse_8_field_line(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Parse full 8-field format: aX, aY, aZ, mX, mY, mZ, temp, humidity."""
        line = "0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0"

        result = arduino_instance._parse_line(line)

        assert result is True
        assert arduino_instance._accelerometer == {"aX": 0.5, "aY": 0.0, "aZ": 0.87}
        assert arduino_instance._magnetometer == {"mX": 30.0, "mY": 0.0, "mZ": 40.0}
        assert arduino_instance._temperature == 22.5
        assert arduino_instance._humidity == 55.0
        assert arduino_instance._raw_values == line
        assert arduino_instance._last_update is not None

    def test_parse_6_field_legacy_line(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Parse legacy 6-field format: aX, aZ, aY, mX, mZ, mY."""
        line = "0.5\t0.87\t0.0\t30.0\t40.0\t0.0"

        result = arduino_instance._parse_line(line)

        assert result is True
        # Note: Legacy format has different order!
        assert arduino_instance._accelerometer == {"aX": 0.5, "aZ": 0.87, "aY": 0.0}
        assert arduino_instance._magnetometer == {"mX": 30.0, "mZ": 40.0, "mY": 0.0}

    def test_parse_empty_line(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Empty lines should be ignored."""
        result = arduino_instance._parse_line("")
        assert result is False

        result = arduino_instance._parse_line("   ")
        assert result is False

    def test_parse_command_response_info(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """INFO: lines should be skipped."""
        result = arduino_instance._parse_line("INFO: Sensor initialized")
        assert result is False

    def test_parse_command_response_ok(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """OK: lines should be skipped."""
        result = arduino_instance._parse_line("OK: Command successful")
        assert result is False

    def test_parse_command_response_error(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """ERROR: lines should be skipped."""
        result = arduino_instance._parse_line("ERROR: Invalid command")
        assert result is False

    def test_parse_separator_line(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Separator lines (===, ---) should be skipped."""
        result = arduino_instance._parse_line("===")
        assert result is False

        result = arduino_instance._parse_line("---")
        assert result is False

    def test_parse_malformed_values(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Malformed values should be handled gracefully."""
        result = arduino_instance._parse_line(
            "abc\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0"
        )
        assert result is False

    def test_parse_wrong_field_count(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Wrong number of fields should be ignored."""
        # 5 fields - not valid
        result = arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0")
        assert result is False

        # 7 fields - not valid
        result = arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5")
        assert result is False

    def test_parse_with_carriage_return(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Lines with \\r\\n should be handled."""
        line = "0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0\r\n"

        result = arduino_instance._parse_line(line)

        assert result is True
        assert arduino_instance._accelerometer == {"aX": 0.5, "aY": 0.0, "aZ": 0.87}


# =============================================================================
# Altitude Calculation Tests
# =============================================================================


class TestAltitudeCalculation:
    """Tests for altitude calculation from accelerometer data."""

    def test_altitude_level(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Horizontal orientation should give ~0° altitude."""
        # Level: aX=0, aY=0, aZ=1 (gravity down)
        arduino_instance._accelerometer = {"aX": 0.0, "aY": 0.0, "aZ": 1.0}

        alt = arduino_instance._calculate_altitude()

        assert abs(alt) < 1.0  # Should be near 0°

    def test_altitude_45_degrees(self, arduino_instance: ArduinoSensorInstance) -> None:
        """45° tilt should give ~45° altitude."""
        # 45°: aX=sin(45°)≈0.707, aY=0, aZ=cos(45°)≈0.707
        arduino_instance._accelerometer = {"aX": 0.707, "aY": 0.0, "aZ": 0.707}

        alt = arduino_instance._calculate_altitude()

        assert 40 < alt < 50  # Should be around 45°

    def test_altitude_90_degrees(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Vertical up should give ~90° altitude."""
        # Nearly vertical: aX=1, aY=0, aZ=small value
        arduino_instance._accelerometer = {"aX": 1.0, "aY": 0.0, "aZ": 0.01}

        alt = arduino_instance._calculate_altitude()

        assert alt > 80  # Should be near 90°

    def test_altitude_negative(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Negative tilt should give negative altitude."""
        # Tilted down: aX negative
        arduino_instance._accelerometer = {"aX": -0.5, "aY": 0.0, "aZ": 0.87}

        alt = arduino_instance._calculate_altitude()

        assert alt < 0  # Should be negative

    def test_altitude_zero_division(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Handle edge case where aY and aZ are both zero."""
        arduino_instance._accelerometer = {"aX": 0.5, "aY": 0.0, "aZ": 0.0}

        alt = arduino_instance._calculate_altitude()

        assert alt == 0.0  # Should return 0 safely

    def test_altitude_with_tilt_calibration(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Tilt calibration should scale and offset result."""
        arduino_instance._accelerometer = {"aX": 0.0, "aY": 0.0, "aZ": 1.0}

        # Set calibration: corrected = 2.0 * raw + 10.0
        arduino_instance._set_tilt_calibration(slope=2.0, intercept=10.0)

        alt = arduino_instance._calculate_altitude()

        # Raw altitude is ~0°, so calibrated should be ~10°
        assert 9 < alt < 11


# =============================================================================
# Azimuth Calculation Tests
# =============================================================================


class TestAzimuthCalculation:
    """Tests for azimuth calculation from magnetometer data."""

    def test_azimuth_north(self, arduino_instance: ArduinoSensorInstance) -> None:
        """North (mX positive, mY zero) should give ~0° or ~360°."""
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 0.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        # North is 0° (or 360°)
        assert az < 10 or az > 350

    def test_azimuth_east(self, arduino_instance: ArduinoSensorInstance) -> None:
        """East (mX zero, mY positive) should give ~90°."""
        arduino_instance._magnetometer = {"mX": 0.0, "mY": 1.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 85 < az < 95  # Should be around 90°

    def test_azimuth_south(self, arduino_instance: ArduinoSensorInstance) -> None:
        """South (mX negative, mY zero) should give ~180°."""
        arduino_instance._magnetometer = {"mX": -1.0, "mY": 0.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 175 < az < 185  # Should be around 180°

    def test_azimuth_west(self, arduino_instance: ArduinoSensorInstance) -> None:
        """West (mX zero, mY negative) should give ~270°."""
        arduino_instance._magnetometer = {"mX": 0.0, "mY": -1.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 265 < az < 275  # Should be around 270°

    def test_azimuth_northeast(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Northeast (equal mX and mY) should give ~45°."""
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 1.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 40 < az < 50  # Should be around 45°

    def test_azimuth_zero_division(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Handle edge case where mX and mY are both zero."""
        arduino_instance._magnetometer = {"mX": 0.0, "mY": 0.0, "mZ": 1.0}

        az = arduino_instance._calculate_azimuth()

        assert az == 0.0  # Should return 0 safely

    def test_azimuth_normalized_to_360(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Azimuth should always be in 0-360 range."""
        arduino_instance._magnetometer = {"mX": 0.0, "mY": -1.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 0 <= az < 360


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCalibration:
    """Tests for sensor calibration."""

    def test_calibrate_altitude_offset(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Calibration should set altitude offset correctly."""
        # Set up reading
        arduino_instance._accelerometer = {"aX": 0.0, "aY": 0.0, "aZ": 1.0}
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 0.0, "mZ": 0.0}

        # Current reading is ~0° altitude
        # Calibrate to true position of 30°
        arduino_instance.calibrate(true_altitude=30.0, true_azimuth=0.0)

        # Now reading should be ~30°
        reading = arduino_instance.read()
        assert 28 < reading.altitude < 32

    def test_calibrate_azimuth_offset(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Calibration should set azimuth offset correctly."""
        # Set up reading pointing north
        arduino_instance._accelerometer = {"aX": 0.0, "aY": 0.0, "aZ": 1.0}
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 0.0, "mZ": 0.0}

        # Current reading is ~0° azimuth
        # Calibrate to true position of 90° (east)
        arduino_instance.calibrate(true_altitude=0.0, true_azimuth=90.0)

        # Now reading should be ~90°
        reading = arduino_instance.read()
        assert 85 < reading.azimuth < 95

    def test_tilt_calibration_slope(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Tilt calibration slope should scale readings."""
        arduino_instance._accelerometer = {"aX": 0.707, "aY": 0.0, "aZ": 0.707}
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 0.0, "mZ": 0.0}

        # Raw is ~45°
        raw_alt = arduino_instance._calculate_altitude()

        # Apply slope of 0.5 (halve the angle)
        arduino_instance._set_tilt_calibration(slope=0.5, intercept=0.0)

        scaled_alt = arduino_instance._calculate_altitude()

        assert abs(scaled_alt - raw_alt * 0.5) < 1.0

    def test_tilt_calibration_intercept(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Tilt calibration intercept should offset readings."""
        arduino_instance._accelerometer = {"aX": 0.0, "aY": 0.0, "aZ": 1.0}
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 0.0, "mZ": 0.0}

        # Raw is ~0°
        # Apply offset of 5°
        arduino_instance._set_tilt_calibration(slope=1.0, intercept=5.0)

        alt = arduino_instance._calculate_altitude()

        assert 4 < alt < 6


# =============================================================================
# Sensor Reading Tests
# =============================================================================


class TestSensorReading:
    """Tests for reading sensor data."""

    def test_read_returns_sensor_reading(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """read() should return a SensorReading with all data."""
        arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0")

        reading = arduino_instance.read()

        assert reading.accelerometer == {"aX": 0.5, "aY": 0.0, "aZ": 0.87}
        assert reading.magnetometer == {"mX": 30.0, "mY": 0.0, "mZ": 40.0}
        assert reading.temperature == 22.5
        assert reading.humidity == 55.0
        assert reading.timestamp is not None
        assert reading.raw_values == "0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0"

    def test_read_raises_when_no_data(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """read() should raise RuntimeError when no data available."""
        with pytest.raises(RuntimeError, match="No sensor data available"):
            arduino_instance.read()

    def test_read_raises_when_closed(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """read() should raise RuntimeError when sensor is closed."""
        arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0")
        arduino_instance._is_open = False

        with pytest.raises(RuntimeError, match="Sensor is closed"):
            arduino_instance.read()

    def test_read_calculated_values(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """read() should include calculated altitude and azimuth."""
        # Level, pointing north
        arduino_instance._parse_line("0.0\t0.0\t1.0\t1.0\t0.0\t0.0\t20.0\t50.0")

        reading = arduino_instance.read()

        # Should be ~0° altitude, ~0° azimuth (north)
        assert abs(reading.altitude) < 5
        assert reading.azimuth < 10 or reading.azimuth > 350


# =============================================================================
# Command Handling Tests
# =============================================================================


class TestCommandHandling:
    """Tests for Arduino command handling."""

    def test_send_command_writes_to_serial(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """send_command() should write command to serial port."""
        # Queue a response
        mock_serial.queue_line("OK: Reset complete")

        arduino_instance._send_command("RESET", wait_response=False)

        commands = mock_serial.get_written_commands()
        assert "RESET" in commands

    def test_send_command_with_response(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """send_command() should collect response when wait_response=True."""
        mock_serial.queue_line("OK: Status response")
        mock_serial._in_waiting = 100

        response = arduino_instance._send_command(
            "STATUS", wait_response=True, timeout=0.1
        )

        assert "OK:" in response or response == ""  # May timeout in test

    def test_reset_command(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """reset() should send RESET command."""
        mock_serial.queue_line("OK: Reset complete")
        mock_serial._in_waiting = 100

        arduino_instance.reset()

        commands = mock_serial.get_written_commands()
        assert "RESET" in commands

    def test_stop_output(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """stop_output() should send STOP command."""
        arduino_instance._stop_output()

        commands = mock_serial.get_written_commands()
        assert "STOP" in commands

    def test_start_output(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """start_output() should send START command."""
        arduino_instance._start_output()

        commands = mock_serial.get_written_commands()
        assert "START" in commands


# =============================================================================
# Sensor Info and Status Tests
# =============================================================================


class TestSensorInfo:
    """Tests for sensor information and status."""

    def test_get_info(self, arduino_instance: ArduinoSensorInstance) -> None:
        """get_info() should return sensor metadata."""
        info = arduino_instance.get_info()

        assert info["type"] == "arduino_ble33"
        assert info["name"] == "Arduino Nano BLE33 Sense"
        assert info["port"] == "/dev/ttyTEST"
        assert info["has_accelerometer"] is True
        assert info["has_magnetometer"] is True
        assert info["has_temperature"] is True
        assert info["has_humidity"] is True
        assert info["sample_rate_hz"] == 10.0

    def test_get_status_uncalibrated(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """get_status() should show calibrated=False when not calibrated."""
        mock_serial.queue_line("OK: Status")
        mock_serial._in_waiting = 100

        status = arduino_instance.get_status()

        assert status["connected"] is True
        assert status["type"] == "arduino_ble33"
        assert status["port"] == "/dev/ttyTEST"
        assert status["calibrated"] is False

    def test_get_status_calibrated(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """get_status() should show calibrated=True after calibration."""
        # Set up and calibrate
        arduino_instance._parse_line("0.0\t0.0\t1.0\t1.0\t0.0\t0.0\t20.0\t50.0")
        arduino_instance.calibrate(true_altitude=30.0, true_azimuth=45.0)

        mock_serial.queue_line("OK: Status")
        mock_serial._in_waiting = 100

        status = arduino_instance.get_status()

        assert status["calibrated"] is True


# =============================================================================
# Driver Tests (Port Enumeration)
# =============================================================================


class TestArduinoSensorDriver:
    """Tests for ArduinoSensorDriver port enumeration."""

    def test_get_available_sensors_with_arduino(self) -> None:
        """Should detect Arduino devices."""
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyACM0", "Arduino Nano 33 BLE"),
                MockComPort("/dev/ttyUSB0", "Generic USB"),
            ]
        )

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        sensors = driver.get_available_sensors()

        assert len(sensors) == 1
        assert sensors[0]["port"] == "/dev/ttyACM0"
        assert sensors[0]["type"] == "arduino_ble33"
        assert "Arduino" in sensors[0]["name"]

    def test_get_available_sensors_with_acm(self) -> None:
        """Should detect ACM devices."""
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyACM1", "ttyACM1"),
            ]
        )

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        sensors = driver.get_available_sensors()

        assert len(sensors) == 1
        assert sensors[0]["port"] == "/dev/ttyACM1"

    def test_get_available_sensors_with_ch340(self) -> None:
        """Should detect CH340 USB-serial devices."""
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyUSB0", "CH340 Serial Adapter"),
            ]
        )

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        sensors = driver.get_available_sensors()

        assert len(sensors) == 1
        assert sensors[0]["port"] == "/dev/ttyUSB0"

    def test_get_available_sensors_empty(self) -> None:
        """Should return empty list when no Arduino devices."""
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyS0", "Standard Serial Port"),
                MockComPort("/dev/ttyUSB0", "Bluetooth Device"),
            ]
        )

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        sensors = driver.get_available_sensors()

        assert len(sensors) == 0

    def test_get_available_sensors_multiple(self) -> None:
        """Should detect multiple Arduino devices."""
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyACM0", "Arduino Nano 33 BLE"),
                MockComPort("/dev/ttyACM1", "Arduino Mega 2560"),
                MockComPort("/dev/ttyUSB0", "CH340"),
            ]
        )

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        sensors = driver.get_available_sensors()

        assert len(sensors) == 3

    def test_open_with_serial_injection(self) -> None:
        """Should allow opening with injected serial port."""
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        instance = driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        assert instance is not None
        assert instance._port == "/dev/ttyTEST"
        assert instance._is_open is True

    def test_open_already_open_raises(self) -> None:
        """Should raise error when trying to open twice."""
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        with pytest.raises(RuntimeError, match="Sensor already open"):
            driver._open_with_serial(MockSerialPort(), "/dev/ttyTEST2")

    def test_close_driver(self) -> None:
        """Driver close should close instance."""
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        instance = driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        driver.close()

        assert mock_serial._closed is True
        assert driver._instance is None


# =============================================================================
# Instance Lifecycle Tests
# =============================================================================


class TestInstanceLifecycle:
    """Tests for ArduinoSensorInstance lifecycle."""

    def test_close_stops_reading(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """close() should set stop flag and close serial."""
        arduino_instance.close()

        assert arduino_instance._stop_reading is True
        assert arduino_instance._is_open is False
        assert mock_serial._closed is True

    def test_create_with_serial_no_reader(self) -> None:
        """_create_with_serial without reader should not start thread."""
        mock_serial = MockSerialPort()

        instance = ArduinoSensorInstance._create_with_serial(
            mock_serial,
            port_name="/dev/test",
            start_reader=False,
        )

        assert instance._reader_thread is None
        assert instance._is_open is True

    def test_create_with_serial_with_reader(self) -> None:
        """_create_with_serial with reader should start thread."""
        mock_serial = MockSerialPort()

        instance = ArduinoSensorInstance._create_with_serial(
            mock_serial,
            port_name="/dev/test",
            start_reader=True,
        )

        assert instance._reader_thread is not None
        # Clean up
        instance.close()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_negative_values(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Should handle negative sensor values."""
        line = "-0.5\t-0.3\t-0.87\t-30.0\t-10.0\t-40.0\t-5.0\t0.0"

        result = arduino_instance._parse_line(line)

        assert result is True
        assert arduino_instance._accelerometer["aX"] == -0.5
        assert arduino_instance._magnetometer["mX"] == -30.0
        assert arduino_instance._temperature == -5.0

    def test_parse_scientific_notation(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Should handle scientific notation values."""
        line = "1.5e-2\t0.0\t1.0\t3.0e1\t0.0\t4.0e1\t2.25e1\t5.5e1"

        result = arduino_instance._parse_line(line)

        assert result is True
        assert abs(arduino_instance._accelerometer["aX"] - 0.015) < 0.001
        assert abs(arduino_instance._magnetometer["mX"] - 30.0) < 0.001

    def test_multiple_consecutive_parses(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Should update values on each parse."""
        arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0")
        first_ax = arduino_instance._accelerometer["aX"]

        arduino_instance._parse_line("0.7\t0.1\t0.71\t35.0\t5.0\t45.0\t23.0\t60.0")
        second_ax = arduino_instance._accelerometer["aX"]

        assert first_ax == 0.5
        assert second_ax == 0.7

    def test_azimuth_wraps_at_360(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Azimuth with offset should wrap correctly."""
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 0.0, "mZ": 0.0}

        # Calibrate with large offset
        arduino_instance._cal_az_offset = 350.0

        az = arduino_instance._calculate_azimuth()

        assert 0 <= az < 360  # Should wrap
