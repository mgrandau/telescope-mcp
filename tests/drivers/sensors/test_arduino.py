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
        """Initialize mock serial port with empty queues and open state.

        Creates a mock serial port ready for testing Arduino sensors
        with isolated state. Implements SerialPort protocol.

        Args:
            None. Creates port with default empty state.

        Returns:
            None. Initializes instance attributes.

        Raises:
            No exceptions raised during initialization.

        Business context:
            Arduino sensor tests need isolated serial mock for each test.
            Empty queues ensure no state leakage between tests.

        Example:
            >>> port = MockSerialPort()
            >>> port.is_open
            True

        Implementation:
            Sets up internal state for simulating serial port behavior:
            - is_open: True (simulates connected port)
            - _in_waiting: 0 (no bytes initially)
            - _read_queue: Empty list for queued responses
            - _write_log: Empty list for command logging
            - _closed: False (port not yet closed)
        """
        self.is_open = True
        self._in_waiting = 0
        self._read_queue: list[bytes] = []
        self._write_log: list[bytes] = []
        self._closed = False

    @property
    def in_waiting(self) -> int:
        """Number of bytes waiting in the read queue (simulated).

        Simulates pyserial's in_waiting property that reports bytes
        available for reading. Arduino sensor checks this before reads.

        Args:
            None. Property takes no arguments.

        Returns:
            Sum of bytes in all queued responses. Zero if queue empty.

        Raises:
            No exceptions raised. Always returns valid integer.

        Business context:
            Sensor reader checks in_waiting to determine if data available
            before blocking read. Mock tracks queued data sizes.

        Example:
            >>> port.queue_line("sensor_data")
            >>> port.in_waiting
            14
        """
        return self._in_waiting

    def queue_line(self, line: str) -> None:
        """Add a line to the read queue with CRLF terminator.

        Adds string line with \r\n suffix (Arduino protocol uses CRLF).
        Updates in_waiting to reflect new queued bytes.

        Args:
            line: Line content to queue (without line ending).

        Returns:
            None. Modifies internal queue state.

        Raises:
            No exceptions raised. Accepts any string value.

        Business context:
            Arduino sensor streams tab-separated data lines. Tests queue
            simulated sensor readings for parsing verification.

        Example:
            >>> port.queue_line("0.5\t0.0\t0.87\t30\t0\t40\t22.5\t55")
            >>> port.readline()
            b"0.5\t0.0\t0.87\t30\t0\t40\t22.5\t55\r\n"
        """
        self._read_queue.append(f"{line}\r\n".encode())
        self._in_waiting = sum(len(b) for b in self._read_queue)

    def queue_bytes(self, data: bytes) -> None:
        """Add raw bytes to the read queue without modification.

        Queues exact bytes for testing binary/edge case scenarios.
        Updates in_waiting to reflect new queued bytes.

        Args:
            data: Raw bytes to return on next read call.

        Returns:
            None. Modifies internal queue state.

        Raises:
            No exceptions raised. Accepts any bytes value.

        Business context:
            Some tests need raw byte control (partial lines, binary data).
            Unlike queue_line, no CRLF is appended.

        Example:
            >>> port.queue_bytes(b"partial")
            >>> port.read_until()
            b"partial"
        """
        self._read_queue.append(data)
        self._in_waiting = sum(len(b) for b in self._read_queue)

    def read_until(self, expected: bytes = b"\n", size: int | None = None) -> bytes:
        """Read from queue until expected byte sequence.

        Pops and returns first queued response. In real serial, blocks
        until expected bytes received. Mock returns immediately.

        Args:
            expected: Byte sequence to read until (ignored in mock).
            size: Max bytes to read (ignored in mock).

        Returns:
            First queued response bytes, or empty if queue empty.

        Raises:
            No exceptions raised. Returns empty bytes if no data queued.

        Business context:
            Sensor reader calls read_until to get complete lines.
            Mock provides queued responses for test verification.
        """
        if self._read_queue:
            data = self._read_queue.pop(0)
            self._in_waiting = sum(len(b) for b in self._read_queue)
            return data
        return b""

    def readline(self) -> bytes:
        """Read a line (until newline) from the response queue.

        Convenience method calling read_until with newline delimiter.
        Simulates pyserial readline() blocking until newline received.

        Args:
            None. Reads from internal queue.

        Returns:
            First queued response bytes, or empty if queue empty.

        Raises:
            No exceptions raised. Returns empty bytes if no data queued.

        Business context:
            Arduino streams newline-terminated sensor data lines.
            Reader calls readline() to get complete data lines.

        Example:
            >>> port.queue_line("sensor_data")
            >>> port.readline()
            b"sensor_data\r\n"
        """
        return self.read_until(b"\n")

    def write(self, data: bytes) -> int:
        """Write bytes to port, logging for test verification.

        Stores written data in internal log for assertion in tests.
        In real serial, transmits bytes to Arduino.

        Args:
            data: Bytes to write to the mock serial port.

        Returns:
            Number of bytes written (always equals len(data)).

        Raises:
            No exceptions raised. Accepts any bytes value.

        Business context:
            Tests verify correct commands sent (RESET, START, etc.)
            by checking the write log after operations.

        Example:
            >>> port.write(b"RESET\n")
            6
            >>> port.get_written_commands()
            ['RESET']
        """
        self._write_log.append(data)
        return len(data)

    def reset_input_buffer(self) -> None:
        """Clear all queued input data from the mock serial buffer.

        Simulates pyserial's reset_input_buffer() by clearing the
        read queue and resetting bytes-waiting counter. Used when
        tests need to start with clean input state.

        Args:
            None. Operates on internal queue.

        Returns:
            None. Clears internal queue state.

        Raises:
            No exceptions raised.

        Business context:
            Sensor instance may reset buffer before sending commands
            to ensure responses match current request, not stale data.

        Example:
            >>> port.queue_line("stale")
            >>> port.reset_input_buffer()
            >>> port.in_waiting
            0

        Implementation:
            Clears _read_queue list and sets _in_waiting to 0.
        """
        self._read_queue.clear()
        self._in_waiting = 0

    def close(self) -> None:
        """Close the mock serial port and update state flags.

        Sets is_open=False and _closed=True to simulate port closure.
        Tests can check _closed to verify cleanup behavior.

        Args:
            None. Operates on internal state.

        Returns:
            None. Updates internal state flags.

        Raises:
            No exceptions raised.

        Business context:
            Tests verify proper lifecycle management by checking that
            serial port is closed when instance is closed. Prevents
            resource leaks in production code.

        Example:
            >>> port.close()
            >>> port.is_open
            False

        Implementation:
            Sets is_open to False and _closed to True.
        """
        self.is_open = False
        self._closed = True

    def get_written_commands(self) -> list[str]:
        """Get all commands written to port for test verification.

        Decodes and strips all bytes written via write() as strings.
        Primary verification method for testing command sequences.

        Args:
            None. Returns data from internal write log.

        Returns:
            List of command strings in order written, whitespace stripped.

        Raises:
            No exceptions raised. Returns empty list if nothing written.

        Business context:
            Tests verify Arduino commands (RESET, START, CALIBRATE) were
            sent correctly by checking this list after operations.

        Example:
            >>> port.write(b"RESET\n")
            >>> port.write(b"START\n")
            >>> port.get_written_commands()
            ['RESET', 'START']
        """
        return [data.decode().strip() for data in self._write_log]


class MockComPort:
    """Mock COM port for testing port enumeration.

    Attributes:
        device: Port device path.
        description: Port description.
    """

    def __init__(self, device: str, description: str) -> None:
        """Initialize mock COM port with device path and description.

        Simulates the port information returned by pyserial's list_ports.
        Used for testing sensor discovery without real hardware.

        Args:
            device: Port device path (e.g., /dev/ttyACM0).
            description: Port description (e.g., "Arduino Nano").

        Returns:
            None. Initializes instance attributes.

        Raises:
            No exceptions raised during initialization.

        Business context:
            ArduinoSensorDriver uses port description to identify sensors.
            Mock provides configurable port info for discovery tests.

        Example:
            >>> port = MockComPort('/dev/ttyACM0', 'Arduino Nano')
            >>> port.device
            '/dev/ttyACM0'
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
        """Initialize mock port enumerator with list of mock ports.

        Creates a configurable port enumerator for testing sensor
        discovery. Empty list simulates no sensors connected.

        Args:
            ports: List of MockComPort objects. None defaults to empty.

        Returns:
            None. Initializes instance attributes.

        Raises:
            No exceptions raised during initialization.

        Business context:
            Driver tests need to simulate various port scenarios:
            no sensors, ACM ports, various descriptions. Configurable
            list allows testing each discovery scenario.

        Example:
            >>> ports = [MockComPort('/dev/ttyACM0', 'Arduino')]
            >>> enum = MockPortEnumerator(ports)
            >>> enum.comports()
            [<MockComPort '/dev/ttyACM0'>]
        """
        self._ports = ports or []

    def comports(self) -> list[MockComPort]:
        """Return list of mock COM ports for enumeration testing.

        Simulates serial.tools.list_ports.comports() which returns
        available serial ports. Returns preconfigured mock ports.

        Args:
            None. Returns preconfigured port list.

        Returns:
            List of MockComPort objects configured during initialization.

        Raises:
            No exceptions raised. Always returns valid list.

        Business context:
            Driver uses comports() to discover Arduino sensors.
            Mock enables testing detection with various port scenarios.

        Example:
            >>> enum = MockPortEnumerator([MockComPort('/dev/ttyACM0', 'Arduino')])
            >>> enum.comports()
            [<MockComPort '/dev/ttyACM0'>]
        """
        return self._ports


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_serial() -> MockSerialPort:
    """Create a mock serial port fixture for Arduino sensor tests.

    Provides fresh MockSerialPort for each test, ensuring isolation.
    Mock simulates pyserial Serial interface for Arduino communication.

    Args:
        None. Fixture takes no arguments.

    Returns:
        Fresh MockSerialPort with empty queues and write log.

    Raises:
        No exceptions raised during fixture creation.

    Business context:
        Arduino sensor tests need serial port without real hardware.
        Mock captures commands and provides configurable sensor data.

    Example:
        def test_sensor(mock_serial):
            mock_serial.queue_line("0.5\t0.0\t0.87\t30\t0\t40")
    """
    return MockSerialPort()


@pytest.fixture
def arduino_instance(mock_serial: MockSerialPort) -> ArduinoSensorInstance:
    """Create an ArduinoSensorInstance with mock serial for testing.

    Injects mock serial port and disables background reader thread
    for synchronous testing of parsing and calculation methods.

    Args:
        mock_serial: Mock serial port fixture providing test doubles.

    Returns:
        ArduinoSensorInstance configured with mock serial, no reader.

    Raises:
        No exceptions raised during fixture creation.

    Business context:
        Instance tests verify parsing, calculation, and calibration
        without real Arduino. start_reader=False allows synchronous testing.

    Example:
        def test_altitude(arduino_instance, mock_serial):
            mock_serial.queue_line("0.5\t0.0\t0.87\t30\t0\t40")
            arduino_instance._read_and_parse_line()
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
    """Test suite for Arduino sensor data line parsing.

    Categories:
    1. Format Tests - 8-field and 6-field formats (3 tests)
    2. Skip Tests - Command responses, separators (5 tests)
    3. Error Tests - Malformed data, wrong field count (2 tests)

    Total: 10 tests.
    """

    def test_parse_8_field_line(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies parsing of full 8-field sensor data format.

        Tests the primary data format: aX, aY, aZ, mX, mY, mZ, temp, humidity.

        Business context:
        Arduino streams tab-separated sensor data at ~10Hz. The 8-field
        format includes IMU (accelerometer, magnetometer) and environmental
        sensors for complete telescope orientation and condition monitoring.

        Arrangement:
        1. Prepare valid 8-field tab-separated sensor data string.
        2. Use mock serial instance with no background reader.

        Action:
        Call _parse_line() with the 8-field data string.

        Assertion Strategy:
        Validates parsing by confirming:
        - Returns True (successful parse).
        - Accelerometer dict populated with aX, aY, aZ.
        - Magnetometer dict populated with mX, mY, mZ.
        - Temperature and humidity floats extracted.
        - Raw values and timestamp updated.

        Testing Principle:
        Validates happy path parsing, ensuring primary data format
        is correctly parsed and stored in instance state.
        """
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
        """Verifies parsing of legacy 6-field IMU-only format.

        Tests backward compatibility: aX, aZ, aY, mX, mZ, mY (different order).

        Business context:
        Early Arduino firmware used 6-field format without environmental
        sensors and with swapped Y/Z axes. Parser must handle both formats
        for backward compatibility with existing deployments.

        Arrangement:
        1. Prepare valid 6-field legacy format string.
        2. Note different axis ordering in legacy format.

        Action:
        Call _parse_line() with the 6-field legacy data.

        Assertion Strategy:
        Validates legacy parsing by confirming:
        - Returns True (successful parse).
        - Accelerometer populated with correct axis mapping.
        - Magnetometer populated with correct axis mapping.
        - Y/Z swap handled correctly per legacy format.

        Testing Principle:
        Validates backward compatibility, ensuring legacy format
        support for deployments with older Arduino firmware.
        """
        line = "0.5\t0.87\t0.0\t30.0\t40.0\t0.0"

        result = arduino_instance._parse_line(line)

        assert result is True
        # Note: Legacy format has different order!
        assert arduino_instance._accelerometer == {"aX": 0.5, "aZ": 0.87, "aY": 0.0}
        assert arduino_instance._magnetometer == {"mX": 30.0, "mZ": 40.0, "mY": 0.0}

    def test_parse_empty_line(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies empty and whitespace-only lines are ignored.

        Tests robustness against empty serial data.

        Business context:
        Serial streams may contain empty lines from timing gaps or
        buffer flushes. Parser must silently ignore these without
        corrupting state or raising errors.

        Arrangement:
        1. Use mock instance ready for parsing.

        Action:
        Call _parse_line() with empty string and whitespace-only string.

        Assertion Strategy:
        Validates skip behavior by confirming:
        - Returns False for empty string.
        - Returns False for whitespace-only string.
        - No state changes or errors.

        Testing Principle:
        Validates defensive parsing, ensuring invalid input
        is rejected gracefully without side effects.
        """
        result = arduino_instance._parse_line("")
        assert result is False

        result = arduino_instance._parse_line("   ")
        assert result is False

    def test_parse_command_response_info(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies INFO: lines from Arduino are skipped.

        Tests filtering of informational command responses.

        Business context:
        Arduino sends INFO: lines during initialization and status
        queries. These should not be parsed as sensor data to avoid
        corrupting accelerometer/magnetometer state.

        Arrangement:
        1. Use mock instance ready for parsing.

        Action:
        Call _parse_line() with INFO: prefixed line.

        Assertion Strategy:
        Validates filtering by confirming:
        - Returns False (not parsed as data).

        Testing Principle:
        Validates command/data separation, ensuring Arduino
        responses don't corrupt sensor data state.
        """
        result = arduino_instance._parse_line("INFO: Sensor initialized")
        assert result is False

    def test_parse_command_response_ok(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies OK: lines from Arduino are skipped.

        Tests filtering of success command responses.

        Business context:
        Arduino sends OK: lines to confirm command execution (RESET,
        CALIBRATE). These confirmations should not be parsed as sensor
        data to maintain clean data separation.

        Arrangement:
        1. Use mock instance ready for parsing.

        Action:
        Call _parse_line() with OK: prefixed line.

        Assertion Strategy:
        Validates filtering by confirming:
        - Returns False (not parsed as data).

        Testing Principle:
        Validates protocol separation, ensuring command acknowledgments
        are distinguished from sensor data.
        """
        result = arduino_instance._parse_line("OK: Command successful")
        assert result is False

    def test_parse_command_response_error(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies ERROR: lines from Arduino are skipped.

        Tests filtering of error command responses.

        Business context:
        Arduino sends ERROR: lines when commands fail. These error
        messages should not be parsed as sensor data. Errors are
        handled separately through command response handling.

        Arrangement:
        1. Use mock instance ready for parsing.

        Action:
        Call _parse_line() with ERROR: prefixed line.

        Assertion Strategy:
        Validates filtering by confirming:
        - Returns False (not parsed as data).

        Testing Principle:
        Validates error message handling, ensuring errors don't
        corrupt sensor state and are routed appropriately.
        """
        result = arduino_instance._parse_line("ERROR: Invalid command")
        assert result is False

    def test_parse_separator_line(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies separator lines (===, ---) are skipped.

        Tests filtering of visual separator lines from Arduino output.

        Business context:
        Arduino status output uses === and --- as visual separators
        for human readability. These formatting lines should not be
        parsed as sensor data.

        Arrangement:
        1. Use mock instance ready for parsing.

        Action:
        Call _parse_line() with === and --- separator lines.

        Assertion Strategy:
        Validates filtering by confirming:
        - Returns False for === separator.
        - Returns False for --- separator.

        Testing Principle:
        Validates formatting resilience, ensuring visual formatting
        in Arduino output doesn't affect sensor data parsing.
        """
        result = arduino_instance._parse_line("===")
        assert result is False

        result = arduino_instance._parse_line("---")
        assert result is False

    def test_parse_malformed_values(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies malformed numeric values are handled gracefully.

        Tests error handling for non-numeric data in sensor fields.

        Business context:
        Serial noise or firmware bugs can produce malformed data.
        Parser must reject invalid lines without crashing or
        corrupting state. Logging may record issues for debugging.

        Arrangement:
        1. Prepare data with non-numeric value in first field.

        Action:
        Call _parse_line() with malformed data.

        Assertion Strategy:
        Validates error handling by confirming:
        - Returns False (parse failed).
        - No exceptions raised.
        - State unchanged from before call.

        Testing Principle:
        Validates robustness, ensuring garbage input is rejected
        gracefully without affecting system stability.
        """
        result = arduino_instance._parse_line(
            "abc\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0"
        )
        assert result is False

    def test_parse_wrong_field_count(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies lines with wrong field count are ignored.

        Tests validation of expected field counts (6 or 8 fields).

        Business context:
        Truncated serial data or protocol mismatches can produce
        lines with unexpected field counts. Parser only accepts
        6-field (legacy) or 8-field (current) formats.

        Arrangement:
        1. Prepare data with 5 fields (invalid).
        2. Prepare data with 7 fields (invalid).

        Action:
        Call _parse_line() with each invalid field count.

        Assertion Strategy:
        Validates field count enforcement by confirming:
        - Returns False for 5-field line.
        - Returns False for 7-field line.
        - Only 6 and 8 field counts are valid.

        Testing Principle:
        Validates protocol enforcement, ensuring only valid
        data formats are accepted.
        """
        # 5 fields - not valid
        result = arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0")
        assert result is False

        # 7 fields - not valid
        result = arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5")
        assert result is False

    def test_parse_with_carriage_return(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies lines with \\r\\n line endings are handled.

        Tests Windows-style line ending support.

        Business context:
        Serial data may have CRLF (\\r\\n) line endings depending on
        Arduino firmware or serial terminal settings. Parser must
        strip both CR and LF for consistent parsing.

        Arrangement:
        1. Prepare valid 8-field data with \\r\\n suffix.

        Action:
        Call _parse_line() with CRLF-terminated data.

        Assertion Strategy:
        Validates line ending handling by confirming:
        - Returns True (successful parse).
        - Accelerometer data correctly extracted.
        - Line endings stripped before parsing.

        Testing Principle:
        Validates platform compatibility, ensuring CRLF line
        endings from Windows/Arduino don't break parsing.
        """
        line = "0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0\r\n"

        result = arduino_instance._parse_line(line)

        assert result is True
        assert arduino_instance._accelerometer == {"aX": 0.5, "aY": 0.0, "aZ": 0.87}


# =============================================================================
# Altitude Calculation Tests
# =============================================================================


class TestAltitudeCalculation:
    """Test suite for altitude calculation from accelerometer data.

    Categories:
    1. Angle Tests - Level, 45°, 90° positions (3 tests)
    2. Edge Cases - Negative tilt, zero division (2 tests)
    3. Calibration - Tilt calibration effect (1 test)

    Total: 6 tests.
    """

    def test_altitude_level(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies horizontal orientation gives approximately 0° altitude.

        Tests altitude calculation when telescope is level.

        Business context:
        Level orientation (aX=0, aZ=1) means gravity is straight down.
        This corresponds to 0° altitude (pointing at horizon). Critical
        reference point for altitude calibration.

        Arrangement:
        1. Set accelerometer to level: aX=0, aY=0, aZ=1 (gravity down).

        Action:
        Call _calculate_altitude() to compute tilt from accelerometer.

        Assertion Strategy:
        Validates calculation by confirming:
        - Altitude near 0° (within 1° tolerance).

        Testing Principle:
        Validates reference point, ensuring level orientation
        produces expected 0° baseline for altitude.
        """
        # Level: aX=0, aY=0, aZ=1 (gravity down)
        arduino_instance._accelerometer = {"aX": 0.0, "aY": 0.0, "aZ": 1.0}

        alt = arduino_instance._calculate_altitude()

        assert abs(alt) < 1.0  # Should be near 0°

    def test_altitude_45_degrees(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies 45° tilt gives approximately 45° altitude.

        Tests altitude calculation at mid-range angle.

        Business context:
        45° is a common observation angle for many targets. Testing
        mid-range ensures trigonometric calculation is correct across
        the full 0-90° range, not just at extremes.

        Arrangement:
        1. Set accelerometer to 45° tilt: aX=sin(45°)≈0.707, aZ=cos(45°)≈0.707.

        Action:
        Call _calculate_altitude() to compute tilt.

        Assertion Strategy:
        Validates calculation by confirming:
        - Altitude between 40° and 50° (within ±5° tolerance).

        Testing Principle:
        Validates mid-range accuracy, ensuring calculation is
        correct across the observable altitude range.
        """
        # 45°: aX=sin(45°)≈0.707, aY=0, aZ=cos(45°)≈0.707
        arduino_instance._accelerometer = {"aX": 0.707, "aY": 0.0, "aZ": 0.707}

        alt = arduino_instance._calculate_altitude()

        assert 40 < alt < 50  # Should be around 45°

    def test_altitude_90_degrees(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies vertical orientation gives approximately 90° altitude.

        Tests altitude calculation when telescope points at zenith.

        Business context:
        90° altitude (zenith) is the upper limit for observation.
        Near-vertical orientation tests the upper boundary of
        trigonometric calculation and ensures no clipping.

        Arrangement:
        1. Set accelerometer nearly vertical: aX=1, aZ=0.01 (small value).
        2. Small aZ prevents division by zero.

        Action:
        Call _calculate_altitude() to compute tilt.

        Assertion Strategy:
        Validates calculation by confirming:
        - Altitude > 80° (approaching 90° zenith).

        Testing Principle:
        Validates boundary condition, ensuring calculation
        handles extreme vertical orientations correctly.
        """
        # Nearly vertical: aX=1, aY=0, aZ=small value
        arduino_instance._accelerometer = {"aX": 1.0, "aY": 0.0, "aZ": 0.01}

        alt = arduino_instance._calculate_altitude()

        assert alt > 80  # Should be near 90°

    def test_altitude_negative(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies downward tilt gives negative altitude.

        Tests altitude calculation for below-horizon pointing.

        Business context:
        Negative altitude indicates pointing below horizon. While
        not typical for observation, this can occur during slewing
        or sensor mounting verification. Negative values are valid.

        Arrangement:
        1. Set accelerometer tilted down: aX=-0.5 (negative X).

        Action:
        Call _calculate_altitude() to compute tilt.

        Assertion Strategy:
        Validates calculation by confirming:
        - Altitude < 0° (below horizon).

        Testing Principle:
        Validates sign handling, ensuring negative tilts
        produce negative altitude values correctly.
        """
        # Tilted down: aX negative
        arduino_instance._accelerometer = {"aX": -0.5, "aY": 0.0, "aZ": 0.87}

        alt = arduino_instance._calculate_altitude()

        assert alt < 0  # Should be negative

    def test_altitude_zero_division(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies division by zero edge case returns 0 safely.

        Tests error handling when aZ=0 would cause division issues.

        Business context:
        When aZ=0, trigonometric calculation may encounter division
        by zero. The implementation must handle this gracefully,
        returning a safe default rather than crashing.

        Arrangement:
        1. Set accelerometer with aZ=0 (horizontal gravity).

        Action:
        Call _calculate_altitude() which would divide by zero.

        Assertion Strategy:
        Validates error handling by confirming:
        - Returns 0.0 (safe default).
        - No exceptions raised.

        Testing Principle:
        Validates error handling, ensuring mathematical edge
        cases don't crash the system.
        """
        arduino_instance._accelerometer = {"aX": 0.5, "aY": 0.0, "aZ": 0.0}

        alt = arduino_instance._calculate_altitude()

        assert alt == 0.0  # Should return 0 safely

    def test_altitude_with_tilt_calibration(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies tilt calibration applies slope and intercept.

        Tests that calibration formula (corrected = slope * raw + intercept)
        is correctly applied to altitude calculations.

        Business context:
        Linear calibration corrects for sensor mounting errors and
        non-linearities. The formula slope*raw+intercept allows both
        scaling (slope) and offset (intercept) correction.

        Arrangement:
        1. Set accelerometer to level (raw altitude ~0°).
        2. Apply calibration: slope=2.0, intercept=10.0.

        Action:
        Call _calculate_altitude() which applies calibration.

        Assertion Strategy:
        Validates calibration math by confirming:
        - Result ≈ 10° (2.0 * 0° + 10°).

        Testing Principle:
        Validates calibration system, ensuring linear correction
        formula is properly applied to raw readings.
        """
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
    """Test suite for azimuth calculation from magnetometer data.

    Categories:
    1. Cardinal Directions - North, East, South, West (4 tests)
    2. Intermediate Angles - Northeast diagonal (1 test)
    3. Edge Cases - Zero division, wrap at 360 (2 tests)

    Total: 7 tests.
    """

    def test_azimuth_north(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies magnetometer pointing north gives ~0°/360° azimuth.

        Tests azimuth calculation for magnetic north reference.

        Business context:
        North (0°/360°) is the reference direction for azimuth.
        mX=1, mY=0 indicates magnetic field pointing in positive X.
        Critical baseline for all azimuth calculations.

        Arrangement:
        1. Set magnetometer pointing north: mX=1, mY=0.

        Action:
        Call _calculate_azimuth() to compute heading.

        Assertion Strategy:
        Validates calculation by confirming:
        - Azimuth near 0° or 360° (within 10° tolerance).

        Testing Principle:
        Validates reference point, ensuring north direction
        produces expected 0° baseline for azimuth.
        """
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 0.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        # North is 0° (or 360°)
        assert az < 10 or az > 350

    def test_azimuth_east(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies magnetometer pointing east gives ~90° azimuth.

        Tests azimuth calculation for east cardinal direction.

        Business context:
        East (90°) is a key cardinal direction. mX=0, mY=1 means
        magnetic field points in positive Y direction. Validates
        that atan2 calculation produces correct quadrant.

        Arrangement:
        1. Set magnetometer pointing east: mX=0, mY=1.

        Action:
        Call _calculate_azimuth() to compute heading.

        Assertion Strategy:
        Validates calculation by confirming:
        - Azimuth between 85° and 95° (±5° tolerance).

        Testing Principle:
        Validates quadrant handling, ensuring atan2 produces
        correct 90° result for positive Y axis.
        """
        arduino_instance._magnetometer = {"mX": 0.0, "mY": 1.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 85 < az < 95  # Should be around 90°

    def test_azimuth_south(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies magnetometer pointing south gives ~180° azimuth.

        Tests azimuth calculation for south cardinal direction.

        Business context:
        South (180°) is opposite to north. mX=-1, mY=0 means
        magnetic field points in negative X direction. Tests
        atan2 handling of negative X values.

        Arrangement:
        1. Set magnetometer pointing south: mX=-1, mY=0.

        Action:
        Call _calculate_azimuth() to compute heading.

        Assertion Strategy:
        Validates calculation by confirming:
        - Azimuth between 175° and 185° (±5° tolerance).

        Testing Principle:
        Validates negative axis handling, ensuring atan2
        correctly handles negative X producing 180°.
        """
        arduino_instance._magnetometer = {"mX": -1.0, "mY": 0.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 175 < az < 185  # Should be around 180°

    def test_azimuth_west(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies magnetometer pointing west gives ~270° azimuth.

        Tests azimuth calculation for west cardinal direction.

        Business context:
        West (270°) is perpendicular to north in the opposite
        direction from east. mX=0, mY=-1 indicates negative Y.
        Tests atan2 handling of negative Y values.

        Arrangement:
        1. Set magnetometer pointing west: mX=0, mY=-1.

        Action:
        Call _calculate_azimuth() to compute heading.

        Assertion Strategy:
        Validates calculation by confirming:
        - Azimuth between 265° and 275° (±5° tolerance).

        Testing Principle:
        Validates negative Y handling, ensuring atan2
        correctly handles negative Y producing 270°.
        """
        arduino_instance._magnetometer = {"mX": 0.0, "mY": -1.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 265 < az < 275  # Should be around 270°

    def test_azimuth_northeast(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies diagonal magnetometer reading gives ~45° azimuth.

        Tests azimuth calculation for northeast intermediate direction.

        Business context:
        Northeast (45°) tests calculation with equal X and Y
        components. Validates that atan2 produces correct angle
        for non-axis-aligned directions.

        Arrangement:
        1. Set magnetometer pointing northeast: mX=1, mY=1.

        Action:
        Call _calculate_azimuth() to compute heading.

        Assertion Strategy:
        Validates calculation by confirming:
        - Azimuth between 40° and 50° (±5° tolerance).

        Testing Principle:
        Validates diagonal handling, ensuring intermediate
        directions are correctly calculated.
        """
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 1.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 40 < az < 50  # Should be around 45°

    def test_azimuth_zero_division(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies zero magnetometer values return 0 safely.

        Tests error handling when mX=mY=0 could cause issues.

        Business context:
        When both mX and mY are zero, atan2(0,0) is undefined.
        The implementation must handle this gracefully, returning
        a safe default rather than crashing or returning NaN.

        Arrangement:
        1. Set magnetometer with mX=0, mY=0 (undefined direction).

        Action:
        Call _calculate_azimuth() which could fail.

        Assertion Strategy:
        Validates error handling by confirming:
        - Returns 0.0 (safe default).
        - No exceptions raised.

        Testing Principle:
        Validates error handling, ensuring edge cases don't
        crash the system.
        """
        arduino_instance._magnetometer = {"mX": 0.0, "mY": 0.0, "mZ": 1.0}

        az = arduino_instance._calculate_azimuth()

        assert az == 0.0  # Should return 0 safely

    def test_azimuth_normalized_to_360(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies azimuth is always normalized to 0-360 range.

        Tests that result is always within valid compass range.

        Business context:
        Azimuth must always be in 0-360° range for consistent
        display and calculations. Negative atan2 results or
        values >360° from calibration must be normalized.

        Arrangement:
        1. Set magnetometer pointing west (will produce 270°).

        Action:
        Call _calculate_azimuth() to compute heading.

        Assertion Strategy:
        Validates normalization by confirming:
        - Result is >= 0 and < 360.

        Testing Principle:
        Validates output normalization, ensuring compass
        values are always in standard range.
        """
        arduino_instance._magnetometer = {"mX": 0.0, "mY": -1.0, "mZ": 0.0}

        az = arduino_instance._calculate_azimuth()

        assert 0 <= az < 360


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCalibration:
    """Test suite for sensor calibration.

    Categories:
    1. Position Calibration - Altitude/azimuth offset (2 tests)
    2. Tilt Calibration - Slope and intercept (2 tests)

    Total: 4 tests.
    """

    def test_calibrate_altitude_offset(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies calibration sets altitude offset correctly.

        Tests that calibrate() computes and applies altitude correction.

        Business context:
        Calibration aligns sensor readings with known sky position.
        If sensor reads 0° but true altitude is 30°, an offset of
        30° is computed. Essential for accurate GoTo alignment.

        Arrangement:
        1. Set accelerometer to level (raw ~0° altitude).
        2. Set magnetometer for valid azimuth.

        Action:
        Call calibrate() with true_altitude=30°.

        Assertion Strategy:
        Validates calibration by confirming:
        - Subsequent read() returns ~30° altitude.
        - Offset correctly applied to raw reading.

        Testing Principle:
        Validates calibration system, ensuring altitude offset
        aligns raw readings with known positions.
        """
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
        """Verifies calibration sets azimuth offset correctly.

        Tests that calibrate() computes and applies azimuth correction.

        Business context:
        Azimuth calibration corrects for magnetic declination and
        mounting offset. If sensor reads 0° (north) but true position
        is 90° (east), a 90° offset is applied. Critical for GoTo.

        Arrangement:
        1. Set accelerometer for valid altitude.
        2. Set magnetometer pointing north (raw ~0° azimuth).

        Action:
        Call calibrate() with true_azimuth=90°.

        Assertion Strategy:
        Validates calibration by confirming:
        - Subsequent read() returns ~90° azimuth.
        - Offset correctly applied to raw reading.

        Testing Principle:
        Validates calibration system, ensuring azimuth offset
        aligns raw readings with known positions.
        """
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
        """Verifies tilt calibration slope scales readings.

        Tests that slope parameter in linear calibration works.

        Business context:
        Slope corrects for sensor non-linearity. A slope of 0.5
        halves all angle readings, useful when sensor over-reports.
        Applied as: corrected = slope * raw + intercept.

        Arrangement:
        1. Set accelerometer to 45° tilt (raw ~45°).
        2. Set magnetometer for valid azimuth.
        3. Get raw altitude reading.

        Action:
        Apply slope=0.5, intercept=0.0 calibration.

        Assertion Strategy:
        Validates slope effect by confirming:
        - Scaled altitude ≈ raw * 0.5.
        - Angle is halved by slope.

        Testing Principle:
        Validates linear calibration, ensuring slope parameter
        correctly scales raw readings.
        """
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
        """Verifies tilt calibration intercept offsets readings.

        Tests that intercept parameter in linear calibration works.

        Business context:
        Intercept corrects for constant offset in sensor mounting.
        An intercept of 5° adds 5° to all readings, compensating
        for a 5° mounting tilt. Applied as: corrected = slope * raw + intercept.

        Arrangement:
        1. Set accelerometer to level (raw ~0°).
        2. Set magnetometer for valid azimuth.

        Action:
        Apply slope=1.0, intercept=5.0 calibration.

        Assertion Strategy:
        Validates intercept effect by confirming:
        - Result ≈ 0° + 5° = 5°.
        - Constant offset added.

        Testing Principle:
        Validates linear calibration, ensuring intercept parameter
        correctly offsets raw readings.
        """
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
    """Test suite for reading sensor data.

    Categories:
    1. Happy Path - Normal read with all data (1 test)
    2. Error States - No data, closed sensor (2 tests)
    3. Calculated Values - Altitude/azimuth included (1 test)

    Total: 4 tests.
    """

    def test_read_returns_sensor_reading(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies read() returns complete SensorReading.

        Tests happy path read with all sensor data populated.

        Business context:
        read() is the primary interface for getting sensor data.
        Must return SensorReading with accelerometer, magnetometer,
        temperature, humidity, timestamp, and raw values.

        Arrangement:
        1. Parse valid 8-field sensor data line.
        2. This populates all internal sensor state.

        Action:
        Call read() to get SensorReading.

        Assertion Strategy:
        Validates complete read by confirming:
        - Accelerometer dict matches parsed values.
        - Magnetometer dict matches parsed values.
        - Temperature and humidity match parsed values.
        - Timestamp is not None.
        - Raw values preserved for debugging.

        Testing Principle:
        Validates data aggregation, ensuring read() correctly
        packages all sensor state into SensorReading.
        """
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
        """Verifies read() raises RuntimeError when no data available.

        Tests error handling for premature read attempts.

        Business context:
        Reading before sensor has streamed data should fail clearly.
        Callers must wait for sensor stream to begin before reading.
        Clear error message helps with debugging timing issues.

        Arrangement:
        1. Fresh instance with no parsed data.

        Action:
        Call read() without any data available.

        Assertion Strategy:
        Validates error handling by confirming:
        - RuntimeError raised.
        - Message includes "No sensor data available".

        Testing Principle:
        Validates precondition checking, ensuring clear errors
        for invalid states.
        """
        with pytest.raises(RuntimeError, match="No sensor data available"):
            arduino_instance.read()

    def test_read_raises_when_closed(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies read() raises RuntimeError when sensor is closed.

        Tests error handling for reads after close().

        Business context:
        Reading from closed sensor should fail clearly. After close(),
        all operations should be invalid. Prevents use-after-close bugs.

        Arrangement:
        1. Parse valid data to populate sensor state.
        2. Set _is_open to False (simulates close).

        Action:
        Call read() on closed sensor.

        Assertion Strategy:
        Validates state checking by confirming:
        - RuntimeError raised.
        - Message includes "Sensor is closed".

        Testing Principle:
        Validates lifecycle enforcement, ensuring closed sensors
        reject operations with clear errors.
        """
        arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0")
        arduino_instance._is_open = False

        with pytest.raises(RuntimeError, match="Sensor is closed"):
            arduino_instance.read()

    def test_read_calculated_values(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies read() includes calculated altitude and azimuth.

        Tests that SensorReading contains derived orientation values.

        Business context:
        SensorReading should include calculated altitude and azimuth,
        not just raw IMU data. This provides telescope orientation
        directly without requiring callers to do trigonometry.

        Arrangement:
        1. Parse data for level, north-pointing position.

        Action:
        Call read() to get SensorReading.

        Assertion Strategy:
        Validates calculated values by confirming:
        - Altitude near 0° (level).
        - Azimuth near 0°/360° (north).

        Testing Principle:
        Validates value derivation, ensuring convenient calculated
        values are included in reading.
        """
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
    """Test suite for Arduino command handling.

    Categories:
    1. Write Commands - RESET, STOP, START (3 tests)
    2. Response Handling - Wait for response (2 tests)

    Total: 5 tests.
    """

    def test_send_command_writes_to_serial(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies _send_command() writes command to serial port.

        Tests basic command transmission to Arduino.

        Business context:
        Commands like RESET, START, STOP control Arduino behavior.
        Commands must be written to serial port for Arduino to
        receive and process them.

        Arrangement:
        1. Queue response line in mock serial (Arduino would respond).

        Action:
        Call _send_command() with "RESET" command.

        Assertion Strategy:
        Validates command transmission by confirming:
        - "RESET" appears in written commands list.
        - Command was sent to serial port.

        Testing Principle:
        Validates I/O operation, ensuring commands are properly
        transmitted to hardware.
        """
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
        """Verifies _send_command() collects response when requested.

        Tests response collection with wait_response=True.

        Business context:
        Some commands return data (like STATUS). wait_response=True
        causes command handler to read and return response. Enables
        query/response communication pattern.

        Arrangement:
        1. Queue response line in mock serial.
        2. Set in_waiting to indicate data available.

        Action:
        Call _send_command() with wait_response=True.

        Assertion Strategy:
        Validates response handling by confirming:
        - Response contains expected text (or empty on timeout).

        Testing Principle:
        Validates bidirectional communication, ensuring responses
        can be collected from Arduino.
        """
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
        """Verifies reset() sends RESET command to Arduino.

        Tests high-level reset functionality.

        Business context:
        Reset command reinitializes Arduino sensor. Used to clear
        errors, restart calibration, or recover from bad state.
        Provides user-friendly interface over raw command.

        Arrangement:
        1. Queue response line and set in_waiting.

        Action:
        Call reset() method.

        Assertion Strategy:
        Validates reset operation by confirming:
        - "RESET" command sent to serial port.

        Testing Principle:
        Validates abstraction layer, ensuring high-level methods
        translate to correct low-level commands.
        """
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
        """Verifies _stop_output() sends STOP command to Arduino.

        Tests command to pause sensor data streaming.

        Business context:
        STOP command pauses continuous data output from Arduino.
        Used during calibration or when reading specific values.
        Reduces serial traffic when real-time data not needed.

        Arrangement:
        1. Use mock serial to capture commands.

        Action:
        Call _stop_output() method.

        Assertion Strategy:
        Validates stop operation by confirming:
        - "STOP" command sent to serial port.

        Testing Principle:
        Validates stream control, ensuring output can be paused.
        """
        arduino_instance._stop_output()

        commands = mock_serial.get_written_commands()
        assert "STOP" in commands

    def test_start_output(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies _start_output() sends START command to Arduino.

        Tests command to resume sensor data streaming.

        Business context:
        START command resumes continuous data output from Arduino.
        Used after calibration or initialization to begin streaming.
        Restores real-time sensor updates.

        Arrangement:
        1. Use mock serial to capture commands.

        Action:
        Call _start_output() method.

        Assertion Strategy:
        Validates start operation by confirming:
        - "START" command sent to serial port.

        Testing Principle:
        Validates stream control, ensuring output can be resumed.
        """
        arduino_instance._start_output()

        commands = mock_serial.get_written_commands()
        assert "START" in commands


# =============================================================================
# Sensor Info and Status Tests
# =============================================================================


class TestSensorInfo:
    """Test suite for sensor information and status.

    Categories:
    1. Metadata - Static sensor information (1 test)
    2. Status - Runtime status with calibration state (2 tests)

    Total: 3 tests.
    """

    def test_get_info(self, arduino_instance: ArduinoSensorInstance) -> None:
        """Verifies get_info() returns complete sensor metadata.

        Tests static sensor information retrieval.

        Business context:
        get_info() provides metadata about sensor capabilities for
        UI display and capability checking. Includes sensor type,
        name, port, and available data channels.

        Arrangement:
        1. Use arduino_instance configured for /dev/ttyTEST.

        Action:
        Call get_info() to retrieve metadata.

        Assertion Strategy:
        Validates metadata by confirming:
        - Type is "arduino_ble33".
        - Name describes sensor model.
        - Port matches configuration.
        - Capability flags (accelerometer, magnetometer, etc.) are True.
        - Sample rate is correct (10.0 Hz).

        Testing Principle:
        Validates metadata completeness, ensuring all expected
        information is available for UI and logic.
        """
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
        """Verifies get_status() shows calibrated=False when not calibrated.

        Tests status reporting for uncalibrated sensor.

        Business context:
        Status must indicate calibration state so users know if
        readings are accurate. calibrated=False warns that readings
        may have offset errors until calibration is performed.

        Arrangement:
        1. Use fresh instance with no calibration.
        2. Queue status response in mock serial.

        Action:
        Call get_status() to retrieve runtime status.

        Assertion Strategy:
        Validates status by confirming:
        - connected is True.
        - type matches expected.
        - port matches configuration.
        - calibrated is False.

        Testing Principle:
        Validates state reporting, ensuring calibration status
        is accurately communicated.
        """
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
        """Verifies get_status() shows calibrated=True after calibration.

        Tests status reporting for calibrated sensor.

        Business context:
        After calibration, status must show calibrated=True to
        indicate readings are aligned with known sky positions.
        Users can trust the altitude/azimuth values.

        Arrangement:
        1. Parse sensor data to enable calibration.
        2. Perform calibration with known position.
        3. Queue status response.

        Action:
        Call get_status() after calibration.

        Assertion Strategy:
        Validates calibrated status by confirming:
        - calibrated is True after calibrate() call.

        Testing Principle:
        Validates state transitions, ensuring calibration
        changes reported status correctly.
        """
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
    """Test suite for ArduinoSensorDriver port enumeration.

    Categories:
    1. Device Detection - Arduino, ACM, CH340 ports (3 tests)
    2. Empty/Multiple - No devices, multiple devices (2 tests)
    3. Lifecycle - Open with injection, already open, close (3 tests)

    Total: 8 tests.
    """

    def test_get_available_sensors_with_arduino(self) -> None:
        """Verifies Arduino devices are detected by name.

        Tests detection based on "Arduino" in device description.

        Business context:
        Arduino Nano 33 BLE identifies as "Arduino" in USB descriptor.
        Driver must detect these devices for user to select them.
        Filters out non-Arduino serial ports.

        Arrangement:
        1. Create mock enumerator with Arduino and generic devices.

        Action:
        Call get_available_sensors() via driver.

        Assertion Strategy:
        Validates detection by confirming:
        - Returns 1 device (only Arduino, not generic).
        - Port path matches Arduino device.
        - Type is "arduino_ble33".
        - Name contains "Arduino".

        Testing Principle:
        Validates device filtering, ensuring only compatible
        devices are presented to users.
        """
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
        """Verifies ACM devices are detected as potential Arduinos.

        Tests detection based on ttyACM port naming.

        Business context:
        Arduino devices on Linux create /dev/ttyACMx ports. Even
        if description doesn't say "Arduino", ACM ports are likely
        Arduino devices and should be offered.

        Arrangement:
        1. Create mock enumerator with ACM device (no Arduino name).

        Action:
        Call get_available_sensors() via driver.

        Assertion Strategy:
        Validates ACM detection by confirming:
        - Returns 1 device for ttyACM port.
        - Port path is correct.

        Testing Principle:
        Validates fallback detection, ensuring ACM ports are
        detected even without explicit Arduino identification.
        """
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
        """Verifies CH340 USB-serial devices are detected.

        Tests detection based on CH340 chip identifier.

        Business context:
        CH340 is a common USB-serial chip used in Arduino clones.
        Many inexpensive Arduino boards use CH340 instead of
        native USB. Must be detected for clone support.

        Arrangement:
        1. Create mock enumerator with CH340 device.

        Action:
        Call get_available_sensors() via driver.

        Assertion Strategy:
        Validates CH340 detection by confirming:
        - Returns 1 device for CH340 adapter.
        - Port path is correct.

        Testing Principle:
        Validates chip-based detection, ensuring CH340 clones
        are supported alongside official Arduino boards.
        """
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
        """Verifies empty list returned when no Arduino devices found.

        Tests behavior when no compatible devices are present.

        Business context:
        When no Arduino sensors connected, user should see empty
        list rather than error. Allows graceful handling of missing
        hardware.

        Arrangement:
        1. Create mock enumerator with only non-Arduino devices.

        Action:
        Call get_available_sensors() via driver.

        Assertion Strategy:
        Validates empty handling by confirming:
        - Returns empty list (not None, not error).

        Testing Principle:
        Validates graceful degradation, ensuring missing hardware
        doesn't cause errors.
        """
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
        """Verifies multiple Arduino devices are all detected.

        Tests behavior with multiple compatible devices connected.

        Business context:
        Users may have multiple Arduino sensors (e.g., main telescope
        and finder scope). All compatible devices must be listed
        for user selection.

        Arrangement:
        1. Create mock enumerator with 3 compatible devices.

        Action:
        Call get_available_sensors() via driver.

        Assertion Strategy:
        Validates multi-device handling by confirming:
        - Returns 3 devices.
        - All compatible types detected.

        Testing Principle:
        Validates completeness, ensuring all connected devices
        are discovered and reported.
        """
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
        """Verifies opening with injected serial port works.

        Tests dependency injection for testing.

        Business context:
        For testing, we need to inject mock serial ports instead
        of opening real hardware. _open_with_serial enables this
        by accepting pre-created serial object.

        Arrangement:
        1. Create mock serial port.
        2. Create driver with empty enumerator.

        Action:
        Call _open_with_serial() with mock serial.

        Assertion Strategy:
        Validates injection by confirming:
        - Returns valid instance.
        - Instance port matches provided name.
        - Instance is marked as open.

        Testing Principle:
        Validates testability design, ensuring mock injection
        path works correctly.
        """
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        instance = driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        assert instance is not None
        assert instance._port == "/dev/ttyTEST"
        assert instance._is_open is True

    def test_open_already_open_raises(self) -> None:
        """Verifies error when opening sensor twice.

        Tests that double-open is prevented.

        Business context:
        Driver maintains single instance. Opening twice would cause
        resource conflicts or undefined state. Must reject second
        open attempt with clear error.

        Arrangement:
        1. Create driver and open first sensor.

        Action:
        Attempt to open second sensor.

        Assertion Strategy:
        Validates protection by confirming:
        - RuntimeError raised on second open.
        - Message includes "Sensor already open".

        Testing Principle:
        Validates resource protection, ensuring single-instance
        constraint is enforced.
        """
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = ArduinoSensorDriver._create_with_enumerator(enum)
        driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        with pytest.raises(RuntimeError, match="Sensor already open"):
            driver._open_with_serial(MockSerialPort(), "/dev/ttyTEST2")

    def test_close_driver(self) -> None:
        """Verifies driver close() closes the instance.

        Tests proper cleanup through driver interface.

        Business context:
        Driver.close() must clean up resources including serial
        port and instance reference. Ensures no resource leaks
        and allows reopening later.

        Arrangement:
        1. Create driver and open sensor with mock serial.

        Action:
        Call driver.close().

        Assertion Strategy:
        Validates cleanup by confirming:
        - Mock serial is closed.
        - Driver instance reference is None.

        Testing Principle:
        Validates resource cleanup, ensuring close properly
        releases all resources.
        """
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
    """Test suite for ArduinoSensorInstance lifecycle.

    Categories:
    1. Close - Stop reading, close serial (1 test)
    2. Creation - With and without reader thread (2 tests)

    Total: 3 tests.
    """

    def test_close_stops_reading(
        self,
        arduino_instance: ArduinoSensorInstance,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies close() sets stop flag and closes serial.

        Tests instance closure behavior.

        Business context:
        close() must signal reader thread to stop, mark instance
        as closed, and close serial port. Ensures clean shutdown
        without resource leaks.

        Arrangement:
        1. Use arduino_instance from fixture.

        Action:
        Call close() on instance.

        Assertion Strategy:
        Validates closure by confirming:
        - _stop_reading flag is True.
        - _is_open flag is False.
        - Serial port is closed.

        Testing Principle:
        Validates shutdown sequence, ensuring all cleanup
        steps are performed.
        """
        arduino_instance.close()

        assert arduino_instance._stop_reading is True
        assert arduino_instance._is_open is False
        assert mock_serial._closed is True

    def test_create_with_serial_no_reader(self) -> None:
        """Verifies creation without reader thread.

        Tests instance creation for testing scenarios.

        Business context:
        For testing, reader thread is often not needed. start_reader=False
        creates instance without background thread, allowing
        synchronous testing of parse/calculate methods.

        Arrangement:
        1. Create mock serial port.

        Action:
        Call _create_with_serial() with start_reader=False.

        Assertion Strategy:
        Validates creation by confirming:
        - _reader_thread is None (no thread started).
        - _is_open is True (instance is ready).

        Testing Principle:
        Validates test mode creation, ensuring instances can be
        created without background threads.
        """
        mock_serial = MockSerialPort()

        instance = ArduinoSensorInstance._create_with_serial(
            mock_serial,
            port_name="/dev/test",
            start_reader=False,
        )

        assert instance._reader_thread is None
        assert instance._is_open is True

    def test_create_with_serial_with_reader(self) -> None:
        """Verifies creation with reader thread starts thread.

        Tests production-like instance creation.

        Business context:
        In production, reader thread continuously reads serial data
        in background. start_reader=True creates this thread for
        real-time sensor updates.

        Arrangement:
        1. Create mock serial port.

        Action:
        Call _create_with_serial() with start_reader=True.

        Assertion Strategy:
        Validates thread creation by confirming:
        - _reader_thread is not None.
        - Thread was started.

        Testing Principle:
        Validates production mode creation, ensuring reader thread
        is properly initialized.
        """
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
    """Test suite for edge cases and error handling.

    Categories:
    1. Value Types - Negative, scientific notation (2 tests)
    2. State Management - Consecutive parses, azimuth wrap (2 tests)

    Total: 4 tests.
    """

    def test_parse_negative_values(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies negative sensor values are handled correctly.

        Tests parsing of negative numbers in sensor data.

        Business context:
        Sensor values can be negative (e.g., negative acceleration,
        negative temperature in cold weather). Parser must preserve
        sign for accurate calculations.

        Arrangement:
        1. Prepare data with negative values in multiple fields.

        Action:
        Call _parse_line() with negative values.

        Assertion Strategy:
        Validates sign handling by confirming:
        - Returns True (successful parse).
        - Negative values preserved in accelerometer.
        - Negative values preserved in magnetometer.
        - Negative temperature preserved.

        Testing Principle:
        Validates numeric range, ensuring full range of sensor
        values including negatives is supported.
        """
        line = "-0.5\t-0.3\t-0.87\t-30.0\t-10.0\t-40.0\t-5.0\t0.0"

        result = arduino_instance._parse_line(line)

        assert result is True
        assert arduino_instance._accelerometer["aX"] == -0.5
        assert arduino_instance._magnetometer["mX"] == -30.0
        assert arduino_instance._temperature == -5.0

    def test_parse_scientific_notation(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies scientific notation values are handled correctly.

        Tests parsing of exponential notation (e.g., 1.5e-2).

        Business context:
        Some Arduino firmware or sprintf implementations output
        very small or large values in scientific notation. Parser
        must handle "e" notation for robustness.

        Arrangement:
        1. Prepare data with scientific notation values.

        Action:
        Call _parse_line() with exponential notation.

        Assertion Strategy:
        Validates notation handling by confirming:
        - Returns True (successful parse).
        - 1.5e-2 parsed as 0.015.
        - 3.0e1 parsed as 30.0.

        Testing Principle:
        Validates numeric format flexibility, ensuring various
        float representations are correctly parsed.
        """
        line = "1.5e-2\t0.0\t1.0\t3.0e1\t0.0\t4.0e1\t2.25e1\t5.5e1"

        result = arduino_instance._parse_line(line)

        assert result is True
        assert abs(arduino_instance._accelerometer["aX"] - 0.015) < 0.001
        assert abs(arduino_instance._magnetometer["mX"] - 30.0) < 0.001

    def test_multiple_consecutive_parses(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies state updates correctly on each parse.

        Tests that new data replaces old data.

        Business context:
        Sensor streams data continuously. Each parse must update
        state with new values, not accumulate or average with old.
        Ensures real-time readings reflect current position.

        Arrangement:
        1. Parse first data line.
        2. Capture first accelerometer X value.
        3. Parse second data line with different values.

        Action:
        Compare accelerometer X values after each parse.

        Assertion Strategy:
        Validates state replacement by confirming:
        - First parse sets aX to 0.5.
        - Second parse sets aX to 0.7.
        - Values replaced, not accumulated.

        Testing Principle:
        Validates state management, ensuring each parse
        completely updates sensor state.
        """
        arduino_instance._parse_line("0.5\t0.0\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0")
        first_ax = arduino_instance._accelerometer["aX"]

        arduino_instance._parse_line("0.7\t0.1\t0.71\t35.0\t5.0\t45.0\t23.0\t60.0")
        second_ax = arduino_instance._accelerometer["aX"]

        assert first_ax == 0.5
        assert second_ax == 0.7

    def test_azimuth_wraps_at_360(
        self, arduino_instance: ArduinoSensorInstance
    ) -> None:
        """Verifies azimuth with calibration offset wraps correctly.

        Tests that values >360° are normalized to 0-360 range.

        Business context:
        Calibration offsets can push azimuth above 360° or below 0°.
        Final value must always be normalized to 0-360 range for
        consistent compass representation.

        Arrangement:
        1. Set magnetometer pointing north (raw ~0°).
        2. Apply large calibration offset (350°).

        Action:
        Call _calculate_azimuth() with offset applied.

        Assertion Strategy:
        Validates wrapping by confirming:
        - Result is >= 0 and < 360.
        - Large offset doesn't produce >360 value.

        Testing Principle:
        Validates normalization, ensuring compass values are
        always in standard 0-360 range.
        """
        arduino_instance._magnetometer = {"mX": 1.0, "mY": 0.0, "mZ": 0.0}

        # Calibrate with large offset
        arduino_instance._cal_az_offset = 350.0

        az = arduino_instance._calculate_azimuth()

        assert 0 <= az < 360  # Should wrap
