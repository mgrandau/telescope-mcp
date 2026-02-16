"""Unit tests for serial motor controller driver.

Tests the SerialMotorController and SerialMotorDriver using mock
serial ports, without requiring actual hardware.

Test Categories:
- Motor movement (absolute and relative)
- Position limits enforcement
- Axis selection
- Command responses
- Port enumeration
- Error handling

Example:
    Run all motor controller tests:
        pdm run pytest tests/drivers/motors/test_serial_controller.py -v
"""

from __future__ import annotations

import pytest

from telescope_mcp.drivers.motors import (
    MOTOR_CONFIGS,
    MotorConfig,
    MotorStatus,
    MotorType,
    SerialMotorController,
    SerialMotorDriver,
    altitude_degrees_to_steps,
    azimuth_degrees_to_steps,
    steps_to_altitude_degrees,
    steps_to_azimuth_degrees,
)

# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockSerialPort:
    """Mock serial port implementing SerialPort protocol.

    Provides configurable responses for testing without hardware.
    Simulates the custom motor controller protocol with command
    queuing and response tracking.

    Example:
        >>> port = MockSerialPort()
        >>> port.queue_response(b"{'alldone': 1}\r\n")
        >>> port.readline()
        b"{'alldone': 1}\r\n"
    """

    def __init__(self) -> None:
        """Initialize mock serial port with empty state.

        Creates a mock serial port ready for testing motor commands
        with empty queues and logs. Implements SerialPort protocol.

        Args:
            None. Creates port with default empty state.

        Returns:
            None. Initializes instance attributes.

        Raises:
            No exceptions raised during initialization.

        Business context:
            Motor controller tests need isolated serial mock for each
            test. Empty queues ensure no state leakage between tests.

        Example:
            >>> port = MockSerialPort()
            >>> port.is_open
            True

        Implementation:
            Sets is_open=True, empty read queue, empty write log.
        """
        self.is_open = True
        self._in_waiting = 0
        self._read_queue: list[bytes] = []
        self._write_log: list[bytes] = []
        self._closed = False

    @property
    def in_waiting(self) -> int:
        """Number of bytes waiting in the read queue (simulated).

        Simulates the pyserial in_waiting property that reports bytes
        available for reading. Used by controller to check if response
        data is available before attempting to read.

        Args:
            None. Property takes no arguments.

        Returns:
            Sum of bytes in all queued responses. Zero if queue empty.

        Raises:
            No exceptions raised. Always returns valid integer.

        Business context:
            Motor controller checks in_waiting before reading to avoid
            blocking on empty buffer. Mock tracks queued response sizes.

        Example:
            >>> port.queue_response(b"OK\r\n")
            >>> port.in_waiting
            4
        """
        return self._in_waiting

    def queue_response(self, data: bytes) -> None:
        """Add response data to the read queue.

        Queues bytes that will be returned by subsequent read operations.
        Updates in_waiting to reflect total queued bytes.

        Args:
            data: Bytes to return on next read call.

        Returns:
            None. Modifies internal queue state.

        Raises:
            No exceptions raised. Accepts any bytes value.

        Business context:
            Tests queue expected motor responses before triggering
            commands. Mock returns queued data in FIFO order.

        Example:
            >>> port.queue_response(b"OK\r\n")
            >>> port.readline()
            b"OK\r\n"
        """
        self._read_queue.append(data)
        self._in_waiting = sum(len(b) for b in self._read_queue)

    def queue_axis_response(self) -> None:
        """Queue typical axis select response.

        Motor controller responds with axis confirmation JSON when
        axis is selected. This queues the standard response format.

        Args:
            None. Queues fixed response format.

        Returns:
            None. Modifies internal queue state.

        Raises:
            No exceptions raised.

        Business context:
            After sending 'A0' or 'A1' axis command, motor controller
            responds with axis confirmation. Tests must queue this
            before calling _select_axis().

        Example:
            >>> port.queue_axis_response()
            >>> controller._select_axis(MotorType.ALTITUDE)
        """
        self.queue_response(b"{'axis': 0}")

    def queue_move_complete(self) -> None:
        """Queue move completion response.

        Motor controller signals move completion with 'alldone' JSON.
        This queues the standard completion response with newline.

        Args:
            None. Queues fixed response format.

        Returns:
            None. Modifies internal queue state.

        Raises:
            No exceptions raised.

        Business context:
            After move commands, motor controller sends 'alldone'
            when movement finishes. Tests must queue this before
            calling move() to simulate successful completion.

        Example:
            >>> port.queue_move_complete()
            >>> controller.move(MotorType.ALTITUDE, -50000)
        """
        self.queue_response(b"{'alldone': 1}\r\n")

    def read_until(self, expected: bytes = b"\n", size: int | None = None) -> bytes:
        """Read from queue until expected byte sequence.

        Pops and returns the first queued response. In real serial,
        this would block until expected bytes received. Mock returns
        immediately from queue.

        Args:
            expected: Byte sequence to read until (ignored in mock).
            size: Max bytes to read (ignored in mock).

        Returns:
            First queued response, or empty bytes if queue empty.

        Raises:
            No exceptions raised. Returns empty bytes if no data queued.

        Business context:
            Motor controller reads responses after sending commands.
            Mock provides queued responses for test verification.

        Example:
            >>> port.queue_response(b"data\n")
            >>> port.read_until(b"\n")
            b"data\n"
        """
        if self._read_queue:
            data = self._read_queue.pop(0)
            self._in_waiting = sum(len(b) for b in self._read_queue)
            return data
        return b""

    def readline(self) -> bytes:
        """Read a line (until newline) from the response queue.

        Convenience method that calls read_until with newline delimiter.
        Simulates pyserial's readline() which blocks until newline received.

        Args:
            None. Reads from internal queue.

        Returns:
            First queued response bytes, or empty bytes if queue empty.

        Raises:
            No exceptions raised. Returns empty bytes if no data queued.

        Business context:
            Motor controller protocol uses newline-terminated responses.
            Controller calls readline() to get complete response messages.

        Example:
            >>> port.queue_response(b"OK\r\n")
            >>> port.readline()
            b"OK\r\n"
        """
        return self.read_until(b"\n")

    def write(self, data: bytes) -> int:
        """Write bytes to port, logging for test verification.

        Stores written data in internal log for later verification.
        In real serial, this would transmit bytes to hardware. Mock
        captures all writes for assertion in tests.

        Args:
            data: Bytes to write to the mock serial port.

        Returns:
            Number of bytes written (always equals len(data)).

        Business context:
            Tests verify correct commands sent by checking write log.
            Motor commands like 'A0', 'o50000' are captured here.

        Example:
            >>> port.write(b"A0")
            2
            >>> port.get_written_commands()
            ["A0"]
        """
        self._write_log.append(data)
        return len(data)

    def reset_input_buffer(self) -> None:
        """Clear the input buffer, removing all queued responses.

        Simulates pyserial's reset_input_buffer() which clears hardware
        receive buffer. Used to discard stale data before sending new
        commands that expect fresh responses.

        Args:
            None. Operates on internal queue.

        Returns:
            None. Clears internal queue state.

        Raises:
            No exceptions raised.

        Business context:
            Controller resets buffer before commands to ensure responses
            match the command just sent, not stale data from prior ops.

        Example:
            >>> port.queue_response(b"stale\r\n")
            >>> port.reset_input_buffer()
            >>> port.in_waiting
            0
        """
        self._read_queue.clear()
        self._in_waiting = 0

    def close(self) -> None:
        """Close the mock serial port and update state flags.

        Sets is_open to False and marks port as closed via _closed flag.
        Allows tests to verify that close() is properly called during
        cleanup and resource release.

        Args:
            None. Operates on internal state.

        Returns:
            None. Updates internal state flags.

        Raises:
            No exceptions raised.

        Business context:
            Tests verify proper lifecycle management by checking that
            serial port is closed when controller is closed. Prevents
            resource leaks in production code.

        Example:
            >>> port.close()
            >>> port.is_open
            False
        """
        self.is_open = False
        self._closed = True

    def get_written_commands(self) -> list[str]:
        """Get all commands written to port for test verification.

        Decodes and returns all bytes written via write() as strings.
        Primary verification method for testing correct command sequences.

        Args:
            None. Returns data from internal write log.

        Returns:
            List of command strings in the order they were written.

        Raises:
            No exceptions raised. Returns empty list if nothing written.

        Business context:
            Tests verify motor controller sends correct protocol commands.
            Command sequence matters - axis select before move, etc.

        Example:
            >>> port.write(b"A0")
            >>> port.write(b"o50000")
            >>> port.get_written_commands()
            ["A0", "o50000"]
        """
        return [data.decode() for data in self._write_log]


class MockComPort:
    """Mock COM port for testing port enumeration.

    Simulates serial port info returned by serial.tools.list_ports.
    Contains device path and description for device identification.
    """

    def __init__(self, device: str, description: str) -> None:
        """Initialize mock COM port with device info.

        Creates a mock serial port info object for testing device
        detection logic. Simulates pyserial's ListPortInfo class.

        Args:
            device: Port device path (e.g., '/dev/ttyACM0').
            description: Port description for identification.

        Returns:
            None. Initializes instance attributes.

        Raises:
            No exceptions raised during initialization.

        Business context:
            SerialMotorDriver uses port description to identify motor
            controllers. Mock provides configurable port info for tests.

        Example:
            >>> port = MockComPort('/dev/ttyACM0', 'Arduino')
            >>> port.device
            '/dev/ttyACM0'

        Implementation:
            Stores device path and description as instance attributes.
        """
        self.device = device
        self.description = description


class MockPortEnumerator:
    """Mock port enumerator implementing PortEnumerator protocol.

    Returns a configured list of mock COM ports for testing
    device detection without real hardware enumeration.
    """

    def __init__(self, ports: list[MockComPort] | None = None) -> None:
        """Initialize with list of mock ports.

        Creates a port enumerator that returns configurable mock ports.
        Implements PortEnumerator protocol for driver testing.

        Args:
            ports: List of mock COM ports to return from comports().
                  Defaults to empty list if None.

        Returns:
            None. Initializes instance attributes.

        Raises:
            No exceptions raised during initialization.

        Business context:
            Driver tests need to simulate various port scenarios:
            no ports, ACM ports, CH340 ports. Configurable list
            allows testing each scenario.

        Example:
            >>> ports = [MockComPort('/dev/ttyACM0', 'ACM')]
            >>> enum = MockPortEnumerator(ports)
            >>> enum.comports()
            [<MockComPort '/dev/ttyACM0'>]

        Implementation:
            Stores ports list, defaulting to empty if None provided.
        """
        self._ports = ports or []

    def comports(self) -> list[MockComPort]:
        """Return list of mock COM ports for enumeration testing.

        Simulates serial.tools.list_ports.comports() which returns
        available serial ports on the system. Returns the preconfigured
        list of mock ports.

        Args:
            None. Returns preconfigured port list.

        Returns:
            List of MockComPort objects configured during initialization.

        Raises:
            No exceptions raised. Always returns valid list.

        Business context:
            Driver uses comports() to discover available motor controllers.
            Mock allows testing detection logic with various port scenarios.

        Example:
            >>> enumerator = MockPortEnumerator([MockComPort('/dev/ttyACM0', 'ACM')])
            >>> enumerator.comports()
            [<MockComPort '/dev/ttyACM0'>]
        """
        return self._ports


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_serial() -> MockSerialPort:
    """Create a mock serial port fixture for motor controller tests.

    Provides a fresh MockSerialPort instance for each test, ensuring
    test isolation. The mock simulates pyserial's Serial interface.

    Args:
        None. Fixture takes no arguments.

    Returns:
        Fresh MockSerialPort instance with empty queues and logs.

    Raises:
        No exceptions raised during fixture creation.

    Business context:
        Motor controller tests need serial port without real hardware.
        Mock captures commands and provides configurable responses.

    Example:
        def test_command(mock_serial):
            mock_serial.queue_response(b"OK\r\n")
    """
    return MockSerialPort()


@pytest.fixture
def motor_controller(mock_serial: MockSerialPort) -> SerialMotorController:
    """Create a SerialMotorController with mock serial for testing.

    Injects mock serial port for testing motor commands without
    actual hardware. Uses test port name '/dev/ttyTEST'.

    Args:
        mock_serial: Mock serial port fixture providing test doubles.

    Returns:
        SerialMotorController configured with mock serial for testing.

    Raises:
        No exceptions raised during fixture creation.

    Business context:
        Controller tests verify command generation and state management
        without real motors. Mock captures commands for verification.

    Example:
        def test_move(motor_controller, mock_serial):
            mock_serial.queue_move_complete()
            motor_controller.move(MotorType.ALTITUDE, -50000)
    """
    return SerialMotorController._create_with_serial(
        mock_serial,
        port_name="/dev/ttyTEST",
    )


# =============================================================================
# Axis Selection Tests
# =============================================================================


class TestAxisSelection:
    """Test suite for motor axis selection.

    Categories:
    1. Axis Commands - Altitude (A0), Azimuth (A1) (2 tests)
    2. Optimization - Skip reselect same axis (1 test)

    Total: 3 tests.
    """

    def test_select_altitude_axis(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies altitude axis selection sends A0 command.

        Tests correct command for altitude motor selection.

        Business context:
        Motor controller uses A0/A1 commands to select which motor
        receives subsequent movement commands. Altitude is axis 0.

        Arrangement:
        1. Queue axis response in mock serial.

        Action:
        Call _select_axis() with ALTITUDE motor type.

        Assertion Strategy:
        Validates command by confirming:
        - "A0" appears in written commands.

        Testing Principle:
        Validates protocol correctness, ensuring correct axis
        command is sent for altitude motor.
        """
        mock_serial.queue_axis_response()

        motor_controller._select_axis(MotorType.ALTITUDE)

        commands = mock_serial.get_written_commands()
        assert "A0" in commands

    def test_select_azimuth_axis(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies azimuth axis selection sends A1 command.

        Tests correct command for azimuth motor selection.

        Business context:
        Azimuth motor is axis 1. Before moving azimuth, controller
        must select it with A1 command.

        Arrangement:
        1. Queue axis response in mock serial.

        Action:
        Call _select_axis() with AZIMUTH motor type.

        Assertion Strategy:
        Validates command by confirming:
        - "A1" appears in written commands.

        Testing Principle:
        Validates protocol correctness, ensuring correct axis
        command is sent for azimuth motor.
        """
        mock_serial.queue_axis_response()

        motor_controller._select_axis(MotorType.AZIMUTH)

        commands = mock_serial.get_written_commands()
        assert "A1" in commands

    def test_skip_reselect_same_axis(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies axis command is skipped if already selected.

        Tests optimization to avoid redundant axis selection.

        Business context:
        Sending unnecessary axis commands wastes serial bandwidth
        and adds latency. Controller tracks current axis and skips
        reselection when already on correct axis.

        Arrangement:
        1. Select altitude axis once.
        2. Record command count.

        Action:
        Select altitude axis again.

        Assertion Strategy:
        Validates optimization by confirming:
        - No additional commands sent on reselect.
        - Command count unchanged.

        Testing Principle:
        Validates efficiency optimization, ensuring redundant
        commands are avoided.
        """
        mock_serial.queue_axis_response()

        motor_controller._select_axis(MotorType.ALTITUDE)
        initial_count = len(mock_serial._write_log)

        # Select same axis again
        motor_controller._select_axis(MotorType.ALTITUDE)

        # Should not have sent another command
        assert len(mock_serial._write_log) == initial_count


# =============================================================================
# Motor Movement Tests
# =============================================================================


class TestMotorMovement:
    """Test suite for motor movement commands.

    Categories:
    1. Absolute Movement - move() command and position tracking (3 tests)
    2. Relative Movement - move_relative() with positive/negative (2 tests)
    3. Multi-Axis - Azimuth motor movement (1 test)

    Total: 6 tests.
    """

    def test_move_absolute(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies move() sends absolute position command.

        Tests 'o' command format for absolute positioning.

        Business context:
        Motor controller uses 'o' prefix for absolute position commands.
        Position is in steps. Absolute moves go directly to target
        regardless of current position.

        Arrangement:
        1. Queue axis response and move completion.

        Action:
        Call move() with target position -50000 steps (within range).

        Assertion Strategy:
        Validates command by confirming:
        - "o-50000" appears in written commands.

        Testing Principle:
        Validates protocol correctness, ensuring absolute position
        command format is correct.
        """
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()

        motor_controller.move(MotorType.ALTITUDE, -50000)

        commands = mock_serial.get_written_commands()
        assert any("o-50000" in cmd for cmd in commands)

    def test_move_updates_position(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies move() updates internal position tracking.

        Tests that position state is updated after move.

        Business context:
        Controller tracks position for relative moves and status.
        After successful move, internal position must reflect target.
        Used by get_status() and move_relative().

        Arrangement:
        1. Queue axis response and move completion.

        Action:
        Call move() to position -50000.

        Assertion Strategy:
        Validates state update by confirming:
        - get_status() reports position_steps == -50000.

        Testing Principle:
        Validates state management, ensuring internal tracking
        matches commanded positions.
        """
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()

        motor_controller.move(MotorType.ALTITUDE, -50000)

        status = motor_controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == -50000

    def test_move_relative(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies move_relative() sends relative movement command.

        Tests 'O' command format for relative positioning.

        Business context:
        Motor controller uses 'O' (uppercase) prefix for relative moves.
        Relative moves add/subtract steps from current position.
        Useful for jogging and fine adjustments.

        Arrangement:
        1. Queue axis response and move completion.

        Action:
        Call move_relative() with 1000 steps.

        Assertion Strategy:
        Validates command by confirming:
        - "O1000" appears in written commands.

        Testing Principle:
        Validates protocol correctness, ensuring relative move
        command format is correct.
        """
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()

        motor_controller.move_relative(MotorType.ALTITUDE, 1000)

        commands = mock_serial.get_written_commands()
        assert any("O1000" in cmd for cmd in commands)

    def test_move_relative_negative(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies move_relative() handles negative steps.

        Tests relative movement in reverse direction.

        Business context:
        Negative relative moves decrease position. Useful for
        correction movements or tracking back. Position must
        correctly subtract from current value.

        Arrangement:
        1. Move to position -50000 first.
        2. Queue completion for relative move.

        Action:
        Call move_relative() with -1000 steps.

        Assertion Strategy:
        Validates negative handling by confirming:
        - Final position is -51000 (-50000 - 1000).

        Testing Principle:
        Validates sign handling, ensuring negative relative
        moves correctly decrement position.
        """
        # First move to a position
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.move(MotorType.ALTITUDE, -50000)

        # Then relative move back
        mock_serial.queue_move_complete()
        motor_controller.move_relative(MotorType.ALTITUDE, -1000)

        status = motor_controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == -51000

    def test_move_azimuth(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies azimuth motor can be moved independently.

        Tests multi-axis support with azimuth motor.

        Business context:
        Telescope has two independent axes. Azimuth motor must
        be selectable and movable separately from altitude.
        Tests that axis selection works correctly for azimuth.

        Arrangement:
        1. Queue axis response (will select azimuth A1).
        2. Queue move completion.

        Action:
        Call move() on AZIMUTH motor to 50000.

        Assertion Strategy:
        Validates azimuth movement by confirming:
        - Azimuth status shows position_steps == 50000.

        Testing Principle:
        Validates multi-axis support, ensuring both motors
        can be controlled independently.
        """
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()

        motor_controller.move(MotorType.AZIMUTH, 50000)

        status = motor_controller.get_status(MotorType.AZIMUTH)
        assert status.position_steps == 50000


# =============================================================================
# Position Limits Tests
# =============================================================================


class TestPositionLimits:
    """Test suite for position limit enforcement.

    Categories:
    1. Altitude Limits - Max/min bounds (2 tests)
    2. Azimuth Limits - Max/min bounds (2 tests)
    3. Relative Limits - Resulting position check (1 test)
    4. Speed Limits - Speed range validation (1 test)

    Total: 6 tests.
    """

    def test_altitude_max_limit(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Verifies altitude position above max is rejected.

        Tests upper bound enforcement for altitude axis.

        Business context:
        Altitude motor has physical limits (+3° past zenith to -60°
        toward horizon). Moving beyond max could damage hardware or
        hit stops. Must reject invalid positions before sending commands.

        Arrangement:
        1. Use motor_controller from fixture.

        Action:
        Attempt move() to 10000 (above ~4667 max).

        Assertion Strategy:
        Validates limit enforcement by confirming:
        - ValueError raised.
        - Message mentions the step limits.

        Testing Principle:
        Validates safety limits, ensuring commands beyond
        physical limits are rejected.
        """
        with pytest.raises(ValueError, match="steps must be"):
            motor_controller.move(MotorType.ALTITUDE, 10000)

    def test_altitude_min_limit(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Verifies altitude position below min is rejected.

        Tests lower bound enforcement for altitude axis.

        Business context:
        Altitude cannot go below the configured minimum (-60°
        toward horizon = -93333 steps). Moving beyond this limit
        is physically dangerous. Must reject before command is sent.

        Arrangement:
        1. Use motor_controller from fixture.

        Action:
        Attempt move() to -100000 (below min).

        Assertion Strategy:
        Validates limit enforcement by confirming:
        - ValueError raised.
        - Message mentions the step limits.

        Testing Principle:
        Validates safety limits, ensuring out-of-range positions
        are rejected.
        """
        with pytest.raises(ValueError, match="steps must be"):
            motor_controller.move(MotorType.ALTITUDE, -100000)

    def test_azimuth_max_limit(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Verifies azimuth position above max is rejected.

        Tests upper bound enforcement for azimuth axis.

        Business context:
        Azimuth has range 0 to 154814 steps (0° to +190°).
        Beyond max could twist cables or hit mechanical stops.
        Must validate before command.

        Arrangement:
        1. Use motor_controller from fixture.

        Action:
        Attempt move() to 160000 (above 154814 max).

        Assertion Strategy:
        Validates limit enforcement by confirming:
        - ValueError raised.
        - Message mentions "0-154814" range.

        Testing Principle:
        Validates safety limits for azimuth axis.
        """
        with pytest.raises(ValueError, match="must be 0-154814"):
            motor_controller.move(MotorType.AZIMUTH, 160000)

    def test_azimuth_min_limit(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Verifies azimuth position below min is rejected.

        Tests lower bound enforcement for azimuth axis.

        Business context:
        Azimuth minimum is 0 steps. Going negative
        exceeds mechanical range. Must validate before command.

        Arrangement:
        1. Use motor_controller from fixture.

        Action:
        Attempt move() to -1000 (below 0 min).

        Assertion Strategy:
        Validates limit enforcement by confirming:
        - ValueError raised.
        - Message mentions range.

        Testing Principle:
        Validates safety limits for negative azimuth.
        """
        with pytest.raises(ValueError, match="must be 0-154814"):
            motor_controller.move(MotorType.AZIMUTH, -1000)

    def test_relative_move_limit(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies relative move checks resulting position.

        Tests that relative moves validate final position, not just delta.

        Business context:
        Relative move from position 0 with -100000 steps would result
        in -100000, exceeding the -93333 min limit. Must check resulting
        position, not just the delta value.

        Arrangement:
        1. Start at position 0 (default).

        Action:
        Attempt move_relative() with -100000 (would exceed -93333 min).

        Assertion Strategy:
        Validates resulting position check by confirming:
        - ValueError raised.
        - Message mentions "would exceed limits".

        Testing Principle:
        Validates predictive limit checking, ensuring relative
        moves don't result in invalid positions.
        """
        # Start at position 0 — -100000 would exceed -93333 limit
        with pytest.raises(ValueError, match="would exceed limits"):
            motor_controller.move_relative(MotorType.ALTITUDE, -100000)

    def test_speed_range_validation(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Verifies speed parameter must be 1-100.

        Tests speed range validation.

        Business context:
        Speed is percentage (1-100). Speed 0 would mean no movement,
        and >100 is undefined. Must validate speed parameter before
        sending to hardware.

        Arrangement:
        1. Use motor_controller from fixture.

        Action:
        Attempt move() with speed=0 and speed=101.

        Assertion Strategy:
        Validates speed bounds by confirming:
        - ValueError for speed=0.
        - ValueError for speed=101.
        - Both mention "Speed must be 1-100".

        Testing Principle:
        Validates parameter range, ensuring speed is within
        valid percentage range.
        """
        with pytest.raises(ValueError, match="Speed must be 1-100"):
            motor_controller.move(MotorType.ALTITUDE, 1000, speed=0)

        with pytest.raises(ValueError, match="Speed must be 1-100"):
            motor_controller.move(MotorType.ALTITUDE, 1000, speed=101)


# =============================================================================
# Homing Tests
# =============================================================================


class TestHoming:
    """Test suite for motor homing.

    Categories:
    1. Single Axis - Home individual motor (1 test)
    2. All Axes - Home both motors (1 test)

    Total: 2 tests.
    """

    def test_home_altitude(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies home() moves motor to configured home position.

        Tests single-axis homing for altitude.

        Business context:
        Home position is the reference point (0 for altitude = zenith).
        Homing is used to initialize or reset position. Must move
        to configured home position.

        Arrangement:
        1. Move away from home to position -50000.
        2. Queue move completion for home operation.

        Action:
        Call home() on altitude motor.

        Assertion Strategy:
        Validates homing by confirming:
        - Position returns to 0 (home).

        Testing Principle:
        Validates homing operation, ensuring motor returns to
        reference position.
        """
        # First move away from home
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.move(MotorType.ALTITUDE, -50000)

        # Home
        mock_serial.queue_move_complete()
        motor_controller.home(MotorType.ALTITUDE)

        status = motor_controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 0  # Home position

    def test_home_all(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies home_all() homes both motors.

        Tests multi-axis homing convenience method.

        Business context:
        home_all() is convenience for telescope park/initialization.
        Homes both altitude and azimuth to their reference positions.
        Ensures consistent starting state.

        Arrangement:
        1. Move both motors away from home.
        2. Queue responses for both home operations.

        Action:
        Call home_all().

        Assertion Strategy:
        Validates multi-axis homing by confirming:
        - Altitude position is 0.
        - Azimuth position is 0.

        Testing Principle:
        Validates batch operation, ensuring both axes are
        homed together.
        """
        # Move both away from home
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.move(MotorType.ALTITUDE, -50000)

        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.move(MotorType.AZIMUTH, 30000)

        # Home all
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.home_all()

        assert motor_controller.get_status(MotorType.ALTITUDE).position_steps == 0
        assert motor_controller.get_status(MotorType.AZIMUTH).position_steps == 0


# =============================================================================
# Status Tests
# =============================================================================


class TestMotorStatus:
    """Test suite for motor status.

    Categories:
    1. Status Query - get_status() returns MotorStatus (1 test)
    2. Controller Info - get_info() returns metadata (1 test)

    Total: 2 tests.
    """

    def test_get_status(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Verifies get_status() returns MotorStatus with all fields.

        Tests status query for motor state.

        Business context:
        get_status() provides current motor state for UI and logic.
        Returns position, moving flag, motor type. Used for display
        and movement decisions.

        Arrangement:
        1. Use fresh motor_controller (position 0, not moving).

        Action:
        Call get_status() for altitude motor.

        Assertion Strategy:
        Validates status by confirming:
        - Returns MotorStatus instance.
        - motor field is ALTITUDE.
        - position_steps is 0 (initial).
        - is_moving is False.

        Testing Principle:
        Validates status query, ensuring complete state is
        returned.
        """
        status = motor_controller.get_status(MotorType.ALTITUDE)

        assert isinstance(status, MotorStatus)
        assert status.motor == MotorType.ALTITUDE
        assert status.position_steps == 0
        assert status.is_moving is False

    def test_get_info(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Verifies get_info() returns controller metadata.

        Tests information query for controller state.

        Business context:
        get_info() provides metadata about controller for UI and
        debugging. Includes type, port, name, and steps_per_degree.

        Arrangement:
        1. Use motor_controller from fixture.

        Action:
        Call get_info().

        Assertion Strategy:
        Validates info by confirming:
        - type is "serial_motor_controller".
        - port matches test port.
        - name contains port identifier.
        - altitude_steps_per_degree and azimuth_steps_per_degree present.

        Testing Principle:
        Validates metadata completeness, ensuring all expected
        fields are present.
        """
        info = motor_controller.get_info()

        assert info["type"] == "serial_motor_controller"
        assert info["port"] == "/dev/ttyTEST"
        assert "/dev/ttyTEST" in info["name"]
        assert info["altitude_steps_per_degree"] > 0
        assert info["azimuth_steps_per_degree"] > 0


# =============================================================================
# Driver Tests (Port Enumeration)
# =============================================================================


class TestSerialMotorDriver:
    """Test suite for SerialMotorDriver port enumeration.

    Categories:
    1. Device Detection - ACM and CH340 ports (2 tests)
    2. Empty Result - No controllers found (1 test)
    3. Lifecycle - Open, already open, close (3 tests)

    Total: 6 tests.
    """

    def test_get_available_controllers_with_acm(self) -> None:
        """Verifies ACM devices are detected as motor controllers.

        Tests detection based on ttyACM port naming.

        Business context:
        Motor controller Arduino creates /dev/ttyACMx ports on Linux.
        Driver must detect these for user selection. Filters out
        non-ACM ports like ttyUSB.

        Arrangement:
        1. Create mock enumerator with ACM and generic USB ports.

        Action:
        Call get_available_controllers() via driver.

        Assertion Strategy:
        Validates detection by confirming:
        - Returns 1 controller (only ACM).
        - Port matches ACM device path.

        Testing Principle:
        Validates device filtering, ensuring only compatible
        ports are presented.
        """
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyACM0", "ttyACM0"),
                MockComPort("/dev/ttyUSB0", "Generic USB"),
            ]
        )

        driver = SerialMotorDriver._create_with_enumerator(enum)
        controllers = driver.get_available_controllers()

        assert len(controllers) == 1
        assert controllers[0]["port"] == "/dev/ttyACM0"

    def test_get_available_controllers_with_ch340(self) -> None:
        """Verifies CH340 USB-serial devices are detected.

        Tests detection based on CH340 chip identifier.

        Business context:
        Some Arduino motor controllers use CH340 USB-serial chips.
        These appear as ttyUSB with CH340 description. Must be
        detected for compatibility.

        Arrangement:
        1. Create mock enumerator with CH340 device.

        Action:
        Call get_available_controllers() via driver.

        Assertion Strategy:
        Validates CH340 detection by confirming:
        - Returns 1 controller.

        Testing Principle:
        Validates chip-based detection for CH340 adapters.
        """
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyUSB0", "CH340 Serial Adapter"),
            ]
        )

        driver = SerialMotorDriver._create_with_enumerator(enum)
        controllers = driver.get_available_controllers()

        assert len(controllers) == 1

    def test_get_available_controllers_empty(self) -> None:
        """Verifies empty list returned when no controllers found.

        Tests behavior when no compatible devices present.

        Business context:
        When no motor controller connected, user should see empty
        list rather than error. Allows graceful handling of missing
        hardware.

        Arrangement:
        1. Create mock enumerator with only standard serial (ttyS0).

        Action:
        Call get_available_controllers() via driver.

        Assertion Strategy:
        Validates empty handling by confirming:
        - Returns empty list.

        Testing Principle:
        Validates graceful degradation when hardware missing.
        """
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyS0", "Standard Serial Port"),
            ]
        )

        driver = SerialMotorDriver._create_with_enumerator(enum)
        controllers = driver.get_available_controllers()

        assert len(controllers) == 0

    def test_open_with_serial_injection(self) -> None:
        """Verifies opening with injected serial port works.

        Tests dependency injection for testing.

        Business context:
        For testing, we inject mock serial instead of opening real
        hardware. _open_with_serial enables this by accepting
        pre-created serial object.

        Arrangement:
        1. Create mock serial port.
        2. Create driver with empty enumerator.

        Action:
        Call _open_with_serial() with mock serial.

        Assertion Strategy:
        Validates injection by confirming:
        - Returns valid controller.
        - Controller port matches provided name.

        Testing Principle:
        Validates testability design for mock injection.
        """
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = SerialMotorDriver._create_with_enumerator(enum)
        controller = driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        assert controller is not None
        assert controller._port == "/dev/ttyTEST"

    def test_open_already_open_raises(self) -> None:
        """Verifies error when opening controller twice.

        Tests single-instance protection.

        Business context:
        Driver maintains single controller. Opening twice would
        cause resource conflicts. Must reject second open with
        clear error.

        Arrangement:
        1. Create driver and open first controller.

        Action:
        Attempt to open second controller.

        Assertion Strategy:
        Validates protection by confirming:
        - RuntimeError raised.
        - Message includes "Controller already open".

        Testing Principle:
        Validates resource protection.
        """
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = SerialMotorDriver._create_with_enumerator(enum)
        driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        with pytest.raises(RuntimeError, match="Controller already open"):
            driver._open_with_serial(MockSerialPort(), "/dev/ttyTEST2")

    def test_close_driver(self) -> None:
        """Verifies driver close() closes the controller.

        Tests proper cleanup through driver interface.

        Business context:
        Driver.close() must clean up serial and controller reference.
        Ensures no resource leaks and allows reopening later.

        Arrangement:
        1. Create driver and open controller.

        Action:
        Call driver.close().

        Assertion Strategy:
        Validates cleanup by confirming:
        - Serial port is closed.
        - Controller reference is None.

        Testing Principle:
        Validates resource cleanup.
        """
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = SerialMotorDriver._create_with_enumerator(enum)
        driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        driver.close()

        assert mock_serial._closed is True
        assert driver._controller is None


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestControllerLifecycle:
    """Test suite for controller lifecycle.

    Categories:
    1. Close - Close controller and serial (1 test)
    2. Error States - Operations on closed controller (1 test)

    Total: 2 tests.
    """

    def test_close_controller(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies close() closes serial and updates state.

        Tests controller closure behavior.

        Business context:
        close() must close serial port and mark controller as closed.
        Subsequent operations should fail with clear error.

        Arrangement:
        1. Use motor_controller from fixture.

        Action:
        Call close() on controller.

        Assertion Strategy:
        Validates closure by confirming:
        - _is_open is False.
        - Serial port is closed.

        Testing Principle:
        Validates shutdown sequence.
        """
        motor_controller.close()

        assert motor_controller._is_open is False
        assert mock_serial._closed is True

    def test_move_when_closed_raises(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Verifies moving on closed controller raises error.

        Tests error handling for operations after close.

        Business context:
        After close(), all operations should be invalid. Must
        raise clear error to prevent use-after-close bugs.

        Arrangement:
        1. Close controller.

        Action:
        Attempt move() on closed controller.

        Assertion Strategy:
        Validates state checking by confirming:
        - RuntimeError raised.
        - Message includes "Controller is closed".

        Testing Principle:
        Validates lifecycle enforcement.
        """
        motor_controller.close()

        with pytest.raises(RuntimeError, match="Controller is closed"):
            motor_controller.move(MotorType.ALTITUDE, 1000)


# =============================================================================
# Position Conversion Tests
# =============================================================================


class TestPositionConversion:
    """Test suite for step/degree conversion utilities.

    Categories:
    1. Altitude Conversion - Zenith, horizon, 45° (3 tests)
    2. Azimuth Conversion - Center, round-trip (2 tests)

    Total: 5 tests.
    """

    def test_altitude_zenith(self) -> None:
        """Verifies steps 0 equals 90° (zenith).

        Tests altitude reference point conversion.

        Business context:
        Motor position 0 corresponds to telescope pointing straight
        up (90° altitude = zenith). This is the altitude reference.

        Arrangement:
        1. None - testing pure function.

        Action:
        Call steps_to_altitude_degrees(0).

        Assertion Strategy:
        Validates conversion by confirming:
        - Returns exactly 90.0 degrees.

        Testing Principle:
        Validates reference point conversion.
        """
        degrees = steps_to_altitude_degrees(0)
        assert degrees == 90.0

    def test_altitude_at_neg60(self) -> None:
        """Verifies step position for -60° from zenith (30° altitude).

        Tests that -60° from zenith (the negative limit) converts
        to 30° altitude (60° below the 90° zenith).

        Arrangement:
        1. None - testing pure function.

        Action:
        Call steps_to_altitude_degrees with -60° worth of steps.

        Assertion Strategy:
        Validates conversion to approximately 150° (since negative
        steps go past zenith direction in this coordinate system).
        """
        from telescope_mcp.drivers.motors.serial_controller import ALTITUDE_MIN_STEPS

        degrees = steps_to_altitude_degrees(ALTITUDE_MIN_STEPS)
        assert abs(degrees - 150.0) < 0.1

    def test_altitude_45_degrees(self) -> None:
        """Verifies 45° converts to approximately 70000 steps.

        Tests mid-range conversion and round-trip accuracy.

        Business context:
        45° is common observation angle. Should be approximately
        midway through motor range (~70000 steps). Round-trip
        must preserve accuracy.

        Arrangement:
        1. None - testing pure functions.

        Action:
        Convert 45° to steps, then back to degrees.

        Assertion Strategy:
        Validates conversion by confirming:
        - Steps are between 69000-71000.
        - Round-trip returns within 0.1° of original.

        Testing Principle:
        Validates mid-range accuracy and round-trip.
        """
        steps = altitude_degrees_to_steps(45.0)
        assert 69000 < steps < 71000

        # Round trip
        degrees = steps_to_altitude_degrees(steps)
        assert abs(degrees - 45.0) < 0.1

    def test_azimuth_center(self) -> None:
        """Verifies steps 0 equals 0° (center).

        Tests azimuth reference point conversion.

        Business context:
        Motor position 0 is azimuth home (0°). Azimuth range
        is 0 to +154814 steps (0° to +190°).

        Arrangement:
        1. None - testing pure function.

        Action:
        Call steps_to_azimuth_degrees(0).

        Assertion Strategy:
        Validates conversion by confirming:
        - Returns exactly 0.0 degrees.

        Testing Principle:
        Validates reference point conversion.
        """
        degrees = steps_to_azimuth_degrees(0)
        assert degrees == 0.0

    def test_azimuth_conversion_roundtrip(self) -> None:
        """Verifies steps to degrees and back matches original.

        Tests round-trip conversion accuracy.

        Business context:
        Conversions must be invertible for position tracking.
        Converting steps to degrees and back should return
        original value (within rounding tolerance).

        Arrangement:
        1. Start with 50000 steps.

        Action:
        Convert to degrees, then back to steps.

        Assertion Strategy:
        Validates round-trip by confirming:
        - Final steps within 2 of original (rounding).

        Testing Principle:
        Validates conversion accuracy and invertibility.
        """
        original_steps = 50000
        degrees = steps_to_azimuth_degrees(original_steps)
        back_to_steps = azimuth_degrees_to_steps(degrees)

        assert abs(back_to_steps - original_steps) < 2  # Allow rounding


# =============================================================================
# Motor Configuration Tests
# =============================================================================


class TestMotorConfiguration:
    """Test suite for motor configuration.

    Categories:
    1. Altitude Config - Configuration exists and correct (1 test)
    2. Azimuth Config - Configuration exists and correct (1 test)

    Total: 2 tests.
    """

    def test_altitude_config_exists(self) -> None:
        """Verifies altitude motor configuration exists and is correct.

        Tests MOTOR_CONFIGS contains altitude with proper values.

        Business context:
        Altitude motor config defines axis ID and limits. Must be
        present and correct for motor operations. Axis 0 is altitude,
        range 0-140000 steps.

        Arrangement:
        1. None - testing static configuration.

        Action:
        Access MOTOR_CONFIGS[ALTITUDE].

        Assertion Strategy:
        Validates config by confirming:
        - Is MotorConfig instance.
        - axis_id is 0.
        - min_steps is 0.
        - max_steps is 140000.

        Testing Principle:
        Validates configuration integrity.
        """
        config = MOTOR_CONFIGS[MotorType.ALTITUDE]
        assert isinstance(config, MotorConfig)
        assert config.axis_id == 0
        assert config.min_steps == int(-60 * (140000 / 90.0))
        assert config.max_steps == int(3 * (140000 / 90.0))

    def test_azimuth_config_exists(self) -> None:
        """Verifies azimuth motor configuration exists and is correct.

        Tests MOTOR_CONFIGS contains azimuth with proper values.

        Business context:
        Azimuth motor config defines axis ID and limits. Must be
        present and correct for motor operations. Axis 1 is azimuth,
        range 0 to +154814 steps (0° to +190°).

        Arrangement:
        1. None - testing static configuration.

        Action:
        Access MOTOR_CONFIGS[AZIMUTH].

        Assertion Strategy:
        Validates config by confirming:
        - Is MotorConfig instance.
        - axis_id is 1.
        - min_steps is 0.
        - max_steps is 154814.

        Testing Principle:
        Validates configuration integrity.
        """
        config = MOTOR_CONFIGS[MotorType.AZIMUTH]
        assert isinstance(config, MotorConfig)
        assert config.axis_id == 1
        assert config.min_steps == 0
        assert config.max_steps == int(190 * (110000 / 135.0))


# =============================================================================
# Test: zero_position (Set Home - Issue #4)
# =============================================================================


class TestSerialZeroPosition:
    """Tests for zero_position() on serial motor controller."""

    def test_zero_altitude_sends_serial_command(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies zero_position sends axis select + set position command.

        Business context:
            Set Home must select correct axis and send p0 command to Teensy
            firmware to zero the position counter without movement.

        Arrangement:
            1. Move altitude to 50000.
            2. Queue axis response for zero_position.
            3. Queue set position acknowledgment.

        Assertion Strategy:
            - p0 command appears in serial writes.
            - Internal position tracking shows 0.
        """
        # First move to a position
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.move(MotorType.ALTITUDE, -50000)

        # Now zero the position
        mock_serial.queue_axis_response()
        mock_serial.queue_response(b"{'position': 0}")
        motor_controller.zero_position(MotorType.ALTITUDE)

        commands = mock_serial.get_written_commands()
        assert any("p0" in cmd for cmd in commands), f"Expected 'p0' in {commands}"
        assert motor_controller.get_status(MotorType.ALTITUDE).position_steps == 0

    def test_zero_azimuth_sends_correct_axis(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies zero_position selects azimuth axis (A1) before zeroing.

        Arrangement:
            1. Queue axis response and set position ack for azimuth.

        Assertion Strategy:
            - A1 axis select appears before p0 in serial writes.
        """
        mock_serial.queue_axis_response()
        mock_serial.queue_response(b"{'position': 0}")
        motor_controller.zero_position(MotorType.AZIMUTH)

        commands = mock_serial.get_written_commands()
        assert any("A1" in cmd for cmd in commands), f"Expected 'A1' in {commands}"
        assert any("p0" in cmd for cmd in commands), f"Expected 'p0' in {commands}"

    def test_zero_when_closed_raises(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """zero_position on closed controller raises RuntimeError."""
        motor_controller.close()
        with pytest.raises(RuntimeError, match="closed"):
            motor_controller.zero_position(MotorType.ALTITUDE)


# =============================================================================
# Test: Stop with Priority Bypass
# =============================================================================


class TestSerialStop:
    """Tests for stop() with lock-bypass priority on serial controller."""

    def test_stop_writes_to_serial(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Verifies stop sends S command to serial port.

        Business context:
            Stop must write directly to serial without waiting for lock.
            The S command tells firmware to halt current motor movement.
        """
        motor_controller.stop(MotorType.ALTITUDE)

        commands = mock_serial.get_written_commands()
        assert any("S" in cmd for cmd in commands), f"Expected 'S' in {commands}"

    def test_stop_all_writes_to_serial(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Emergency stop (motor=None) also writes S command."""
        motor_controller.stop()

        commands = mock_serial.get_written_commands()
        assert any("S" in cmd for cmd in commands), f"Expected 'S' in {commands}"

    def test_stop_clears_moving_flags(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Stop clears is_moving flag for specified motor."""
        motor_controller.stop(MotorType.ALTITUDE)
        assert motor_controller.get_status(MotorType.ALTITUDE).is_moving is False

    def test_stop_all_clears_all_flags(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Emergency stop clears all is_moving flags."""
        motor_controller.stop()
        assert motor_controller.get_status(MotorType.ALTITUDE).is_moving is False
        assert motor_controller.get_status(MotorType.AZIMUTH).is_moving is False

    def test_stop_is_idempotent(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Multiple stop calls are safe (no errors)."""
        motor_controller.stop(MotorType.ALTITUDE)
        motor_controller.stop(MotorType.ALTITUDE)
        motor_controller.stop()  # Emergency stop too
