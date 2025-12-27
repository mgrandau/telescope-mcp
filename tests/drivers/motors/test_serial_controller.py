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
    Simulates the custom motor controller protocol.
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

    def queue_response(self, data: bytes) -> None:
        """Add response to the read queue.

        Args:
            data: Bytes to return on next read call.
        """
        self._read_queue.append(data)
        self._in_waiting = sum(len(b) for b in self._read_queue)

    def queue_axis_response(self) -> None:
        """Queue typical axis select response."""
        self.queue_response(b"{'axis': 0}")

    def queue_move_complete(self) -> None:
        """Queue move completion response."""
        self.queue_response(b"{'alldone': 1}\r\n")

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
        """Read a line (until \\n)."""
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
        return [data.decode() for data in self._write_log]


class MockComPort:
    """Mock COM port for testing port enumeration."""

    def __init__(self, device: str, description: str) -> None:
        """Initialize mock COM port.

        Args:
            device: Port device path.
            description: Port description.
        """
        self.device = device
        self.description = description


class MockPortEnumerator:
    """Mock port enumerator implementing PortEnumerator protocol."""

    def __init__(self, ports: list[MockComPort] | None = None) -> None:
        """Initialize with list of mock ports."""
        self._ports = ports or []

    def comports(self) -> list[MockComPort]:
        """Return list of mock COM ports."""
        return self._ports


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_serial() -> MockSerialPort:
    """Create a mock serial port."""
    return MockSerialPort()


@pytest.fixture
def motor_controller(mock_serial: MockSerialPort) -> SerialMotorController:
    """Create a SerialMotorController with mock serial."""
    return SerialMotorController._create_with_serial(
        mock_serial,
        port_name="/dev/ttyTEST",
    )


# =============================================================================
# Axis Selection Tests
# =============================================================================


class TestAxisSelection:
    """Tests for axis selection."""

    def test_select_altitude_axis(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Should send A0 command for altitude axis."""
        mock_serial.queue_axis_response()

        motor_controller._select_axis(MotorType.ALTITUDE)

        commands = mock_serial.get_written_commands()
        assert "A0" in commands

    def test_select_azimuth_axis(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Should send A1 command for azimuth axis."""
        mock_serial.queue_axis_response()

        motor_controller._select_axis(MotorType.AZIMUTH)

        commands = mock_serial.get_written_commands()
        assert "A1" in commands

    def test_skip_reselect_same_axis(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Should not resend axis command if already selected."""
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
    """Tests for motor movement commands."""

    def test_move_absolute(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """move() should send absolute position command."""
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()

        motor_controller.move(MotorType.ALTITUDE, 70000)

        commands = mock_serial.get_written_commands()
        assert any("o70000" in cmd for cmd in commands)

    def test_move_updates_position(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """move() should update internal position tracking."""
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()

        motor_controller.move(MotorType.ALTITUDE, 50000)

        status = motor_controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 50000

    def test_move_relative(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """move_relative() should send relative movement command."""
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
        """move_relative() should handle negative steps."""
        # First move to a position
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.move(MotorType.ALTITUDE, 50000)

        # Then relative move back
        mock_serial.queue_move_complete()
        motor_controller.move_relative(MotorType.ALTITUDE, -1000)

        status = motor_controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 49000

    def test_move_azimuth(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Should be able to move azimuth motor."""
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()

        motor_controller.move(MotorType.AZIMUTH, 50000)

        status = motor_controller.get_status(MotorType.AZIMUTH)
        assert status.position_steps == 50000


# =============================================================================
# Position Limits Tests
# =============================================================================


class TestPositionLimits:
    """Tests for position limit enforcement."""

    def test_altitude_max_limit(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Should reject altitude position above max."""
        with pytest.raises(ValueError, match="must be 0-140000"):
            motor_controller.move(MotorType.ALTITUDE, 150000)

    def test_altitude_min_limit(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Should reject negative altitude position."""
        with pytest.raises(ValueError, match="must be 0-140000"):
            motor_controller.move(MotorType.ALTITUDE, -100)

    def test_azimuth_max_limit(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Should reject azimuth position above max."""
        with pytest.raises(ValueError, match="must be -110000-110000"):
            motor_controller.move(MotorType.AZIMUTH, 120000)

    def test_azimuth_min_limit(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Should reject azimuth position below min."""
        with pytest.raises(ValueError, match="must be -110000-110000"):
            motor_controller.move(MotorType.AZIMUTH, -120000)

    def test_relative_move_limit(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """Relative move should check resulting position."""
        # Start at position 0
        with pytest.raises(ValueError, match="would exceed limits"):
            motor_controller.move_relative(MotorType.ALTITUDE, -100)

    def test_speed_range_validation(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Should validate speed is 1-100."""
        with pytest.raises(ValueError, match="Speed must be 1-100"):
            motor_controller.move(MotorType.ALTITUDE, 1000, speed=0)

        with pytest.raises(ValueError, match="Speed must be 1-100"):
            motor_controller.move(MotorType.ALTITUDE, 1000, speed=101)


# =============================================================================
# Homing Tests
# =============================================================================


class TestHoming:
    """Tests for motor homing."""

    def test_home_altitude(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """home() should move to configured home position."""
        # First move away from home
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.move(MotorType.ALTITUDE, 50000)

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
        """home_all() should home both motors."""
        # Move both away from home
        mock_serial.queue_axis_response()
        mock_serial.queue_move_complete()
        motor_controller.move(MotorType.ALTITUDE, 50000)

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
    """Tests for motor status."""

    def test_get_status(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """get_status() should return MotorStatus."""
        status = motor_controller.get_status(MotorType.ALTITUDE)

        assert isinstance(status, MotorStatus)
        assert status.motor == MotorType.ALTITUDE
        assert status.position_steps == 0
        assert status.is_moving is False

    def test_get_info(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """get_info() should return controller information."""
        info = motor_controller.get_info()

        assert info["type"] == "serial_motor_controller"
        assert info["port"] == "/dev/ttyTEST"
        assert info["is_open"] is True
        assert "altitude" in info
        assert "azimuth" in info


# =============================================================================
# Driver Tests (Port Enumeration)
# =============================================================================


class TestSerialMotorDriver:
    """Tests for SerialMotorDriver port enumeration."""

    def test_get_available_controllers_with_acm(self) -> None:
        """Should detect ACM devices as motor controllers."""
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
        """Should detect CH340 USB-serial devices."""
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyUSB0", "CH340 Serial Adapter"),
            ]
        )

        driver = SerialMotorDriver._create_with_enumerator(enum)
        controllers = driver.get_available_controllers()

        assert len(controllers) == 1

    def test_get_available_controllers_empty(self) -> None:
        """Should return empty list when no controllers found."""
        enum = MockPortEnumerator(
            [
                MockComPort("/dev/ttyS0", "Standard Serial Port"),
            ]
        )

        driver = SerialMotorDriver._create_with_enumerator(enum)
        controllers = driver.get_available_controllers()

        assert len(controllers) == 0

    def test_open_with_serial_injection(self) -> None:
        """Should allow opening with injected serial port."""
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = SerialMotorDriver._create_with_enumerator(enum)
        controller = driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        assert controller is not None
        assert controller._port == "/dev/ttyTEST"

    def test_open_already_open_raises(self) -> None:
        """Should raise error when trying to open twice."""
        mock_serial = MockSerialPort()
        enum = MockPortEnumerator([])

        driver = SerialMotorDriver._create_with_enumerator(enum)
        driver._open_with_serial(mock_serial, "/dev/ttyTEST")

        with pytest.raises(RuntimeError, match="Controller already open"):
            driver._open_with_serial(MockSerialPort(), "/dev/ttyTEST2")

    def test_close_driver(self) -> None:
        """Driver close should close controller."""
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
    """Tests for controller lifecycle."""

    def test_close_controller(
        self,
        motor_controller: SerialMotorController,
        mock_serial: MockSerialPort,
    ) -> None:
        """close() should close serial and update state."""
        motor_controller.close()

        assert motor_controller._is_open is False
        assert mock_serial._closed is True

    def test_move_when_closed_raises(
        self,
        motor_controller: SerialMotorController,
    ) -> None:
        """Should raise when moving on closed controller."""
        motor_controller.close()

        with pytest.raises(RuntimeError, match="Controller is closed"):
            motor_controller.move(MotorType.ALTITUDE, 1000)


# =============================================================================
# Position Conversion Tests
# =============================================================================


class TestPositionConversion:
    """Tests for step/degree conversion utilities."""

    def test_altitude_zenith(self) -> None:
        """Steps 0 should be 90 degrees (zenith)."""
        degrees = steps_to_altitude_degrees(0)
        assert degrees == 90.0

    def test_altitude_horizon(self) -> None:
        """Steps 140000 should be 0 degrees (horizon)."""
        degrees = steps_to_altitude_degrees(140000)
        assert abs(degrees) < 0.1  # Close to 0

    def test_altitude_45_degrees(self) -> None:
        """45 degrees should be ~70000 steps."""
        steps = altitude_degrees_to_steps(45.0)
        assert 69000 < steps < 71000

        # Round trip
        degrees = steps_to_altitude_degrees(steps)
        assert abs(degrees - 45.0) < 0.1

    def test_azimuth_center(self) -> None:
        """Steps 0 should be 0 degrees (center)."""
        degrees = steps_to_azimuth_degrees(0)
        assert degrees == 0.0

    def test_azimuth_conversion_roundtrip(self) -> None:
        """Steps to degrees and back should match."""
        original_steps = 50000
        degrees = steps_to_azimuth_degrees(original_steps)
        back_to_steps = azimuth_degrees_to_steps(degrees)

        assert abs(back_to_steps - original_steps) < 2  # Allow rounding


# =============================================================================
# Motor Configuration Tests
# =============================================================================


class TestMotorConfiguration:
    """Tests for motor configuration."""

    def test_altitude_config_exists(self) -> None:
        """Altitude motor config should exist."""
        config = MOTOR_CONFIGS[MotorType.ALTITUDE]
        assert isinstance(config, MotorConfig)
        assert config.axis_id == 0
        assert config.min_steps == 0
        assert config.max_steps == 140000

    def test_azimuth_config_exists(self) -> None:
        """Azimuth motor config should exist."""
        config = MOTOR_CONFIGS[MotorType.AZIMUTH]
        assert isinstance(config, MotorConfig)
        assert config.axis_id == 1
        assert config.min_steps == -110000
        assert config.max_steps == 110000
