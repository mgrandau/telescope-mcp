"""Serial motor controller driver.

Communicates with custom stepper motor controller over serial USB to control
altitude and azimuth motors for telescope positioning.

Hardware:
- Custom stepper motor controller board
- NEMA 23 stepper for altitude (0-140000 steps = 0°-90°)
- NEMA 17 stepper for azimuth (belt-driven, ~270° range)

Serial Protocol:
- Baud rate: 9600
- Commands:
  - `?` - Display help
  - `A0` / `A1` - Select axis (0=altitude, 1=azimuth)
  - `o{steps}` - Absolute move to position
  - `O{steps}` - Relative move by steps
- Response format: JSON-like `{'alldone': 1}\\r\\n` on completion

Axis Configuration:
- Axis 0 (altitude): 0=zenith (90°), 140000=horizon (0°)
- Axis 1 (azimuth): 0=center/home, ±110000 safe range

Example:
    from telescope_mcp.drivers.motors import SerialMotorDriver

    driver = SerialMotorDriver()
    ports = driver.get_available_controllers()

    if ports:
        controller = driver.open(ports[0]["port"])
        controller.move(MotorType.ALTITUDE, 70000)  # Move to 45°
        controller.close()

Testing:
    The driver supports dependency injection for testing without hardware:

    from telescope_mcp.drivers.serial import SerialPort
    from telescope_mcp.drivers.motors.serial_controller import (
        SerialMotorController,
    )

    # Create mock serial port
    mock = MockSerialPort()
    mock.queue_response(b"{'alldone': 1}\\r\\n")

    controller = SerialMotorController._create_with_serial(mock)
    controller.move(MotorType.ALTITUDE, 1000)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from telescope_mcp.drivers.motors.types import MotorStatus, MotorType
from telescope_mcp.drivers.serial import PortEnumerator, SerialPort
from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# Motor configuration constants
ALTITUDE_MAX_STEPS = 140000  # 0° (horizon)
ALTITUDE_MIN_STEPS = 0  # 90° (zenith)
AZIMUTH_MAX_STEPS = 110000
AZIMUTH_MIN_STEPS = -110000
DEFAULT_BAUDRATE = 9600
DEFAULT_TIMEOUT = 30.0  # Motor moves can take a while


@dataclass
class MotorConfig:
    """Configuration for a motor axis.

    Attributes:
        axis_id: Controller axis number (0=altitude, 1=azimuth).
        min_steps: Minimum allowed position in steps.
        max_steps: Maximum allowed position in steps.
        home_position: Home position in steps.
        steps_per_degree: Steps per degree of rotation.
    """

    axis_id: int
    min_steps: int
    max_steps: int
    home_position: int
    steps_per_degree: float


# Default motor configurations
MOTOR_CONFIGS = {
    MotorType.ALTITUDE: MotorConfig(
        axis_id=0,
        min_steps=ALTITUDE_MIN_STEPS,
        max_steps=ALTITUDE_MAX_STEPS,
        home_position=0,  # Zenith
        steps_per_degree=ALTITUDE_MAX_STEPS / 90.0,  # ~1555 steps/degree
    ),
    MotorType.AZIMUTH: MotorConfig(
        axis_id=1,
        min_steps=AZIMUTH_MIN_STEPS,
        max_steps=AZIMUTH_MAX_STEPS,
        home_position=0,  # Center
        steps_per_degree=AZIMUTH_MAX_STEPS / 135.0,  # ~815 steps/degree (270° range)
    ),
}


class SerialMotorController:
    """Serial motor controller for telescope stepper motors.

    Implements MotorController protocol using serial communication
    with custom stepper controller board.

    Thread-safe: Uses lock for serial communication.
    """

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE) -> None:
        """Open serial connection to motor controller.

        Args:
            port: Serial port (e.g., /dev/ttyACM0).
            baudrate: Serial baud rate (default 9600).

        Raises:
            RuntimeError: If serial connection fails.
        """
        try:
            import serial as serial_module

            serial_port = serial_module.Serial(port, baudrate=baudrate, timeout=1.0)
        except ImportError:
            raise RuntimeError("pyserial not installed. Run: pdm add pyserial")
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {port}: {e}")

        self._init_with_serial(serial_port, port)
        logger.info("Motor controller connected", port=port)

    @classmethod
    def _create_with_serial(
        cls,
        serial_port: SerialPort,
        port_name: str = "/dev/mock",
    ) -> SerialMotorController:
        """Create instance with injected serial port (for testing).

        Args:
            serial_port: Mock or real serial port implementing SerialPort protocol.
            port_name: Port name for identification.

        Returns:
            Configured SerialMotorController.
        """
        instance = cls.__new__(cls)
        instance._init_with_serial(serial_port, port_name)
        return instance

    def _init_with_serial(self, serial_port: SerialPort, port_name: str) -> None:
        """Initialize instance state with given serial port.

        Args:
            serial_port: Serial port to use for communication.
            port_name: Port name for identification.
        """
        self._serial: SerialPort = serial_port
        self._port = port_name
        self._is_open = True

        # Track current axis and positions
        self._current_axis: int | None = None
        self._positions = {
            MotorType.ALTITUDE: 0,
            MotorType.AZIMUTH: 0,
        }
        self._is_moving = {
            MotorType.ALTITUDE: False,
            MotorType.AZIMUTH: False,
        }

        # Thread safety
        self._lock = threading.Lock()

    def _select_axis(self, motor: MotorType) -> None:
        """Select the axis for subsequent commands.

        Args:
            motor: Motor to select.
        """
        config = MOTOR_CONFIGS[motor]

        if self._current_axis == config.axis_id:
            return  # Already selected

        # Send axis select command
        cmd = f"A{config.axis_id}".encode()
        self._serial.write(cmd)

        # Wait for response (ends with })
        response = self._serial.read_until(b"}")
        logger.debug(
            "Axis selected",
            axis=config.axis_id,
            response=response.decode(errors="ignore"),
        )

        self._current_axis = config.axis_id

    def _send_move_command(
        self,
        command: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> str:
        """Send move command and wait for completion.

        Args:
            command: Movement command (o{steps} or O{steps}).
            timeout: Seconds to wait for completion.

        Returns:
            Response string from controller.

        Raises:
            RuntimeError: If move times out.
        """
        self._serial.write(command.encode())
        logger.debug("Sent move command", command=command)

        # Wait for completion response
        start_time = time.time()
        response_data = b""

        while (time.time() - start_time) < timeout:
            chunk = self._serial.read_until(b"\r\n")
            response_data += chunk

            if b"alldone" in response_data:
                break

            time.sleep(0.01)
        else:
            raise RuntimeError(f"Move command timed out after {timeout}s")

        response = response_data.decode(errors="ignore")
        logger.debug("Move complete", response=response.strip())
        return response

    def move(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Move motor to absolute position.

        Args:
            motor: Which motor to move.
            steps: Absolute position in steps.
            speed: Speed percentage (1-100). Note: Custom controller may not support.

        Raises:
            ValueError: If steps outside valid range or speed invalid.
            RuntimeError: If controller not connected.
        """
        if not self._is_open:
            raise RuntimeError("Controller is closed")

        if not 1 <= speed <= 100:
            raise ValueError(f"Speed must be 1-100, got {speed}")

        config = MOTOR_CONFIGS[motor]

        # Enforce limits
        if not config.min_steps <= steps <= config.max_steps:
            raise ValueError(
                f"{motor.value} steps must be {config.min_steps}-{config.max_steps}, "
                f"got {steps}"
            )

        with self._lock:
            self._is_moving[motor] = True
            try:
                self._select_axis(motor)

                # Absolute move command
                cmd = f"o{steps}"
                self._send_move_command(cmd)

                self._positions[motor] = steps
            finally:
                self._is_moving[motor] = False

    def move_relative(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Move motor by relative steps.

        Args:
            motor: Which motor to move.
            steps: Steps to move (positive or negative).
            speed: Speed percentage (1-100).

        Raises:
            ValueError: If resulting position outside valid range.
            RuntimeError: If controller not connected.
        """
        if not self._is_open:
            raise RuntimeError("Controller is closed")

        config = MOTOR_CONFIGS[motor]
        new_position = self._positions[motor] + steps

        # Enforce limits
        if not config.min_steps <= new_position <= config.max_steps:
            raise ValueError(
                f"{motor.value} move would exceed limits: "
                f"{self._positions[motor]} + {steps} = {new_position}"
            )

        with self._lock:
            self._is_moving[motor] = True
            try:
                self._select_axis(motor)

                # Relative move command
                cmd = f"O{steps}"
                self._send_move_command(cmd)

                self._positions[motor] = new_position
            finally:
                self._is_moving[motor] = False

    def stop(self, motor: MotorType | None = None) -> None:
        """Stop motor(s) immediately.

        Note: Custom controller may not support stop during move.
        This implementation is a placeholder.

        Args:
            motor: Motor to stop, or None for all.
        """
        logger.warning(
            "Stop command sent - custom controller may not support interrupt",
            motor=motor.value if motor else "all",
        )
        # Custom controller doesn't have a stop command documented
        # Movement blocks until complete

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get current motor status.

        Args:
            motor: Which motor to query.

        Returns:
            MotorStatus with position and state.
        """
        return MotorStatus(
            motor=motor,
            is_moving=self._is_moving[motor],
            position_steps=self._positions[motor],
            speed=0,  # Speed tracking not implemented
        )

    def home(self, motor: MotorType) -> None:
        """Move motor to home position.

        Args:
            motor: Which motor to home.
        """
        config = MOTOR_CONFIGS[motor]
        self.move(motor, config.home_position)
        logger.info("Motor homed", motor=motor.value)

    def home_all(self) -> None:
        """Home both motors to their home positions."""
        self.home(MotorType.ALTITUDE)
        self.home(MotorType.AZIMUTH)
        logger.info("All motors homed")

    def get_help(self) -> str:
        """Get controller help text.

        Returns:
            Help text from controller.
        """
        with self._lock:
            self._serial.write(b"?")
            response = self._serial.read_until(b"?\tdisplay this help screen\r\n")
            return response.decode(errors="ignore")

    def get_info(self) -> dict:
        """Get controller information.

        Returns:
            Dict with controller type, port, and configuration.
        """
        return {
            "type": "serial_motor_controller",
            "port": self._port,
            "is_open": self._is_open,
            "altitude": {
                "position": self._positions[MotorType.ALTITUDE],
                "min": MOTOR_CONFIGS[MotorType.ALTITUDE].min_steps,
                "max": MOTOR_CONFIGS[MotorType.ALTITUDE].max_steps,
            },
            "azimuth": {
                "position": self._positions[MotorType.AZIMUTH],
                "min": MOTOR_CONFIGS[MotorType.AZIMUTH].min_steps,
                "max": MOTOR_CONFIGS[MotorType.AZIMUTH].max_steps,
            },
        }

    def close(self) -> None:
        """Close the serial connection."""
        self._is_open = False

        if self._serial and self._serial.is_open:
            self._serial.close()

        logger.info("Motor controller closed", port=self._port)


class SerialMotorDriver:
    """Driver for serial motor controllers.

    Discovers motor controllers on serial ports and creates connections.

    Example:
        driver = SerialMotorDriver()
        controllers = driver.get_available_controllers()

        for ctrl in controllers:
            print(f"Found: {ctrl['name']} on {ctrl['port']}")

        if controllers:
            controller = driver.open(controllers[0]["port"])
    """

    def __init__(self, baudrate: int = DEFAULT_BAUDRATE) -> None:
        """Initialize driver with baud rate.

        Args:
            baudrate: Serial baud rate for controller communication.
        """
        self._baudrate = baudrate
        self._controller: SerialMotorController | None = None
        self._port_enumerator: PortEnumerator | None = None

    @classmethod
    def _create_with_enumerator(
        cls,
        port_enumerator: PortEnumerator,
        baudrate: int = DEFAULT_BAUDRATE,
    ) -> SerialMotorDriver:
        """Create driver with injected port enumerator (for testing).

        Args:
            port_enumerator: Object with comports() method.
            baudrate: Serial baud rate.

        Returns:
            Configured SerialMotorDriver.
        """
        driver = cls.__new__(cls)
        driver._baudrate = baudrate
        driver._controller = None
        driver._port_enumerator = port_enumerator
        return driver

    def get_available_controllers(self) -> list[dict]:
        """List available motor controllers on serial ports.

        Scans serial ports for potential motor controller devices.

        Returns:
            List of controller info dicts with id, type, name, port.
        """
        # Use injected enumerator or real pyserial
        if self._port_enumerator is not None:
            ports = self._port_enumerator.comports()
        else:
            try:
                import serial.tools.list_ports
            except ImportError:
                logger.warning("pyserial not installed, cannot scan ports")
                return []
            ports = serial.tools.list_ports.comports()

        controllers = []

        for i, port in enumerate(ports):
            # Check for likely motor controller devices
            desc = port.description.lower()
            if any(x in desc for x in ["acm", "usb serial", "ch340", "motor"]):
                controllers.append(
                    {
                        "id": i,
                        "type": "serial_motor_controller",
                        "name": f"Motor Controller ({port.device})",
                        "port": port.device,
                        "description": port.description,
                    }
                )

        logger.debug("Found controllers", count=len(controllers))
        return controllers

    def open(self, port: str) -> SerialMotorController:
        """Open connection to motor controller.

        Args:
            port: Serial port path (e.g., /dev/ttyACM0).

        Returns:
            SerialMotorController for controlling motors.

        Raises:
            RuntimeError: If connection fails or already open.
        """
        if self._controller is not None and self._controller._is_open:
            raise RuntimeError("Controller already open")

        self._controller = SerialMotorController(port, self._baudrate)
        return self._controller

    def _open_with_serial(
        self,
        serial_port: SerialPort,
        port_name: str = "/dev/mock",
    ) -> SerialMotorController:
        """Open with injected serial port (for testing).

        Args:
            serial_port: Mock serial port implementing SerialPort protocol.
            port_name: Port name for identification.

        Returns:
            SerialMotorController configured with mock serial.
        """
        if self._controller is not None and self._controller._is_open:
            raise RuntimeError("Controller already open")

        self._controller = SerialMotorController._create_with_serial(
            serial_port, port_name
        )
        return self._controller

    def close(self) -> None:
        """Close the current controller."""
        if self._controller is not None:
            self._controller.close()
            self._controller = None


# Utility functions for position conversion


def steps_to_altitude_degrees(steps: int) -> float:
    """Convert altitude steps to degrees.

    Args:
        steps: Position in steps (0=zenith, 140000=horizon).

    Returns:
        Altitude in degrees (90=zenith, 0=horizon).
    """
    # Steps 0 = zenith (90°), steps 140000 = horizon (0°)
    return 90.0 - (steps / MOTOR_CONFIGS[MotorType.ALTITUDE].steps_per_degree)


def altitude_degrees_to_steps(degrees: float) -> int:
    """Convert altitude degrees to steps.

    Args:
        degrees: Altitude in degrees (90=zenith, 0=horizon).

    Returns:
        Position in steps.
    """
    steps = (90.0 - degrees) * MOTOR_CONFIGS[MotorType.ALTITUDE].steps_per_degree
    return int(round(steps))


def steps_to_azimuth_degrees(steps: int) -> float:
    """Convert azimuth steps to degrees from center.

    Args:
        steps: Position in steps (0=center, negative=CW, positive=CCW).

    Returns:
        Azimuth offset in degrees from center.
    """
    return steps / MOTOR_CONFIGS[MotorType.AZIMUTH].steps_per_degree


def azimuth_degrees_to_steps(degrees: float) -> int:
    """Convert azimuth degrees to steps.

    Args:
        degrees: Azimuth offset in degrees from center.

    Returns:
        Position in steps.
    """
    steps = degrees * MOTOR_CONFIGS[MotorType.AZIMUTH].steps_per_degree
    return int(round(steps))
