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

from telescope_mcp.drivers.motors.types import (
    AvailableMotorController,
    MotorInfo,
    MotorInstance,
    MotorStatus,
    MotorType,
)
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

        Creates serial connection and initializes controller state.

        Business context: Motor controller connects via serial port to
        custom stepper board. This constructor handles port opening and
        error reporting for common issues (missing pyserial, port in use).

        Args:
            port: Serial port (e.g., /dev/ttyACM0).
            baudrate: Serial baud rate (default 9600).

        Returns:
            None. Controller connected and ready.

        Raises:
            RuntimeError: If serial connection fails (port busy, pyserial
                not installed, or hardware error).

        Example:
            >>> controller = SerialMotorController("/dev/ttyACM0")
            >>> controller.move(MotorType.ALTITUDE, 1000)
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
        """Create instance with injected serial port for testing.

        Factory method that bypasses normal serial port initialization,
        allowing tests to inject mock serial ports. Essential for unit
        testing motor controller logic without hardware.

        Business context: Hardware-independent testing enables CI/CD
        pipelines and rapid development iteration. Mock serial ports
        can simulate various response scenarios including errors.

        Args:
            serial_port: Object implementing SerialPort protocol. Can be
                real pyserial port or mock with read/write methods.
            port_name: Identifier string for logging. Defaults to '/dev/mock'.

        Returns:
            SerialMotorController: Fully initialized controller ready for
                move operations. Uses injected serial_port for all I/O.

        Raises:
            No exceptions during creation. Serial errors occur on use.

        Example:
            >>> mock_serial = MockSerialPort(responses=[b"OK\\r\\n"])
            >>> controller = SerialMotorController._create_with_serial(mock_serial)
            >>> controller.move(MotorType.ALTITUDE, 1000)
        """
        instance = cls.__new__(cls)
        instance._init_with_serial(serial_port, port_name)
        return instance

    def _init_with_serial(self, serial_port: SerialPort, port_name: str) -> None:
        """Initialize instance state with given serial port.

        Sets up all instance variables needed for motor control operations.
        Called by both __init__ (with real serial) and _create_with_serial
        (with mock serial for testing).

        Business context: Centralizes initialization logic so both production
        and test code paths use identical setup. Ensures consistent state
        regardless of how instance was created.

        Implementation: Stores serial port reference, initializes position
        tracking dicts for both axes (ALTITUDE, AZIMUTH), sets movement
        flags to False, and marks controller as open. No serial I/O
        performed during init - first communication happens on move/status.

        Args:
            serial_port: Serial port to use for communication. Must implement
                SerialPort protocol (read, write, read_until methods).
            port_name: Port name for identification in logs.

        Returns:
            None. Instance state initialized.

        Raises:
            No exceptions raised during initialization.

        Example:
            >>> # Called internally by factory methods
            >>> instance = cls.__new__(cls)
            >>> instance._init_with_serial(mock_serial, "/dev/mock")
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
        """Select the axis for subsequent movement commands.

        Sends axis selection command to the motor controller if not already
        selected. The controller remembers the selected axis, so this is
        optimized to skip redundant selections.

        Business context: The motor controller uses a single serial connection
        for both axes. Axis selection must precede move commands. This method
        tracks state to minimize unnecessary serial traffic.

        Implementation: Checks _current_axis against motor's axis_id. If
        different, sends 'A{axis_id}' command and waits for '}' response.
        Updates _current_axis on success.

        Args:
            motor: Motor to select (ALTITUDE or AZIMUTH).

        Returns:
            None. Internal _current_axis updated.

        Raises:
            No exceptions raised. Serial errors may occur on write.

        Example:
            >>> controller._select_axis(MotorType.ALTITUDE)
            >>> controller._current_axis
            1
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
        """Send move command and wait for completion response.

        Writes command to serial port and blocks until controller responds
        with completion or timeout expires. Handles multi-line responses
        accumulating until completion marker received.

        Business context: Motor movements take variable time depending on
        distance and speed. This method abstracts the blocking wait,
        allowing callers to treat moves as synchronous operations. Critical
        for coordinated multi-axis movements where sequencing matters.

        Args:
            command: Movement command (o{steps} or O{steps}).
                Lowercase 'o' for relative, uppercase 'O' for absolute.
            timeout: Seconds to wait for completion. Default 30s allows
                for full-range slews.

        Returns:
            Response string from controller (typically "OK" or position).

        Raises:
            RuntimeError: If move times out before completion response.

        Example:
            >>> controller._send_move_command("o1000", timeout=10.0)
            'OK'
            >>> controller._send_move_command("O0", timeout=30.0)  # Home
            'OK'
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
        """Move motor to absolute step position.

        Commands the specified motor to move to an absolute position and
        blocks until movement completes. Position limits are enforced to
        prevent mechanical damage.

        Business context: Core telescope pointing operation. Used by MCP
        tools and automation scripts to slew the telescope to target
        positions. Blocking behavior ensures moves complete before
        subsequent operations.

        Args:
            motor: Which motor to move. Use MotorType.ALTITUDE for
                elevation axis or MotorType.AZIMUTH for rotation.
            steps: Absolute target position in steps.
                Altitude: 0 (zenith) to 140000 (horizon)
                Azimuth: -500000 to +500000 (relative to center)
            speed: Speed percentage 1-100. Note: Current controller
                firmware may not support variable speed. Default 100.

        Returns:
            None. Method blocks until move completes.

        Raises:
            ValueError: If steps outside motor's valid range or
                speed not in 1-100.
            RuntimeError: If controller not connected.

        Example:
            >>> controller.move(MotorType.ALTITUDE, 70000)  # 45° up
            >>> controller.move(MotorType.AZIMUTH, 35000, speed=50)
            >>> # Multi-axis move (sequential)
            >>> controller.move(MotorType.ALTITUDE, 50000)
            >>> controller.move(MotorType.AZIMUTH, 0)  # Center
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
        """Move motor by relative step offset from current position.

        Commands the specified motor to move by a relative number of steps
        and blocks until movement completes. Validates that resulting
        position stays within motor limits.

        Business context: Enables incremental adjustments during alignment
        and fine-tuning. Preferred for small corrections where absolute
        position isn't needed. Used in hand-controller style interfaces.

        Args:
            motor: Which motor to move. Use MotorType.ALTITUDE for
                elevation axis or MotorType.AZIMUTH for rotation.
            steps: Number of steps to move from current position.
                Positive = higher steps (altitude: down, azimuth: CCW)
                Negative = lower steps (altitude: up, azimuth: CW)
            speed: Speed percentage 1-100. Note: Current controller
                firmware may not support variable speed. Default 100.

        Returns:
            None. Method blocks until move completes.

        Raises:
            ValueError: If resulting position would exceed motor limits.
            RuntimeError: If controller not connected.

        Example:
            >>> controller.move_relative(MotorType.ALTITUDE, 1000)   # Nudge down
            >>> controller.move_relative(MotorType.ALTITUDE, -1000)  # Nudge up
            >>> controller.move_relative(MotorType.AZIMUTH, -5000)   # Rotate CW
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
        """Request motor(s) to stop immediately (best-effort).

        Sends stop request to the controller. Note that the current firmware
        may not support interrupting moves in progress - this is a placeholder
        for future hardware capability.

        Business context: Emergency stop capability is important for safety.
        If the telescope is moving toward an obstruction or user needs to
        abort, this method signals intent to stop. Currently logs warning
        since hardware support is limited.

        Args:
            motor: Specific motor to stop (ALTITUDE or AZIMUTH),
                or None to request stopping all motors.

        Returns:
            None. Stop request logged but may not interrupt active moves.

        Raises:
            No exceptions raised. Best-effort operation.

        Example:
            >>> controller.stop(MotorType.ALTITUDE)  # Stop one motor
            >>> controller.stop()  # Stop all motors
        """
        logger.warning(
            "Stop command sent - custom controller may not support interrupt",
            motor=motor.value if motor else "all",
        )
        # Custom controller doesn't have a stop command documented
        # Movement blocks until complete

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get current motor status including position and movement state.

        Queries the internal state tracking for the specified motor.
        Returns cached position from last move command - does not query
        hardware directly.

        Business context: UI components and automation scripts need real-time
        motor status. This method provides the current believed position
        for display and decision-making during slew operations.

        Args:
            motor: Which motor to query. Use MotorType.ALTITUDE for
                the elevation axis or MotorType.AZIMUTH for rotation.

        Returns:
            MotorStatus: Dataclass containing:
                - motor (MotorType): The queried motor
                - is_moving (bool): True if motor currently in motion
                - position_steps (int): Current position in steps
                - speed (int): Always 0 (speed tracking not implemented)

        Raises:
            KeyError: If motor type not recognized (should not occur
                with proper MotorType enum usage).

        Example:
            >>> controller = driver.open("/dev/ttyACM0")
            >>> status = controller.get_status(MotorType.ALTITUDE)
            >>> print(f"Position: {status.position_steps} steps")
            Position: 70000 steps
            >>> if not status.is_moving:
            ...     controller.move(MotorType.ALTITUDE, 80000)
        """
        return MotorStatus(
            motor=motor,
            is_moving=self._is_moving[motor],
            position_steps=self._positions[motor],
            speed=0,  # Speed tracking not implemented
        )

    def home(self, motor: MotorType) -> None:
        """Move motor to its configured home (safe parking) position.

        Moves the specified motor to its predefined home position. For
        altitude, this is typically horizontal (0°). For azimuth, this
        is center position (0 steps). Blocks until move completes.

        Business context: Safe shutdown requires parking the telescope
        to prevent damage. Home positions are configured as safe resting
        states. Called before closing sessions or during emergency stops.

        Args:
            motor: Which motor to home. Use MotorType.ALTITUDE for
                elevation axis or MotorType.AZIMUTH for rotation.

        Returns:
            None. Method blocks until home position reached.

        Raises:
            RuntimeError: If controller not connected.
            ValueError: Should not occur - home positions are always valid.

        Example:
            >>> controller.home(MotorType.ALTITUDE)  # Park horizontal
            >>> controller.home(MotorType.AZIMUTH)   # Center rotation
            >>> # Or use home_all() for both motors
        """
        config = MOTOR_CONFIGS[motor]
        self.move(motor, config.home_position)
        logger.info("Motor homed", motor=motor.value)

    def home_all(self) -> None:
        """Home both altitude and azimuth motors to safe positions.

        Sequentially moves both motors to their configured home positions.
        Altitude homes first (to horizontal), then azimuth (to center).
        Blocks until both moves complete.

        Business context: Standard shutdown procedure for telescope safety.
        Parking at known positions prevents damage during transport or
        storage and provides consistent starting point for next session.

        Args:
            No arguments.

        Returns:
            None. Method blocks until both motors reach home.

        Raises:
            RuntimeError: If controller not connected.

        Example:
            >>> # End of observation session
            >>> controller.home_all()  # Park telescope safely
            >>> controller.close()
        """
        self.home(MotorType.ALTITUDE)
        self.home(MotorType.AZIMUTH)
        logger.info("All motors homed")

    def get_help(self) -> str:
        """Get help text from the motor controller firmware.

        Sends the '?' command to the Arduino motor controller and returns
        the full help response listing all available commands. This is
        useful for debugging and understanding controller capabilities.

        Business context: Motor controllers may have custom firmware with
        varying command sets. This method exposes the firmware's built-in
        help for diagnostics and development. Useful when troubleshooting
        communication issues or exploring controller features.

        Returns:
            str: Multi-line help text from controller firmware.
                Contains list of available commands and syntax.
                Returns partial response if timeout occurs.

        Raises:
            serial.SerialException: If serial communication fails.
            RuntimeError: If controller not connected.

        Example:
            >>> controller = driver.open("/dev/ttyACM0")
            >>> help_text = controller.get_help()
            >>> print(help_text)
            Available commands:
            m <motor> <steps> - move motor
            s <motor> - stop motor
            ? - display this help screen
        """
        with self._lock:
            self._serial.write(b"?")
            response = self._serial.read_until(b"?\tdisplay this help screen\r\n")
            return response.decode(errors="ignore")

    def get_info(self) -> MotorInfo:
        """Get motor controller information and current state.

        Returns comprehensive information about the controller including
        connection details and configuration. Used for status displays
        and diagnostic tools.

        Business context: Remote observatory control requires visibility
        into equipment state. This method provides the information needed
        for dashboards, logging, and MCP tool responses.

        Returns:
            MotorInfo TypedDict containing:
                - type (str): Always 'serial_motor_controller'
                - name (str): Human-readable controller name
                - port (str): Serial port path (e.g., '/dev/ttyACM0')
                - altitude_steps_per_degree (float): Steps per degree for altitude
                - azimuth_steps_per_degree (float): Steps per degree for azimuth

        Raises:
            No exceptions raised.

        Example:
            >>> info = controller.get_info()
            >>> print(f"Connected to {info['port']}")
            Connected to /dev/ttyACM0
        """
        return MotorInfo(
            type="serial_motor_controller",
            name=f"Serial Motor Controller ({self._port})",
            port=self._port,
            altitude_steps_per_degree=MOTOR_CONFIGS[
                MotorType.ALTITUDE
            ].steps_per_degree,
            azimuth_steps_per_degree=MOTOR_CONFIGS[MotorType.AZIMUTH].steps_per_degree,
        )

    def close(self) -> None:
        """Close the serial connection and release hardware resources.

        Marks controller as closed and releases the serial port. Safe to
        call multiple times. After closing, new controller must be opened
        via driver.open().

        Business context: Proper resource cleanup essential for shared
        serial hardware. Unreleased ports prevent reconnection and may
        require system restart. Always close before application exit.

        Args:
            No arguments.

        Returns:
            None. Connection state cleared regardless of outcome.

        Raises:
            No exceptions raised. Errors during close are suppressed.

        Example:
            >>> controller.home_all()  # Park first
            >>> controller.close()     # Release serial port
            >>> # Serial port now available for other applications
        """
        self._is_open = False

        if self._serial and self._serial.is_open:
            self._serial.close()

        logger.info("Motor controller closed", port=self._port)

    @property
    def is_open(self) -> bool:
        """Return True if controller connection is open.

        Returns:
            bool: True if connected, False if closed.

        Example:
            >>> if controller.is_open:
            ...     controller.move(MotorType.ALTITUDE, 1000)
        """
        return self._is_open

    def is_at_limit(self, motor: MotorType) -> str | None:
        """Check if motor is at a position limit.

        For serial controller, checks if current position equals min/max limits.

        Args:
            motor: Which motor to check (ALTITUDE or AZIMUTH).

        Returns:
            'min' if at minimum limit, 'max' if at maximum limit,
            None if within normal operating range.

        Example:
            >>> limit = controller.is_at_limit(MotorType.AZIMUTH)
            >>> if limit == 'min':
            ...     print("At CCW limit")
        """
        config = MOTOR_CONFIGS[motor]
        position = self._positions[motor]

        if position <= config.min_steps:
            return "min"
        elif position >= config.max_steps:
            return "max"
        return None

    def move_until_stall(
        self,
        motor: MotorType,
        direction: int,
        speed: int = 20,
        step_size: int = 100,
    ) -> int:
        """Move motor slowly until stall/slip is detected.

        For serial controller, this moves incrementally until reaching
        configured position limits. Real stall detection requires
        hardware feedback not yet implemented.

        Args:
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            direction: Direction to move. Negative = CCW/down, Positive = CW/up.
            speed: Speed percentage 1-100. Default 20 for slow homing.
            step_size: Steps per increment. Default 100.

        Returns:
            Final position in steps when limit reached.

        Raises:
            RuntimeError: If controller not connected.

        Example:
            >>> ccw_limit = controller.move_until_stall(
            ...     MotorType.AZIMUTH, direction=-1, speed=20
            ... )
        """
        if not self._is_open:
            raise RuntimeError("Controller not connected")

        config = MOTOR_CONFIGS[motor]
        step_delta = step_size if direction > 0 else -step_size

        while True:
            current = self._positions[motor]
            target = current + step_delta

            # Check if we would exceed limits
            if target <= config.min_steps:
                logger.info(
                    "Motor reached minimum limit",
                    motor=motor.value,
                    position=config.min_steps,
                )
                return config.min_steps
            elif target >= config.max_steps:
                logger.info(
                    "Motor reached maximum limit",
                    motor=motor.value,
                    position=config.max_steps,
                )
                return config.max_steps

            # Move incrementally
            self.move_relative(motor, step_delta, speed=speed)


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
        """Initialize driver with baud rate for motor controller.

        Sets up driver configuration without opening any serial ports.
        Call open() or get_available_controllers() to interact with hardware.

        Business context: The driver manages lifecycle of serial connections
        to motor controllers. Baud rate is stored for consistent port
        opening across discovery and connection.

        Implementation: Stores baudrate and initializes _controller to None.
        No hardware interaction occurs during init.

        Args:
            baudrate: Serial baud rate for controller communication.
                Defaults to 9600 (DEFAULT_BAUDRATE).

        Returns:
            None. Driver initialized, ready for open().

        Raises:
            No exceptions raised.

        Example:
            >>> driver = SerialMotorDriver(baudrate=9600)
            >>> controllers = driver.get_available_controllers()
            >>> if controllers:
            ...     motor = driver.open(controllers[0]['port'])
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
        """Create driver with injected port enumerator for testing.

        Factory method that injects a custom port enumerator, allowing
        tests to control which ports are 'discovered' without real hardware.

        Business context: Enables testing of device discovery and selection
        logic in CI/CD. Mock enumerators can simulate various hardware
        configurations (no devices, multiple devices, specific port names).

        Args:
            port_enumerator: Object with comports() method returning list
                of port objects with device and description attributes.
            baudrate: Serial baud rate. Defaults to DEFAULT_BAUDRATE.

        Returns:
            SerialMotorDriver: Configured driver that will use injected
                enumerator for get_available_controllers().

        Raises:
            No exceptions during creation.

        Example:
            >>> class MockPort:
            ...     device = "/dev/ttyMOCK0"
            ...     description = "Mock Motor Controller"
            >>> class MockEnumerator:
            ...     @staticmethod
            ...     def comports():
            ...         return [MockPort()]
            >>> driver = SerialMotorDriver._create_with_enumerator(MockEnumerator())
            >>> controllers = driver.get_available_controllers()
        """
        driver = cls.__new__(cls)
        driver._baudrate = baudrate
        driver._controller = None
        driver._port_enumerator = port_enumerator
        return driver

    def get_available_controllers(self) -> list[AvailableMotorController]:
        """Discover motor controllers available on serial ports.

        Scans system serial ports for devices matching motor controller
        signatures (USB serial adapters, CH340 chips, ACM devices).
        Does not open connections - only enumerates potential devices.

        Business context: Users may have multiple USB devices connected.
        This method enables device selection UI and auto-discovery
        workflows. Called by MCP tools to let clients enumerate and
        select motor controllers without prior configuration.

        Returns:
            list[AvailableMotorController]: List of controller descriptors,
                each containing:
                - id (int): Index for use with open()
                - type (str): Always 'serial_motor_controller'
                - name (str): Human-readable name with port
                - port (str): Serial port path (e.g., '/dev/ttyACM0')
                - description (str): OS-provided device description
            Empty list if no compatible devices found or pyserial
            not installed.

        Raises:
            No exceptions raised. Errors logged and empty list returned.

        Example:
            >>> driver = SerialMotorDriver()
            >>> controllers = driver.get_available_controllers()
            >>> for c in controllers:
            ...     print(f"{c['name']} on {c['port']}")
            Motor Controller (/dev/ttyACM0) on /dev/ttyACM0
            >>> if controllers:
            ...     controller = driver.open(controllers[0]['port'])
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

        controllers: list[AvailableMotorController] = []

        for i, port in enumerate(ports):
            # Check for likely motor controller devices
            desc = port.description.lower()
            if any(x in desc for x in ["acm", "usb serial", "ch340", "motor"]):
                controllers.append(
                    AvailableMotorController(
                        id=i,
                        type="serial_motor_controller",
                        name=f"Motor Controller ({port.device})",
                        port=port.device,
                        description=port.description,
                    )
                )

        logger.debug("Found controllers", count=len(controllers))
        return controllers

    def open(self, controller_id: int | str = 0) -> MotorInstance:
        """Open serial connection to motor controller.

        Creates a SerialMotorController instance connected to the specified
        serial port. Only one controller can be open at a time per driver.

        Business context: Primary entry point for connecting to physical
        motor controller hardware. The returned controller provides all
        motor movement operations. Must be called before any motor commands.

        Args:
            controller_id: Either integer index from get_available_controllers()
                or string port path (e.g., '/dev/ttyACM0' on Linux,
                'COM3' on Windows). Default 0 opens first available.

        Returns:
            MotorInstance: Connected controller ready for use.
                Provides move(), home(), get_status() and other methods.

        Raises:
            RuntimeError: If controller already open or connection fails.
            ValueError: If controller_id is invalid index.
            serial.SerialException: If port cannot be opened (permissions,
                device not found, port in use).

        Example:
            >>> driver = SerialMotorDriver()
            >>> controllers = driver.get_available_controllers()
            >>> if controllers:
            ...     controller = driver.open(controllers[0]['port'])
            ...     controller.move(MotorType.ALTITUDE, 70000)
            ...     driver.close()
        """
        if self._controller is not None and self._controller._is_open:
            raise RuntimeError("Controller already open")

        # Resolve controller_id to port string
        if isinstance(controller_id, int):
            controllers = self.get_available_controllers()
            if controller_id >= len(controllers):
                raise ValueError(
                    f"Controller index {controller_id} out of range "
                    f"(found {len(controllers)} controllers)"
                )
            port = controllers[controller_id]["port"]
        else:
            port = controller_id

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

        Raises:
            RuntimeError: If controller already open.

        Example:
            >>> driver = SerialMotorDriver()
            >>> mock = MockSerialPort(responses=[b"OK\\r\\n"])
            >>> controller = driver._open_with_serial(mock)
            >>> controller.move(MotorType.ALTITUDE, 100)
        """
        if self._controller is not None and self._controller._is_open:
            raise RuntimeError("Controller already open")

        self._controller = SerialMotorController._create_with_serial(
            serial_port, port_name
        )
        return self._controller

    def close(self) -> None:
        """Close the current motor controller connection.

        Closes the underlying SerialMotorController if one is open,
        releasing the serial port. Safe to call when no controller open.

        Business context: Driver-level close for clean resource management.
        Should be called when done with motor operations to release
        hardware for other applications.

        Args:
            No arguments.

        Returns:
            None. Controller reference cleared.

        Raises:
            No exceptions raised.

        Example:
            >>> driver = SerialMotorDriver()
            >>> controller = driver.open("/dev/ttyACM0")
            >>> # ... do motor operations ...
            >>> driver.close()  # Release serial port
        """
        if self._controller is not None:
            self._controller.close()
            self._controller = None


# Utility functions for position conversion


def steps_to_altitude_degrees(steps: int) -> float:
    """Convert motor step position to altitude in degrees.

    Transforms the altitude motor's step count to astronomical altitude
    angle. The altitude motor uses an inverted scale where 0 steps = zenith
    (90°) and maximum steps = horizon (0°).

    Business context: UI components and logging need human-readable angles.
    This conversion enables display of current telescope pointing position
    and is used for reporting in MCP tool responses.

    Args:
        steps: Motor position in steps.
            0 = zenith (90° altitude)
            140000 = horizon (0° altitude)
            Values outside range produce valid but out-of-bounds degrees.

    Returns:
        float: Altitude angle in degrees from horizon.
            90.0 = zenith (pointing straight up)
            0.0 = horizon (pointing at horizon)
            Negative values possible if steps exceed max.

    Raises:
        No exceptions raised. Out-of-range inputs produce out-of-range outputs.

    Example:
        >>> steps_to_altitude_degrees(0)      # Zenith
        90.0
        >>> steps_to_altitude_degrees(70000)  # Mid-sky
        45.0
        >>> steps_to_altitude_degrees(140000) # Horizon
        0.0
    """
    # Steps 0 = zenith (90°), steps 140000 = horizon (0°)
    return 90.0 - (steps / MOTOR_CONFIGS[MotorType.ALTITUDE].steps_per_degree)


def altitude_degrees_to_steps(degrees: float) -> int:
    """Convert altitude in degrees to motor step position.

    Transforms astronomical altitude angle to the motor controller's
    internal step count. The altitude motor uses an inverted scale where
    0 steps = zenith (90°) and maximum steps = horizon (0°).

    Business context: Telescope pointing commands use degrees for human
    readability, but the motor controller operates in discrete steps.
    This conversion is critical for accurate pointing - errors here
    directly affect observation accuracy.

    Args:
        degrees: Altitude angle in degrees from horizon.
            Valid range 0.0 (horizon) to 90.0 (zenith).
            Values outside range are accepted but may exceed
            motor limits.

    Returns:
        int: Motor position in steps, rounded to nearest integer.
            Higher step values = lower altitude.
            0 steps = 90° (zenith)
            140000 steps ≈ 0° (horizon)

    Raises:
        No exceptions raised. Out-of-range values not clamped.

    Example:
        >>> altitude_degrees_to_steps(90.0)  # Zenith
        0
        >>> altitude_degrees_to_steps(45.0)  # Mid-sky
        70000
        >>> altitude_degrees_to_steps(0.0)   # Horizon
        140000
    """
    steps = (90.0 - degrees) * MOTOR_CONFIGS[MotorType.ALTITUDE].steps_per_degree
    return int(round(steps))


def steps_to_azimuth_degrees(steps: int) -> float:
    """Convert motor step position to azimuth offset in degrees.

    Transforms the azimuth motor's step count to degrees offset from
    center position. Unlike altitude, azimuth is bidirectional allowing
    rotation in both CW and CCW directions.

    Business context: UI components need to display current azimuth
    position. This conversion enables human-readable angle display and
    is used in MCP tool responses for position reporting.

    Args:
        steps: Motor position in steps.
            0 = center position
            Positive = counter-clockwise from center
            Negative = clockwise from center
            Range limited only by motor physical stops.

    Returns:
        float: Azimuth offset in degrees from center position.
            Positive = CCW rotation
            Negative = CW rotation

    Raises:
        No exceptions raised. Any step value produces valid degrees.

    Example:
        >>> steps_to_azimuth_degrees(0)       # Center
        0.0
        >>> steps_to_azimuth_degrees(70000)   # 45° CCW
        45.0
        >>> steps_to_azimuth_degrees(-140000) # 90° CW
        -90.0
    """
    return steps / MOTOR_CONFIGS[MotorType.AZIMUTH].steps_per_degree


def azimuth_degrees_to_steps(degrees: float) -> int:
    """Convert azimuth offset in degrees to motor step position.

    Transforms azimuth angle offset from center to motor step count.
    Unlike altitude, azimuth is relative to a center position, allowing
    for rotation in both directions (CW and CCW).

    Business context: The azimuth motor tracks east-west movement.
    Center position (0 steps) typically aligns with magnetic north
    or a calibrated reference. This conversion enables accurate
    tracking of objects across the sky.

    Args:
        degrees: Azimuth offset in degrees from center position.
            Positive = counter-clockwise, negative = clockwise.
            No fixed range - limited only by motor physical stops.

    Returns:
        int: Motor position in steps, rounded to nearest integer.
            Positive steps = CCW from center
            Negative steps = CW from center

    Raises:
        No exceptions raised. Values not range-checked.

    Example:
        >>> azimuth_degrees_to_steps(0.0)    # Center
        0
        >>> azimuth_degrees_to_steps(45.0)   # 45° CCW
        70000
        >>> azimuth_degrees_to_steps(-90.0)  # 90° CW
        -140000
    """
    steps = degrees * MOTOR_CONFIGS[MotorType.AZIMUTH].steps_per_degree
    return int(round(steps))
