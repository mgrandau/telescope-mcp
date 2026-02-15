"""Motor type definitions and protocols.

This module contains base types, enums, data classes, and protocols for motor
drivers. By keeping these in a separate module, we avoid circular imports
when implementation modules need to reference these types.

Types defined here:
- MotorType: Enum for motor axis selection
- MotorStatus: Data class for motor status
- MotorInfo: TypedDict for motor hardware information
- AvailableMotorController: TypedDict for discovered motor controller
- MotorController: Protocol for motor controller implementations
- MotorInstance: Protocol for connected motor instances (mirrors SensorInstance)
- MotorDriver: Protocol for motor drivers (mirrors SensorDriver)

Example:
    from telescope_mcp.drivers.motors.types import (
        MotorType,
        MotorStatus,
        MotorController,
        MotorDriver,
        MotorInstance,
    )

    class MyMotorDriver:
        def open(self, controller_id: int | str = 0) -> MotorInstance:
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypedDict, runtime_checkable

__all__ = [
    "MotorType",
    "MotorStatus",
    "MotorInfo",
    "AvailableMotorController",
    "MotorController",
    "MotorInstance",
    "MotorDriver",
]


class MotorType(Enum):
    """Motor axis selection."""

    ALTITUDE = "altitude"
    AZIMUTH = "azimuth"


@dataclass
class MotorStatus:
    """Current motor status.

    Attributes:
        motor: Which motor axis this status describes.
        is_moving: True if motor is currently executing a move command.
        position_steps: Current position in steps from home (can be negative).
        speed: Current speed percentage (0 if stopped, 1-100 if moving).
        target_steps: Target position for current move (None if not moving).
        error: Error message if motor in error state, None otherwise.
        stalled: True if motor stalled (hit limit or slipped). Cleared on next move.
        at_limit: Which limit reached if any ('min', 'max', or None).
    """

    motor: MotorType
    is_moving: bool
    position_steps: int
    speed: int = 0
    target_steps: int | None = None
    error: str | None = None
    stalled: bool = False
    at_limit: str | None = None


class MotorInfo(TypedDict, total=False):
    """Type for motor controller hardware information.

    Keys:
        type: Controller type (e.g., "serial_motor_controller", "digital_twin").
        name: Human-readable controller name.
        port: Connection port/path (e.g., "/dev/ttyACM0").
        firmware: Firmware version (if available).
        altitude_steps_per_degree: Steps per degree for altitude axis.
        azimuth_steps_per_degree: Steps per degree for azimuth axis.
    """

    type: str
    name: str
    port: str
    firmware: str
    altitude_steps_per_degree: float
    azimuth_steps_per_degree: float


class AvailableMotorController(TypedDict, total=False):
    """Type for discovered motor controller descriptor.

    Keys:
        id: Integer index for selection.
        type: Controller type (e.g., "serial_motor_controller").
        name: Human-readable name.
        port: Connection path (e.g., "/dev/ttyACM0").
        description: Hardware description from OS.
    """

    id: int
    type: str
    name: str
    port: str
    description: str


@runtime_checkable
class MotorController(Protocol):  # pragma: no cover
    """Protocol for motor controller implementations.

    Defines the interface for controlling telescope mount motors.
    Implementations can use GPIO, serial, or network communication.
    """

    def move(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Move motor by specified steps for telescope pointing adjustment.

        Initiates motor movement for precise telescope positioning. Positive steps
        move altitude up or rotate azimuth clockwise (looking down). Negative steps
        reverse direction. Movement is non-blocking and asynchronous.

        Business context: Core positioning interface for telescope mount control,
        enabling automated tracking and goto functionality. Step count determines
        angular movement based on motor/gearbox configuration.

        Args:
            motor: Which motor to move (altitude or azimuth). Determines axis.
            steps: Number of steps to move. Positive=up/CW, negative=down/CCW.
                Range depends on motor/driver specs (typically ±100000).
            speed: Speed percentage from 1-100. Default 100 (max speed).
                Lower values provide smoother movement but take longer.

        Returns:
            None. Movement occurs asynchronously. Use get_status() to monitor.

        Raises:
            ValueError: If speed outside 1-100 range.
            RuntimeError: If motor hardware not connected or failed to respond.
            MotorError: If motor is already at limit switch position.

        Example:
            >>> controller.move(MotorType.ALTITUDE, 1000, speed=50)
            >>> status = controller.get_status(MotorType.ALTITUDE)
            >>> while status.is_moving:
            ...     time.sleep(0.1)
            ...     status = controller.get_status(MotorType.ALTITUDE)
        """
        ...

    def stop(self, motor: MotorType | None = None) -> None:
        """Stop motor(s) immediately for safety or user interrupt.

        Halts motor movement with rapid deceleration. Used for emergency stops,
        user-initiated cancellation, or before switching control modes. Stopping
        is immediate and may cause position loss in non-encoder systems.

        Business context: Safety-critical function for preventing collisions,
        protecting equipment, and responding to user control. Essential for
        manual intervention during automated tracking.

        Args:
            motor: Motor to stop, or None to emergency stop all motors.
                Passing None is recommended for safety situations to ensure
                all movement ceases immediately.

        Returns:
            None. Motor should halt within milliseconds.

        Raises:
            RuntimeError: If motor hardware fails to respond to stop command.
            CommunicationError: If unable to reach motor controller.

        Example:
            >>> # Emergency stop all motors
            >>> controller.stop(None)
            >>> # Stop just altitude motor
            >>> controller.stop(MotorType.ALTITUDE)
        """
        ...

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get current motor status for monitoring and control decisions.

        Queries real-time motor state including position, movement status, and
        speed. Position is in steps from last home or power-on. Use this to
        monitor movement progress or verify idle state before new commands.

        Business context: Essential for implementing tracking loops, verifying
        movement completion, and building higher-level control logic. Position
        tracking enables absolute pointing when combined with homing.

        Args:
            motor: Which motor to query (altitude or azimuth).

        Returns:
            MotorStatus dataclass containing:
            - motor: Echo of queried motor type
            - is_moving: True if motor currently executing move command
            - position_steps: Current position in steps from home (can be negative)
            - speed: Current speed percentage (0 if stopped)

        Raises:
            RuntimeError: If motor hardware not responding.
            CommunicationError: If unable to query motor controller.

        Example:
            >>> status = controller.get_status(MotorType.AZIMUTH)
            >>> print(f"Position: {status.position_steps} steps")
            >>> if status.is_moving:
            ...     print(f"Moving at {status.speed}% speed")
        """
        ...

    def home(self, motor: MotorType) -> None:
        """Move motor to home position for absolute positioning reference.

        Executes homing sequence to establish position zero reference. May use
        limit switches, encoders, or move to mechanical stop depending on
        hardware. Blocks until homing complete. Position counter resets to 0.

        Business context: Required for absolute pointing systems. Must be run
        after power-on or if position tracking is lost. Enables repeatable
        pointing and goto functionality by establishing known reference.

        Args:
            motor: Which motor to home (altitude or azimuth). Each axis must be
                homed independently for full telescope calibration.

        Returns:
            None. Blocks until homing complete and position reset.

        Raises:
            RuntimeError: If homing sequence fails or timeout exceeded.
            HardwareError: If limit switch not found or encoder malfunction.
            ValueError: If motor already in motion (must stop first).

        Example:
            >>> # Home both axes on startup
            >>> controller.home(MotorType.ALTITUDE)
            >>> controller.home(MotorType.AZIMUTH)
            >>> # Position now calibrated at (0, 0)
        """
        ...

    def zero_position(self, motor: MotorType) -> None:
        """Zero the position counter at current physical location.

        Sets the motor's internal position counter to 0 without any physical
        movement. Used to establish the current telescope position as the
        reference origin (0,0) at the start of an observing session.

        Business context: Essential for session setup. Operator physically
        positions telescope to a known reference (e.g., level, pointed north),
        then calls zero_position to establish that as (0,0). All subsequent
        position readouts are relative to this home. Equivalent to pressing
        "Set Home" on the dashboard.

        Args:
            motor: Which motor to zero (ALTITUDE or AZIMUTH).

        Returns:
            None. Position counter set to 0 immediately.

        Raises:
            RuntimeError: If motor controller not connected.

        Example:
            >>> controller.zero_position(MotorType.ALTITUDE)
            >>> controller.zero_position(MotorType.AZIMUTH)
            >>> # Both axes now read 0 steps
        """
        ...


@runtime_checkable
class MotorInstance(Protocol):  # pragma: no cover
    """Protocol for connected motor controller instances.

    A MotorInstance represents an open connection to a motor controller device.
    It provides methods to move motors, query status, and control the hardware.
    Mirrors SensorInstance pattern for consistency.

    Business context: Defines the interface for all telescope motor controllers.
    Implementations include SerialMotorController (real Teensy + AMIS hardware)
    and DigitalTwinMotorInstance (simulation). Enables polymorphic use
    across different controller types.
    """

    def get_info(self) -> MotorInfo:
        """Get motor controller identification and capability information.

        Returns static information about the motor controller hardware and its
        configuration. Used for logging, diagnostics, and UI display.

        Business context: Enables identification of controller type for
        appropriate handling. UI can display controller model, firmware
        version. Logs include controller info for debugging.

        Returns:
            MotorInfo TypedDict containing:
            - type: Controller type string (e.g., "serial_motor_controller")
            - name: Human-readable name
            - port: Connection port/path (if applicable)
            - firmware: Firmware version (if available)
            - altitude_steps_per_degree: Conversion factor for altitude
            - azimuth_steps_per_degree: Conversion factor for azimuth

        Raises:
            RuntimeError: If controller connection is closed.

        Example:
            >>> info = controller.get_info()
            >>> print(f"Controller: {info['type']} on {info['port']}")
        """
        ...

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
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            steps: Absolute target position in steps.
            speed: Speed percentage 1-100. Default 100.

        Returns:
            None. Method blocks until move completes.

        Raises:
            ValueError: If steps outside motor's valid range or speed invalid.
            RuntimeError: If controller not connected.

        Example:
            >>> controller.move(MotorType.ALTITUDE, 70000)  # 45° up
        """
        ...

    def move_relative(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Move motor by relative step offset from current position.

        Commands the specified motor to move by a relative number of steps
        and blocks until movement completes.

        Business context: Enables incremental adjustments during alignment
        and fine-tuning. Preferred for small corrections.

        Args:
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            steps: Number of steps to move from current position.
            speed: Speed percentage 1-100. Default 100.

        Returns:
            None. Method blocks until move completes.

        Raises:
            ValueError: If resulting position would exceed motor limits.
            RuntimeError: If controller not connected.

        Example:
            >>> controller.move_relative(MotorType.ALTITUDE, 1000)  # Nudge
        """
        ...

    def stop(self, motor: MotorType | None = None) -> None:
        """Stop motor(s) immediately.

        Sends stop request to halt motor movement. Essential for safety.

        Args:
            motor: Motor to stop, or None for emergency stop all.

        Returns:
            None.

        Raises:
            No exceptions raised. Best-effort operation.

        Example:
            >>> controller.stop(MotorType.ALTITUDE)
            >>> controller.stop()  # Stop all
        """
        ...

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get current motor status including position and movement state.

        Args:
            motor: Which motor to query (ALTITUDE or AZIMUTH).

        Returns:
            MotorStatus dataclass with position, is_moving, speed.

        Raises:
            KeyError: If motor type not recognized.

        Example:
            >>> status = controller.get_status(MotorType.ALTITUDE)
            >>> print(f"Position: {status.position_steps}")
        """
        ...

    def home(self, motor: MotorType) -> None:
        """Move motor to its configured home position.

        Args:
            motor: Which motor to home.

        Returns:
            None. Blocks until home position reached.

        Raises:
            RuntimeError: If controller not connected.

        Example:
            >>> controller.home(MotorType.ALTITUDE)
        """
        ...

    def home_all(self) -> None:
        """Home both altitude and azimuth motors to safe positions.

        Returns:
            None. Blocks until both motors reach home.

        Example:
            >>> controller.home_all()
        """
        ...

    def zero_position(self, motor: MotorType) -> None:
        """Zero the position counter at current physical location.

        Sets the motor's internal position counter to 0 without any physical
        movement. Used to establish the current telescope position as the
        reference origin (0,0) at the start of an observing session.

        Business context: Essential for session setup. Operator physically
        positions telescope to a known reference (e.g., level, pointed north),
        then calls zero_position to establish that as (0,0). All subsequent
        position readouts are relative to this home. Maps to stepper_amis
        setPosition(axis, 0) on Teensy hardware.

        Args:
            motor: Which motor to zero (ALTITUDE or AZIMUTH).

        Returns:
            None. Position counter set to 0 immediately.

        Raises:
            RuntimeError: If controller not connected.

        Example:
            >>> controller.zero_position(MotorType.ALTITUDE)
            >>> status = controller.get_status(MotorType.ALTITUDE)
            >>> assert status.position_steps == 0
        """
        ...

    def close(self) -> None:
        """Close the controller connection and release resources.

        Returns:
            None. Safe to call multiple times.

        Example:
            >>> controller.close()
        """
        ...

    @property
    def is_open(self) -> bool:
        """Return True if controller connection is open.

        Returns:
            bool: True if connected, False if closed.

        Example:
            >>> if controller.is_open:
            ...     controller.move(MotorType.ALTITUDE, 1000)
        """
        ...

    def is_at_limit(self, motor: MotorType) -> str | None:
        """Check if motor is at a position limit.

        Useful for detecting when motor has reached its mechanical range
        or when approaching limits during homing sequences.

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
        ...

    def move_until_stall(
        self,
        motor: MotorType,
        direction: int,
        speed: int = 20,
        step_size: int = 100,
    ) -> int:
        """Move motor slowly until stall/slip is detected.

        Moves motor incrementally in the specified direction at low speed
        until the motor stalls (reaches limit and cannot continue). Used
        for homing sequences to find mechanical end stops without damage.

        Business context: Homing procedure requires finding mechanical limits
        without damaging belt or mount. Moving slowly allows detecting slip
        when belt reaches end of travel. Backing off 2-3 steps from detected
        limit establishes safe operating boundary.

        Args:
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            direction: Direction to move. Negative = CCW/down, Positive = CW/up.
            speed: Speed percentage 1-100. Default 20 for slow homing.
            step_size: Steps per increment. Default 100.

        Returns:
            Final position in steps when stall detected.

        Raises:
            RuntimeError: If controller not connected.

        Example:
            >>> # Find CCW limit
            >>> ccw_limit = controller.move_until_stall(
            ...     MotorType.AZIMUTH, direction=-1, speed=20
            ... )
            >>> print(f"CCW limit at {ccw_limit} steps")
        """
        ...


@runtime_checkable
class MotorDriver(Protocol):  # pragma: no cover
    """Protocol for motor drivers.

    A MotorDriver handles discovery and connection to motor controller devices.
    It can enumerate available controllers and open connections.
    Mirrors SensorDriver pattern for consistency.

    Business context: Factory pattern for motor controller creation. Enables
    automatic discovery of available controllers and uniform connection
    interface. Implementations include SerialMotorDriver and
    DigitalTwinMotorDriver.
    """

    def get_available_controllers(self) -> list[AvailableMotorController]:
        """Enumerate available motor controller devices.

        Scans for compatible controllers (serial ports, simulated devices)
        and returns information about each. Used for device discovery
        and selection UI.

        Business context: Enables automatic controller discovery without
        manual configuration. Users select from discovered devices
        rather than typing port names.

        Returns:
            List of controller info dicts, each containing:
            - id: Integer index for selection
            - type: Controller type (e.g., "serial_motor_controller")
            - name: Human-readable name
            - port: Connection path (e.g., "/dev/ttyACM0")
            - description: Hardware description

        Raises:
            None. Returns empty list if no controllers found.

        Example:
            >>> controllers = driver.get_available_controllers()
            >>> for c in controllers:
            ...     print(f"{c['id']}: {c['name']} ({c['port']})")
        """
        ...

    def open(self, controller_id: int | str = 0) -> MotorInstance:
        """Open connection to a motor controller device.

        Creates and returns a connected MotorInstance for the specified
        controller. Only one controller can be open at a time per driver.

        Business context: Primary factory method for creating controller
        connections. Accepts either index from get_available_controllers()
        or direct port path for advanced users.

        Args:
            controller_id: Either integer index from get_available_controllers()
                or string port path (e.g., "/dev/ttyACM0"). Default 0
                opens first available controller.

        Returns:
            MotorInstance connected and ready for movement commands.

        Raises:
            RuntimeError: If controller already open, not found,
                or connection fails.
            ValueError: If controller_id is invalid index.

        Example:
            >>> driver = SerialMotorDriver()
            >>> controller = driver.open(0)  # First controller
            >>> # Or by port:
            >>> controller = driver.open("/dev/ttyACM0")
        """
        ...

    def close(self) -> None:
        """Close the currently open motor controller.

        Closes any controller opened by this driver. Safe to call even
        if no controller is open.

        Returns:
            None.

        Raises:
            None. Safe to call multiple times.

        Example:
            >>> driver.open(0)
            >>> # ... use controller ...
            >>> driver.close()
        """
        ...
