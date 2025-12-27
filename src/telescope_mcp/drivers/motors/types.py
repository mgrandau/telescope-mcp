"""Motor type definitions and protocols.

This module contains base types, enums, data classes, and protocols for motor
drivers. By keeping these in a separate module, we avoid circular imports
when implementation modules need to reference these types.

Types defined here:
- MotorType: Enum for motor axis selection
- MotorStatus: Data class for motor status
- MotorController: Protocol for motor controller implementations

Example:
    from telescope_mcp.drivers.motors.types import (
        MotorType,
        MotorStatus,
        MotorController,
    )

    class MyMotorController:
        def move(self, motor: MotorType, steps: int, speed: int = 100) -> None:
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class MotorType(Enum):
    """Motor axis selection."""

    ALTITUDE = "altitude"
    AZIMUTH = "azimuth"


@dataclass
class MotorStatus:
    """Current motor status."""

    motor: MotorType
    is_moving: bool
    position_steps: int
    speed: int = 0


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
