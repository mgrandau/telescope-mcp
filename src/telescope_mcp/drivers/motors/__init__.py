"""Motor control driver.

Controls:
- NEMA 23 stepper for altitude (via controller TBD)
- NEMA 17 stepper for azimuth (via controller TBD)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class MotorType(Enum):
    ALTITUDE = "altitude"
    AZIMUTH = "azimuth"


@dataclass
class MotorStatus:
    """Current motor status."""
    motor: MotorType
    is_moving: bool
    position_steps: int
    speed: int = 0


class MotorController(Protocol):
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


class StubMotorController:
    """Stub implementation for development without hardware.
    
    Simulates motor movements by tracking position internally.
    Prints actions to console for debugging.
    """

    def __init__(self) -> None:
        """Initialize stub motor controller for testing.
        
        Creates simulated motor controller with altitude and azimuth motors at position 0.
        No hardware communication. Position tracking purely mathematical for testing control logic.
        
        Business context: Enables development of telescope control systems without stepper motors,
        motor drivers, or power supplies. UI developers test motor controls, goto algorithms validate
        pointing calculations, CI/CD tests tracking logic. Critical for rapid iteration without
        hardware dependencies.
        
        Implementation details: Initializes _positions dict mapping MotorType.ALTITUDE/AZIMUTH to
        int position counters (starts 0). move() adds steps to counters instantly. No limits, no
        delays, no actual motion. Used by DriverFactory in DIGITAL_TWIN mode.
        
        Args:
            None.
        
        Returns:
            None. Controller ready for simulated move() calls.
        
        Raises:
            None. Stub never fails.
        
        Example:
            >>> controller = StubMotorController()
            >>> controller.move(MotorType.ALTITUDE, 1000, speed=50)  # Instant position change
            >>> print(controller.get_position(MotorType.ALTITUDE))  # 1000
        """
        self._positions = {
            MotorType.ALTITUDE: 0,
            MotorType.AZIMUTH: 0,
        }

    def move(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Simulate movement by updating internal position counter.
        
        Updates simulated position instantly without actual delay, allowing
        rapid testing of control logic. Prints action for visual debugging.
        Position tracking is purely mathematical.
        
        Business context: Enables development and testing of telescope control
        logic without physical hardware. Useful for UI development, algorithm
        testing, and CI/CD automated testing.
        
        Args:
            motor: Which motor to move.
            steps: Number of steps (adds to current position). No limits enforced.
            speed: Speed 0-100 (logged for debugging but not used in timing).
        
        Returns:
            None. Position update is immediate.
        
        Raises:
            None. Stub implementation never fails.
        
        Example:
            >>> stub = StubMotorController()
            >>> stub.move(MotorType.ALTITUDE, 500, speed=75)
            [STUB] Moving altitude by 500 steps at speed 75
            >>> status = stub.get_status(MotorType.ALTITUDE)
            >>> print(status.position_steps)  # 500
        """
        self._positions[motor] += steps
        print(f"[STUB] Moving {motor.value} by {steps} steps at speed {speed}")

    def stop(self, motor: MotorType | None = None) -> None:
        """Simulate stop by logging action without position change.
        
        Prints stop command for debugging but doesn't modify state. In real
        hardware, this would halt movement mid-step. Stub is always "stopped"
        since movement is instant.
        
        Business context: Tests emergency stop UI flows and safety logic
        without hardware. Verifies stop commands reach controller layer.
        
        Args:
            motor: Motor to stop, or None for emergency stop all. Both cases
                logged for debugging.
        
        Returns:
            None.
        
        Raises:
            None. Stub implementation never fails.
        
        Example:
            >>> stub = StubMotorController()
            >>> stub.stop(MotorType.AZIMUTH)
            [STUB] Stopping azimuth
            >>> stub.stop(None)
            [STUB] Emergency stop all motors
        """
        if motor:
            print(f"[STUB] Stopping {motor.value}")
        else:
            print("[STUB] Emergency stop all motors")

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get simulated motor status showing tracked position.
        
        Returns status with current position from move() calls. is_moving is
        always False since stub movement is instant. Speed is always 0.
        Position can be negative if moved in negative direction.
        
        Business context: Enables testing of status-dependent control logic
        and UI status displays without hardware. Position tracking validates
        cumulative movement calculations.
        
        Args:
            motor: Which motor to query.
        
        Returns:
            MotorStatus with:
            - motor: Echo of queried motor
            - is_moving: Always False (instant movement)
            - position_steps: Sum of all move() steps since init or home
            - speed: Always 0 (not moving)
        
        Raises:
            None. Stub implementation never fails.
        
        Example:
            >>> stub = StubMotorController()
            >>> stub.move(MotorType.ALTITUDE, 1000)
            >>> stub.move(MotorType.ALTITUDE, -200)
            >>> status = stub.get_status(MotorType.ALTITUDE)
            >>> print(status.position_steps)  # 800
        """
        return MotorStatus(
            motor=motor,
            is_moving=False,
            position_steps=self._positions[motor],
        )

    def home(self, motor: MotorType) -> None:
        """Simulate homing by resetting position counter to 0.
        
        Instantly resets position without movement delay, mimicking successful
        homing sequence. Prints action for debugging. Use to test absolute
        positioning logic.
        
        Business context: Validates homing workflows and position reset logic
        in development. Tests UI flows for startup calibration without
        waiting for real motor movement.
        
        Args:
            motor: Which motor to home. Position reset is independent per axis.
        
        Returns:
            None. Reset is immediate.
        
        Raises:
            None. Stub implementation never fails.
        
        Example:
            >>> stub = StubMotorController()
            >>> stub.move(MotorType.AZIMUTH, 5000)
            >>> stub.home(MotorType.AZIMUTH)
            [STUB] Homing azimuth
            >>> status = stub.get_status(MotorType.AZIMUTH)
            >>> print(status.position_steps)  # 0
        """
        self._positions[motor] = 0
        print(f"[STUB] Homing {motor.value}")


# TODO: Implement real motor controllers
# - GPIOMotorController (for Raspberry Pi GPIO)
# - SerialMotorController (for Arduino/serial-based controllers)
