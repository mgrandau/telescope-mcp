"""Motor control driver.

Controls:
- NEMA 23 stepper for altitude (via serial controller)
- NEMA 17 stepper for azimuth (via serial controller)

Example:
    from telescope_mcp.drivers.motors import (
        MotorType,
        SerialMotorDriver,
        DigitalTwinMotorDriver,
        StubMotorController,
    )

    # For testing (no hardware) - Full digital twin
    driver = DigitalTwinMotorDriver()
    controller = driver.open()
    controller.move(MotorType.ALTITUDE, 70000)  # 45 degrees
    driver.close()

    # For testing (simple stub)
    controller = StubMotorController()
    controller.move(MotorType.ALTITUDE, 1000)

    # For real hardware
    driver = SerialMotorDriver()
    controllers = driver.get_available_controllers()
    if controllers:
        controller = driver.open(controllers[0]["port"])
        controller.move(MotorType.ALTITUDE, 70000)  # 45 degrees
"""

# Base types (no circular import issues - import first)
# Serial motor controller
from telescope_mcp.drivers.motors.serial_controller import (
    MOTOR_CONFIGS,
    MotorConfig,
    SerialMotorController,
    SerialMotorDriver,
    altitude_degrees_to_steps,
    azimuth_degrees_to_steps,
    steps_to_altitude_degrees,
    steps_to_azimuth_degrees,
)

# Digital twin motor driver
from telescope_mcp.drivers.motors.twin import (
    DigitalTwinMotorConfig,
    DigitalTwinMotorDriver,
    DigitalTwinMotorInstance,
)
from telescope_mcp.drivers.motors.types import (
    AvailableMotorController,
    MotorController,
    MotorDriver,
    MotorInfo,
    MotorInstance,
    MotorStatus,
    MotorType,
)


class StubMotorController:
    """Stub implementation for development without hardware.

    Simulates motor movements by tracking position internally.
    Prints actions to console for debugging.
    """

    def __init__(self) -> None:
        """Initialize stub motor controller for testing.

        Creates simulated motor controller with altitude and azimuth motors
        at position 0. No hardware communication. Position tracking purely
        mathematical for testing control logic.

        Business context: Enables development of telescope control systems
        without stepper motors, motor drivers, or power supplies. UI developers
        test motor controls, goto algorithms validate pointing calculations,
        CI/CD tests tracking logic. Critical for rapid iteration without
        hardware dependencies.

        Implementation details: Initializes _positions dict mapping
        MotorType.ALTITUDE/AZIMUTH to int position counters (starts 0).
        move() adds steps to counters instantly. No limits, no delays,
        no actual motion. Used by DriverFactory in DIGITAL_TWIN mode.

        Args:
            None.

        Returns:
            None. Controller ready for simulated move() calls.

        Raises:
            None. Stub never fails.

        Example:
            >>> controller = StubMotorController()
            >>> controller.move(MotorType.ALTITUDE, 1000, speed=50)
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

    def zero_position(self, motor: MotorType) -> None:
        """Zero the position counter at current physical location.

        Sets the specified motor's position counter to 0 without movement.
        Used to establish the current telescope position as the reference
        origin for an observing session.

        Business context: Called when user presses 'Set Home' on the
        dashboard. Records current physical position as (0,0) reference.

        Args:
            motor: Which motor to zero (ALTITUDE or AZIMUTH).

        Returns:
            None. Position counter set to 0 immediately.

        Example:
            >>> stub = StubMotorController()
            >>> stub.move(MotorType.ALTITUDE, 5000)
            >>> stub.zero_position(MotorType.ALTITUDE)
            [STUB] Zeroing altitude position
            >>> status = stub.get_status(MotorType.ALTITUDE)
            >>> print(status.position_steps)  # 0
        """
        self._positions[motor] = 0
        print(f"[STUB] Zeroing {motor.value} position")


__all__ = [
    # Enums and data classes
    "MotorType",
    "MotorStatus",
    "MotorInfo",
    "MotorConfig",
    "AvailableMotorController",
    "MOTOR_CONFIGS",
    # Protocols
    "MotorController",
    "MotorDriver",
    "MotorInstance",
    # Stub (for simple testing)
    "StubMotorController",
    # Digital twin (for realistic simulation)
    "DigitalTwinMotorConfig",
    "DigitalTwinMotorDriver",
    "DigitalTwinMotorInstance",
    # Serial driver (real hardware)
    "SerialMotorController",
    "SerialMotorDriver",
    # Utility functions
    "steps_to_altitude_degrees",
    "altitude_degrees_to_steps",
    "steps_to_azimuth_degrees",
    "azimuth_degrees_to_steps",
]
