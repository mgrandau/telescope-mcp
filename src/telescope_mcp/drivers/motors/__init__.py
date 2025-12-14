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
    """Protocol for motor controller implementations."""

    def move(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Move motor by specified steps."""
        ...

    def stop(self, motor: MotorType | None = None) -> None:
        """Stop motor(s). None means stop all."""
        ...

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get motor status."""
        ...

    def home(self, motor: MotorType) -> None:
        """Move motor to home position."""
        ...


class StubMotorController:
    """Stub implementation for development without hardware."""

    def __init__(self) -> None:
        self._positions = {
            MotorType.ALTITUDE: 0,
            MotorType.AZIMUTH: 0,
        }

    def move(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Simulate movement."""
        self._positions[motor] += steps
        print(f"[STUB] Moving {motor.value} by {steps} steps at speed {speed}")

    def stop(self, motor: MotorType | None = None) -> None:
        """Simulate stop."""
        if motor:
            print(f"[STUB] Stopping {motor.value}")
        else:
            print("[STUB] Emergency stop all motors")

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get simulated status."""
        return MotorStatus(
            motor=motor,
            is_moving=False,
            position_steps=self._positions[motor],
        )

    def home(self, motor: MotorType) -> None:
        """Simulate homing."""
        self._positions[motor] = 0
        print(f"[STUB] Homing {motor.value}")


# TODO: Implement real motor controllers
# - GPIOMotorController (for Raspberry Pi GPIO)
# - SerialMotorController (for Arduino/serial-based controllers)
