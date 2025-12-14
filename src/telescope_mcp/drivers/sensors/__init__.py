"""Position sensor driver.

Reads altitude and azimuth position from sensors (TBD - accelerometer, encoder, etc.)
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TelescopePosition:
    """Current telescope position in degrees."""
    altitude: float  # 0-90 degrees
    azimuth: float   # 0-360 degrees


class PositionSensor(Protocol):
    """Protocol for position sensor implementations."""

    def read(self) -> TelescopePosition:
        """Read current position."""
        ...

    def calibrate(self, altitude: float, azimuth: float) -> None:
        """Calibrate sensor to known position."""
        ...


class StubPositionSensor:
    """Stub implementation for development without hardware."""

    def __init__(self) -> None:
        self._position = TelescopePosition(altitude=45.0, azimuth=180.0)

    def read(self) -> TelescopePosition:
        """Return simulated position."""
        return self._position

    def calibrate(self, altitude: float, azimuth: float) -> None:
        """Set simulated position."""
        self._position = TelescopePosition(altitude=altitude, azimuth=azimuth)
        print(f"[STUB] Calibrated to alt={altitude}, az={azimuth}")


# TODO: Implement real sensor drivers
# - AccelerometerSensor (for IMU-based sensing)
# - EncoderSensor (for rotary encoder-based sensing)
