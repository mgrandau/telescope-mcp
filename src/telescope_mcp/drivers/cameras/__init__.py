"""Camera driver module.

Provides camera control for ASI cameras (real hardware) and digital twin
simulation for development without hardware.
"""

from typing import Protocol

from telescope_mcp.drivers.cameras.stub import StubCameraDriver, StubCameraInstance


class CameraDriver(Protocol):
    """Protocol for camera drivers (real or simulated)."""

    def get_connected_cameras(self) -> dict:
        """List connected cameras."""
        ...

    def open(self, camera_id: int) -> "CameraInstance":
        """Open a camera."""
        ...


class CameraInstance(Protocol):
    """Protocol for an opened camera."""

    def get_info(self) -> dict:
        """Get camera info."""
        ...

    def get_controls(self) -> dict:
        """Get available controls."""
        ...

    def set_control(self, control: str, value: int) -> dict:
        """Set a control value."""
        ...

    def get_control(self, control: str) -> dict:
        """Get a control value."""
        ...

    def capture(self, exposure_us: int) -> bytes:
        """Capture a frame, return JPEG bytes."""
        ...

    def close(self) -> None:
        """Close the camera."""
        ...


__all__ = [
    "CameraDriver",
    "StubCameraDriver",
]
