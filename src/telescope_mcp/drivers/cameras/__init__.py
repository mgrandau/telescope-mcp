"""Camera driver module.

Provides camera control for ASI cameras (real hardware) and digital twin
simulation for development without hardware.
"""

from typing import Protocol

from telescope_mcp.drivers.cameras.twin import (
    DigitalTwinCameraDriver,
    DigitalTwinConfig,
    ImageSource,
    create_directory_camera,
    create_file_camera,
)


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


class CameraDriver(Protocol):
    """Protocol for camera drivers (real or simulated).
    
    Drivers can return either:
    - dict[int, dict]: Raw info dicts (legacy, converted by registry)
    - dict[int, CameraInfo]: Structured info (preferred)
    """

    def get_connected_cameras(self) -> dict:
        """List connected cameras.
        
        Returns:
            Dict mapping camera_id to camera info (dict or CameraInfo)
        """
        ...

    def open(self, camera_id: int) -> CameraInstance:
        """Open a camera.
        
        Args:
            camera_id: ID of camera to open
            
        Returns:
            CameraInstance for the opened camera
        """
        ...


__all__ = [
    "CameraDriver",
    "CameraInstance",
    "DigitalTwinCameraDriver",
    "DigitalTwinConfig",
    "ImageSource",
    "create_directory_camera",
    "create_file_camera",
]
