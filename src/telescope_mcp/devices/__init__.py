"""Logical device layer - hardware-agnostic device abstractions."""

from telescope_mcp.devices.camera import (
    Camera,
    CameraConfig,
    CameraInfo,
    CaptureResult,
)

__all__ = [
    "Camera",
    "CameraConfig",
    "CameraInfo",
    "CaptureResult",
]
