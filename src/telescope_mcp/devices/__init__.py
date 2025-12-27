"""Logical device layer - hardware-agnostic device abstractions."""

from telescope_mcp.devices.camera import (
    Camera,
    CameraConfig,
    CameraDisconnectedError,
    CameraError,
    CameraHooks,
    CameraInfo,
    CaptureOptions,
    CaptureResult,
    Clock,
    NullRecoveryStrategy,
    NullRenderer,
    OverlayConfig,
    OverlayRenderer,
    RecoveryStrategy,
    StreamFrame,
    SystemClock,
)
from telescope_mcp.devices.controller import (
    CameraController,
    SyncCaptureConfig,
    SyncCaptureResult,
)
from telescope_mcp.devices.registry import (
    CameraNotInRegistryError,
    CameraRegistry,
    get_registry,
    init_registry,
    shutdown_registry,
)
from telescope_mcp.devices.sensor import (
    Sensor,
    SensorConfig,
    SensorInfo,
)

__all__ = [
    # Camera
    "Camera",
    "CameraConfig",
    "CameraDisconnectedError",
    "CameraError",
    "CameraHooks",
    "CameraInfo",
    "CaptureOptions",
    "CaptureResult",
    "OverlayConfig",
    "OverlayRenderer",
    "NullRenderer",
    "StreamFrame",
    # Recovery
    "RecoveryStrategy",
    "NullRecoveryStrategy",
    # Clock (shared)
    "Clock",
    "SystemClock",
    # Controller
    "CameraController",
    "SyncCaptureConfig",
    "SyncCaptureResult",
    # Registry
    "CameraRegistry",
    "CameraNotInRegistryError",
    "init_registry",
    "get_registry",
    "shutdown_registry",
    # Sensor
    "Sensor",
    "SensorConfig",
    "SensorInfo",
]
