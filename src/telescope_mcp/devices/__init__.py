"""Logical device layer - hardware-agnostic device abstractions.

This module provides the public API for telescope device management:
- Camera: Image capture with overlays, streaming, and recovery
- CameraController: Multi-camera synchronization for alignment
- CameraRegistry: Discovery and singleton management
- Sensor: Environmental (temperature, humidity) and positional (alt/az) data
- CoordinateProvider: Automatic coordinate injection into frame metadata

Example:
    >>> from telescope_mcp.devices import Camera, CameraRegistry, Sensor
    >>> registry = CameraRegistry(driver)
    >>> camera = registry.get(0, auto_connect=True)
    >>> result = camera.capture()

All public symbols are explicitly listed in __all__. Submodules (camera,
controller, registry, sensor) are implementation details and not part of
the public API.
"""

# Use explicit imports to avoid polluting namespace with submodule names.
# The 'from .camera import X' style would add 'camera' to dir(module).
from telescope_mcp.devices.camera import (
    Camera,
    CameraConfig,
    CameraDisconnectedError,
    CameraError,
    CameraHooks,
    CameraInfo,
    CaptureCoordinates,
    CaptureOptions,
    CaptureResult,
    Clock,
    CoordinateProvider,
    NullCoordinateProvider,
    NullRecoveryStrategy,
    NullRenderer,
    OverlayConfig,
    OverlayRenderer,
    RecoveryStrategy,
    StreamFrame,
    SystemClock,
)
from telescope_mcp.devices.camera_controller import (
    CameraController,
    SyncCaptureConfig,
    SyncCaptureResult,
)
from telescope_mcp.devices.camera_registry import (
    CameraNotInRegistryError,
    CameraRegistry,
    get_registry,
    init_registry,
    shutdown_registry,
)
from telescope_mcp.devices.coordinate_provider import (
    LocationConfig,
    SensorCoordinateProvider,
)
from telescope_mcp.devices.sensor import (
    DEFAULT_SAMPLE_RATE_HZ,
    STATUS_SETTLE_DELAY_SEC,
    DeviceSensorInfo,
    Sensor,
    SensorConfig,
    SensorDeviceStatus,
)

__all__ = [
    # Camera (19 exports)
    "Camera",
    "CameraConfig",
    "CameraDisconnectedError",
    "CameraError",
    "CameraHooks",
    "CameraInfo",
    "CaptureCoordinates",
    "CaptureOptions",
    "CaptureResult",
    "Clock",
    "CoordinateProvider",
    "NullCoordinateProvider",
    "NullRecoveryStrategy",
    "NullRenderer",
    "OverlayConfig",
    "OverlayRenderer",
    "RecoveryStrategy",
    "StreamFrame",
    "SystemClock",
    # Controller (3 exports)
    "CameraController",
    "SyncCaptureConfig",
    "SyncCaptureResult",
    # Registry (5 exports)
    "CameraNotInRegistryError",
    "CameraRegistry",
    "get_registry",
    "init_registry",
    "shutdown_registry",
    # Coordinate Provider (2 exports)
    "LocationConfig",
    "SensorCoordinateProvider",
    # Sensor (6 exports)
    "DEFAULT_SAMPLE_RATE_HZ",
    "STATUS_SETTLE_DELAY_SEC",
    "DeviceSensorInfo",
    "Sensor",
    "SensorConfig",
    "SensorDeviceStatus",
]  # Total: 35 exports
