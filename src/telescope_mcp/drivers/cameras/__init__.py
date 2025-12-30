"""Camera driver module.

Provides camera control for ASI cameras (real hardware) and digital twin
simulation for development without hardware.

Protocols:
    CameraDriver: Interface for camera discovery and connection
    CameraInstance: Interface for capture and control operations

Implementations:
    ASICameraDriver/ASICameraInstance: Real ZWO ASI cameras via SDK
    DigitalTwinCameraDriver/DigitalTwinCameraInstance: Simulated cameras

TypedDicts:
    CameraInfo: Camera hardware information
    ControlInfo: Control definition with ranges
    ControlValue: Current control state
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from telescope_mcp.drivers.cameras.asi import (
    ASICameraDriver,
    ASICameraInstance,
)
from telescope_mcp.drivers.cameras.twin import (
    DEFAULT_CAMERAS,
    ControlInfo,
    ControlValue,
    DigitalTwinCameraDriver,
    DigitalTwinCameraInstance,
    DigitalTwinConfig,
    ImageSource,
    TwinCameraInfo,
    create_directory_camera,
    create_file_camera,
)


@runtime_checkable
class CameraInstance(Protocol):  # pragma: no cover
    """Protocol for an opened camera instance.

    Represents a connection to a specific camera for capture and control.
    Implemented by ASICameraInstance and DigitalTwinCameraInstance.

    See README.md for full API documentation and usage examples.
    """

    def get_info(self) -> dict[str, Any]:
        """Get camera hardware info (name, resolution, pixel size, etc.).

        Returns:
            Dict with: name, max_width, max_height, pixel_size_um, is_color, bit_depth

        Raises:
            RuntimeError: If camera disconnected.
        """
        ...

    def get_controls(self) -> dict[str, dict[str, Any]]:
        """Get all available camera controls with ranges.

        Returns:
            Dict[control_name, {min_value, max_value, default_value, is_auto_supported}]

        Raises:
            RuntimeError: If camera disconnected.
        """
        ...

    def set_control(self, control: str, value: int) -> dict[str, Any]:
        """Set a camera control value.

        Args:
            control: Control name (e.g., "Gain", "Exposure").
            value: Value to set (units depend on control).

        Returns:
            Dict with {value: int, auto: bool} - actual value set.

        Raises:
            ValueError: If control unknown or value out of range.
        """
        ...

    def get_control(self, control: str) -> dict[str, Any]:
        """Get current value of a camera control.

        Args:
            control: Control name to query.

        Returns:
            Dict with {value: int, auto: bool}.

        Raises:
            ValueError: If control unknown.
        """
        ...

    def capture(self, exposure_us: int) -> bytes:
        """Capture a single frame. Blocking for exposure duration.

        Args:
            exposure_us: Exposure time in microseconds (1 to 3,600,000,000).

        Returns:
            JPEG-encoded image bytes.

        Raises:
            ValueError: If exposure out of range.
            RuntimeError: If capture fails or times out.
        """
        ...

    def close(self) -> None:
        """Close camera and release hardware resources. Idempotent."""
        ...

    def stop_exposure(self) -> None:
        """Stop an in-progress exposure. No-op if none running."""
        ...

    def __enter__(self) -> CameraInstance:
        """Enter context manager, returning self."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing camera."""
        ...


@runtime_checkable
class CameraDriver(Protocol):  # pragma: no cover
    """Protocol for camera drivers (hardware abstraction).

    Implemented by: ASICameraDriver (real HW), DigitalTwinCameraDriver (simulation).
    See README.md for full documentation.
    """

    def get_connected_cameras(self) -> Mapping[int, Mapping[str, Any]]:
        """Discover connected cameras.

        Returns:
            Mapping[camera_id, camera_info] - empty if no cameras found.

        Raises:
            RuntimeError: If SDK init failed.
        """
        ...

    def open(self, camera_id: int) -> CameraInstance:
        """Open camera for exclusive access.

        Args:
            camera_id: ID from get_connected_cameras().

        Returns:
            CameraInstance for capture/control.

        Raises:
            ValueError: If camera_id invalid.
            RuntimeError: If camera already open or HW failure.
        """
        ...


__all__ = [
    # Protocols
    "CameraDriver",
    "CameraInstance",
    # ASI implementation
    "ASICameraDriver",
    "ASICameraInstance",
    # Digital twin implementation
    "DigitalTwinCameraDriver",
    "DigitalTwinCameraInstance",
    "DigitalTwinConfig",
    "ImageSource",
    "DEFAULT_CAMERAS",
    "create_directory_camera",
    "create_file_camera",
    # TypedDicts
    "TwinCameraInfo",
    "ControlInfo",
    "ControlValue",
]
