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

    Business context: Defines the contract for camera operations enabling
    hardware abstraction. Applications code against this protocol, allowing
    seamless switching between real ASI cameras and digital twin simulation.
    Essential for testing imaging pipelines without physical hardware.

    See README.md for full API documentation and usage examples.
    """

    def get_info(self) -> dict[str, Any]:
        """Get camera hardware information dictionary.

        Retrieves camera properties including sensor resolution, pixel size,
        color capability, and hardware features. Used for UI display, exposure
        calculations, and determining sensor capabilities.

        Business context: Camera info drives field-of-view calculations, plate
        solving parameters, and determines whether debayering is needed for
        color cameras. Essential for adaptive applications that work across
        different camera models.

        Returns:
            Dict with camera properties:
            - name: str - Camera model name
            - max_width: int - Sensor width in pixels
            - max_height: int - Sensor height in pixels
            - pixel_size_um: float - Pixel size in micrometers
            - is_color: bool - True for Bayer color sensors
            - bit_depth: int - ADC bit depth (8, 12, 14, 16)

        Raises:
            RuntimeError: If camera disconnected or hardware error.

        Example:
            >>> info = camera.get_info()
            >>> print(f"{info['name']}: {info['max_width']}x{info['max_height']}")
        """
        ...

    def get_controls(self) -> dict[str, dict[str, Any]]:
        """Get all available camera controls with their ranges.

        Returns control definitions from the camera including value ranges
        and capabilities for each adjustable parameter.

        Business context: Control definitions enable UI sliders with proper
        min/max bounds, validate user inputs before sending to hardware, and
        determine which features are available (e.g., cooling, auto-exposure).

        Returns:
            Dict mapping control name to definition:
            - min_value: int - Minimum allowed value
            - max_value: int - Maximum allowed value
            - default_value: int - Factory default
            - is_auto_supported: bool - Auto mode availability

        Raises:
            RuntimeError: If camera disconnected or hardware error.

        Example:
            >>> controls = camera.get_controls()
            >>> gain = controls['Gain']
            >>> print(f"Gain range: {gain['min_value']}-{gain['max_value']}")
        """
        ...

    def set_control(self, control: str, value: int) -> dict[str, Any]:
        """Set a camera control value.

        Applies the specified value to the camera control and returns the
        actual value set by hardware (which may differ due to clamping).

        Business context: Direct hardware control for real-time adjustment
        of imaging parameters. Used by exposure algorithms, gain control,
        and manual optimization during imaging sessions.

        Args:
            control: Control name (e.g., "Gain", "Exposure", "Offset").
            value: Integer value to set. Valid range depends on control.

        Returns:
            Dict with actual state after setting:
            - value: int - Value actually set by hardware
            - auto: bool - Whether auto mode is enabled

        Raises:
            ValueError: If control name unknown or value out of range.

        Example:
            >>> result = camera.set_control("Gain", 100)
            >>> print(f"Gain set to {result['value']}")
        """
        ...

    def get_control(self, control: str) -> dict[str, Any]:
        """Get current value of a camera control.

        Queries the camera for the current control value and auto status.

        Business context: Real-time control readback enables UI synchronization,
        verifies settings were applied correctly, and monitors read-only controls
        like sensor temperature.

        Args:
            control: Control name to query (e.g., "Gain", "Temperature").

        Returns:
            Dict with current state:
            - value: int - Current control value
            - auto: bool - Whether auto mode is enabled

        Raises:
            ValueError: If control name unknown.

        Example:
            >>> temp = camera.get_control("Temperature")
            >>> print(f"Sensor: {temp['value'] / 10}°C")
        """
        ...

    def capture(self, exposure_us: int) -> bytes:
        """Capture a single frame from the camera.

        Triggers exposure, waits for completion, and returns JPEG-encoded
        image data. Blocking call for the duration of the exposure.

        Business context: Core imaging function for astrophotography. Exposure
        times range from microseconds (planetary) to minutes (deep sky). JPEG
        encoding enables efficient network transmission for preview/streaming.

        Args:
            exposure_us: Exposure time in microseconds (1 to 3,600,000,000).
                Common values: 1000 (1ms) for planets, 5000000 (5s) deep sky.

        Returns:
            JPEG-encoded image bytes ready for display or storage.

        Raises:
            ValueError: If exposure_us out of valid range.
            RuntimeError: If capture fails, times out, or encoding fails.

        Example:
            >>> jpeg_data = camera.capture(5_000_000)  # 5 second exposure
            >>> with open("frame.jpg", "wb") as f:
            ...     f.write(jpeg_data)
        """
        ...

    def close(self) -> None:
        """Close camera and release hardware resources.

        Releases exclusive camera access, allowing other applications to
        use the device. Safe to call multiple times (idempotent).

        Business context: Essential for multi-application workflows where
        cameras are shared between tools. Without proper close(), cameras
        remain locked until process termination.

        Args:
            self: Camera instance (implicit).

        Returns:
            None. Camera resources released.

        Raises:
            None. Errors during close are logged but not raised.

        Example:
            >>> camera.close()  # Release hardware
        """
        ...

    def stop_exposure(self) -> None:
        """Stop an in-progress exposure immediately.

        Aborts any ongoing exposure, useful for cancellation when conditions
        change. Safe to call when no exposure is running (no-op).

        Business context: Allows users to abort long exposures without waiting
        for completion. Used by autofocus routines that need quick retries.

        Args:
            self: Camera instance (implicit).

        Returns:
            None. Exposure aborted if running.

        Raises:
            None. Safe to call at any time.

        Example:
            >>> camera.stop_exposure()  # Abort current capture
        """
        ...

    def __enter__(self) -> CameraInstance:
        """Enter context manager, returning self for capture operations.

        Enables the with-statement pattern for automatic resource cleanup.

        Returns:
            Self for use in with-block.

        Raises:
            None. Entry always succeeds for opened cameras.

        Example:
            >>> with driver.open(0) as camera:
            ...     data = camera.capture(100000)
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing camera automatically.

        Called when exiting with-block, ensures camera is closed regardless
        of whether an exception occurred.

        Args:
            exc_type: Exception type if raised in with-block, else None.
            exc_val: Exception instance if raised, else None.
            exc_tb: Exception traceback if raised, else None.

        Returns:
            None. Does not suppress exceptions.

        Example:
            >>> with driver.open(0) as camera:
            ...     camera.capture(100000)
            # camera.close() called automatically here
        """
        ...


@runtime_checkable
class CameraDriver(Protocol):  # pragma: no cover
    """Protocol for camera drivers (hardware abstraction layer).

    Defines the interface for camera discovery and connection management.
    Implemented by ASICameraDriver (real hardware) and DigitalTwinCameraDriver
    (simulation for testing).

    Business context: Enables hardware-agnostic telescope control applications.
    Production code uses ASICameraDriver with real ZWO cameras while tests and
    CI/CD pipelines use DigitalTwinCameraDriver without physical hardware.

    See README.md for full documentation and usage examples.
    """

    def get_connected_cameras(self) -> Mapping[int, Mapping[str, Any]]:
        """Discover and enumerate connected cameras.

        Scans for available cameras and returns basic information about each.
        For USB cameras, triggers device enumeration. For digital twin, returns
        pre-configured simulated cameras.

        Business context: Essential for multi-camera setups (guide + imaging),
        auto-discovery in UIs, and device health monitoring. Enables plug-and-play
        camera configuration without manual ID assignment.

        Returns:
            Mapping of camera_id to camera info. Empty mapping if no cameras.
            Each camera info contains at minimum:
            - camera_id: int - Unique identifier for opening
            - name: str - Camera model name

        Raises:
            RuntimeError: If SDK initialization fails (hardware mode only).

        Example:
            >>> cameras = driver.get_connected_cameras()
            >>> for cam_id, info in cameras.items():
            ...     print(f"Camera {cam_id}: {info['name']}")
        """
        ...

    def open(self, camera_id: int) -> CameraInstance:
        """Open a camera for exclusive access.

        Establishes connection to the specified camera and returns an instance
        for capture and control operations. Camera remains locked until closed.

        Business context: Cameras require exclusive access - only one application
        can control a camera at a time. Opening claims the device, preventing
        conflicts in multi-application environments.

        Args:
            camera_id: Camera identifier from get_connected_cameras(). Must be
                a valid ID returned by discovery.

        Returns:
            CameraInstance for capture and control. Supports context manager
            protocol for automatic cleanup.

        Raises:
            ValueError: If camera_id is invalid (negative or not found).
            RuntimeError: If camera already open by another process or
                hardware communication fails.

        Example:
            >>> cameras = driver.get_connected_cameras()
            >>> if 0 in cameras:
            ...     with driver.open(0) as camera:
            ...         data = camera.capture(100000)
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
