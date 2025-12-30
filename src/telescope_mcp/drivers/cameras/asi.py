"""ASI Camera Driver - Real Hardware Implementation.

Wraps the zwoasi library to provide camera control for ZWO ASI cameras
following the CameraDriver protocol.

Types:
    CameraInfo: TypedDict for camera hardware information
    ControlInfo: TypedDict for camera control definitions
    ControlValue: TypedDict for current control state
    DiscoveredCamera: TypedDict for camera discovery results
    ASISDKProtocol: Protocol for SDK abstraction (enables testing)

Classes:
    ASICameraInstance: Opened camera instance for capture/control
    ASICameraDriver: Driver for discovering and opening cameras

Example:
    from telescope_mcp.drivers.cameras.asi import ASICameraDriver

    driver = ASICameraDriver()
    cameras = driver.get_connected_cameras()
    if cameras:
        with driver.open(0) as camera:
            info = camera.get_info()
            jpeg_data = camera.capture(100000)  # 100ms exposure
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, final, runtime_checkable

import cv2
import numpy as np
import zwoasi as asi

from telescope_mcp.drivers.asi_sdk import get_sdk_library_path
from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from telescope_mcp.drivers.cameras import CameraInstance

logger = get_logger(__name__)

__all__ = [
    "ASICameraDriver",
    "ASICameraInstance",
    "CameraInfo",
    "ControlInfo",
    "ControlValue",
    "DiscoveredCamera",
    "CONTROL_MAP",
]

# =============================================================================
# Constants
# =============================================================================

# Exposure polling constants
_POLL_INTERVAL_SEC = 0.01  # Time between exposure status checks (seconds)
_MAX_POLL_ATTEMPTS = 100  # Maximum polls before timeout (count)
_EXPOSURE_BUFFER_SEC = 0.1  # Buffer time added after expected exposure (seconds)

# Exposure limits (microseconds)
_MIN_EXPOSURE_US = 1  # Minimum exposure: 1 microsecond (µs)
_MAX_EXPOSURE_US = 3_600_000_000  # Maximum exposure: 1 hour (µs)

# JPEG encoding settings
_JPEG_QUALITY = 90  # JPEG compression quality (%)

# Control name to ASI constant mapping (immutable)
CONTROL_MAP: Mapping[str, int] = MappingProxyType(
    {
        "Gain": asi.ASI_GAIN,
        "Exposure": asi.ASI_EXPOSURE,
        "Gamma": asi.ASI_GAMMA,
        "WB_R": asi.ASI_WB_R,
        "WB_B": asi.ASI_WB_B,
        "Brightness": asi.ASI_BRIGHTNESS,
        "Offset": asi.ASI_OFFSET,
        "BandwidthOverload": asi.ASI_BANDWIDTHOVERLOAD,
        "Temperature": asi.ASI_TEMPERATURE,
        "Flip": asi.ASI_FLIP,
        "HighSpeedMode": asi.ASI_HIGH_SPEED_MODE,
    }
)


# =============================================================================
# TypedDicts for Type Safety
# =============================================================================


class CameraInfo(TypedDict, total=False):
    """Camera hardware information returned by get_info().

    Keys:
        camera_id: Camera ID (0-indexed).
        name: Camera model name (e.g., "ZWO ASI183MM Pro").
        max_width: Maximum sensor width in pixels.
        max_height: Maximum sensor height in pixels.
        pixel_size_um: Physical pixel size in micrometers.
        is_color: True for color (Bayer) cameras.
        bayer_pattern: Bayer pattern code if color camera.
        supported_bins: List of supported binning modes.
        supported_formats: List of supported image formats.
        bit_depth: Sensor bit depth (8, 12, 14, 16).
        is_usb3: True if USB3 camera.
        has_cooler: True if camera has cooling capability.
        has_st4_port: True if camera has ST4 autoguider port.
    """

    camera_id: int
    name: str
    max_width: int
    max_height: int
    pixel_size_um: float
    is_color: bool
    bayer_pattern: int
    supported_bins: list[int]
    supported_formats: list[int]
    bit_depth: int
    is_usb3: bool
    has_cooler: bool
    has_st4_port: bool


class ControlInfo(TypedDict):
    """Camera control definition returned by get_controls().

    Keys:
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        default_value: Factory default value.
        is_auto_supported: True if auto mode available.
        is_writable: True if control can be modified.
        description: Human-readable control description.
    """

    min_value: int
    max_value: int
    default_value: int
    is_auto_supported: bool
    is_writable: bool
    description: str


class ControlValue(TypedDict):
    """Current control state returned by get_control() and set_control().

    Keys:
        control: Control name (e.g., "Gain", "Exposure").
        value: Current control value.
        auto: True if auto mode is enabled.
    """

    control: str
    value: int
    auto: bool


class DiscoveredCamera(TypedDict, total=False):
    """Camera info from discovery (may have partial data on errors).

    Keys:
        camera_id: Camera ID (0-indexed).
        name: Camera model name.
        max_width: Maximum sensor width (if available).
        max_height: Maximum sensor height (if available).
        pixel_size_um: Pixel size in µm (if available).
        is_color: True for color cameras (if available).
    """

    camera_id: int
    name: str
    max_width: int
    max_height: int
    pixel_size_um: float
    is_color: bool


# =============================================================================
# SDK Protocol for Dependency Injection
# =============================================================================


@runtime_checkable
class ASICameraProtocol(Protocol):  # pragma: no cover
    """Protocol for ASI camera object (enables mocking in tests).

    Matches the subset of zwoasi.Camera interface used by this driver.
    Implement this protocol to create mock cameras for unit testing.

    All methods defined here are required for full protocol compliance.
    """

    def get_camera_property(self) -> dict[str, Any]:
        """Get camera properties from hardware."""
        ...

    def get_controls(self) -> dict[str, dict[str, Any]]:
        """Get available controls and their ranges."""
        ...

    def set_control_value(self, control_id: int, value: int) -> None:
        """Set a control value."""
        ...

    def get_control_value(self, control_id: int) -> tuple[int, bool]:
        """Get current control value and auto status."""
        ...

    def set_image_type(self, image_type: int) -> None:
        """Set image format (RAW8, RAW16, etc.)."""
        ...

    def start_exposure(self) -> None:
        """Begin exposure."""
        ...

    def get_exposure_status(self) -> int:
        """Get exposure completion status."""
        ...

    def stop_exposure(self) -> None:
        """Stop an in-progress exposure."""
        ...

    def get_data_after_exposure(self) -> bytes:
        """Retrieve image data after exposure completes."""
        ...

    def close(self) -> None:
        """Close camera and release resources."""
        ...


@runtime_checkable
class ASISDKProtocol(Protocol):  # pragma: no cover
    """Protocol for ASI SDK operations (enables mocking in tests).

    Matches the subset of zwoasi module interface used by this driver.
    Implement this protocol to create mock SDK for unit testing without hardware.

    All methods defined here are required for full protocol compliance.

    Example:
        class MockASISDK:
            def init(self, path: str) -> None:
                pass  # No-op for tests

            def get_num_cameras(self) -> int:
                return 1  # Simulate one camera

            def list_cameras(self) -> list[str]:
                return ["Mock ASI Camera"]

            def open_camera(self, camera_id: int) -> ASICameraProtocol:
                return MockCamera()

        driver = ASICameraDriver(sdk=MockASISDK())
    """

    def init(self, library_path: str) -> None:
        """Initialize SDK with library path."""
        ...

    def get_num_cameras(self) -> int:
        """Return number of connected cameras."""
        ...

    def list_cameras(self) -> list[str]:
        """Return list of camera names."""
        ...

    def open_camera(self, camera_id: int) -> ASICameraProtocol:
        """Open camera by ID and return camera object.

        Note: Named open_camera to satisfy PEP8 (N802). The real zwoasi
        module uses Camera() which is automatically wrapped.
        """
        ...


class _ASISDKWrapper:
    """Wrapper adapting real zwoasi module to ASISDKProtocol.

    Bridges between the protocol interface (using open_camera) and the
    actual zwoasi module (using Camera). This avoids N802 naming issues
    while maintaining type safety.
    """

    def init(self, library_path: str) -> None:
        """Initialize SDK with library path."""
        asi.init(library_path)

    def get_num_cameras(self) -> int:
        """Return number of connected cameras."""
        result: int = asi.get_num_cameras()
        return result

    def list_cameras(self) -> list[str]:
        """Return list of camera names."""
        result: list[str] = asi.list_cameras()
        return result

    def open_camera(self, camera_id: int) -> ASICameraProtocol:
        """Open camera by ID and return camera object."""
        camera: ASICameraProtocol = asi.Camera(camera_id)
        return camera


def _wrap_asi_module() -> ASISDKProtocol:
    """Create SDK wrapper for the real zwoasi module.

    Returns:
        ASISDKProtocol implementation wrapping the real SDK.
    """
    return _ASISDKWrapper()


# =============================================================================
# ASI Camera Instance
# =============================================================================


@final
class ASICameraInstance:
    """Opened ASI camera instance.

    Wraps a zwoasi.Camera and implements the CameraInstance protocol.
    Created by ASICameraDriver.open() and should be closed when done.

    Supports context manager protocol for automatic cleanup:
        with driver.open(0) as camera:
            data = camera.capture(100000)
    """

    __slots__ = ("_camera_id", "_camera", "_info", "_controls", "_closed")

    def __init__(self, camera_id: int, camera: ASICameraProtocol) -> None:
        """Create camera instance from opened zwoasi camera.

        Queries camera properties and controls from the hardware.
        Should not be called directly - use ASICameraDriver.open().

        Business context: Wraps hardware camera with clean protocol interface.
        Abstracts ASI SDK details enabling higher layers (devices/camera.py)
        to work hardware-agnostically. Enables testing by substituting
        DigitalTwinCameraInstance.

        Args:
            camera_id: Camera ID (0-indexed).
            camera: Opened zwoasi.Camera instance or mock implementing protocol.

        Returns:
            None. Instance ready for capture, control operations.

        Raises:
            RuntimeError: If camera property or control query fails.

        Example:
            # Normally via driver:
            >>> driver = ASICameraDriver()
            >>> camera = driver.open(0)  # Creates ASICameraInstance internally
        """
        self._camera_id = camera_id
        self._camera = camera
        self._info = camera.get_camera_property()
        self._controls = camera.get_controls()
        self._closed = False

    def __enter__(self) -> ASICameraInstance:
        """Enter context manager, returning self for capture operations.

        Enables the with-statement pattern for automatic resource cleanup.

        Business context: Ensures camera resources are properly released even
        if exceptions occur during capture. Essential for robust applications
        where camera leaks prevent subsequent operations.

        Returns:
            Self for use in with-block.

        Example:
            >>> with driver.open(0) as camera:
            ...     data = camera.capture(100000)
            # camera.close() called automatically
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing camera.

        Called automatically when exiting with-block. Closes camera
        regardless of whether an exception occurred.

        Args:
            exc_type: Exception type if raised in with-block.
            exc_val: Exception instance if raised.
            exc_tb: Exception traceback if raised.

        Returns:
            None. Does not suppress exceptions.
        """
        self.close()

    def _validate_control(self, control: str) -> int:
        """Validate control name and return SDK control ID.

        Centralizes control validation logic for set_control/get_control.

        Args:
            control: Control name to validate.

        Returns:
            ASI SDK control ID integer.

        Raises:
            ValueError: If control name is not in CONTROL_MAP.
        """
        if control not in CONTROL_MAP:
            raise ValueError(
                f"Unknown control: {control}. Valid: {list(CONTROL_MAP.keys())}"
            )
        return CONTROL_MAP[control]

    def get_info(self) -> dict[str, Any]:
        """Get camera information from ASI hardware.

        Returns hardware properties queried from the ASI SDK including
        sensor resolution, pixel size, color capability, and features.

        Business context: Camera info is essential for determining capture
        capabilities, calculating field of view, plate solving parameters,
        and GUI display. Sensor specs drive exposure calculations and
        determine whether debayering is needed for color cameras.

        Args:
            None. Reads cached info from camera initialization.

        Returns:
            Dict with camera properties. Structure matches CameraInfo TypedDict:
            - camera_id, name, max_width, max_height
            - pixel_size_um, is_color, bayer_pattern
            - supported_bins, bit_depth, is_usb3
            - has_cooler, has_st4_port

            Note: Returns dict[str, Any] for CameraInstance protocol
            compatibility. See CameraInfo TypedDict for key documentation.

        Raises:
            None. Returns cached data from SDK query at open time.

        Example:
            >>> info = camera_instance.get_info()
            >>> print(f"{info['name']}: {info['max_width']}x{info['max_height']}")
            >>> if info['is_color']:
            ...     print(f"Color camera with Bayer pattern {info['bayer_pattern']}")
        """
        return dict(
            camera_id=self._camera_id,
            name=self._info.get("Name", f"ASI Camera {self._camera_id}"),
            max_width=self._info["MaxWidth"],
            max_height=self._info["MaxHeight"],
            pixel_size_um=self._info.get("PixelSize", 0),
            is_color=self._info.get("IsColorCam", False),
            bayer_pattern=self._info.get("BayerPattern", 0),
            supported_bins=self._info.get("SupportedBins", [1]),
            supported_formats=self._info.get("SupportedVideoFormat", []),
            bit_depth=self._info.get("BitDepth", 8),
            is_usb3=self._info.get("IsUSB3Camera", False),
            has_cooler=self._info.get("IsCoolerCam", False),
            has_st4_port=self._info.get("ST4Port", False),
        )

    def get_controls(self) -> dict[str, dict[str, Any]]:
        """Get available camera controls from ASI hardware.

        Returns control definitions from the ASI SDK including value
        ranges and capabilities for each control.

        Business context: Control definitions enable UI sliders with proper
        ranges, validate user inputs before sending to hardware, and determine
        which features are available (e.g., cooler control, auto-exposure).
        Essential for building adaptive UIs that work across different camera
        models.

        Args:
            None. Returns cached control definitions from SDK.

        Returns:
            Dict mapping control name to ControlInfo TypedDict including:
            - min_value, max_value, default_value
            - is_auto_supported, is_writable, description

        Raises:
            None. Returns cached data from SDK query at open time.

        Example:
            >>> controls = camera_instance.get_controls()
            >>> gain = controls['Gain']
            >>> print(f"Gain range: {gain['min_value']}-{gain['max_value']}")
            >>> if gain['is_auto_supported']:
            ...     print("Auto-gain available")
        """
        result: dict[str, dict[str, Any]] = {}
        for name, ctrl in self._controls.items():
            result[name] = dict(
                min_value=ctrl["MinValue"],
                max_value=ctrl["MaxValue"],
                default_value=ctrl["DefaultValue"],
                is_auto_supported=ctrl["IsAutoSupported"],
                is_writable=ctrl["IsWritable"],
                description=ctrl.get("Description", ""),
            )
        return result

    def set_control(self, control: str, value: int) -> dict[str, Any]:
        """Set a camera control value on ASI hardware.

        Applies the control value to the physical camera via ZWO SDK and reads
        back the actual value set by hardware (which may differ due to clamping
        to valid range or hardware rounding). This is the low-level interface
        to ASI camera control hardware.

        Business context: Direct hardware control for ASI cameras enabling
        real-time adjustment of imaging parameters. Used by higher-level Camera
        class for exposure control, gain adjustment, and sensor configuration.
        Critical for automated exposure algorithms, adaptive gain control, and
        manual optimization during imaging sessions. Hardware read-back confirms
        settings were applied successfully.

        Args:
            control: Control name from CONTROL_MAP (Gain, Exposure, WB_R, WB_B,
                Offset, Brightness, Gamma, etc.). Must be one of the supported
                ASI camera controls.
            value: Integer value to set. Valid range depends on control and camera
                model. Hardware will clamp to valid range if out of bounds.

        Returns:
            ControlValue TypedDict with:
            - control: str - Echo of control name
            - value: int - Actual value set by hardware (may differ from requested)
            - auto: bool - Whether auto mode is enabled for this control

        Raises:
            ValueError: If control name is not in CONTROL_MAP (unsupported control).
            RuntimeError: If SDK fails to apply control (camera disconnected, etc.).

        Example:
            >>> result = instance.set_control("Gain", 100)
            >>> print(f"Requested 100, hardware set {result['value']}")
            >>> # Set exposure for 5 second frame
            >>> result = instance.set_control("Exposure", 5_000_000)
        """
        control_id = self._validate_control(control)
        self._camera.set_control_value(control_id, value)

        # Read back to confirm
        current_value, is_auto = self._camera.get_control_value(control_id)

        return dict(
            control=control,
            value=current_value,
            auto=is_auto,
        )

    def get_control(self, control: str) -> dict[str, Any]:
        """Get current value of a camera control from ASI hardware.

        Queries the hardware for the current control value and auto status.

        Business context: Real-time control readback enables UI synchronization,
        verifies settings were applied correctly, and monitors read-only controls
        like sensor temperature for cooler management.

        Args:
            control: Control name (e.g., "Gain", "Exposure", "Temperature").

        Returns:
            ControlValue TypedDict with control name, current value, and auto status.

        Raises:
            ValueError: If control name is not in CONTROL_MAP.

        Example:
            >>> result = camera_instance.get_control("Gain")
            >>> print(f"Current gain: {result['value']}")
            >>> temp = camera_instance.get_control("Temperature")
            >>> print(f"Sensor temp: {temp['value']/10}°C")
        """
        control_id = self._validate_control(control)
        current_value, is_auto = self._camera.get_control_value(control_id)

        return dict(
            control=control,
            value=current_value,
            auto=is_auto,
        )

    def _encode_to_jpeg(
        self,
        img_data: bytes,
        img_type: int,
        jpeg_quality: int,
    ) -> bytes:
        """Convert raw image data to JPEG bytes.

        Handles different ASI image formats (RAW8, RAW16, RGB24) and encodes
        to JPEG with specified quality. Extracted for testability.

        Args:
            img_data: Raw image bytes from camera.
            img_type: ASI image type constant (ASI_IMG_RAW8, etc.).
            jpeg_quality: JPEG compression quality (0-100).

        Returns:
            JPEG-encoded image bytes.

        Raises:
            RuntimeError: If JPEG encoding fails.
        """
        width = self._info["MaxWidth"]
        height = self._info["MaxHeight"]

        if img_type == asi.ASI_IMG_RAW16:
            img_array_16: NDArray[np.uint16] = np.frombuffer(
                img_data, dtype=np.uint16
            ).reshape((height, width))
            img_array: NDArray[np.uint8] = (img_array_16 >> 8).astype(np.uint8)
        elif img_type == asi.ASI_IMG_RGB24:
            img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(
                (height, width, 3)
            )
        else:
            img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))

        success, jpeg_data = cv2.imencode(
            ".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not success:
            raise RuntimeError("Failed to encode image as JPEG")

        return bytes(jpeg_data.tobytes())

    def capture(
        self,
        exposure_us: int,
        image_type: int | None = None,
        jpeg_quality: int = _JPEG_QUALITY,
    ) -> bytes:
        """Capture a frame from ASI camera hardware.

        Sets exposure, triggers capture, waits for completion, then
        retrieves and encodes the image as JPEG.

        Business context: Core imaging function for astrophotography. Handles
        hardware timing, buffer management, and JPEG encoding. Exposure times
        range from microseconds (planetary) to minutes (deep sky). JPEG
        encoding enables efficient network transmission for live preview and
        streaming.

        Args:
            exposure_us: Exposure time in microseconds. Valid range is
                1 to 3,600,000,000 (1 hour). Typical values:
                - 1,000 (1ms) for bright planets
                - 5,000,000 (5s) for deep sky
            image_type: ASI image type constant. Supported formats:
                - asi.ASI_IMG_RAW8: 8-bit mono (default)
                - asi.ASI_IMG_RGB24: 24-bit color
                - asi.ASI_IMG_RAW16: 16-bit mono (converted to 8-bit for JPEG)
                - asi.ASI_IMG_Y8: 8-bit luminance
                Other values passed through to SDK (may fail if unsupported).
            jpeg_quality: JPEG compression quality (0-100). Higher values
                produce better quality but larger files. Defaults to 90.

        Returns:
            JPEG-encoded image data as bytes.

        Raises:
            ValueError: If exposure_us or jpeg_quality out of valid range.
            RuntimeError: If exposure fails, times out, or JPEG encoding fails.

        Example:
            >>> # 5-second exposure for deep sky (default RAW8)
            >>> jpeg_data = camera_instance.capture(5_000_000)
            >>> # 1ms exposure with 16-bit for scientific imaging
            >>> jpeg_data = camera_instance.capture(1000, asi.ASI_IMG_RAW16)
            >>> # Lower quality for streaming/preview
            >>> jpeg_data = camera_instance.capture(1000, jpeg_quality=70)
        """
        # Validate jpeg_quality
        if not 0 <= jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be 0-100, got {jpeg_quality}")

        # Validate exposure
        if exposure_us < _MIN_EXPOSURE_US:
            raise ValueError(
                f"exposure_us must be >= {_MIN_EXPOSURE_US}, got {exposure_us}"
            )
        if exposure_us > _MAX_EXPOSURE_US:
            raise ValueError(
                f"exposure_us must be <= {_MAX_EXPOSURE_US}, got {exposure_us}"
            )

        # Set exposure
        self._camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)

        # Set image format (default RAW8)
        img_type = image_type if image_type is not None else asi.ASI_IMG_RAW8
        self._camera.set_image_type(img_type)

        # Start exposure and wait for completion
        capture_start = time.monotonic()
        self._camera.start_exposure()
        self._wait_for_exposure(exposure_us)
        exposure_elapsed = time.monotonic() - capture_start

        # Get image data and encode to JPEG
        img_data = self._camera.get_data_after_exposure()
        encode_start = time.monotonic()
        jpeg_bytes = self._encode_to_jpeg(img_data, img_type, jpeg_quality)
        encode_elapsed = time.monotonic() - encode_start

        logger.debug(
            "Capture complete",
            camera_id=self._camera_id,
            exposure_us=exposure_us,
            exposure_elapsed_ms=round(exposure_elapsed * 1000, 1),
            encode_elapsed_ms=round(encode_elapsed * 1000, 1),
            jpeg_size_kb=round(len(jpeg_bytes) / 1024, 1),
        )

        return jpeg_bytes

    def _wait_for_exposure(self, exposure_us: int) -> None:
        """Wait for exposure to complete by polling status.

        Polls the camera exposure status until completion, failure, or timeout.
        Extracted from capture() for testability and cleaner separation of concerns.

        Business context: Exposure polling is hardware-dependent and may need
        tuning for different camera models or USB conditions. Extracting this
        logic enables unit testing of timeout behavior without real hardware.

        Args:
            exposure_us: Expected exposure time in microseconds for timeout calc.

        Returns:
            None. Returns when exposure completes successfully.

        Raises:
            RuntimeError: If exposure fails (non-working status) or times out.

        Example:
            >>> self._camera.start_exposure()
            >>> self._wait_for_exposure(100_000)  # Wait for 100ms exposure
        """
        exposure_sec = exposure_us / 1_000_000
        timeout_sec = (
            exposure_sec
            + _EXPOSURE_BUFFER_SEC
            + (_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_SEC)
        )
        start_time = time.monotonic()

        while True:
            status = self._camera.get_exposure_status()

            if status == asi.ASI_EXP_SUCCESS:
                return

            if status != asi.ASI_EXP_WORKING:
                raise RuntimeError(f"Exposure failed with status: {status}")

            if time.monotonic() - start_time > timeout_sec:
                raise RuntimeError(
                    f"Exposure timeout after {timeout_sec:.1f}s "
                    f"(expected {exposure_sec:.1f}s)"
                )

            time.sleep(_POLL_INTERVAL_SEC)

    def stop_exposure(self) -> None:
        """Stop an in-progress exposure.

        Immediately terminates any ongoing exposure, useful for cancellation
        or when exposure conditions change.

        Business context: Essential for responsive imaging control. Allows
        users to abort long exposures (e.g., if clouds roll in) without
        waiting for completion. Also used by autofocus routines that need
        to quickly retry with different settings.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Safe to call even when no exposure is in progress.

        Example:
            >>> instance.capture(30_000_000)  # 30s exposure in thread
            >>> # User cancels
            >>> instance.stop_exposure()
        """
        logger.debug(f"Stopping exposure on ASI camera {self._camera_id}")
        self._camera.stop_exposure()

    def close(self) -> None:
        """Close the ASI camera and release exclusive hardware access.

        Releases the camera hardware by closing the ZWO SDK connection,
        allowing other processes or applications to open the camera.
        Always call this when done with a camera to prevent resource leaks
        and allow camera reuse. Part of proper cleanup workflow.

        Business context: Essential for multi-session workflows where cameras
        are shared between applications (plate solving tool, PHD2 guiding,
        imaging software). Without proper close(), cameras remain locked and
        unusable until process termination. Critical for long-running servers
        and daemons that cycle through cameras.

        Implementation: Delegates to zwoasi library's close() which performs
        USB cleanup and releases kernel driver locks. Safe to call multiple
        times. Errors during close are logged but not raised.

        Args:
            None. Closes the camera opened via constructor.

        Returns:
            None.

        Raises:
            None. Errors during close are logged but suppressed for cleanup safety.

        Example:
            >>> instance = driver.open(0)
            >>> try:
            ...     instance.capture(100000)
            ... finally:
            ...     instance.close()  # Always release in finally block
            >>> # Or use context manager pattern
            >>> with driver.open(0) as instance:
            ...     instance.capture(100000)
        """
        if self._closed:
            logger.debug(f"ASI camera {self._camera_id} already closed, skipping")
            return

        try:
            self._camera.close()
            self._closed = True
            logger.info(f"Closed ASI camera {self._camera_id}")
        except Exception as e:
            logger.warning(f"Error closing ASI camera {self._camera_id}: {e}")
            self._closed = True


# =============================================================================
# ASI Camera Driver
# =============================================================================


@final
class ASICameraDriver:
    """ASI Camera Driver for real ZWO hardware.

    Wraps the zwoasi library and implements the CameraDriver protocol.
    Handles SDK initialization automatically on first use.

    Supports dependency injection for testing via sdk parameter.

    Example:
        # Production use
        driver = ASICameraDriver()
        cameras = driver.get_connected_cameras()

        # Testing with mock
        mock_sdk = MockASISDK()
        driver = ASICameraDriver(sdk=mock_sdk)
    """

    __slots__ = ("_sdk", "_sdk_initialized")

    def __init__(self, sdk: ASISDKProtocol | None = None) -> None:
        """Create ASI camera driver for ZWO hardware.

        SDK initialized lazily on first camera operation (not during
        construction). Allows driver creation when cameras disconnected,
        enabling graceful handling of missing hardware.

        Business context: Lazy initialization enables application startup even
        when cameras offline (network issues, power off, USB disconnected).
        Application can show UI, handle MCP requests, provide diagnostics.
        When cameras reconnected, operations succeed. Critical for robust
        production systems where hardware may be temporarily unavailable.

        Implementation details: Sets _sdk_initialized=False. Actual SDK
        loading (_ensure_sdk_initialized) happens on first
        get_connected_cameras() or open() call. SDK loads libASICamera2.so
        via ctypes, initializes USB context. Construction never fails - SDK
        errors surface during operations.

        Args:
            sdk: Optional SDK implementation for dependency injection.
                If None (default), uses real zwoasi module. Pass mock
                implementation for unit testing without hardware.

        Returns:
            None. Driver ready for lazy SDK init on first use.

        Raises:
            None. Construction always succeeds. SDK errors (library not found,
            USB issues) occur during first camera operation.

        Example:
            >>> driver = ASICameraDriver()  # Succeeds even without cameras
            >>> cameras = driver.get_connected_cameras()  # SDK init happens here
            >>> # For testing:
            >>> mock_sdk = MockASISDK()
            >>> test_driver = ASICameraDriver(sdk=mock_sdk)
        """
        self._sdk: ASISDKProtocol = sdk if sdk is not None else _wrap_asi_module()
        self._sdk_initialized = sdk is not None  # Mock SDKs are pre-initialized

    def _ensure_sdk_initialized(self) -> None:
        """Initialize ASI SDK if not already done.

        Loads the ASI SDK library from the bundled location and
        initializes it. Called automatically before camera operations.

        Business context: Lazy init pattern enables graceful handling of
        missing cameras. Application starts even when hardware offline. First
        camera operation triggers SDK load, failing cleanly with diagnostics
        if library missing. Production systems stay responsive during hardware
        issues.

        Args:
            None. Uses internal _sdk_initialized flag.

        Returns:
            None. Sets _sdk_initialized=True on success.

        Raises:
            RuntimeError: If SDK initialization fails.

        Example:
            # Called internally:
            >>> driver = ASICameraDriver()
            >>> cameras = driver.get_connected_cameras()  # Triggers init
        """
        if not self._sdk_initialized:
            try:
                sdk_path = get_sdk_library_path()
                self._sdk.init(sdk_path)
                self._sdk_initialized = True
                logger.info(f"ASI SDK initialized from {sdk_path}")
            except Exception as e:
                logger.error(f"Failed to initialize ASI SDK: {e}")
                raise RuntimeError(f"ASI SDK initialization failed: {e}") from e

    def get_connected_cameras(self) -> dict[int, DiscoveredCamera]:
        """Discover connected ASI cameras via USB enumeration.

        Scans USB for connected ZWO cameras, queries basic info
        from each, and returns a summary. Each camera is briefly
        opened to get full properties.

        Business context: Essential for multi-camera setups (guide + imaging),
        auto-discovery in UIs, and device health monitoring. Enables plug-and-play
        camera configuration without manual ID assignment.

        Note: Each camera is briefly opened to query full properties.
        This may cause USB re-enumeration delays on some systems.

        Args:
            None. Scans all USB-connected ASI cameras.

        Returns:
            Dict mapping camera_id (0-indexed) to DiscoveredCamera TypedDict
            containing camera_id, name, resolution, and capabilities.

        Raises:
            RuntimeError: If ASI SDK initialization fails.

        Example:
            >>> driver = ASICameraDriver()
            >>> cameras = driver.get_connected_cameras()
            >>> for cam_id, info in cameras.items():
            ...     print(f"Camera {cam_id}: {info['name']}")
            >>> # Returns {} if no cameras connected
        """
        self._ensure_sdk_initialized()

        num_cameras = self._sdk.get_num_cameras()
        if num_cameras == 0:
            logger.info("No ASI cameras detected")
            return {}

        camera_names = self._sdk.list_cameras()

        result: dict[int, DiscoveredCamera] = {}
        for camera_id, name in enumerate(camera_names):
            # Open camera briefly to get full info
            try:
                temp_camera = self._sdk.open_camera(camera_id)
                info = temp_camera.get_camera_property()
                temp_camera.close()

                result[camera_id] = DiscoveredCamera(
                    camera_id=camera_id,
                    name=info.get("Name", name),
                    max_width=info["MaxWidth"],
                    max_height=info["MaxHeight"],
                    pixel_size_um=info.get("PixelSize", 0),
                    is_color=info.get("IsColorCam", False),
                )
            except Exception as e:
                logger.warning(f"Failed to get info for camera {camera_id}: {e}")
                # Fallback to minimal info
                result[camera_id] = DiscoveredCamera(
                    camera_id=camera_id,
                    name=name,
                )

        logger.info(f"Discovered {len(result)} ASI camera(s)")
        return result

    def open(self, camera_id: int) -> CameraInstance:
        """Open an ASI camera for exclusive access.

        Establishes connection to the camera hardware and returns
        an instance for control and capture operations.

        Business context: Camera must be opened before any control or capture
        operations. Opening claims exclusive hardware access - only one process
        can open a camera at a time. Essential for preventing conflicts in
        multi-application environments.

        Args:
            camera_id: ID of camera to open (0-based index from discovery).
                Must be >= 0.

        Returns:
            CameraInstance for capture and control operations.
            Supports context manager protocol for automatic cleanup.

        Raises:
            ValueError: If camera_id is negative.
            RuntimeError: If camera cannot be opened (disconnected, in use).

        Example:
            >>> driver = ASICameraDriver()
            >>> cameras = driver.get_connected_cameras()
            >>> if 0 in cameras:
            ...     with driver.open(0) as instance:
            ...         info = instance.get_info()
            ...         data = instance.capture(100000)
        """
        if camera_id < 0:
            raise ValueError(f"camera_id must be >= 0, got {camera_id}")

        self._ensure_sdk_initialized()

        try:
            camera = self._sdk.open_camera(camera_id)
            logger.info(f"Opened ASI camera {camera_id}")
            return ASICameraInstance(camera_id, camera)
        except Exception as e:
            logger.error(f"Failed to open camera {camera_id}: {e}")
            raise RuntimeError(f"Cannot open ASI camera {camera_id}: {e}") from e
