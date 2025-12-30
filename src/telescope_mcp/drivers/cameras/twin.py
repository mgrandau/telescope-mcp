"""Digital Twin Camera Driver - Simulated Hardware for Testing.

Provides simulated camera responses for development and testing without
physical hardware. Follows the CameraDriver protocol for drop-in replacement.

Image Sources:
    Synthetic: Generate test patterns (gradient, noise, checkerboard)
    Directory: Cycle through images in a folder
    File: Return same image repeatedly

Types:
    TwinCameraInfo: TypedDict for camera hardware information (PascalCase)
    ControlInfo: TypedDict for camera control definitions
    ControlValue: TypedDict for current control state

Enums:
    ImageSource: Source type for simulated captures

Classes:
    DigitalTwinConfig: Configuration dataclass for image sources
    DigitalTwinCameraDriver: Driver for discovering and opening simulated cameras
    DigitalTwinCameraInstance: Opened camera instance for capture/control

Constants:
    DEFAULT_CAMERAS: Pre-configured simulated camera specs (ASI120MC-S, ASI482MC)

Factory Functions:
    create_file_camera: Create twin that returns single image
    create_directory_camera: Create twin that cycles through image directory

Example:
    from telescope_mcp.drivers.cameras.twin import (
        DigitalTwinCameraDriver,
        create_directory_camera,
    )

    # Synthetic patterns (default)
    driver = DigitalTwinCameraDriver()
    with driver.open(0) as camera:
        jpeg_data = camera.capture(100000)

    # Cycle through sky frames
    driver = create_directory_camera("/data/sky_frames/")
    with driver.open(0) as camera:
        jpeg_data = camera.capture(100000)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, TypedDict, final

import cv2
import numpy as np

from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "DigitalTwinCameraDriver",
    "DigitalTwinCameraInstance",
    "DigitalTwinConfig",
    "ImageSource",
    "DEFAULT_CAMERAS",
    "TwinCameraInfo",
    "ControlInfo",
    "ControlValue",
    "create_file_camera",
    "create_directory_camera",
]


class ImageSource(Enum):
    """Image source for digital twin camera."""

    SYNTHETIC = "synthetic"  # Generate test patterns
    DIRECTORY = "directory"  # Cycle through images in a folder
    FILE = "file"  # Return same image repeatedly


# =============================================================================
# TypedDicts for Type Safety
# =============================================================================


class TwinCameraInfo(TypedDict, total=False):
    """Digital twin camera info (PascalCase keys matching ASI SDK raw output).

    Note: Different from asi.CameraInfo which uses snake_case for normalized output.
    This TypedDict matches the raw ASI SDK property names for simulation fidelity.
    """

    Name: str
    MaxWidth: int
    MaxHeight: int
    IsColorCam: bool
    PixelSize: float
    SensorWidth: float
    SensorHeight: float
    BayerPattern: str
    BitDepth: int
    ElecPerADU: float
    USB3Host: bool
    LensFOV: int
    FOVPerPixel: float
    FocalLength: int
    FOVWidth: float
    FOVHeight: float
    Purpose: str


class ControlInfo(TypedDict):
    """Camera control definition returned by get_controls().

    Keys:
        Name: Control name.
        MinValue: Minimum allowed value.
        MaxValue: Maximum allowed value.
        DefaultValue: Factory default value.
        IsAutoSupported: True if auto mode available.
        IsWritable: True if control can be modified.
    """

    Name: str
    MinValue: int
    MaxValue: int
    DefaultValue: int
    IsAutoSupported: bool
    IsWritable: bool


class ControlValue(TypedDict):
    """Current control state returned by get_control() and set_control().

    Keys:
        value: Current control value.
        auto: True if auto mode is enabled.
    """

    value: int
    auto: bool


@dataclass
class DigitalTwinConfig:
    """Configuration for digital twin camera behavior."""

    image_source: ImageSource = ImageSource.SYNTHETIC
    image_path: Path | None = None  # Directory or file path
    cycle_images: bool = True  # Loop through directory images


# =============================================================================
# Constants
# =============================================================================

# Default fallback exposure for synthetic generation (microseconds)
_DEFAULT_FALLBACK_EXPOSURE_US = 100_000

# Synthetic image generation constants
_SYNTHETIC_GRID_SPACING = 50
_CROSSHAIR_RADIUS = 100

# JPEG encoding quality (0-100)
_DEFAULT_JPEG_QUALITY = 90

# Control maximum values by partial name match
_CONTROL_MAX_VALUES: dict[str, int] = {
    "GAIN": 600,
    "EXPOSURE": 60_000_000,  # 60 seconds in microseconds
    "WB": 99,
    "GAMMA": 100,
    "TEMPERATURE": 1000,  # 100.0°C
}

# Default camera specifications (matching real ASI cameras)
# Camera 0: ASI120MC-S with 150° All-Sky Lens (finder/spotter scope)
# Camera 1: ASI482MC through telescope optics (main imaging camera)
#
# Specifications from astrophotography_camera_exposure_times.ipynb:
# - ASI120MC-S: 1.2MP, 1280x960, 3.75µm pixels, 4.8x3.6mm sensor
# - ASI482MC: 2.07MP, 1920x1080, 5.8µm pixels, 11.13x6.26mm sensor
DEFAULT_CAMERAS: Mapping[int, TwinCameraInfo] = MappingProxyType(
    {
        0: TwinCameraInfo(
            # Finder camera: ASI120MC-S with 150° All-Sky Lens
            Name="ASI120MC-S (Finder - 150deg All-Sky)",
            MaxWidth=1280,
            MaxHeight=960,
            IsColorCam=True,
            PixelSize=3.75,  # micrometers
            SensorWidth=4.8,  # mm
            SensorHeight=3.6,  # mm
            BayerPattern="RGGB",
            BitDepth=8,
            ElecPerADU=0.21,  # electrons per ADU
            USB3Host=True,
            # All-sky lens specific
            LensFOV=150,  # degrees (150° all-sky lens)
            FOVPerPixel=421.875,  # arcseconds (150*3600/1280)
            Purpose="finder",
        ),
        1: TwinCameraInfo(
            # Main camera: ASI482MC through 1600mm telescope optics
            Name="ASI482MC (Main - Through Optics)",
            MaxWidth=1920,
            MaxHeight=1080,
            IsColorCam=True,
            PixelSize=5.8,  # micrometers
            SensorWidth=11.13,  # mm
            SensorHeight=6.26,  # mm
            BayerPattern="RGGB",
            BitDepth=12,  # 12-bit ADC
            ElecPerADU=0.16,
            USB3Host=True,
            # Telescope optics specific
            FocalLength=1600,  # mm (telescope focal length)
            FOVPerPixel=0.748,  # arcseconds (calculated from focal length)
            FOVWidth=23.9,  # arcminutes
            FOVHeight=13.4,  # arcminutes
            Purpose="main",
        ),
    }
)


@final
class DigitalTwinCameraDriver:
    """Digital twin camera driver for development without hardware.

    Provides simulated camera responses for testing and development without
    requiring physical ASI cameras. Supports multiple image sources including
    synthetic test patterns, single file, or cycling through a directory.

    Example:
        # Basic synthetic camera (test patterns)
        driver = DigitalTwinCameraDriver()

        # Camera with images from directory
        config = DigitalTwinConfig(
            image_source=ImageSource.DIRECTORY,
            image_path=Path("/data/test_images"),
        )
        driver = DigitalTwinCameraDriver(config=config)
    """

    __slots__ = ("config", "_cameras")

    def __init__(
        self,
        config: DigitalTwinConfig | None = None,
        cameras: Mapping[int, TwinCameraInfo] | None = None,
    ) -> None:
        """Initialize digital twin camera driver.

        Sets up the simulated camera driver with the specified configuration.
        By default, simulates two cameras matching ASI120MC-S (finder) and
        ASI482MC (main) specifications.

        Args:
            config: Configuration for image source behavior. Defaults to
                synthetic test pattern generation.
            cameras: Custom camera definitions mapping camera_id to properties.
                Defaults to DEFAULT_CAMERAS (ASI120MC-S and ASI482MC).

        Returns:
            None. Initializes internal camera registry for simulation.

        Raises:
            None. Configuration validation deferred to camera operations.

        Business context: Enables development without physical ASI cameras. CI/CD
        runs full integration tests with simulated cameras at high speed. Developers
        iterate rapidly without USB hardware setup. Demonstrations run reliably
        without equipment.

        Example:
            # Custom camera specifications
            cameras = {
                0: {"Name": "Test Camera", "MaxWidth": 640, "MaxHeight": 480}
            }
            driver = DigitalTwinCameraDriver(cameras=cameras)
        """
        self.config = config or DigitalTwinConfig()
        self._cameras: dict[int, TwinCameraInfo] = (
            dict(cameras) if cameras else dict(DEFAULT_CAMERAS)
        )
        logger.info(
            "Digital twin camera driver initialized",
            image_source=self.config.image_source.value,
            num_cameras=len(self._cameras),
        )

    def __repr__(self) -> str:
        """Return human-readable string representation for debugging.

        Produces a concise summary of driver configuration useful in logs,
        REPL sessions, and debugging output.

        Business context: Debugging distributed systems requires quick
        identification of driver configuration. repr() shows image source
        and available camera IDs at a glance.

        Args:
            self: Driver instance (implicit).

        Returns:
            String in format: DigitalTwinCameraDriver(source=X, cameras=[...])

        Raises:
            None. Always succeeds.

        Example:
            >>> driver = DigitalTwinCameraDriver()
            >>> print(repr(driver))
            DigitalTwinCameraDriver(source=synthetic, cameras=[0, 1])
        """
        return (
            f"DigitalTwinCameraDriver("
            f"source={self.config.image_source.value}, "
            f"cameras={list(self._cameras.keys())})"
        )

    def get_connected_cameras(self) -> dict[int, TwinCameraInfo]:
        """Return simulated camera list (digital twin discovery).

        Returns copy of configured camera definitions, simulating ASI SDK camera
        enumeration. No hardware scanning - returns pre-configured simulated
        cameras instantly.

        Business context: Enables development and testing of multi-camera systems
        without hardware. CI/CD pipelines can test discovery and registration
        logic. Supports offline development, demo systems, and automated testing
        without physical cameras.

        Implementation details: Returns self._cameras.copy() dict built in
        __init__. Keys are camera_id, values match ASI camera property structure.

        Args:
            None.

        Returns:
            dict[int, dict] mapping camera_id to camera properties.

        Raises:
            None. Always succeeds returning configured cameras.

        Example:
            >>> cameras = driver.get_connected_cameras()
            >>> print(f"Found {len(cameras)} simulated cameras")
        """
        logger.debug("Listing simulated cameras", count=len(self._cameras))
        return self._cameras.copy()

    def open(self, camera_id: int) -> DigitalTwinCameraInstance:
        """Open a simulated camera.

        Creates a DigitalTwinCameraInstance for the specified camera ID,
        simulating the process of opening a connection to physical hardware.

        Args:
            camera_id: Camera identifier (0 for finder, 1 for main by default).

        Returns:
            DigitalTwinCameraInstance configured with camera properties
            and the driver's image source configuration.

        Raises:
            ValueError: If camera_id is not in the configured cameras dict.

        Example:
            camera = driver.open(0)  # Open finder camera
            frame = camera.capture(100_000)
        """
        if camera_id not in self._cameras:
            logger.error("Camera not found", camera_id=camera_id)
            raise ValueError(f"Camera {camera_id} not found")
        logger.info("Opening simulated camera", camera_id=camera_id)
        return DigitalTwinCameraInstance(
            camera_id,
            self._cameras[camera_id],
            self.config,
        )


@final
class DigitalTwinCameraInstance:
    """Digital twin camera instance for simulated capture operations.

    Represents an open connection to a simulated camera. Provides the same
    interface as ASICameraInstance for capture, control get/set, and info
    queries. Images can come from synthetic generation, a file, or directory.

    Supports context manager protocol for API compatibility:
        with driver.open(0) as camera:
            data = camera.capture(100000)
    """

    __slots__ = (
        "_camera_id",
        "_info",
        "_config",
        "_controls",
        "_image_files",
        "_image_index",
    )

    def __init__(
        self,
        camera_id: int,
        info: TwinCameraInfo,
        config: DigitalTwinConfig,
    ) -> None:
        """Initialize digital twin camera instance (simulated camera connection).

        Sets up simulated camera with default control values (Gain=50,
        Exposure=100000, etc.) matching typical ASI defaults. For directory mode,
        loads available image files.

        Business context: Represents opening simulated camera analogous to opening
        real USB camera. Enables testing full capture workflows without physical
        hardware. Critical for CI/CD integration tests, rapid development
        iteration, and demos without hardware setup.

        Implementation details: Stores camera_id, info dict, config reference.
        Initializes _controls with default ASI values. For directory mode, loads
        image files list. No hardware operations.

        Args:
            camera_id: 0-based identifier (0=finder, 1=main typically).
            info: Camera properties dict (Name, MaxWidth, MaxHeight, IsColorCam, etc.).
            config: Image source and capture delay configuration.

        Returns:
            None. Instance ready for capture operations.

        Raises:
            None. Directory scanning errors ignored (falls back to synthetic mode).

        Example:
            >>> instance = DigitalTwinCameraInstance(0, info_dict, config)
            >>> frame = instance.capture(100_000)
        """
        self._camera_id = camera_id
        self._info = info
        self._config = config
        # Control names match ASI driver's CONTROL_MAP keys (unprefixed)
        self._controls = {
            "Gain": {"value": 50, "auto": False},
            "Exposure": {"value": 100000, "auto": False},
            "WB_R": {"value": 52, "auto": False},
            "WB_B": {"value": 95, "auto": False},
            "Gamma": {"value": 50, "auto": False},
            "Brightness": {"value": 50, "auto": False},
            "Offset": {"value": 0, "auto": False},
            "BandWidth": {"value": 80, "auto": False},
            "Flip": {"value": 0, "auto": False},
            "HighSpeedMode": {"value": 0, "auto": False},
            "Temperature": {"value": 250, "auto": False},  # 25.0°C * 10
        }

        # For directory mode: track image index
        self._image_files: list[Path] = []
        self._image_index = 0
        self._load_image_files()

    def __enter__(self) -> DigitalTwinCameraInstance:
        """Enter context manager, returning self for capture operations.

        Enables the with-statement pattern for API compatibility with
        ASICameraInstance. Digital twin has no resources to acquire.

        Business context: Provides identical API to real cameras, enabling
        code to work with either implementation unchanged. Essential for
        testing camera workflows with simulated hardware.

        Args:
            self: Camera instance (implicit).

        Returns:
            Self for use in with-block.

        Raises:
            None. Entry always succeeds for digital twin.

        Example:
            >>> with driver.open(0) as camera:
            ...     data = camera.capture(100000)
            # close() called automatically (no-op for twin)
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing simulated camera.

        Called when exiting with-block. Delegates to close() which is a
        no-op for digital twin since there are no hardware resources.

        Business context: Maintains API compatibility with real cameras.
        Code using context managers works identically with real and
        simulated cameras.

        Args:
            exc_type: Exception type if raised in with-block, else None.
            exc_val: Exception instance if raised, else None.
            exc_tb: Exception traceback if raised, else None.

        Returns:
            None. Does not suppress exceptions.

        Raises:
            None. close() is a safe no-op.

        Example:
            >>> with driver.open(0) as camera:
            ...     camera.capture(100000)
            # __exit__ calls close() - no-op for digital twin
        """
        self.close()

    def _load_image_files(self) -> None:
        """Load list of image files for directory mode.

        Scans the configured image_path directory for supported image files
        and stores sorted paths in _image_files. Supports JPEG, PNG, TIFF,
        and FITS formats. No-op if not in DIRECTORY mode or path is invalid.

        Business context: Enables replaying real sky captures in testing.
        Developers test plate solving, star detection, stacking with actual
        astronomy data. CI/CD validates algorithms against known good images.
        Demos show realistic output without live capture.

        Args:
            None. Uses _config.image_path and _config.image_source.

        Returns:
            None. Populates _image_files list with sorted paths.

        Raises:
            None. Errors ignored - empty list triggers synthetic fallback.

        Example:
            # Automatically called during __init__:
            >>> config = DigitalTwinConfig(
            ...     image_source=ImageSource.DIRECTORY, image_path="/data/sky"
            ... )
            >>> instance = DigitalTwinCameraInstance(0, info, config)
            >>> # _load_image_files() ran, _image_files now populated
        """
        if self._config.image_source != ImageSource.DIRECTORY:
            return

        if self._config.image_path is None:
            return

        path = Path(self._config.image_path)
        if not path.is_dir():
            return

        # Find image files
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fits"}
        self._image_files = sorted(
            f for f in path.iterdir() if f.suffix.lower() in extensions
        )

    def __repr__(self) -> str:
        """Return human-readable string representation for debugging.

        Produces a concise summary of instance configuration useful in logs,
        REPL sessions, and debugging output.

        Business context: Debugging capture pipelines requires quick
        identification of which camera instance and image source is active.
        repr() shows camera ID and source at a glance.

        Args:
            self: Camera instance (implicit).

        Returns:
            String in format: DigitalTwinCameraInstance(camera_id=X, source=Y)

        Raises:
            None. Always succeeds.

        Example:
            >>> instance = driver.open(0)
            >>> print(repr(instance))
            DigitalTwinCameraInstance(camera_id=0, source=synthetic)
        """
        return (
            f"DigitalTwinCameraInstance("
            f"camera_id={self._camera_id}, "
            f"source={self._config.image_source.value})"
        )

    def get_info(self) -> dict[str, Any]:
        """Get simulated camera information from digital twin.

        Returns a copy of the camera properties dictionary containing
        resolution, color mode, pixel size, and other specifications.
        Business context: Enables testing camera-dependent logic without hardware.
        Applications can validate field-of-view calculations, plate solving setup,
        and UI camera selection using realistic simulated camera specifications.

        Args:
            None. Returns cached camera properties from digital twin config.

        Returns:
            Dictionary with camera properties (Name, MaxWidth, MaxHeight,
            IsColorCam, PixelSize, BayerPattern, etc.).

        Raises:
            None. Always returns valid dictionary.

        Example:
            >>> twin = DigitalTwinCameraInstance(...)
            >>> info = twin.get_info()
            >>> print(f"{info['Name']}: {info['MaxWidth']}x{info['MaxHeight']}")
            >>> if info['IsColorCam']:
            ...     print(f"Simulating color camera")
        """
        return dict(self._info)

    def get_controls(self) -> dict[str, dict[str, Any]]:
        """Get available simulated camera controls from digital twin.

        Returns control definitions matching the ASI SDK control format.
        Each control includes name, min/max values, default, and flags.
        Business context: Enables testing control UI and validation logic
        without hardware. Ensures control range validation, slider bounds,
        and auto-mode toggles work correctly across different camera types.

        Args:
            None. Returns simulated control definitions.

        Returns:
            Dictionary mapping control names to their definitions.
            Each definition includes Name, MinValue, MaxValue, DefaultValue,
            IsAutoSupported, and IsWritable.

        Raises:
            None. Always returns valid control definitions.

        Example:
            >>> twin = DigitalTwinCameraInstance(...)
            >>> controls = twin.get_controls()
            >>> gain = controls['Gain']
            >>> print(f"Gain range: {gain['MinValue']}-{gain['MaxValue']}")
        """
        return {
            name: {
                "Name": name,
                "MinValue": 0,
                "MaxValue": self._get_control_max(name),
                "DefaultValue": 50,
                "IsAutoSupported": name not in {"Temperature"},
                "IsWritable": name not in {"Temperature"},
            }
            for name in self._controls
        }

    def _get_control_max(self, control: str) -> int:
        """Get maximum value for a simulated control.

        Returns appropriate maximum values based on control type to match
        typical ASI camera specifications. Uses _CONTROL_MAX_VALUES dict
        for data-driven lookup.

        Business context: Provides realistic control ranges for testing
        validation logic. Ensures test code uses valid ranges that match
        actual hardware capabilities.

        Args:
            control: Control name (e.g., "Gain", "Exposure").

        Returns:
            Maximum allowed value for the control (e.g., 600 for gain,
            60000000 for exposure in microseconds).

        Raises:
            None. Returns 100 as default for unknown control types.

        Example:
            >>> twin = DigitalTwinCameraInstance(...)
            >>> max_gain = twin._get_control_max("Gain")
            >>> print(f"Max gain: {max_gain}")  # 600
        """
        control_upper = control.upper()
        for key, value in _CONTROL_MAX_VALUES.items():
            if key in control_upper:
                return value
        return 100

    def set_control(self, control: str, value: int) -> dict[str, Any]:
        """Set a simulated camera control value for digital twin testing.

        Updates the internal control state in the digital twin simulation.
        Values are stored and some affect synthetic image generation behavior.
        For example, gain affects noise level in generated images, exposure
        affects brightness simulation.

        Business context: Enables realistic testing of camera control workflows
        without physical hardware. Developers can test exposure/gain adjustment
        algorithms, auto-exposure logic, and UI controls using predictable
        simulated camera responses. Essential for CI/CD testing, offline
        development, and demonstrating telescope control features in
        presentations or training without actual cameras.

        Args:
            control: Control name (e.g., "Gain", "Exposure", "WB_R").
                Must be one of the controls defined in _controls dict.
            value: Integer value to set. Not validated against min/max ranges
                in digital twin (simulation accepts any value for flexibility).

        Returns:
            Updated control state dict with:
            - value: int - The value just set
            - auto: bool - Always False for digital twin (no auto modes simulated)

        Raises:
            ValueError: If control name is not in _controls (matches ASI behavior).

        Example:
            >>> twin = DigitalTwinCameraInstance(...)
            >>> twin.set_control("Gain", 80)
            >>> twin.set_control("Exposure", 500000)  # 500ms
            >>> result = twin.get_control("Gain")
            >>> print(result['value'])  # 80
        """
        if control not in self._controls:
            raise ValueError(
                f"Unknown control: {control}. "
                f"Valid controls: {list(self._controls.keys())}"
            )
        self._controls[control]["value"] = value
        return self.get_control(control)

    def get_control(self, control: str) -> dict[str, Any]:
        """Get a simulated control's current value from digital twin.

        Retrieves the current state of a camera control including its
        value and auto mode setting.

        Business context: Enables testing control readback logic and UI
        synchronization without hardware. Validates that control state
        management works correctly in multi-threaded environments.

        Args:
            control: Control name (e.g., "Gain", "Temperature").

        Returns:
            Dictionary with 'value' (int) and 'auto' (bool) keys.

        Raises:
            ValueError: If control name is not in _controls (matches ASI behavior).

        Example:
            >>> twin = DigitalTwinCameraInstance(...)
            >>> twin.set_control("Gain", 100)
            >>> result = twin.get_control("Gain")
            >>> print(f"Gain is now {result['value']}")
        """
        if control not in self._controls:
            raise ValueError(
                f"Unknown control: {control}. "
                f"Valid controls: {list(self._controls.keys())}"
            )
        return self._controls[control]

    def capture(self, exposure_us: int) -> bytes:
        """Capture a frame based on configured image source.

        Generates or loads an image based on the camera's ImageSource config.
        SYNTHETIC mode creates test patterns, FILE mode loads a single image,
        DIRECTORY mode cycles through images in a folder.

        Args:
            exposure_us: Exposure time in microseconds. Used by synthetic mode
                to display on the test pattern; ignored for file/directory.

        Returns:
            JPEG-encoded image bytes at the camera's native resolution.

        Raises:
            cv2.error: If image encoding fails (rare).

        Note:
            Digital twin intentionally accepts any exposure value for testing
            flexibility. Unlike ASICameraInstance, no bounds validation is
            performed - this allows testing edge cases and error handling.

        Example:
            frame = camera.capture(100_000)  # 100ms exposure
            with open("frame.jpg", "wb") as f:
                f.write(frame)
        """
        import time

        source = self._config.image_source
        capture_start = time.monotonic()

        if source == ImageSource.FILE:
            data = self._capture_from_file()
        elif source == ImageSource.DIRECTORY:
            data = self._capture_from_directory()
        else:
            data = self._capture_synthetic(exposure_us)

        capture_elapsed = time.monotonic() - capture_start

        logger.debug(
            "Capture complete",
            camera_id=self._camera_id,
            source=source.value,
            exposure_us=exposure_us,
            capture_elapsed_ms=round(capture_elapsed * 1000, 1),
            jpeg_size_kb=round(len(data) / 1024, 1),
        )
        return data

    def _capture_from_file(self) -> bytes:
        """Load and return image from configured file path.

        Reads the image file specified in config.image_path, resizes it
        to match camera resolution if needed, and returns JPEG bytes.
        Falls back to synthetic generation if file is missing or unreadable.

        Business context: Single-file mode useful for testing specific scenarios
        repeatedly. Developers debug plate solving with known star field, test
        overlay rendering with familiar frame, validate processing pipeline with
        reference image.

        Args:
            None. Uses self._config.image_path.

        Returns:
            JPEG-encoded image bytes at camera resolution.

        Raises:
            None. File errors trigger synthetic fallback.

        Example:
            # FILE mode returns same image every capture:
            >>> config = DigitalTwinConfig(
            ...     image_source=ImageSource.FILE, image_path="m31.jpg"
            ... )
            >>> instance = DigitalTwinCameraInstance(0, info, config)
            >>> frame1 = instance.capture(100_000)  # From m31.jpg
            >>> frame2 = instance.capture(200_000)  # Same m31.jpg
        """
        if self._config.image_path is None:
            return self._capture_synthetic(_DEFAULT_FALLBACK_EXPOSURE_US)

        path = Path(self._config.image_path)
        if not path.is_file():
            return self._capture_synthetic(_DEFAULT_FALLBACK_EXPOSURE_US)

        img = cv2.imread(str(path))
        if img is None:
            return self._capture_synthetic(_DEFAULT_FALLBACK_EXPOSURE_US)

        # Resize to match camera resolution if needed
        img = self._resize_to_camera(img)

        _, jpeg = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _DEFAULT_JPEG_QUALITY]
        )
        return jpeg.tobytes()

    def _capture_from_directory(self) -> bytes:
        """Load next image from directory and advance index.

        Returns the next image in the sorted file list, then advances
        the index. If cycle_images is True, loops back to start after
        the last image. Falls back to synthetic if directory is empty.

        Business context: Directory mode simulates time-series captures for
        stacking/timelapse. Developers test image stacking algorithms, validate
        frame averaging, debug timelapse generation. CI/CD tests sequences of
        real sky captures showing star motion.

        Args:
            None. Uses _image_files and _image_index state.

        Returns:
            JPEG-encoded image bytes at camera resolution.

        Raises:
            None. Empty directory or read errors trigger synthetic fallback.

        Example:
            # DIRECTORY mode cycles through images:
            >>> config = DigitalTwinConfig(
            ...     image_source=ImageSource.DIRECTORY,
            ...     image_path="/sky",
            ...     cycle_images=True,
            ... )
            >>> instance = DigitalTwinCameraInstance(0, info, config)
            >>> frame1 = instance.capture(100_000)  # sky/001.jpg
            >>> frame2 = instance.capture(100_000)  # sky/002.jpg
            >>> # ... eventually loops back to sky/001.jpg
        """
        if not self._image_files:
            return self._capture_synthetic(_DEFAULT_FALLBACK_EXPOSURE_US)

        # Get current image
        image_path = self._image_files[self._image_index]

        # Advance index
        self._image_index += 1
        if self._config.cycle_images:
            self._image_index %= len(self._image_files)
        else:
            self._image_index = min(self._image_index, len(self._image_files) - 1)

        img = cv2.imread(str(image_path))
        if img is None:
            return self._capture_synthetic(_DEFAULT_FALLBACK_EXPOSURE_US)

        # Resize to match camera resolution
        img = self._resize_to_camera(img)

        _, jpeg = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _DEFAULT_JPEG_QUALITY]
        )
        return jpeg.tobytes()

    def _resize_to_camera(self, img: NDArray[Any]) -> NDArray[Any]:
        """Resize image to match camera resolution.

        Scales the input image to the camera's MaxWidth x MaxHeight if
        dimensions don't match. Uses OpenCV resize with default interpolation.

        Business context: Enables mixing arbitrary images with simulated cameras.
        Developers use high-res DSLR captures in 1280x720 finder simulation, or
        phone photos in main camera tests. Automatic resizing means image source
        resolution doesn't need to match camera specs.

        Args:
            img: Input image as numpy array (BGR format from cv2.imread).

        Returns:
            Image resized to camera resolution, or original if already correct.

        Raises:
            None. OpenCV resize failures (e.g., corrupted array) bubble up.

        Example:
            >>> img = cv2.imread("4000x3000.jpg")  # High-res image
            >>> resized = instance._resize_to_camera(img)  # Now 1920x1080
        """
        target_width = self._info["MaxWidth"]
        target_height = self._info["MaxHeight"]

        h, w = img.shape[:2]
        if w != target_width or h != target_height:
            img = cv2.resize(img, (target_width, target_height))

        return img

    def _capture_synthetic(self, exposure_us: int) -> bytes:
        """Generate a synthetic test pattern image.

        Creates a test image with grid lines, center crosshair, diagnostic
        text overlay, and simulated noise based on gain setting. Useful for
        testing capture pipelines without real hardware.

        Business context: Default mode enabling instant testing without image
        files. Developers validate UI layouts, test overlay rendering, verify
        streaming infrastructure. CI/CD runs full integration tests with zero
        external dependencies. Demos work anywhere.

        Args:
            exposure_us: Exposure time displayed on the test pattern.

        Returns:
            JPEG-encoded image bytes at camera resolution.

        Raises:
            None. Synthetic generation always succeeds.

        Example:
            # SYNTHETIC mode (default):
            >>> config = DigitalTwinConfig()  # Defaults to ImageSource.SYNTHETIC
            >>> instance = DigitalTwinCameraInstance(0, info, config)
            >>> frame = instance.capture(100_000)  # Test pattern with exposure info

        Note:
            Noise level increases with gain setting to simulate real camera
            behavior. Text overlay shows camera ID, exposure, and gain.
        """
        width = self._info["MaxWidth"]
        height = self._info["MaxHeight"]

        # Create base image (type widened for cv2 compatibility)
        img: NDArray[Any] = np.zeros((height, width, 3), dtype=np.uint8)

        # Add grid pattern
        img[::_SYNTHETIC_GRID_SPACING, :] = [50, 50, 50]
        img[:, ::_SYNTHETIC_GRID_SPACING] = [50, 50, 50]

        # Add center crosshair
        cv2.line(img, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
        cv2.line(img, (0, height // 2), (width, height // 2), (0, 255, 0), 1)

        # Add circle at center
        cv2.circle(img, (width // 2, height // 2), _CROSSHAIR_RADIUS, (0, 100, 0), 1)

        # Add text overlay
        cv2.putText(
            img,
            f"DIGITAL TWIN - Camera {self._camera_id}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Exposure: {exposure_us}us",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            img,
            f"Gain: {self._controls['Gain']['value']}",
            (50, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
        )

        # Simulate noise based on gain
        gain = self._controls["Gain"]["value"]
        if gain > 0:
            noise_level = int(gain) // 10
            noise = np.random.randint(0, noise_level + 1, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)

        # Encode as JPEG
        _, jpeg = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _DEFAULT_JPEG_QUALITY]
        )
        return jpeg.tobytes()

    def stop_exposure(self) -> None:
        """Stop an in-progress exposure (no-op for digital twin).

        Exists for API compatibility with ASICameraInstance. Digital twin
        has no actual exposure to stop.

        Business context: Enables code to call stop_exposure() regardless
        of whether using real or simulated camera. Essential for testing
        exposure cancellation workflows.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Safe to call at any time.

        Example:
            >>> twin.stop_exposure()  # No-op, always succeeds
        """
        logger.debug(f"stop_exposure called on digital twin camera {self._camera_id}")

    def close(self) -> None:
        """Close the simulated camera connection.

        No-op for digital twin since there's no actual hardware to release.
        Exists for API compatibility with real camera drivers, ensuring
        test code and production code can use identical patterns.

        Business context: Maintains consistent resource management patterns
        across hardware and simulated cameras. Enables test code to use
        try/finally blocks and context managers without modification when
        switching between real and simulated hardware.

        Implementation: Empty function that does nothing. Safe to call multiple
        times. No state changes occur. Provided solely for API compatibility.

        Args:
            None. No resources to clean up in simulation.

        Returns:
            None.

        Raises:
            None. Never fails.

        Example:
            >>> twin = driver.open(0)
            >>> try:
            ...     twin.capture(100000)
            ... finally:
            ...     twin.close()  # Safe no-op for digital twin
        """
        pass


# Convenience function for creating pre-configured twins
def create_file_camera(image_path: Path | str) -> DigitalTwinCameraDriver:
    """Create a digital twin camera that returns a single image.

    Convenience factory for creating a camera that always returns the same
    image. Useful for testing with a known reference image.

    Args:
        image_path: Path to the image file (JPEG, PNG, TIFF, or FITS).

    Returns:
        Configured DigitalTwinCameraDriver in FILE mode.

    Raises:
        None. Path validation deferred to capture time; missing files
        trigger synthetic fallback instead of raising exceptions.

    Example:
        driver = create_file_camera("/data/reference.jpg")
        camera = driver.open(0)
        frame = camera.capture(100_000)  # Always returns reference.jpg
    """
    config = DigitalTwinConfig(
        image_source=ImageSource.FILE,
        image_path=Path(image_path),
    )
    return DigitalTwinCameraDriver(config=config)


def create_directory_camera(
    image_dir: Path | str,
    cycle: bool = True,
) -> DigitalTwinCameraDriver:
    """Create a digital twin camera that cycles through images in a directory.

    Convenience factory for creating a camera that returns images from a
    folder in sorted order. Useful for testing with a sequence of frames
    or simulating time-lapse capture.

    Args:
        image_dir: Path to directory containing images.
        cycle: Whether to loop back to start after last image (default True).

    Returns:
        Configured DigitalTwinCameraDriver in DIRECTORY mode.

    Raises:
        None. Path validation deferred to capture time; missing directories
        trigger synthetic fallback instead of raising exceptions.

    Example:
        driver = create_directory_camera("/data/test_frames/", cycle=True)
        camera = driver.open(0)
        for i in range(10):
            frame = camera.capture(100_000)  # Cycles through directory
    """
    config = DigitalTwinConfig(
        image_source=ImageSource.DIRECTORY,
        image_path=Path(image_dir),
        cycle_images=cycle,
    )
    return DigitalTwinCameraDriver(config=config)
