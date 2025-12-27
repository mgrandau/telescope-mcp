"""Digital twin camera driver for testing without hardware.

Provides simulated camera responses for development and testing.
Supports multiple image sources:
- Synthetic: Generate test patterns (default)
- Directory: Cycle through images in a folder
- File: Return same image repeatedly
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class ImageSource(Enum):
    """Image source for digital twin camera."""

    SYNTHETIC = "synthetic"  # Generate test patterns
    DIRECTORY = "directory"  # Cycle through images in a folder
    FILE = "file"  # Return same image repeatedly


@dataclass
class DigitalTwinConfig:
    """Configuration for digital twin camera behavior."""

    image_source: ImageSource = ImageSource.SYNTHETIC
    image_path: Path | None = None  # Directory or file path
    cycle_images: bool = True  # Loop through directory images


# Default camera specifications (matching real ASI cameras)
# Camera 0: ASI120MC-S with 150° All-Sky Lens (finder/spotter scope)
# Camera 1: ASI482MC through telescope optics (main imaging camera)
#
# Specifications from astrophotography_camera_exposure_times.ipynb:
# - ASI120MC-S: 1.2MP, 1280x960, 3.75µm pixels, 4.8x3.6mm sensor
# - ASI482MC: 2.07MP, 1920x1080, 5.8µm pixels, 11.13x6.26mm sensor
DEFAULT_CAMERAS: dict[int, dict] = {
    0: {
        # Finder camera: ASI120MC-S with 150° All-Sky Lens
        "Name": b"ASI120MC-S (Finder - 150deg All-Sky)",
        "MaxWidth": 1280,
        "MaxHeight": 960,
        "IsColorCam": True,
        "PixelSize": 3.75,  # micrometers
        "SensorWidth": 4.8,  # mm
        "SensorHeight": 3.6,  # mm
        "BayerPattern": "RGGB",
        "BitDepth": 8,
        "ElecPerADU": 0.21,  # electrons per ADU
        "USB3Host": True,
        # All-sky lens specific
        "LensFOV": 150,  # degrees (150° all-sky lens)
        "FOVPerPixel": 421.875,  # arcseconds (150*3600/1280)
        "Purpose": "finder",
    },
    1: {
        # Main camera: ASI482MC through 1600mm telescope optics
        "Name": b"ASI482MC (Main - Through Optics)",
        "MaxWidth": 1920,
        "MaxHeight": 1080,
        "IsColorCam": True,
        "PixelSize": 5.8,  # micrometers
        "SensorWidth": 11.13,  # mm
        "SensorHeight": 6.26,  # mm
        "BayerPattern": "RGGB",
        "BitDepth": 12,  # 12-bit ADC
        "ElecPerADU": 0.16,
        "USB3Host": True,
        # Telescope optics specific
        "FocalLength": 1600,  # mm (telescope focal length)
        "FOVPerPixel": 0.748,  # arcseconds (calculated from focal length)
        "FOVWidth": 23.9,  # arcminutes
        "FOVHeight": 13.4,  # arcminutes
        "Purpose": "main",
    },
}


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

    def __init__(
        self,
        config: DigitalTwinConfig | None = None,
        cameras: dict[int, dict] | None = None,
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
                0: {"Name": b"Test Camera", "MaxWidth": 640, "MaxHeight": 480}
            }
            driver = DigitalTwinCameraDriver(cameras=cameras)
        """
        self.config = config or DigitalTwinConfig()
        self._cameras = cameras or DEFAULT_CAMERAS.copy()
        logger.info(
            "Digital twin camera driver initialized",
            image_source=self.config.image_source.value,
            num_cameras=len(self._cameras),
        )

    def get_connected_cameras(self) -> dict:
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


class DigitalTwinCameraInstance:
    """Digital twin camera instance for simulated capture operations.

    Represents an open connection to a simulated camera. Provides the same
    interface as ASICameraInstance for capture, control get/set, and info
    queries. Images can come from synthetic generation, a file, or directory.
    """

    def __init__(
        self,
        camera_id: int,
        info: dict,
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
        self._controls = {
            "ASI_GAIN": {"value": 50, "auto": False},
            "ASI_EXPOSURE": {"value": 100000, "auto": False},
            "ASI_WB_R": {"value": 52, "auto": False},
            "ASI_WB_B": {"value": 95, "auto": False},
            "ASI_GAMMA": {"value": 50, "auto": False},
            "ASI_BRIGHTNESS": {"value": 50, "auto": False},
            "ASI_OFFSET": {"value": 0, "auto": False},
            "ASI_BANDWIDTHOVERLOAD": {"value": 80, "auto": False},
            "ASI_FLIP": {"value": 0, "auto": False},
            "ASI_HIGH_SPEED_MODE": {"value": 0, "auto": False},
            "ASI_TEMPERATURE": {"value": 250, "auto": False},  # 25.0°C * 10
        }

        # For directory mode: track image index
        self._image_files: list[Path] = []
        self._image_index = 0
        self._load_image_files()

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

    def get_info(self) -> dict:
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
        return self._info.copy()

    def get_controls(self) -> dict:
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
            >>> gain = controls['ASI_GAIN']
            >>> print(f"Gain range: {gain['MinValue']}-{gain['MaxValue']}")
        """
        return {
            name: {
                "Name": name,
                "MinValue": 0,
                "MaxValue": self._get_control_max(name),
                "DefaultValue": 50,
                "IsAutoSupported": name not in {"ASI_TEMPERATURE"},
                "IsWritable": name not in {"ASI_TEMPERATURE"},
            }
            for name in self._controls
        }

    def _get_control_max(self, control: str) -> int:
        """Get maximum value for a simulated control.

        Returns appropriate maximum values based on control type to match
        typical ASI camera specifications.
        Business context: Provides realistic control ranges for testing
        validation logic. Ensures test code uses valid ranges that match
        actual hardware capabilities.

        Args:
            control: Control name (e.g., "ASI_GAIN", "ASI_EXPOSURE").

        Returns:
            Maximum allowed value for the control (e.g., 600 for gain,
            60000000 for exposure in microseconds).

        Raises:
            None. Returns sensible defaults for all control types.

        Example:
            >>> twin = DigitalTwinCameraInstance(...)
            >>> max_gain = twin._get_control_max("ASI_GAIN")
            >>> print(f"Max gain: {max_gain}")  # 600
        """
        if "GAIN" in control:
            return 600
        if "EXPOSURE" in control:
            return 60_000_000  # 60 seconds in microseconds
        if "WB" in control:
            return 99
        if "GAMMA" in control:
            return 100
        if "TEMPERATURE" in control:
            return 1000  # 100.0°C
        return 100

    def set_control(self, control: str, value: int) -> dict:
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
            control: Control name (e.g., "ASI_GAIN", "ASI_EXPOSURE", "ASI_WB_R").
                Must be one of the controls defined in _controls dict. Unknown
                controls are ignored (no error).
            value: Integer value to set. Not validated against min/max ranges
                in digital twin (simulation accepts any value for flexibility).

        Returns:
            Updated control state dict with:
            - value: int - The value just set
            - auto: bool - Always False for digital twin (no auto modes simulated)

        Raises:
            None. Unknown controls are silently ignored for compatibility.

        Example:
            >>> twin = DigitalTwinCameraInstance(...)
            >>> twin.set_control("ASI_GAIN", 80)
            >>> twin.set_control("ASI_EXPOSURE", 500000)  # 500ms
            >>> result = twin.get_control("ASI_GAIN")
            >>> print(result['value'])  # 80
        """
        if control in self._controls:
            self._controls[control]["value"] = value
        return self.get_control(control)

    def get_control(self, control: str) -> dict:
        """Get a simulated control's current value from digital twin.

        Retrieves the current state of a camera control including its
        value and auto mode setting.
        Business context: Enables testing control readback logic and UI
        synchronization without hardware. Validates that control state
        management works correctly in multi-threaded environments.

        Args:
            control: Control name (e.g., "ASI_GAIN", "ASI_TEMPERATURE").

        Returns:
            Dictionary with 'value' (int) and 'auto' (bool) keys.
            Returns {'value': 0, 'auto': False} for unknown controls.

        Raises:
            None. Returns default values for unknown controls.

        Example:
            >>> twin = DigitalTwinCameraInstance(...)
            >>> twin.set_control("ASI_GAIN", 100)
            >>> result = twin.get_control("ASI_GAIN")
            >>> print(f"Gain is now {result['value']}")
        """
        return self._controls.get(control, {"value": 0, "auto": False})

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

        Example:
            frame = camera.capture(100_000)  # 100ms exposure
            with open("frame.jpg", "wb") as f:
                f.write(frame)
        """
        source = self._config.image_source
        logger.debug(
            "Capturing frame",
            camera_id=self._camera_id,
            source=source.value,
            exposure_us=exposure_us,
        )

        if source == ImageSource.FILE:
            data = self._capture_from_file()
        elif source == ImageSource.DIRECTORY:
            data = self._capture_from_directory()
        else:
            data = self._capture_synthetic(exposure_us)

        logger.debug(
            "Frame captured",
            camera_id=self._camera_id,
            size_bytes=len(data),
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
            return self._capture_synthetic(100000)

        path = Path(self._config.image_path)
        if not path.is_file():
            return self._capture_synthetic(100000)

        img = cv2.imread(str(path))
        if img is None:
            return self._capture_synthetic(100000)

        # Resize to match camera resolution if needed
        img = self._resize_to_camera(img)

        _, jpeg = cv2.imencode(".jpg", img)
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
            return self._capture_synthetic(100000)

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
            return self._capture_synthetic(100000)

        # Resize to match camera resolution
        img = self._resize_to_camera(img)

        _, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()

    def _resize_to_camera(self, img: NDArray) -> NDArray:
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

        # Create base image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Add grid pattern
        grid_spacing = 50
        img[::grid_spacing, :] = [50, 50, 50]
        img[:, ::grid_spacing] = [50, 50, 50]

        # Add center crosshair
        cv2.line(img, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
        cv2.line(img, (0, height // 2), (width, height // 2), (0, 255, 0), 1)

        # Add circle at center
        cv2.circle(img, (width // 2, height // 2), 100, (0, 100, 0), 1)

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
            f"Gain: {self._controls['ASI_GAIN']['value']}",
            (50, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
        )

        # Simulate noise based on gain
        gain = self._controls["ASI_GAIN"]["value"]
        if gain > 0:
            noise_level = int(gain) // 10
            noise = np.random.randint(0, noise_level + 1, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)

        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()

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
def create_file_camera(
    image_path: Path | str, camera_id: int = 0
) -> DigitalTwinCameraDriver:
    """Create a digital twin camera that returns a single image.

    Convenience factory for creating a camera that always returns the same
    image. Useful for testing with a known reference image.

    Args:
        image_path: Path to the image file (JPEG, PNG, TIFF, or FITS).
        camera_id: Which camera ID to simulate (default 0 for finder).

    Returns:
        Configured DigitalTwinCameraDriver in FILE mode.

    Raises:
        FileNotFoundError: If image_path does not exist (on first capture).

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
    camera_id: int = 0,
    cycle: bool = True,
) -> DigitalTwinCameraDriver:
    """Create a digital twin camera that cycles through images in a directory.

    Convenience factory for creating a camera that returns images from a
    folder in sorted order. Useful for testing with a sequence of frames
    or simulating time-lapse capture.

    Args:
        image_dir: Path to directory containing images.
        camera_id: Which camera ID to simulate (default 0 for finder).
        cycle: Whether to loop back to start after last image (default True).

    Returns:
        Configured DigitalTwinCameraDriver in DIRECTORY mode.

    Raises:
        FileNotFoundError: If image_dir does not exist (on driver creation).

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
