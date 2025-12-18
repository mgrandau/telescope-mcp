"""Digital twin camera driver for testing without hardware.

Provides simulated camera responses for development and testing.
Supports multiple image sources:
- Synthetic: Generate test patterns (default)
- Directory: Cycle through images in a folder
- File: Return same image repeatedly
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Digital twin camera driver for development without hardware."""

    def __init__(
        self,
        config: DigitalTwinConfig | None = None,
        cameras: dict[int, dict] | None = None,
    ) -> None:
        """Initialize digital twin camera driver.

        Args:
            config: Configuration for image source behavior
            cameras: Custom camera definitions (defaults to ASI120MC-S and ASI482MC)
        """
        self.config = config or DigitalTwinConfig()
        self._cameras = cameras or DEFAULT_CAMERAS.copy()
        logger.info(
            "Digital twin camera driver initialized",
            image_source=self.config.image_source.value,
            num_cameras=len(self._cameras),
        )

    def get_connected_cameras(self) -> dict:
        """Return simulated camera list."""
        logger.debug("Listing simulated cameras", count=len(self._cameras))
        return self._cameras.copy()

    def open(self, camera_id: int) -> "DigitalTwinCameraInstance":
        """Open a simulated camera."""
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
    """Digital twin camera instance."""

    def __init__(
        self,
        camera_id: int,
        info: dict,
        config: DigitalTwinConfig,
    ) -> None:
        """Initialize digital twin camera instance.

        Args:
            camera_id: Camera identifier
            info: Camera properties
            config: Image source configuration
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
        """Load list of image files for directory mode."""
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
        """Get camera info."""
        return self._info.copy()

    def get_controls(self) -> dict:
        """Get available controls."""
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
        """Get max value for a control."""
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
        """Set a control value."""
        if control in self._controls:
            self._controls[control]["value"] = value
        return self.get_control(control)

    def get_control(self, control: str) -> dict:
        """Get a control value."""
        return self._controls.get(control, {"value": 0, "auto": False})

    def capture(self, exposure_us: int) -> bytes:
        """Capture a frame based on configured image source.

        Args:
            exposure_us: Exposure time in microseconds (used for synthetic)

        Returns:
            JPEG-encoded image bytes
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
        """Load image from a single file."""
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
        """Load next image from directory."""
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

    def _resize_to_camera(self, img: "NDArray") -> "NDArray":
        """Resize image to match camera resolution."""
        target_width = self._info["MaxWidth"]
        target_height = self._info["MaxHeight"]

        h, w = img.shape[:2]
        if w != target_width or h != target_height:
            img = cv2.resize(img, (target_width, target_height))

        return img

    def _capture_synthetic(self, exposure_us: int) -> bytes:
        """Generate a synthetic test pattern."""
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
            noise_level = gain // 10
            noise = np.random.randint(0, noise_level + 1, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)

        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()

    def close(self) -> None:
        """Close the camera (no-op for stub)."""
        pass


# Convenience function for creating pre-configured twins
def create_file_camera(image_path: Path | str, camera_id: int = 0) -> DigitalTwinCameraDriver:
    """Create a digital twin camera that returns a single image.

    Args:
        image_path: Path to the image file
        camera_id: Which camera ID to simulate

    Returns:
        Configured DigitalTwinCameraDriver
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

    Args:
        image_dir: Path to directory containing images
        camera_id: Which camera ID to simulate
        cycle: Whether to loop back to start after last image

    Returns:
        Configured DigitalTwinCameraDriver
    """
    config = DigitalTwinConfig(
        image_source=ImageSource.DIRECTORY,
        image_path=Path(image_dir),
        cycle_images=cycle,
    )
    return DigitalTwinCameraDriver(config=config)
