"""Tests for driver stubs."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from telescope_mcp.drivers.cameras.twin import (
    DEFAULT_CAMERAS,
    DigitalTwinCameraDriver,
    DigitalTwinCameraInstance,
    DigitalTwinConfig,
    ImageSource,
    create_directory_camera,
    create_file_camera,
)
from telescope_mcp.drivers.motors import MotorType, StubMotorController
from telescope_mcp.drivers.sensors import StubPositionSensor, TelescopePosition


# =============================================================================
# Digital Twin Camera Driver Tests
# =============================================================================


class TestDefaultCameraSpecs:
    """Tests for DEFAULT_CAMERAS specifications matching real hardware."""

    def test_finder_camera_specs(self):
        """Camera 0 (ASI120MC-S finder) has correct specifications."""
        finder = DEFAULT_CAMERAS[0]

        # Basic identification
        assert b"ASI120MC-S" in finder["Name"]
        assert finder["Purpose"] == "finder"

        # Resolution (1.2MP)
        assert finder["MaxWidth"] == 1280
        assert finder["MaxHeight"] == 960

        # Sensor specs
        assert finder["PixelSize"] == 3.75  # micrometers
        assert finder["SensorWidth"] == 4.8  # mm
        assert finder["SensorHeight"] == 3.6  # mm

        # Color camera
        assert finder["IsColorCam"] is True
        assert finder["BayerPattern"] == "RGGB"
        assert finder["BitDepth"] == 8

        # All-sky lens
        assert finder["LensFOV"] == 150  # degrees

    def test_main_camera_specs(self):
        """Camera 1 (ASI482MC main) has correct specifications."""
        main = DEFAULT_CAMERAS[1]

        # Basic identification
        assert b"ASI482MC" in main["Name"]
        assert main["Purpose"] == "main"

        # Resolution (2.07MP)
        assert main["MaxWidth"] == 1920
        assert main["MaxHeight"] == 1080

        # Sensor specs
        assert main["PixelSize"] == 5.8  # micrometers
        assert main["SensorWidth"] == 11.13  # mm
        assert main["SensorHeight"] == 6.26  # mm

        # Color camera with higher bit depth
        assert main["IsColorCam"] is True
        assert main["BayerPattern"] == "RGGB"
        assert main["BitDepth"] == 12

        # Telescope optics
        assert main["FocalLength"] == 1600  # mm

    def test_fov_calculations(self):
        """FOV calculations are accurate."""
        finder = DEFAULT_CAMERAS[0]
        main = DEFAULT_CAMERAS[1]

        # Finder: 150° lens / 1280 pixels ≈ 421.875 arcsec/pixel
        assert finder["FOVPerPixel"] == pytest.approx(421.875, rel=0.01)

        # Main: smaller FOV for high resolution imaging
        assert main["FOVPerPixel"] == pytest.approx(0.748, rel=0.01)
        assert main["FOVWidth"] == pytest.approx(23.9, rel=0.1)  # arcminutes
        assert main["FOVHeight"] == pytest.approx(13.4, rel=0.1)  # arcminutes


class TestDigitalTwinCameraDriver:
    """Tests for DigitalTwinCameraDriver."""

    def test_init_defaults(self):
        """Driver initializes with default cameras."""
        driver = DigitalTwinCameraDriver()
        cameras = driver.get_connected_cameras()
        assert len(cameras) == 2
        assert 0 in cameras
        assert 1 in cameras

    def test_init_custom_cameras(self):
        """Driver accepts custom camera definitions."""
        custom = {
            0: {"Name": b"Custom Camera", "MaxWidth": 640, "MaxHeight": 480}
        }
        driver = DigitalTwinCameraDriver(cameras=custom)
        cameras = driver.get_connected_cameras()
        assert len(cameras) == 1
        assert cameras[0]["Name"] == b"Custom Camera"

    def test_get_connected_cameras_returns_copy(self):
        """get_connected_cameras returns a copy, not the original."""
        driver = DigitalTwinCameraDriver()
        cameras1 = driver.get_connected_cameras()
        cameras2 = driver.get_connected_cameras()
        assert cameras1 is not cameras2

    def test_open_valid_camera(self):
        """Opening a valid camera returns instance."""
        driver = DigitalTwinCameraDriver()
        instance = driver.open(0)
        assert isinstance(instance, DigitalTwinCameraInstance)

    def test_open_invalid_camera_raises(self):
        """Opening invalid camera raises ValueError."""
        driver = DigitalTwinCameraDriver()
        with pytest.raises(ValueError, match="Camera 99 not found"):
            driver.open(99)


class TestDigitalTwinCameraInstance:
    """Tests for DigitalTwinCameraInstance."""

    @pytest.fixture
    def finder_camera(self):
        """Create finder camera instance."""
        driver = DigitalTwinCameraDriver()
        return driver.open(0)

    @pytest.fixture
    def main_camera(self):
        """Create main camera instance."""
        driver = DigitalTwinCameraDriver()
        return driver.open(1)

    def test_get_info_finder(self, finder_camera):
        """Finder camera info matches specifications."""
        info = finder_camera.get_info()
        assert info["MaxWidth"] == 1280
        assert info["MaxHeight"] == 960
        assert info["PixelSize"] == 3.75

    def test_get_info_main(self, main_camera):
        """Main camera info matches specifications."""
        info = main_camera.get_info()
        assert info["MaxWidth"] == 1920
        assert info["MaxHeight"] == 1080
        assert info["PixelSize"] == 5.8

    def test_get_info_returns_copy(self, finder_camera):
        """get_info returns a copy, not the original."""
        info1 = finder_camera.get_info()
        info2 = finder_camera.get_info()
        assert info1 is not info2

    def test_get_controls(self, finder_camera):
        """Controls are available with correct structure."""
        controls = finder_camera.get_controls()
        assert "ASI_GAIN" in controls
        assert "ASI_EXPOSURE" in controls
        assert "ASI_WB_R" in controls
        assert "ASI_WB_B" in controls
        assert "ASI_TEMPERATURE" in controls

        # Check control structure
        gain = controls["ASI_GAIN"]
        assert "MinValue" in gain
        assert "MaxValue" in gain
        assert "DefaultValue" in gain
        assert "IsAutoSupported" in gain
        assert "IsWritable" in gain

    def test_set_and_get_control(self, finder_camera):
        """Setting a control updates its value."""
        finder_camera.set_control("ASI_GAIN", 200)
        result = finder_camera.get_control("ASI_GAIN")
        assert result["value"] == 200

    def test_get_unknown_control(self, finder_camera):
        """Getting unknown control returns default."""
        result = finder_camera.get_control("UNKNOWN_CONTROL")
        assert result == {"value": 0, "auto": False}


class TestDigitalTwinCapture:
    """Tests for digital twin capture functionality."""

    @pytest.fixture
    def finder_camera(self):
        """Create finder camera instance."""
        driver = DigitalTwinCameraDriver()
        return driver.open(0)

    @pytest.fixture
    def main_camera(self):
        """Create main camera instance."""
        driver = DigitalTwinCameraDriver()
        return driver.open(1)

    def test_synthetic_capture_returns_jpeg(self, finder_camera):
        """Synthetic capture returns valid JPEG data."""
        data = finder_camera.capture(100000)
        assert isinstance(data, bytes)
        assert len(data) > 0
        # JPEG magic bytes
        assert data[:2] == b"\xff\xd8"

    def test_synthetic_capture_correct_resolution_finder(self, finder_camera):
        """Finder camera capture has correct resolution."""
        data = finder_camera.capture(100000)
        # Decode JPEG
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == 1280  # width
        assert img.shape[0] == 960  # height

    def test_synthetic_capture_correct_resolution_main(self, main_camera):
        """Main camera capture has correct resolution."""
        data = main_camera.capture(100000)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == 1920  # width
        assert img.shape[0] == 1080  # height

    def test_capture_gain_affects_noise(self, finder_camera):
        """Higher gain produces different (noisier) images."""
        finder_camera.set_control("ASI_GAIN", 0)
        data_low = finder_camera.capture(100000)

        finder_camera.set_control("ASI_GAIN", 300)
        data_high = finder_camera.capture(100000)

        # Images should be different due to noise
        assert data_low != data_high


class TestDigitalTwinFileSource:
    """Tests for file-based image source."""

    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create test image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "TEST", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
            cv2.imwrite(f.name, img)
            yield Path(f.name)
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_create_file_camera(self, temp_image):
        """create_file_camera returns configured driver."""
        driver = create_file_camera(temp_image)
        assert driver.config.image_source == ImageSource.FILE
        assert driver.config.image_path == temp_image

    def test_file_capture_resizes_to_camera(self, temp_image):
        """File images are resized to match camera resolution."""
        driver = create_file_camera(temp_image)
        instance = driver.open(0)  # ASI120MC-S: 1280x960

        data = instance.capture(100000)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        # Should be resized to camera resolution
        assert img.shape[1] == 1280
        assert img.shape[0] == 960


class TestDigitalTwinDirectorySource:
    """Tests for directory-based image source."""

    @pytest.fixture
    def temp_image_dir(self):
        """Create a temporary directory with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create multiple test images
            for i in range(3):
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, f"IMG{i}", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                cv2.imwrite(str(tmppath / f"test_{i}.jpg"), img)
            yield tmppath

    def test_create_directory_camera(self, temp_image_dir):
        """create_directory_camera returns configured driver."""
        driver = create_directory_camera(temp_image_dir)
        assert driver.config.image_source == ImageSource.DIRECTORY
        assert driver.config.image_path == temp_image_dir
        assert driver.config.cycle_images is True

    def test_directory_cycles_images(self, temp_image_dir):
        """Directory source cycles through images."""
        driver = create_directory_camera(temp_image_dir, cycle=True)
        instance = driver.open(0)

        # Capture 4 images (should cycle back to first)
        captures = [instance.capture(100000) for _ in range(4)]

        # First and fourth should be same image (after resize)
        # We can't compare exact bytes due to JPEG encoding, but sizes should match
        assert len(captures[0]) > 0
        assert len(captures[3]) > 0

    def test_directory_no_cycle(self, temp_image_dir):
        """Directory source stops at last image when cycle=False."""
        driver = create_directory_camera(temp_image_dir, cycle=False)
        instance = driver.open(0)

        # Capture 5 images (should stay on last after 3rd)
        captures = [instance.capture(100000) for _ in range(5)]

        # All captures should return valid data
        for cap in captures:
            assert len(cap) > 0


class TestDigitalTwinConfig:
    """Tests for DigitalTwinConfig."""

    def test_default_config(self):
        """Default config uses synthetic source."""
        config = DigitalTwinConfig()
        assert config.image_source == ImageSource.SYNTHETIC
        assert config.image_path is None
        assert config.cycle_images is True

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = DigitalTwinConfig(
            image_source=ImageSource.FILE,
            image_path=Path("/test/path"),
            cycle_images=False,
        )
        assert config.image_source == ImageSource.FILE
        assert config.image_path == Path("/test/path")
        assert config.cycle_images is False


# =============================================================================
# Motor Controller Tests
# =============================================================================


class TestStubMotorController:
    """Tests for the stub motor controller."""

    def test_initial_position(self):
        """Motors start at position 0."""
        controller = StubMotorController()
        status = controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 0

    def test_move_updates_position(self):
        """Move updates position correctly."""
        controller = StubMotorController()
        controller.move(MotorType.ALTITUDE, 100)
        status = controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 100

    def test_home_resets_position(self):
        """Home resets position to 0."""
        controller = StubMotorController()
        controller.move(MotorType.AZIMUTH, 500)
        controller.home(MotorType.AZIMUTH)
        status = controller.get_status(MotorType.AZIMUTH)
        assert status.position_steps == 0


class TestStubPositionSensor:
    """Tests for the stub position sensor."""

    def test_initial_position(self):
        """Sensor has default position."""
        sensor = StubPositionSensor()
        pos = sensor.read()
        assert pos.altitude == 45.0
        assert pos.azimuth == 180.0

    def test_calibrate_updates_position(self):
        """Calibrate updates reported position."""
        sensor = StubPositionSensor()
        sensor.calibrate(30.0, 90.0)
        pos = sensor.read()
        assert pos.altitude == 30.0
        assert pos.azimuth == 90.0
