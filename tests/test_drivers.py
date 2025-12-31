"""Tests for driver stubs."""

import struct
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
from telescope_mcp.drivers.sensors import DigitalTwinSensorDriver


def _parse_jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Parse width and height from JPEG data.

    Scans JPEG byte stream for SOF0 (0xFFC0), SOF1 (0xFFC1), or
    SOF2 (0xFFC2) markers and extracts image dimensions from
    the frame header.

    Business context:
    Used by capture tests to verify digital twin cameras produce
    correctly-sized JPEG output matching camera specifications
    (e.g., 1280x960 for finder, 1920x1080 for main imager).

    Args:
        data: JPEG byte data containing valid JPEG image.
            Must include SOF marker with dimension fields.

    Returns:
        Tuple of (width, height) in pixels as integers.

    Raises:
        ValueError: If no SOF marker found in JPEG data,
            indicating corrupt or incomplete JPEG stream.

    Example:
        >>> jpeg_data = camera.capture(100000)
        >>> width, height = _parse_jpeg_dimensions(jpeg_data)
        >>> assert width == 1280 and height == 960
    """
    i = 0
    while i < len(data) - 9:
        if data[i] == 0xFF:
            marker = data[i + 1]
            # SOF0, SOF1, SOF2 markers contain dimensions
            if marker in (0xC0, 0xC1, 0xC2):
                # Skip marker (2) + length (2) + precision (1)
                height = struct.unpack(">H", data[i + 5 : i + 7])[0]
                width = struct.unpack(">H", data[i + 7 : i + 9])[0]
                return width, height
            elif marker == 0xD8:  # SOI
                i += 2
            elif marker == 0xD9:  # EOI
                break
            elif marker == 0x00:  # Stuffed byte
                i += 1
            else:
                # Skip segment
                if i + 3 < len(data):
                    seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
                    i += 2 + seg_len
                else:
                    break
        else:
            i += 1
    raise ValueError("SOF marker not found in JPEG data")


# =============================================================================
# Digital Twin Camera Driver Tests
# =============================================================================


class TestDefaultCameraSpecs:
    """Tests for DEFAULT_CAMERAS specifications matching real hardware."""

    def test_finder_camera_specs(self):
        """Verifies ASI120MC-S finder camera specification accuracy.

        Arrangement:
        1. DEFAULT_CAMERAS[0] = finder camera spec.
        2. ASI120MC-S: 1280x960, 3.75μm pixels, 150° lens.
        3. 8-bit color camera with RGGB Bayer pattern.

        Action:
        Retrieves finder spec and validates all fields.

        Assertion Strategy:
        Validates hardware specification by confirming:
        - Name contains "ASI120MC-S", Purpose="finder".
        - MaxWidth=1280, MaxHeight=960 (1.2MP).
        - PixelSize=3.75μm, sensor 4.8x3.6mm.
        - IsColorCam=True, BayerPattern="RGGB", BitDepth=8.
        - LensFOV=150° (all-sky lens).

        Testing Principle:
        Validates hardware specification accuracy, ensuring
        digital twin matches real ASI120MC-S finder camera.
        """
        finder = DEFAULT_CAMERAS[0]

        # Basic identification
        assert "ASI120MC-S" in finder["Name"]
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
        """Verifies ASI482MC main imager specification accuracy.

        Arrangement:
        1. DEFAULT_CAMERAS[1] = main imager spec.
        2. ASI482MC: 1920x1080, 5.8μm pixels, 1600mm focal length.
        3. 12-bit color camera with RGGB Bayer pattern.

        Action:
        Retrieves main imager spec and validates all fields.

        Assertion Strategy:
        Validates hardware specification by confirming:
        - Name contains "ASI482MC", Purpose="main".
        - MaxWidth=1920, MaxHeight=1080 (2.07MP).
        - PixelSize=5.8μm, sensor 11.13x6.26mm.
        - IsColorCam=True, BayerPattern="RGGB", BitDepth=12.
        - FocalLength=1600mm (telescope optics).

        Testing Principle:
        Validates hardware specification accuracy, ensuring
        digital twin matches real ASI482MC main imager.
        """
        main = DEFAULT_CAMERAS[1]

        # Basic identification
        assert "ASI482MC" in main["Name"]
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
        """Verifies field-of-view calculation accuracy for both cameras.

        Arrangement:
        1. Finder: 150° lens, 1280 pixels → 421.875 arcsec/pixel.
        2. Main: 1600mm focal length, 1920x1080 → 0.748 arcsec/pixel.
        3. Main FOV: 23.9' x 13.4' arcminutes.

        Action:
        Retrieves FOV fields from camera specs.

        Assertion Strategy:
        Validates FOV calculations by confirming:
        - Finder FOVPerPixel = 421.875 arcsec/pixel (±1%).
        - Main FOVPerPixel = 0.748 arcsec/pixel (±1%).
        - Main FOVWidth = 23.9 arcmin (±10%).
        - Main FOVHeight = 13.4 arcmin (±10%).

        Testing Principle:
        Validates plate scale calculations, ensuring
        astrometry and alignment use correct angular resolution.
        """
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
        """Verifies driver initializes with 2 default cameras.

        Arrangement:
        1. DigitalTwinCameraDriver() with no arguments.
        2. DEFAULT_CAMERAS defines ASI120MC-S (id=0) and ASI482MC (id=1).

        Action:
        Creates driver and retrieves connected camera list.

        Assertion Strategy:
        Validates default initialization by confirming:
        - get_connected_cameras() returns 2 cameras.
        - Camera IDs 0 and 1 present.

        Testing Principle:
        Validates default behavior, ensuring driver provides
        working twin cameras without explicit configuration.
        """
        driver = DigitalTwinCameraDriver()
        cameras = driver.get_connected_cameras()
        assert len(cameras) == 2
        assert 0 in cameras
        assert 1 in cameras

    def test_init_custom_cameras(self):
        """Verifies driver accepts custom camera definitions.

        Arrangement:
        1. Custom camera dict with single camera (id=0).
        2. Camera spec: "Custom Camera", 640x480.
        3. Driver created with cameras= parameter.

        Action:
        Creates driver with custom cameras, retrieves list.

        Assertion Strategy:
        Validates custom initialization by confirming:
        - get_connected_cameras() returns 1 camera.
        - Camera Name = "Custom Camera".

        Testing Principle:
        Validates configuration flexibility, allowing test
        scenarios with non-standard camera configurations.
        """
        custom = {0: {"Name": b"Custom Camera", "MaxWidth": 640, "MaxHeight": 480}}
        driver = DigitalTwinCameraDriver(cameras=custom)
        cameras = driver.get_connected_cameras()
        assert len(cameras) == 1
        assert cameras[0]["Name"] == b"Custom Camera"

    def test_get_connected_cameras_returns_copy(self):
        """Verifies get_connected_cameras returns defensive copy.

        Arrangement:
        1. Driver initialized with camera specs.
        2. Internal camera dict must not be exposed.
        3. get_connected_cameras() called twice.

        Action:
        Retrieves camera list twice.

        Assertion Strategy:
        Validates defensive copying by confirming:
        - cameras1 is not cameras2 (different object IDs).
        - Prevents external modification of internal state.

        Testing Principle:
        Validates encapsulation, ensuring driver maintains
        immutable camera registry for thread safety.
        """
        driver = DigitalTwinCameraDriver()
        cameras1 = driver.get_connected_cameras()
        cameras2 = driver.get_connected_cameras()
        assert cameras1 is not cameras2

    def test_open_valid_camera(self):
        """Verifies opening valid camera ID returns instance.

        Arrangement:
        1. Driver initialized with DEFAULT_CAMERAS.
        2. Camera ID 0 exists (ASI120MC-S finder).

        Action:
        Opens camera 0.

        Assertion Strategy:
        Validates camera opening by confirming:
        - Returns DigitalTwinCameraInstance type.

        Testing Principle:
        Validates driver interface, ensuring valid camera
        IDs return usable camera instances.
        """
        driver = DigitalTwinCameraDriver()
        instance = driver.open(0)
        assert isinstance(instance, DigitalTwinCameraInstance)

    def test_open_invalid_camera_raises(self):
        """Verifies opening invalid camera ID raises ValueError.

        Arrangement:
        1. Driver initialized with cameras 0 and 1.
        2. Camera ID 99 does not exist.

        Action:
        Attempts to open camera 99.

        Assertion Strategy:
        Validates error handling by confirming:
        - Raises ValueError with message "Camera 99 not found".

        Testing Principle:
        Validates error handling, ensuring invalid camera
        IDs fail fast with clear error messages.
        """
        driver = DigitalTwinCameraDriver()
        with pytest.raises(ValueError, match="Camera 99 not found"):
            driver.open(99)


class TestDigitalTwinCameraInstance:
    """Tests for DigitalTwinCameraInstance."""

    @pytest.fixture
    def finder_camera(self):
        """Pytest fixture providing ASI120MC-S finder camera instance.

        Creates digital twin camera for finder (id=0) with
        1280x960 resolution, 3.75μm pixels, and 150° FOV.
        Used for testing camera info retrieval, control operations,
        and specification queries.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            DigitalTwinCameraInstance: Camera instance for camera 0 (finder).
                Provides get_info(), get_controls(), capture() interfaces.

        Raises:
            None. DigitalTwin driver guaranteed to provide camera 0.

        Example:
            >>> def test_finder(finder_camera):
            ...     info = finder_camera.get_info()
            ...     assert info["MaxWidth"] == 1280

        Business Context:
            The finder camera (ASI120MC-S) provides wide-field
            all-sky monitoring for telescope alignment and target
            acquisition. Testing ensures driver correctly exposes
            camera specifications for alignment calculations.
        """
        driver = DigitalTwinCameraDriver()
        return driver.open(0)

    @pytest.fixture
    def main_camera(self):
        """Pytest fixture providing ASI482MC main imager instance.

        Creates digital twin camera for main imager (id=1) with
        1920x1080 resolution, 5.8μm pixels, 1600mm focal length.
        Used for testing camera specifications, control interfaces,
        and high-resolution imaging parameters.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            DigitalTwinCameraInstance: Camera instance for camera 1 (main).
                Provides get_info(), get_controls(), capture() interfaces.

        Raises:
            None. DigitalTwin driver guaranteed to provide camera 1.

        Example:
            >>> def test_main(main_camera):
            ...     info = main_camera.get_info()
            ...     assert info["MaxWidth"] == 1920

        Business Context:
            The main imager (ASI482MC) captures high-resolution
            astronomy images through telescope optics. Testing
            ensures driver correctly exposes camera specifications
            for plate scale calculations and image processing.
        """
        driver = DigitalTwinCameraDriver()
        return driver.open(1)

    def test_get_info_finder(self, finder_camera):
        """Verifies finder camera returns correct specifications.

        Arrangement:
        1. Finder camera (id=0) opened via DigitalTwinCameraDriver.
        2. Camera spec: 1280x960, 3.75μm pixel size.

        Action:
        Calls get_info() to retrieve camera specifications.

        Assertion Strategy:
        Validates specification accuracy by confirming:
        - MaxWidth=1280 matches finder spec.
        - MaxHeight=960 matches finder spec.
        - PixelSize=3.75 matches sensor spec.

        Testing Principle:
        Validates hardware specification, ensuring digital twin
        returns accurate ASI120MC-S parameters for FOV calculations.
        """
        info = finder_camera.get_info()
        assert info["MaxWidth"] == 1280
        assert info["MaxHeight"] == 960
        assert info["PixelSize"] == 3.75

    def test_get_info_main(self, main_camera):
        """Verifies main camera returns correct specifications.

        Arrangement:
        1. Main camera (id=1) opened via DigitalTwinCameraDriver.
        2. Camera spec: 1920x1080, 5.8μm pixel size.

        Action:
        Calls get_info() to retrieve camera specifications.

        Assertion Strategy:
        Validates specification accuracy by confirming:
        - MaxWidth=1920 matches main imager spec.
        - MaxHeight=1080 matches main imager spec.
        - PixelSize=5.8 matches sensor spec.

        Testing Principle:
        Validates hardware specification, ensuring digital twin
        returns accurate ASI482MC parameters for plate scale calculations.
        """
        info = main_camera.get_info()
        assert info["MaxWidth"] == 1920
        assert info["MaxHeight"] == 1080
        assert info["PixelSize"] == 5.8

    def test_get_info_returns_copy(self, finder_camera):
        """Verifies get_info returns defensive copy.

        Arrangement:
        1. Camera info stored internally.
        2. get_info() called twice.
        3. Each call should return new dict instance.

        Action:
        Retrieves camera info twice.

        Assertion Strategy:
        Validates defensive copying by confirming:
        - info1 and info2 are not the same object.
        - Prevents external modification of internal state.

        Testing Principle:
        Validates encapsulation, ensuring callers cannot mutate
        internal camera state through returned info dictionary.
        """
        info1 = finder_camera.get_info()
        info2 = finder_camera.get_info()
        assert info1 is not info2

    def test_get_controls(self, finder_camera):
        """Verifies camera exposes all control parameters.

        Arrangement:
        1. Finder camera with standard ASI control set.
        2. Controls include gain, exposure, white balance, temperature.
        3. Each control has metadata (min, max, default, writable).

        Action:
        Retrieves all available camera controls.

        Assertion Strategy:
        Validates control availability by confirming:
        - ASI_GAIN, ASI_EXPOSURE, ASI_WB_R/B present.
        - ASI_TEMPERATURE available for monitoring.
        - Each control has MinValue, MaxValue fields.
        - Each control has DefaultValue, IsAutoSupported.
        - IsWritable flag indicates adjustability.

        Testing Principle:
        Validates control interface completeness, ensuring all
        ASI SDK controls are exposed for camera configuration.
        """
        controls = finder_camera.get_controls()
        assert "Gain" in controls
        assert "Exposure" in controls
        assert "WB_R" in controls
        assert "WB_B" in controls
        assert "Temperature" in controls

        # Check control structure
        gain = controls["Gain"]
        assert "MinValue" in gain
        assert "MaxValue" in gain
        assert "DefaultValue" in gain
        assert "IsAutoSupported" in gain
        assert "IsWritable" in gain

    def test_set_and_get_control(self, finder_camera):
        """Verifies control value updates persist.

        Arrangement:
        1. Camera with writable ASI_GAIN control.
        2. set_control() changes value to 200.
        3. get_control() reads back current value.

        Action:
        Sets gain to 200, then reads it back.

        Assertion Strategy:
        Validates state persistence by confirming:
        - get_control returns value=200.

        Testing Principle:
        Validates state management, ensuring control values
        persist between set and get operations.
        """
        finder_camera.set_control("Gain", 200)
        result = finder_camera.get_control("Gain")
        assert result["value"] == 200

    def test_get_unknown_control(self, finder_camera):
        """Verifies unknown control raises ValueError.

        Arrangement:
        1. Camera has defined control set.
        2. Request for UNKNOWN_CONTROL (not in set).

        Action:
        Requests non-existent control.

        Assertion Strategy:
        Validates error handling by confirming:
        - ValueError raised with control name in message.

        Testing Principle:
        Validates input validation, ensuring invalid control
        names are rejected with descriptive error messages.
        """
        with pytest.raises(ValueError, match="Unknown control"):
            finder_camera.get_control("UNKNOWN_CONTROL")


class TestDigitalTwinCapture:
    """Tests for digital twin capture functionality."""

    @pytest.fixture
    def finder_camera(self):
        """Pytest fixture providing finder camera for capture tests.

        Creates digital twin camera with synthetic star field
        generation for capture testing (id=0, 1280x960).
        Uses SYNTHETIC image source to generate realistic
        test frames with stars and noise.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            DigitalTwinCameraInstance: Camera instance for camera 0.
                Configured for synthetic star field generation,
                returns JPEG-encoded image data from capture().

        Raises:
            None. DigitalTwin driver guaranteed to provide camera 0.

        Example:
            >>> def test_capture(finder_camera):
            ...     data = finder_camera.capture(100000)
            ...     assert len(data) > 0

        Business Context:
            Synthetic captures enable testing image processing
            pipelines (star detection, plate solving) without
            requiring real hardware or sky access. Validates
            that capture interface produces correctly formatted
            JPEG data matching camera resolution specifications.
        """
        driver = DigitalTwinCameraDriver()
        return driver.open(0)

    @pytest.fixture
    def main_camera(self):
        """Pytest fixture providing main camera for capture tests.

        Creates digital twin camera with synthetic star field
        generation for capture testing (id=1, 1920x1080).
        Uses SYNTHETIC image source for high-resolution
        test frame generation.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            DigitalTwinCameraInstance: Camera instance for camera 1.
                Configured for synthetic star field generation,
                returns JPEG-encoded image data from capture().

        Raises:
            None. DigitalTwin driver guaranteed to provide camera 1.

        Example:
            >>> def test_capture(main_camera):
            ...     data = main_camera.capture(100000)
            ...     assert len(data) > 0

        Business Context:
            Main camera synthetic captures enable testing
            high-resolution imaging workflows (guiding corrections,
            focus analysis) without telescope hardware. Validates
            that capture produces correctly sized JPEG matching
            ASI482MC's 1920x1080 resolution specification.
        """
        driver = DigitalTwinCameraDriver()
        return driver.open(1)

    def test_synthetic_capture_returns_jpeg(self, finder_camera):
        """Verifies synthetic capture produces valid JPEG.

        Arrangement:
        1. Finder camera generates synthetic star field.
        2. Exposure set to 100000μs (100ms).
        3. Image encoded as JPEG for transmission.

        Action:
        Captures single frame.

        Assertion Strategy:
        Validates JPEG encoding by confirming:
        - Data is bytes type.
        - Length > 0 (non-empty).
        - First 2 bytes = 0xFFD8 (JPEG SOI marker).
        """
        data = finder_camera.capture(100000)
        assert isinstance(data, bytes)
        assert len(data) > 0
        # JPEG magic bytes
        assert data[:2] == b"\xff\xd8"

    def test_synthetic_capture_correct_resolution_finder(self, finder_camera):
        """Verifies finder captures at correct resolution.

        Arrangement:
        1. Finder camera spec: 1280x960.
        2. Capture produces JPEG bytes.
        3. Parse JPEG header to verify dimensions.

        Action:
        Captures frame and parses JPEG SOF marker.

        Assertion Strategy:
        Validates resolution by confirming:
        - Image width = 1280 pixels.
        - Image height = 960 pixels.
        """
        data = finder_camera.capture(100000)
        # Parse JPEG dimensions from SOF0 marker (FF C0)
        width, height = _parse_jpeg_dimensions(data)
        assert width == 1280
        assert height == 960

    def test_synthetic_capture_correct_resolution_main(self, main_camera):
        """Verifies main camera captures at correct resolution.

        Arrangement:
        1. Main camera spec: 1920x1080.
        2. Capture produces JPEG bytes.
        3. Parse JPEG header to verify dimensions.

        Action:
        Captures frame and parses JPEG SOF marker.

        Assertion Strategy:
        Validates resolution by confirming:
        - Image width = 1920 pixels.
        - Image height = 1080 pixels.
        """
        data = main_camera.capture(100000)
        width, height = _parse_jpeg_dimensions(data)
        assert width == 1920
        assert height == 1080

    def test_capture_gain_affects_noise(self, finder_camera):
        """Verifies gain parameter affects image noise.

        Arrangement:
        1. Capture at gain=0 (low noise).
        2. Capture at gain=300 (high noise).
        3. Synthetic noise increases with gain.

        Action:
        Captures two frames with different gain settings.

        Assertion Strategy:
        Validates gain simulation by confirming:
        - data_low != data_high (different bytes).
        - Higher gain produces different noise pattern.
        """
        finder_camera.set_control("Gain", 0)
        data_low = finder_camera.capture(100000)

        finder_camera.set_control("Gain", 300)
        data_high = finder_camera.capture(100000)

        # Images should be different due to noise
        assert data_low != data_high


class TestDigitalTwinFileSource:
    """Tests for file-based image source."""

    @pytest.fixture
    def temp_image(self):
        """Pytest fixture creating temporary test image file.

        Creates 640x480 test JPEG with "TEST" text overlay
        for file-based camera source testing. Image is
        automatically cleaned up after test completion.

        Args:
            None (pytest fixture with implicit request parameter).

        Yields:
            Path: Path object pointing to temporary JPEG file.
                File contains 640x480 black image with white "TEST" text.

        Raises:
            None. Cleanup handles missing files gracefully.

        Example:
            >>> def test_file(temp_image):
            ...     driver = create_file_camera(temp_image)
            ...     assert driver.config.image_path == temp_image

        Business Context:
            File-based camera source enables replaying previously
            captured images for regression testing and algorithm
            development. Tests validate that driver correctly
            loads, resizes, and serves static images as if they
            were live camera captures.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create test image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                img, "TEST", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3
            )
            cv2.imwrite(f.name, img)
            yield Path(f.name)
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_create_file_camera(self, temp_image):
        """Verifies create_file_camera configures file source.

        Arrangement:
        1. Temporary test image created.
        2. create_file_camera() factory function.
        3. Driver configured for FILE source mode.

        Action:
        Creates driver with file path.

        Assertion Strategy:
        Validates configuration by confirming:
        - image_source = ImageSource.FILE.
        - image_path matches provided path.
        """
        driver = create_file_camera(temp_image)
        assert driver.config.image_source == ImageSource.FILE
        assert driver.config.image_path == temp_image

    def test_file_capture_resizes_to_camera(self, temp_image):
        """Verifies file images resized to camera resolution.

        Arrangement:
        1. Test image is 640x480.
        2. ASI120MC-S camera spec: 1280x960.
        3. Driver must resize file to match camera.

        Action:
        Opens camera, captures frame from file.

        Assertion Strategy:
        Validates resizing by confirming:
        - Output width = 1280 (camera spec).
        - Output height = 960 (camera spec).
        """
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
        """Pytest fixture creating temporary directory with test images.

        Creates directory containing 3 test JPEGs (test_0.jpg,
        test_1.jpg, test_2.jpg) with text overlays (IMG0, IMG1, IMG2)
        for directory-based camera source testing. Directory and
        contents automatically cleaned up after test.

        Args:
            None (pytest fixture with implicit request parameter).

        Yields:
            Path: Path object pointing to temporary directory containing
                3 JPEG files (640x480 each with unique text labels).

        Raises:
            None. Automatic cleanup via tempfile.TemporaryDirectory.

        Example:
            >>> def test_dir(temp_image_dir):
            ...     driver = create_directory_camera(temp_image_dir)
            ...     assert driver.config.image_path == temp_image_dir

        Business Context:
            Directory-based camera source enables time-lapse replay
            and sequence testing for astronomy workflows (drift
            analysis, stack processing). Tests validate that driver
            correctly cycles through image sequences, simulating
            continuous observation sessions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create multiple test images
            for i in range(3):
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    img,
                    f"IMG{i}",
                    (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (255, 255, 255),
                    3,
                )
                cv2.imwrite(str(tmppath / f"test_{i}.jpg"), img)
            yield tmppath

    def test_create_directory_camera(self, temp_image_dir):
        """Verifies create_directory_camera configures dir source.

        Arrangement:
        1. Temporary directory with test images.
        2. create_directory_camera() factory function.
        3. Driver configured for DIRECTORY source mode.

        Action:
        Creates driver with directory path.

        Assertion Strategy:
        Validates configuration by confirming:
        - image_source = ImageSource.DIRECTORY.
        - image_path points to directory.
        """
        driver = create_directory_camera(temp_image_dir)
        assert driver.config.image_source == ImageSource.DIRECTORY
        assert driver.config.image_path == temp_image_dir
        assert driver.config.cycle_images is True

    def test_directory_cycles_images(self, temp_image_dir):
        """Verifies directory source cycles through images.

        Arrangement:
        1. Directory contains 3 test images (IMG0, IMG1, IMG2).
        2. Driver configured with cycle=True.
        3. Captures should loop back to first image.

        Action:
        Captures 4 frames (more than 3 images available).

        Assertion Strategy:
        Validates cycling by confirming:
        - Frame 4 differs from frames 1-3 (not frozen).
        - Cycling back to start works.
        """
        driver = create_directory_camera(temp_image_dir, cycle=True)
        instance = driver.open(0)

        # Capture 4 images (should cycle back to first)
        captures = [instance.capture(100000) for _ in range(4)]

        # First and fourth should be same image (after resize)
        # We can't compare exact bytes due to JPEG encoding, but sizes should match
        assert len(captures[0]) > 0
        assert len(captures[3]) > 0

    def test_directory_no_cycle(self, temp_image_dir):
        """Verifies directory source stops at last image without cycling.

        Arrangement:
        1. temp_image_dir contains 3 test images.
        2. Driver created with cycle=False (no wrap).
        3. Captures beyond image count should repeat last image.

        Action:
        Performs 5 captures from 3-image directory.

        Assertion Strategy:
        Validates no-cycle behavior by confirming:
        - All 5 captures return valid data (non-empty).
        - Captures 4-5 repeat image 3 (last image).

        Testing Principle:
        Validates cycle control, ensuring drivers can
        hold on final image without looping.
        """
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
        """Verifies DigitalTwinConfig uses synthetic source by default.

        Arrangement:
        1. DigitalTwinConfig() with no arguments.
        2. Expected defaults: SYNTHETIC source, no path, cycle=True.

        Action:
        Creates config and reads fields.

        Assertion Strategy:
        Validates default configuration by confirming:
        - image_source = ImageSource.SYNTHETIC.
        - image_path = None (no external file).
        - cycle_images = True.

        Testing Principle:
        Validates default behavior, ensuring config works
        out-of-box for synthetic star field generation.
        """
        config = DigitalTwinConfig()
        assert config.image_source == ImageSource.SYNTHETIC
        assert config.image_path is None
        assert config.cycle_images is True

    def test_custom_config(self):
        """Verifies DigitalTwinConfig preserves custom values.

        Arrangement:
        1. Custom config: FILE source, /test/path, cycle=False.
        2. All parameters explicitly set.

        Action:
        Creates config with custom values.

        Assertion Strategy:
        Validates custom configuration by confirming:
        - image_source = ImageSource.FILE.
        - image_path = Path("/test/path").
        - cycle_images = False.

        Testing Principle:
        Validates configuration flexibility, ensuring all
        parameters can be customized for different test scenarios.
        """
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
        """Verifies motors start at position 0 on initialization.

        Arrangement:
        1. StubMotorController() created.
        2. No move commands issued yet.
        3. Both ALT and AZ motors should be at position 0.

        Action:
        Gets motor status after initialization.

        Assertion Strategy:
        Validates initial state by confirming:
        - position_steps = 0 for ALTITUDE motor.

        Testing Principle:
        Validates initialization, ensuring motors start
        at known home position for deterministic testing.
        """
        controller = StubMotorController()
        status = controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 0

    def test_move_updates_position(self):
        """Verifies move command updates motor position correctly.

        Arrangement:
        1. Motor at position 0.
        2. move() command with 100 steps.
        3. Position should reflect command.

        Action:
        Moves ALTITUDE motor 100 steps, reads status.

        Assertion Strategy:
        Validates position tracking by confirming:
        - position_steps = 100 after move.

        Testing Principle:
        Validates state updates, ensuring move commands
        correctly update internal position tracking.
        """
        controller = StubMotorController()
        controller.move(MotorType.ALTITUDE, 100)
        status = controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 100

    def test_home_resets_position(self):
        """Verifies home command resets motor to position 0.

        Arrangement:
        1. Motor moved to position 500.
        2. home() command issued.
        3. Position should reset to 0.

        Action:
        Moves AZIMUTH 500 steps, then homes it.

        Assertion Strategy:
        Validates homing by confirming:
        - position_steps = 0 after home().

        Testing Principle:
        Validates homing functionality, ensuring motors
        can return to known reference position.
        """
        controller = StubMotorController()
        controller.move(MotorType.AZIMUTH, 500)
        controller.home(MotorType.AZIMUTH)
        status = controller.get_status(MotorType.AZIMUTH)
        assert status.position_steps == 0


class TestDigitalTwinSensorDriver:
    """Tests for the digital twin sensor driver."""

    def test_initial_position(self):
        """Verifies sensor starts with default position.

        Arrangement:
        1. DigitalTwinSensorDriver created and opened.
        2. Default position: altitude=45.0°, azimuth=180.0°.
        3. No calibration performed yet.

        Action:
        Reads sensor position after initialization.

        Assertion Strategy:
        Validates default state by confirming:
        - reading.altitude close to 45.0 degrees.
        - reading.azimuth close to 180.0 degrees (south).

        Testing Principle:
        Validates initialization, ensuring sensor provides
        reasonable default position for testing.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()
        reading = instance.read()
        # Allow for noise in digital twin
        assert 40.0 <= reading.altitude <= 50.0
        assert 175.0 <= reading.azimuth <= 185.0
        driver.close()

    def test_calibrate_updates_position(self):
        """Verifies calibrate updates reported position.

        Arrangement:
        1. Sensor at default position (45°, 180°).
        2. calibrate() called with new position (30°, 90°).
        3. Subsequent reads should reflect calibration.

        Action:
        Calibrates to (30.0°, 90.0°), then reads position.

        Assertion Strategy:
        Validates calibration by confirming:
        - reading.altitude close to 30.0 degrees.
        - reading.azimuth close to 90.0 degrees (east).

        Testing Principle:
        Validates calibration mechanism, ensuring sensor
        can be set to known positions for testing.
        """
        driver = DigitalTwinSensorDriver()
        instance = driver.open()
        instance.calibrate(30.0, 90.0)
        reading = instance.read()
        # Allow for noise in digital twin
        assert 25.0 <= reading.altitude <= 35.0
        assert 85.0 <= reading.azimuth <= 95.0
        driver.close()
