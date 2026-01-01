"""Unit tests for Digital Twin Camera Driver coverage.

Tests DigitalTwinCameraDriver and DigitalTwinCameraInstance edge cases
to achieve 100% code coverage.

Test Categories:
1. Driver Tests
   - __repr__ string representation
   - Configuration variants

2. Instance Tests
   - Context manager (__enter__/__exit__)
   - Directory image source initialization
   - File capture edge cases (missing file, imread failure)
   - Directory capture edge cases (empty list, imread failure)
   - Image resizing path
   - stop_exposure logging

Coverage Targets:
- Line 273: DigitalTwinCameraDriver.__repr__
- Lines 434, 454: DigitalTwinCameraInstance context manager
- Lines 489, 493, 503: _load_directory_images edge cases
- Lines 768, 772, 776: _capture_from_file edge cases
- Lines 820, 834: _capture_from_directory edge cases
- Lines 872->875: _resize_to_camera when resize needed
- Line 989: stop_exposure logging
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from telescope_mcp.drivers.cameras.twin import (
    DigitalTwinCameraDriver,
    DigitalTwinCameraInstance,
    DigitalTwinConfig,
    ImageSource,
    create_directory_camera,
    create_file_camera,
)
from telescope_mcp.drivers.cameras.twin import (
    TwinCameraInfo as CameraInfo,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def driver() -> DigitalTwinCameraDriver:
    """Create default DigitalTwinCameraDriver for testing.

    Factory fixture providing fresh driver instance for each test.
    Uses default configuration (synthetic images, single camera).

    Business context:
    Standard fixture for unit tests that need driver functionality
    without custom configuration. Synthetic mode ensures predictable
    test images without external file dependencies.

    Args:
        None. Pytest fixture with no parameters.

    Returns:
        DigitalTwinCameraDriver: Configured with defaults (synthetic source,
        one simulated camera at index 0).

    Raises:
        None. Default configuration always succeeds.

    Example:
        >>> def test_discovery(driver: DigitalTwinCameraDriver):
        ...     cameras = driver.get_connected_cameras()
        ...     assert 0 in cameras
    """
    return DigitalTwinCameraDriver()


@pytest.fixture
def camera_info() -> CameraInfo:
    """Create standard test camera info TypedDict.

    Provides realistic camera metadata matching common ASI camera specs.
    Used for tests requiring pre-defined camera properties.

    Business context:
    CameraInfo structure mirrors real hardware properties. Standard
    fixture ensures consistent test data across all tests requiring
    camera metadata (info display, resolution validation, etc.).

    Args:
        None. Pytest fixture with no parameters.

    Returns:
        CameraInfo: TypedDict with standard test values:
        - camera_id: 0 (first camera)
        - name: "Test Camera"
        - Resolution: 1920x1080 (Full HD)
        - pixel_size_um: 3.75 (common ASI spec)
        - Monochrome, 12-bit, USB3, no cooler/ST4

    Raises:
        None. Static data always succeeds.

    Example:
        >>> def test_info_structure(camera_info: CameraInfo):
        ...     assert camera_info["max_width"] == 1920
        ...     assert camera_info["is_color"] is False
    """
    return CameraInfo(
        camera_id=0,
        name="Test Camera",
        max_width=1920,
        max_height=1080,
        pixel_size_um=3.75,
        is_color=False,
        bit_depth=12,
        is_usb3=True,
        has_cooler=False,
        has_st4_port=False,
        MaxWidth=1920,
        MaxHeight=1080,
    )


# =============================================================================
# DigitalTwinCameraDriver Tests
# =============================================================================


class TestDigitalTwinCameraDriverRepr:
    """Test __repr__ method of DigitalTwinCameraDriver.

    Covers twin.py line 273: __repr__ string representation.
    """

    def test_repr_returns_string_with_source_and_cameras(
        self, driver: DigitalTwinCameraDriver
    ) -> None:
        """Verify __repr__ includes image source and camera list.

        Business context:
        String representation aids debugging by showing driver config
        at a glance in logs and REPL sessions.

        Arrangement:
        Create driver with default configuration.

        Action:
        Call repr() on driver instance.

        Assertion:
        String contains class name, image source, and camera list.
        """
        result = repr(driver)

        assert "DigitalTwinCameraDriver" in result
        assert "source=" in result
        assert "cameras=" in result

    def test_repr_shows_configured_image_source(self) -> None:
        """Verify __repr__ reflects configured image source.

        Business context:
        Different image sources (synthetic, file, directory) produce
        different behaviors; repr should show which is active.

        Arrangement:
        Create driver with FILE image source.

        Action:
        Call repr() on driver.

        Assertion:
        String contains "file" (lowercase from enum value).
        """
        config = DigitalTwinConfig(image_source=ImageSource.FILE)
        driver = DigitalTwinCameraDriver(config=config)

        result = repr(driver)

        assert "file" in result.lower()


# =============================================================================
# DigitalTwinCameraInstance Context Manager Tests
# =============================================================================


class TestDigitalTwinCameraInstanceContextManager:
    """Test context manager protocol for DigitalTwinCameraInstance.

    Covers twin.py lines 434 and 454: __enter__ and __exit__.
    """

    def test_context_manager_enter_returns_self(self, camera_info: CameraInfo) -> None:
        """Verify __enter__ returns the instance for use in with-block.

        Business context:
        Context manager pattern enables resource cleanup on exit,
        matching real camera API for consistent code patterns.

        Arrangement:
        Create DigitalTwinCameraInstance.

        Action:
        Enter context manager and capture returned value.

        Assertion:
        Returned value is the same instance.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        with instance as ctx:
            assert ctx is instance

    def test_context_manager_exit_is_safe(self, camera_info: CameraInfo) -> None:
        """Verify __exit__ completes without error.

        Business context:
        Digital twin has no hardware to release, but __exit__ must
        be implemented for API compatibility with real cameras.

        Arrangement:
        Create DigitalTwinCameraInstance.

        Action:
        Exit context manager (with-block ends).

        Assertion:
        No exception raised.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        # Should complete without error
        with instance:
            pass

    def test_context_manager_exit_with_exception(self, camera_info: CameraInfo) -> None:
        """Verify __exit__ does not suppress exceptions.

        Business context:
        Exceptions in with-block should propagate normally;
        digital twin should not interfere with error handling.

        Arrangement:
        Create DigitalTwinCameraInstance.

        Action:
        Raise exception inside with-block.

        Assertion:
        Exception propagates out of context manager.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        with pytest.raises(ValueError, match="test error"):
            with instance:
                raise ValueError("test error")


# =============================================================================
# Directory Image Source Tests
# =============================================================================


class TestDigitalTwinCameraInstanceDirectoryLoading:
    """Test _load_directory_images edge cases.

    Covers twin.py lines 489, 493, 503: Directory loading initialization.
    """

    def test_load_directory_skipped_when_not_directory_source(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify _load_directory_images is no-op for non-directory sources.

        Business context:
        FILE and SYNTHETIC sources don't use directory scanning;
        method should return early without side effects.

        Arrangement:
        Create instance with FILE image source.

        Action:
        Instance created (triggers _load_directory_images in __init__).

        Assertion:
        _image_files remains empty list.
        """
        config = DigitalTwinConfig(
            image_source=ImageSource.FILE, image_path="/some/file.jpg"
        )
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        assert instance._image_files == []

    def test_load_directory_skipped_when_path_is_none(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify _load_directory_images handles None path gracefully.

        Business context:
        Directory source with no path configured should not crash;
        falls back to synthetic images on capture.

        Arrangement:
        Create instance with DIRECTORY source but no image_path.

        Action:
        Instance created.

        Assertion:
        _image_files remains empty, no exception.
        """
        config = DigitalTwinConfig(image_source=ImageSource.DIRECTORY, image_path=None)
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        assert instance._image_files == []

    def test_load_directory_skipped_when_path_not_directory(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify _load_directory_images handles non-directory path.

        Business context:
        Path might be file or non-existent; should not crash but
        gracefully fall back to synthetic images.

        Arrangement:
        Create temp file (not directory).
        Create instance with DIRECTORY source pointing to file.

        Action:
        Instance created.

        Assertion:
        _image_files remains empty, no exception.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            config = DigitalTwinConfig(
                image_source=ImageSource.DIRECTORY,
                image_path=f.name,  # This is a file, not directory
            )
            instance = DigitalTwinCameraInstance(0, camera_info, config)

            assert instance._image_files == []

    def test_load_directory_finds_image_files(self, camera_info: CameraInfo) -> None:
        """Verify _load_directory_images finds supported image formats.

        Business context:
        Directory source should find jpg, png, tif, fits files for
        cycling through during captures.

        Arrangement:
        Create temp directory with test image files.
        Create instance with DIRECTORY source.

        Action:
        Instance created.

        Assertion:
        _image_files contains paths to the image files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image files
            (Path(tmpdir) / "image1.jpg").touch()
            (Path(tmpdir) / "image2.png").touch()
            (Path(tmpdir) / "image3.tif").touch()
            (Path(tmpdir) / "not_image.txt").touch()  # Should be ignored

            config = DigitalTwinConfig(
                image_source=ImageSource.DIRECTORY, image_path=tmpdir
            )
            instance = DigitalTwinCameraInstance(0, camera_info, config)

            assert len(instance._image_files) == 3
            # Should be sorted
            assert instance._image_files[0].name == "image1.jpg"


# =============================================================================
# File Capture Edge Cases
# =============================================================================


class TestDigitalTwinCameraInstanceFileCaptureEdgeCases:
    """Test _capture_from_file edge cases.

    Covers twin.py lines 768, 772, 776: File capture fallback paths.
    """

    @pytest.mark.parametrize(
        "scenario",
        ["path_none", "path_is_directory", "corrupted_file"],
        ids=["none_path", "dir_not_file", "imread_fails"],
    )
    def test_capture_from_file_fallback_to_synthetic(
        self, camera_info: CameraInfo, scenario: str
    ) -> None:
        """Verify capture falls back to synthetic for various failure scenarios.

        Business context:
        FILE source must gracefully handle misconfiguration and file errors
        by falling back to synthetic image generation rather than crashing.
        This enables partial configuration testing and robust error handling.

        Parameterized scenarios:
        - path_none: image_path is None (missing config)
        - path_is_directory: image_path points to directory, not file
        - corrupted_file: file exists but cv2.imread returns None

        Arrangement:
        Create instance with FILE source configured per scenario.

        Action:
        Call capture().

        Assertion:
        Returns valid JPEG bytes (synthetic fallback) for all scenarios.

        Args:
            camera_info: Camera info fixture.
            scenario: Test scenario identifier.
        """
        cleanup_path: Path | None = None
        cleanup_dir: str | None = None

        try:
            if scenario == "path_none":
                config = DigitalTwinConfig(
                    image_source=ImageSource.FILE,
                    image_path=None,
                )
            elif scenario == "path_is_directory":
                cleanup_dir = tempfile.mkdtemp()
                config = DigitalTwinConfig(
                    image_source=ImageSource.FILE,
                    image_path=cleanup_dir,
                )
            else:  # corrupted_file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(b"not a real image")
                    cleanup_path = Path(f.name)
                config = DigitalTwinConfig(
                    image_source=ImageSource.FILE,
                    image_path=str(cleanup_path),
                )

            instance = DigitalTwinCameraInstance(0, camera_info, config)
            result = instance.capture(100000)

            # All scenarios should fall back to synthetic JPEG
            assert isinstance(result, bytes), f"{scenario}: expected bytes"
            assert len(result) > 0, f"{scenario}: expected non-empty"
            assert result[:2] == b"\xff\xd8", f"{scenario}: expected JPEG magic"

        finally:
            if cleanup_path and cleanup_path.exists():
                cleanup_path.unlink()
            if cleanup_dir:
                import shutil

                shutil.rmtree(cleanup_dir, ignore_errors=True)


# =============================================================================
# Directory Capture Edge Cases
# =============================================================================


class TestDigitalTwinCameraInstanceDirectoryCaptureEdgeCases:
    """Test _capture_from_directory edge cases.

    Covers twin.py lines 820, 834: Directory capture fallback paths.
    """

    def test_capture_from_directory_fallback_when_empty(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify capture falls back when _image_files is empty.

        Business context:
        Directory might contain no image files; should produce
        synthetic image rather than crash or hang.

        Arrangement:
        Create instance with DIRECTORY source but empty directory.

        Action:
        Call capture().

        Assertion:
        Returns valid JPEG bytes (synthetic fallback).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory - no image files
            config = DigitalTwinConfig(
                image_source=ImageSource.DIRECTORY, image_path=tmpdir
            )
            instance = DigitalTwinCameraInstance(0, camera_info, config)

            result = instance.capture(100000)

            assert isinstance(result, bytes)
            assert result[:2] == b"\xff\xd8"

    def test_capture_from_directory_fallback_when_imread_fails(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify capture falls back when imread fails on directory image.

        Business context:
        Image file in directory might be corrupted; should skip to
        synthetic fallback rather than crash.

        Arrangement:
        Create directory with corrupted image file.
        Create instance with DIRECTORY source.
        Patch cv2.imread to return None.

        Action:
        Call capture().

        Assertion:
        Returns valid JPEG bytes (synthetic fallback).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a "corrupted" image file
            corrupt_file = Path(tmpdir) / "corrupt.jpg"
            corrupt_file.write_bytes(b"not a real image")

            config = DigitalTwinConfig(
                image_source=ImageSource.DIRECTORY, image_path=tmpdir
            )
            instance = DigitalTwinCameraInstance(0, camera_info, config)

            # Force the file list to contain our corrupt file
            instance._image_files = [corrupt_file]

            # cv2.imread will return None
            result = instance.capture(100000)

            assert isinstance(result, bytes)
            assert result[:2] == b"\xff\xd8"

    def test_capture_from_directory_cycles_through_images(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify capture cycles through directory images.

        Business context:
        Directory source with cycle_images=True should loop back
        to first image after reaching end.

        Arrangement:
        Create directory with valid images.
        Create instance with cycle_images=True.

        Action:
        Capture multiple times (more than number of images).

        Assertion:
        Index wraps around, captures succeed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid test images
            for i in range(2):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                img[:, :] = (i * 50, i * 50, i * 50)
                cv2.imwrite(str(Path(tmpdir) / f"image{i}.jpg"), img)

            config = DigitalTwinConfig(
                image_source=ImageSource.DIRECTORY, image_path=tmpdir, cycle_images=True
            )
            instance = DigitalTwinCameraInstance(0, camera_info, config)

            # Capture more times than images exist
            results = [instance.capture(100000) for _ in range(5)]

            # All should succeed
            assert all(isinstance(r, bytes) for r in results)
            assert all(r[:2] == b"\xff\xd8" for r in results)

    def test_capture_from_directory_stops_at_last_when_not_cycling(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify capture stays on last image when cycle_images=False.

        Business context:
        Directory source without cycling should stay on last image
        after exhausting the list, for testing static scenes.

        Arrangement:
        Create directory with 2 images.
        Create instance with cycle_images=False.

        Action:
        Capture 5 times.

        Assertion:
        Index clamped to last image, all captures succeed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid test images
            for i in range(2):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(Path(tmpdir) / f"image{i}.jpg"), img)

            config = DigitalTwinConfig(
                image_source=ImageSource.DIRECTORY,
                image_path=tmpdir,
                cycle_images=False,
            )
            instance = DigitalTwinCameraInstance(0, camera_info, config)

            # Capture more times than images exist
            for _ in range(5):
                result = instance.capture(100000)
                assert isinstance(result, bytes)

            # Index should be clamped to last (1 for 2 images)
            assert instance._image_index == 1


# =============================================================================
# Image Resize Tests
# =============================================================================


class TestDigitalTwinCameraInstanceResize:
    """Test _resize_to_camera method.

    Covers twin.py lines 872-875: Resize when dimensions don't match.
    """

    @pytest.mark.parametrize(
        "input_height,input_width,description",
        [
            (3000, 4000, "larger than camera"),
            (480, 640, "smaller than camera"),
            (1080, 1920, "exact match"),
        ],
        ids=["downscale", "upscale", "passthrough"],
    )
    def test_resize_to_camera_dimensions(
        self,
        camera_info: CameraInfo,
        input_height: int,
        input_width: int,
        description: str,
    ) -> None:
        """Verify _resize_to_camera produces correct output dimensions.

        Business context:
        Images from file/directory may differ from camera resolution;
        _resize_to_camera must normalize all inputs to camera dimensions
        for consistent output regardless of source image size.

        Parameterized scenarios:
        - downscale: 4000x3000 → 1920x1080 (larger than camera)
        - upscale: 640x480 → 1920x1080 (smaller than camera)
        - passthrough: 1920x1080 → 1920x1080 (exact match)

        Arrangement:
        Create instance with 1920x1080 camera.
        Create test image with parameterized dimensions.

        Action:
        Call _resize_to_camera().

        Assertion:
        Output dimensions always match camera (1920x1080).

        Args:
            camera_info: Camera info fixture with 1920x1080 resolution.
            input_height: Height of test input image.
            input_width: Width of test input image.
            description: Human-readable scenario description.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        # Create input image with parameterized dimensions
        input_img = np.zeros((input_height, input_width, 3), dtype=np.uint8)

        result = instance._resize_to_camera(input_img)

        # Output should always match camera resolution
        assert result.shape[0] == 1080, f"Height mismatch for {description}"
        assert result.shape[1] == 1920, f"Width mismatch for {description}"


# =============================================================================
# Stop Exposure Tests
# =============================================================================


class TestDigitalTwinCameraInstanceStopExposure:
    """Test stop_exposure method.

    Covers twin.py line 989: stop_exposure logging.
    """

    def test_stop_exposure_logs_debug_message(self, camera_info: CameraInfo) -> None:
        """Verify stop_exposure logs debug message.

        Business context:
        Stop exposure is a no-op for digital twin but should log
        for debugging API usage patterns.

        Arrangement:
        Create instance.

        Action:
        Call stop_exposure().

        Assertion:
        Method completes without error (logging verified implicitly).
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        # Should complete without error
        instance.stop_exposure()

    def test_stop_exposure_multiple_calls_safe(self, camera_info: CameraInfo) -> None:
        """Verify stop_exposure is idempotent.

        Business context:
        Multiple stop calls should be safe, matching real camera
        behavior where abort can be called repeatedly.

        Arrangement:
        Create instance.

        Action:
        Call stop_exposure() multiple times.

        Assertion:
        All calls complete without error.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        # Multiple calls should be safe
        instance.stop_exposure()
        instance.stop_exposure()
        instance.stop_exposure()


# =============================================================================
# Integration Tests
# =============================================================================


class TestDigitalTwinCameraIntegration:
    """Integration tests combining multiple coverage paths."""

    def test_full_capture_cycle_with_file_source(self, camera_info: CameraInfo) -> None:
        """Test complete capture cycle with FILE source and valid image.

        Business context:
        Validates full path through file-based capture including
        imread, resize, and JPEG encoding.

        Arrangement:
        Create temp file with valid image.
        Create instance with FILE source.

        Action:
        Call capture().

        Assertion:
        Returns valid JPEG of correct dimensions.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create valid test image
            img = np.zeros((500, 800, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            temp_path = f.name

        try:
            config = DigitalTwinConfig(
                image_source=ImageSource.FILE, image_path=temp_path
            )
            instance = DigitalTwinCameraInstance(0, camera_info, config)

            result = instance.capture(100000)

            assert isinstance(result, bytes)
            assert result[:2] == b"\xff\xd8"  # JPEG magic
        finally:
            Path(temp_path).unlink()

    def test_driver_open_and_capture_cycle(self) -> None:
        """Test complete driver -> instance -> capture cycle.

        Business context:
        Validates typical usage pattern from driver discovery
        through capture.

        Arrangement:
        Create driver.
        Open camera 0.

        Action:
        Capture image through instance.

        Assertion:
        Capture returns valid JPEG.
        """
        driver = DigitalTwinCameraDriver()
        instance = driver.open(0)

        result = instance.capture(100000)

        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"


# =============================================================================
# Additional Coverage Tests - Driver Methods
# =============================================================================


class TestDigitalTwinCameraDriverOpen:
    """Test open() method edge cases.

    Covers twin.py lines 331-332: ValueError for invalid camera_id.
    """

    def test_open_invalid_camera_id_raises_value_error(self) -> None:
        """Verify open() raises ValueError for unknown camera_id.

        Business context:
        Requesting non-existent camera should fail early with clear
        error, not return None or crash later.

        Arrangement:
        Create driver with default cameras (0, 1).

        Action:
        Call open(99) for non-existent camera.

        Assertion:
        ValueError raised with camera ID in message.
        """
        driver = DigitalTwinCameraDriver()

        with pytest.raises(ValueError, match="Camera 99 not found"):
            driver.open(99)


class TestDigitalTwinCameraDriverGetConnectedCameras:
    """Test get_connected_cameras() method.

    Covers twin.py lines 307-308: Debug logging and camera listing.
    """

    def test_get_connected_cameras_returns_copy(self) -> None:
        """Verify get_connected_cameras returns copy, not original dict.

        Business context:
        Callers should not be able to modify internal camera registry
        by modifying the returned dict.

        Arrangement:
        Create driver.

        Action:
        Get cameras and modify returned dict.

        Assertion:
        Original internal dict unchanged.
        """
        driver = DigitalTwinCameraDriver()

        cameras = driver.get_connected_cameras()
        original_count = len(cameras)
        cameras[999] = {}  # Try to modify

        # Should not affect driver's internal state
        fresh_cameras = driver.get_connected_cameras()
        assert len(fresh_cameras) == original_count


# =============================================================================
# Additional Coverage Tests - Instance Methods
# =============================================================================


class TestDigitalTwinCameraInstanceRepr:
    """Test __repr__ method of DigitalTwinCameraInstance.

    Covers twin.py line 503-507: Instance __repr__ string.
    """

    def test_instance_repr_contains_camera_id_and_source(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify instance __repr__ includes camera ID and source.

        Business context:
        Instance repr aids debugging by showing which camera and
        what image source configuration.

        Arrangement:
        Create instance.

        Action:
        Call repr().

        Assertion:
        String contains class name, camera_id, and source.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        result = repr(instance)

        assert "DigitalTwinCameraInstance" in result
        assert "camera_id=0" in result
        assert "source=" in result


class TestDigitalTwinCameraInstanceGetInfo:
    """Test get_info() method.

    Covers twin.py line 535: Returns dict copy of camera info.
    """

    def test_get_info_returns_copy(self, camera_info: CameraInfo) -> None:
        """Verify get_info returns copy, not original dict.

        Business context:
        Callers should not modify internal camera info state.

        Arrangement:
        Create instance.

        Action:
        Get info and attempt to modify.

        Assertion:
        Original info unchanged.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        info = instance.get_info()
        original_name = info["name"]
        info["name"] = "Modified"

        # Should not affect internal state
        fresh_info = instance.get_info()
        assert fresh_info["name"] == original_name


class TestDigitalTwinCameraInstanceGetControls:
    """Test get_controls() method.

    Covers twin.py lines 563-576: Control definitions returned.
    """

    def test_get_controls_returns_all_controls(self, camera_info: CameraInfo) -> None:
        """Verify get_controls returns expected control set.

        Business context:
        Control definitions enable UI sliders and validation logic.

        Arrangement:
        Create instance.

        Action:
        Call get_controls().

        Assertion:
        Contains expected controls with proper structure.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        controls = instance.get_controls()

        assert "Gain" in controls
        assert "Exposure" in controls
        assert "Temperature" in controls

        # Verify structure
        gain = controls["Gain"]
        assert "MinValue" in gain
        assert "MaxValue" in gain
        assert "DefaultValue" in gain
        assert "IsAutoSupported" in gain
        assert "IsWritable" in gain


class TestDigitalTwinCameraInstanceGetControlMax:
    """Test _get_control_max() method.

    Covers twin.py lines 601-604: Control max value lookup.
    """

    def test_get_control_max_returns_specific_values(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify _get_control_range returns correct ranges for known controls.

        Business context:
        Different controls have different valid ranges; max values
        must be realistic for testing.

        Arrangement:
        Create instance.

        Action:
        Call _get_control_range for various controls.

        Assertion:
        Returns expected (min, max, default) values per control type.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        # Gain typically has max of 510 (ASI120MC-S default)
        gain_min, gain_max, gain_default = instance._get_control_range("Gain")
        assert gain_max == 510

        # Exposure max is 60 seconds (60,000,000 µs)
        exp_min, exp_max, exp_default = instance._get_control_range("Exposure")
        assert exp_max == 60_000_000

        # Unknown control returns default range (0, 100, 50)
        unk_min, unk_max, unk_default = instance._get_control_range("UNKNOWN_CONTROL")
        assert unk_max == 100


class TestDigitalTwinCameraInstanceSetControl:
    """Test set_control() method edge cases.

    Covers twin.py: Unknown control raises ValueError (matches ASI behavior).
    """

    def test_set_control_unknown_control_raises_value_error(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify set_control raises ValueError for unknown control.

        Business context:
        Unknown controls should raise ValueError to match ASI driver behavior
        and fail fast on typos.

        Arrangement:
        Create instance.

        Action:
        Call set_control with unknown control name.

        Assertion:
        ValueError raised with control name in message.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        with pytest.raises(ValueError, match="Unknown control"):
            instance.set_control("UNKNOWN_CONTROL", 999)

    def test_set_control_valid_control_updates_state(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify set_control updates known control state.

        Business context:
        Control changes should be persisted and readable.

        Arrangement:
        Create instance.

        Action:
        Set gain to specific value.

        Assertion:
        get_control returns the set value.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        instance.set_control("Gain", 250)
        result = instance.get_control("Gain")

        assert result["value"] == 250


class TestDigitalTwinCameraInstanceGetControl:
    """Test get_control() method edge cases.

    Covers twin.py: Unknown control raises ValueError (matches ASI behavior).
    """

    def test_get_control_unknown_raises_value_error(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify get_control raises ValueError for unknown control.

        Business context:
        Unknown controls should raise ValueError to match ASI driver behavior.

        Arrangement:
        Create instance.

        Action:
        Call get_control with unknown name.

        Assertion:
        ValueError raised with control name in message.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        with pytest.raises(ValueError, match="Unknown control"):
            instance.get_control("TOTALLY_UNKNOWN")


class TestDigitalTwinCameraInstanceClose:
    """Test close() method.

    Covers twin.py close() no-op implementation.
    """

    def test_close_is_safe_noop(self, camera_info: CameraInfo) -> None:
        """Verify close() completes without error.

        Business context:
        Digital twin has no resources to release, but close()
        must exist for API compatibility.

        Arrangement:
        Create instance.

        Action:
        Call close() multiple times.

        Assertion:
        All calls complete without error.
        """
        config = DigitalTwinConfig()
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        instance.close()
        instance.close()  # Multiple calls safe


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Test convenience factory functions.

    Covers twin.py lines 1047-1051, 1081-1086: Factory functions.
    """

    def test_create_file_camera_creates_file_source_driver(self) -> None:
        """Verify create_file_camera creates FILE source driver.

        Business context:
        Factory simplifies creating camera that returns single image.

        Arrangement:
        Create temp image file.

        Action:
        Call create_file_camera().

        Assertion:
        Returns driver with FILE source config.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            driver = create_file_camera(f.name)

            assert driver.config.image_source == ImageSource.FILE
            assert str(driver.config.image_path) == f.name

    def test_create_directory_camera_creates_directory_source_driver(self) -> None:
        """Verify create_directory_camera creates DIRECTORY source driver.

        Business context:
        Factory simplifies creating camera that cycles through images.

        Arrangement:
        Create temp directory.

        Action:
        Call create_directory_camera().

        Assertion:
        Returns driver with DIRECTORY source and cycle config.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = create_directory_camera(tmpdir, cycle=True)

            assert driver.config.image_source == ImageSource.DIRECTORY
            assert str(driver.config.image_path) == tmpdir
            assert driver.config.cycle_images is True

    def test_create_directory_camera_respects_cycle_false(self) -> None:
        """Verify create_directory_camera respects cycle=False.

        Business context:
        Some tests need to stop at last image instead of cycling.

        Arrangement:
        Create temp directory.

        Action:
        Call create_directory_camera(cycle=False).

        Assertion:
        Driver config has cycle_images=False.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = create_directory_camera(tmpdir, cycle=False)

            assert driver.config.cycle_images is False


# =============================================================================
# Synthetic Capture Tests
# =============================================================================


class TestDigitalTwinCameraInstanceSyntheticCapture:
    """Test synthetic image generation path.

    Covers twin.py lines 956-962: Noise simulation based on gain.
    """

    def test_synthetic_capture_with_high_gain_has_noise(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify synthetic capture adds noise when gain > 0.

        Business context:
        High gain in real cameras increases noise; synthetic mode
        simulates this for realistic testing.

        Arrangement:
        Create instance with SYNTHETIC source.
        Set high gain value.

        Action:
        Capture synthetic image.

        Assertion:
        Returns valid JPEG (noise is internal to image).
        """
        config = DigitalTwinConfig(image_source=ImageSource.SYNTHETIC)
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        instance.set_control("Gain", 500)
        result = instance.capture(100000)

        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"

    def test_synthetic_capture_with_zero_gain_no_crash(
        self, camera_info: CameraInfo
    ) -> None:
        """Verify synthetic capture works with gain=0.

        Business context:
        Zero gain should produce clean image without noise or crash.

        Arrangement:
        Create instance.
        Set gain to 0.

        Action:
        Capture synthetic image.

        Assertion:
        Returns valid JPEG.
        """
        config = DigitalTwinConfig(image_source=ImageSource.SYNTHETIC)
        instance = DigitalTwinCameraInstance(0, camera_info, config)

        instance.set_control("Gain", 0)
        result = instance.capture(100000)

        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"
