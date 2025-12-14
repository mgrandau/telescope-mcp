"""Tests for driver configuration and digital twin."""

import pytest
from telescope_mcp.drivers import (
    DriverMode,
    DriverConfig,
    DriverFactory,
    get_factory,
    configure,
    use_digital_twin,
    use_hardware,
)
from telescope_mcp.drivers.config import StubCameraDriver, StubCameraInstance


class TestDriverConfig:
    """Tests for driver configuration."""

    def test_default_mode_is_digital_twin(self):
        """Default mode should be digital twin for safety."""
        config = DriverConfig()
        assert config.mode == DriverMode.DIGITAL_TWIN

    def test_configure_changes_factory(self):
        """Configure should update the global factory."""
        use_digital_twin()
        assert get_factory().config.mode == DriverMode.DIGITAL_TWIN
        
        configure(DriverConfig(mode=DriverMode.HARDWARE))
        assert get_factory().config.mode == DriverMode.HARDWARE
        
        # Reset to digital twin
        use_digital_twin()


class TestStubCameraDriver:
    """Tests for the digital twin camera driver."""

    def test_list_cameras(self):
        """Should return simulated camera list."""
        driver = StubCameraDriver()
        cameras = driver.get_connected_cameras()
        assert 0 in cameras
        assert 1 in cameras
        assert b"ASI120MC-S" in cameras[0]["Name"]

    def test_open_camera(self):
        """Should open a simulated camera."""
        driver = StubCameraDriver()
        camera = driver.open(0)
        assert camera is not None
        assert isinstance(camera, StubCameraInstance)

    def test_open_invalid_camera(self):
        """Should raise error for invalid camera ID."""
        driver = StubCameraDriver()
        with pytest.raises(ValueError):
            driver.open(99)


class TestStubCameraInstance:
    """Tests for the digital twin camera instance."""

    def test_get_info(self):
        """Should return camera info."""
        driver = StubCameraDriver()
        camera = driver.open(0)
        info = camera.get_info()
        assert "MaxWidth" in info
        assert "MaxHeight" in info

    def test_get_controls(self):
        """Should return available controls."""
        driver = StubCameraDriver()
        camera = driver.open(0)
        controls = camera.get_controls()
        assert "ASI_GAIN" in controls
        assert "ASI_EXPOSURE" in controls

    def test_set_and_get_control(self):
        """Should set and retrieve control values."""
        driver = StubCameraDriver()
        camera = driver.open(0)
        
        camera.set_control("ASI_GAIN", 75)
        result = camera.get_control("ASI_GAIN")
        assert result["value"] == 75

    def test_capture_returns_jpeg(self):
        """Should return JPEG bytes."""
        driver = StubCameraDriver()
        camera = driver.open(0)
        
        jpeg = camera.capture(exposure_us=100000)
        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        # JPEG magic bytes
        assert jpeg[:2] == b'\xff\xd8'


class TestDriverFactory:
    """Tests for the driver factory."""

    def test_create_camera_driver_digital_twin(self):
        """Should create stub camera driver in digital twin mode."""
        factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
        driver = factory.create_camera_driver()
        assert isinstance(driver, StubCameraDriver)

    def test_create_camera_driver_hardware_not_implemented(self):
        """Should raise NotImplementedError for hardware mode (until implemented)."""
        factory = DriverFactory(DriverConfig(mode=DriverMode.HARDWARE))
        with pytest.raises(NotImplementedError):
            factory.create_camera_driver()

    def test_create_motor_controller_digital_twin(self):
        """Should create stub motor controller in digital twin mode."""
        from telescope_mcp.drivers.motors import StubMotorController
        
        factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
        controller = factory.create_motor_controller()
        assert isinstance(controller, StubMotorController)

    def test_create_position_sensor_digital_twin(self):
        """Should create stub position sensor in digital twin mode."""
        from telescope_mcp.drivers.sensors import StubPositionSensor
        
        factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
        sensor = factory.create_position_sensor()
        assert isinstance(sensor, StubPositionSensor)
