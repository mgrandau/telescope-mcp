"""Driver configuration and factory.

Supports switching between real hardware drivers and digital twin (stub) drivers
for testing and development without physical hardware.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from telescope_mcp.drivers.motors import MotorController, StubMotorController
from telescope_mcp.drivers.sensors import PositionSensor, StubPositionSensor


class DriverMode(Enum):
    """Driver mode selection."""
    HARDWARE = "hardware"  # Real hardware drivers
    DIGITAL_TWIN = "digital_twin"  # Simulated drivers for testing


@dataclass
class DriverConfig:
    """Configuration for driver selection."""
    mode: DriverMode = DriverMode.DIGITAL_TWIN
    
    # Camera settings
    finder_camera_id: int = 0
    main_camera_id: int = 1
    
    # Motor settings (for hardware mode)
    motor_serial_port: str | None = None
    motor_baud_rate: int = 115200
    
    # Sensor settings (for hardware mode)
    sensor_i2c_bus: int = 1
    sensor_i2c_address: int = 0x68


class CameraDriver(Protocol):
    """Protocol for camera drivers (real or simulated)."""
    
    def get_connected_cameras(self) -> dict:
        """List connected cameras."""
        ...
    
    def open(self, camera_id: int) -> "CameraInstance":
        """Open a camera."""
        ...


class CameraInstance(Protocol):
    """Protocol for an opened camera."""
    
    def get_info(self) -> dict:
        """Get camera info."""
        ...
    
    def get_controls(self) -> dict:
        """Get available controls."""
        ...
    
    def set_control(self, control: str, value: int) -> dict:
        """Set a control value."""
        ...
    
    def get_control(self, control: str) -> dict:
        """Get a control value."""
        ...
    
    def capture(self, exposure_us: int) -> bytes:
        """Capture a frame, return JPEG bytes."""
        ...
    
    def close(self) -> None:
        """Close the camera."""
        ...


class StubCameraDriver:
    """Digital twin camera driver for testing without hardware."""
    
    def get_connected_cameras(self) -> dict:
        """Return simulated camera list."""
        return {
            0: {
                "Name": b"ASI120MC-S (Simulated)",
                "MaxWidth": 1280,
                "MaxHeight": 960,
                "IsColorCam": True,
                "PixelSize": 3.75,
            },
            1: {
                "Name": b"ASI482MC (Simulated)",
                "MaxWidth": 1920,
                "MaxHeight": 1080,
                "IsColorCam": True,
                "PixelSize": 5.8,
            },
        }
    
    def open(self, camera_id: int) -> "StubCameraInstance":
        """Open a simulated camera."""
        cameras = self.get_connected_cameras()
        if camera_id not in cameras:
            raise ValueError(f"Camera {camera_id} not found")
        return StubCameraInstance(camera_id, cameras[camera_id])


class StubCameraInstance:
    """Digital twin camera instance."""
    
    def __init__(self, camera_id: int, info: dict):
        self._camera_id = camera_id
        self._info = info
        self._controls = {
            "ASI_GAIN": {"value": 50, "auto": False},
            "ASI_EXPOSURE": {"value": 100000, "auto": False},
            "ASI_WB_R": {"value": 52, "auto": False},
            "ASI_WB_B": {"value": 95, "auto": False},
            "ASI_GAMMA": {"value": 50, "auto": False},
            "ASI_BRIGHTNESS": {"value": 50, "auto": False},
        }
    
    def get_info(self) -> dict:
        """Get camera info."""
        return self._info.copy()
    
    def get_controls(self) -> dict:
        """Get available controls."""
        return {
            name: {
                "Name": name,
                "MinValue": 0,
                "MaxValue": 100 if "GAIN" in name else 1000000,
                "DefaultValue": 50,
                "IsAutoSupported": True,
                "IsWritable": True,
            }
            for name in self._controls
        }
    
    def set_control(self, control: str, value: int) -> dict:
        """Set a control value."""
        if control in self._controls:
            self._controls[control]["value"] = value
        return self.get_control(control)
    
    def get_control(self, control: str) -> dict:
        """Get a control value."""
        return self._controls.get(control, {"value": 0, "auto": False})
    
    def capture(self, exposure_us: int) -> bytes:
        """Return a placeholder image."""
        import numpy as np
        import cv2
        
        # Generate a test pattern
        width = self._info["MaxWidth"]
        height = self._info["MaxHeight"]
        
        # Create gradient with some noise
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add grid pattern
        img[::50, :] = [50, 50, 50]
        img[:, ::50] = [50, 50, 50]
        
        # Add center crosshair
        cv2.line(img, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
        cv2.line(img, (0, height // 2), (width, height // 2), (0, 255, 0), 1)
        
        # Add text
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
        
        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()
    
    def close(self) -> None:
        """Close the camera (no-op for stub)."""
        pass


class DriverFactory:
    """Factory for creating drivers based on configuration."""
    
    def __init__(self, config: DriverConfig | None = None):
        self.config = config or DriverConfig()
    
    def create_camera_driver(self) -> CameraDriver:
        """Create camera driver based on config."""
        if self.config.mode == DriverMode.HARDWARE:
            # TODO: Import and return real pyasi driver
            # from telescope_mcp.drivers.pyasi import ASICameraDriver
            # return ASICameraDriver()
            raise NotImplementedError("Hardware camera driver not yet implemented")
        else:
            return StubCameraDriver()
    
    def create_motor_controller(self) -> MotorController:
        """Create motor controller based on config."""
        if self.config.mode == DriverMode.HARDWARE:
            # TODO: Import and return real motor driver
            raise NotImplementedError("Hardware motor driver not yet implemented")
        else:
            return StubMotorController()
    
    def create_position_sensor(self) -> PositionSensor:
        """Create position sensor based on config."""
        if self.config.mode == DriverMode.HARDWARE:
            # TODO: Import and return real sensor driver
            raise NotImplementedError("Hardware position sensor not yet implemented")
        else:
            return StubPositionSensor()


# Global driver factory instance - can be reconfigured
_factory: DriverFactory | None = None


def get_factory() -> DriverFactory:
    """Get the global driver factory."""
    global _factory
    if _factory is None:
        _factory = DriverFactory()
    return _factory


def configure(config: DriverConfig) -> None:
    """Configure the global driver factory."""
    global _factory
    _factory = DriverFactory(config)


def use_digital_twin() -> None:
    """Switch to digital twin mode."""
    configure(DriverConfig(mode=DriverMode.DIGITAL_TWIN))


def use_hardware() -> None:
    """Switch to hardware mode."""
    configure(DriverConfig(mode=DriverMode.HARDWARE))
