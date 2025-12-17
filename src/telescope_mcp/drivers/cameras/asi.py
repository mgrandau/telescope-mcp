"""ASI Camera Driver - Real Hardware Implementation.

Wraps the zwoasi library to provide camera control for ZWO ASI cameras
following the CameraDriver protocol.

Example:
    from telescope_mcp.drivers.cameras import ASICameraDriver
    from telescope_mcp.devices import CameraRegistry
    
    driver = ASICameraDriver()
    with CameraRegistry(driver) as registry:
        cameras = registry.discover()
        camera = registry.get(0)
        camera.connect()
        result = camera.capture()
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import cv2
import numpy as np
import zwoasi as asi

from telescope_mcp.drivers.asi_sdk import get_sdk_library_path

logger = logging.getLogger(__name__)

# Control name to ASI constant mapping
CONTROL_MAP = {
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


class ASICameraInstance:
    """Opened ASI camera instance.
    
    Wraps a zwoasi.Camera and implements the CameraInstance protocol.
    """
    
    def __init__(self, camera_id: int, camera: asi.Camera):
        """Create camera instance.
        
        Args:
            camera_id: Camera ID
            camera: Opened zwoasi.Camera instance
        """
        self._camera_id = camera_id
        self._camera = camera
        self._info = camera.get_camera_property()
        self._controls = camera.get_controls()
    
    def get_info(self) -> dict:
        """Get camera information.
        
        Returns:
            Dict with camera properties (name, resolution, pixel size, etc.)
        """
        return {
            "camera_id": self._camera_id,
            "name": self._info.get("Name", f"ASI Camera {self._camera_id}"),
            "max_width": self._info["MaxWidth"],
            "max_height": self._info["MaxHeight"],
            "pixel_size_um": self._info.get("PixelSize", 0),
            "is_color": self._info.get("IsColorCam", False),
            "bayer_pattern": self._info.get("BayerPattern", 0),
            "supported_bins": self._info.get("SupportedBins", [1]),
            "supported_formats": self._info.get("SupportedVideoFormat", []),
            "bit_depth": self._info.get("BitDepth", 8),
            "is_usb3": self._info.get("IsUSB3Camera", False),
            "has_cooler": self._info.get("IsCoolerCam", False),
            "has_st4_port": self._info.get("ST4Port", False),
        }
    
    def get_controls(self) -> dict:
        """Get available camera controls.
        
        Returns:
            Dict mapping control name to control info
        """
        result = {}
        for name, ctrl in self._controls.items():
            result[name] = {
                "min_value": ctrl["MinValue"],
                "max_value": ctrl["MaxValue"],
                "default_value": ctrl["DefaultValue"],
                "is_auto_supported": ctrl["IsAutoSupported"],
                "is_writable": ctrl["IsWritable"],
                "description": ctrl.get("Description", ""),
            }
        return result
    
    def set_control(self, control: str, value: int) -> dict:
        """Set a camera control value.
        
        Args:
            control: Control name (e.g., "Gain", "Exposure")
            value: Value to set
            
        Returns:
            Dict with current value and auto status
        """
        if control not in CONTROL_MAP:
            raise ValueError(f"Unknown control: {control}. Valid: {list(CONTROL_MAP.keys())}")
        
        control_id = CONTROL_MAP[control]
        self._camera.set_control_value(control_id, value)
        
        # Read back to confirm
        current_value, is_auto = self._camera.get_control_value(control_id)
        
        return {
            "control": control,
            "value": current_value,
            "auto": is_auto,
        }
    
    def get_control(self, control: str) -> dict:
        """Get current value of a camera control.
        
        Args:
            control: Control name
            
        Returns:
            Dict with current value and auto status
        """
        if control not in CONTROL_MAP:
            raise ValueError(f"Unknown control: {control}. Valid: {list(CONTROL_MAP.keys())}")
        
        control_id = CONTROL_MAP[control]
        current_value, is_auto = self._camera.get_control_value(control_id)
        
        return {
            "control": control,
            "value": current_value,
            "auto": is_auto,
        }
    
    def capture(self, exposure_us: int) -> bytes:
        """Capture a frame.
        
        Args:
            exposure_us: Exposure time in microseconds
            
        Returns:
            JPEG image data as bytes
        """
        # Set exposure
        self._camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
        
        # Set image format to RAW8 for simplicity
        self._camera.set_image_type(asi.ASI_IMG_RAW8)
        
        # Start exposure
        self._camera.start_exposure()
        
        # Wait for exposure to complete
        exposure_sec = exposure_us / 1_000_000
        time.sleep(exposure_sec + 0.1)  # Add buffer
        
        # Poll until exposure is done
        status = self._camera.get_exposure_status()
        max_polls = 100
        polls = 0
        while status == asi.ASI_EXP_WORKING and polls < max_polls:
            time.sleep(0.01)
            status = self._camera.get_exposure_status()
            polls += 1
        
        if status != asi.ASI_EXP_SUCCESS:
            raise RuntimeError(f"Exposure failed with status: {status}")
        
        # Get image data
        img_data = self._camera.get_data_after_exposure()
        
        # Convert to numpy array and reshape
        width = self._info["MaxWidth"]
        height = self._info["MaxHeight"]
        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
        
        # Encode as JPEG
        success, jpeg_data = cv2.imencode('.jpg', img_array)
        if not success:
            raise RuntimeError("Failed to encode image as JPEG")
        
        return jpeg_data.tobytes()
    
    def close(self) -> None:
        """Close the camera."""
        self._camera.close()
        logger.info(f"Closed ASI camera {self._camera_id}")


class ASICameraDriver:
    """ASI Camera Driver for real ZWO hardware.
    
    Wraps the zwoasi library and implements the CameraDriver protocol.
    Handles SDK initialization automatically on first use.
    """
    
    def __init__(self):
        """Create ASI camera driver.
        
        Initializes the ASI SDK on first camera operation.
        """
        self._sdk_initialized = False
    
    def _ensure_sdk_initialized(self) -> None:
        """Initialize ASI SDK if not already done."""
        if not self._sdk_initialized:
            try:
                sdk_path = get_sdk_library_path()
                asi.init(sdk_path)
                self._sdk_initialized = True
                logger.info(f"ASI SDK initialized from {sdk_path}")
            except Exception as e:
                logger.error(f"Failed to initialize ASI SDK: {e}")
                raise RuntimeError(f"ASI SDK initialization failed: {e}") from e
    
    def get_connected_cameras(self) -> dict[int, dict]:
        """Discover connected ASI cameras.
        
        Returns:
            Dict mapping camera_id to camera info dict
        """
        self._ensure_sdk_initialized()
        
        num_cameras = asi.get_num_cameras()
        if num_cameras == 0:
            logger.info("No ASI cameras detected")
            return {}
        
        camera_names = asi.list_cameras()
        
        result = {}
        for camera_id, name in enumerate(camera_names):
            # Open camera briefly to get full info
            try:
                temp_camera = asi.Camera(camera_id)
                info = temp_camera.get_camera_property()
                temp_camera.close()
                
                result[camera_id] = {
                    "camera_id": camera_id,
                    "name": info.get("Name", name),
                    "max_width": info["MaxWidth"],
                    "max_height": info["MaxHeight"],
                    "pixel_size_um": info.get("PixelSize", 0),
                    "is_color": info.get("IsColorCam", False),
                }
            except Exception as e:
                logger.warning(f"Failed to get info for camera {camera_id}: {e}")
                # Fallback to minimal info
                result[camera_id] = {
                    "camera_id": camera_id,
                    "name": name,
                }
        
        logger.info(f"Discovered {len(result)} ASI camera(s)")
        return result
    
    def open(self, camera_id: int) -> ASICameraInstance:
        """Open an ASI camera.
        
        Args:
            camera_id: ID of camera to open (0-based index)
            
        Returns:
            ASICameraInstance for the opened camera
            
        Raises:
            RuntimeError: If camera cannot be opened
        """
        self._ensure_sdk_initialized()
        
        try:
            camera = asi.Camera(camera_id)
            logger.info(f"Opened ASI camera {camera_id}")
            return ASICameraInstance(camera_id, camera)
        except Exception as e:
            logger.error(f"Failed to open camera {camera_id}: {e}")
            raise RuntimeError(f"Cannot open ASI camera {camera_id}: {e}") from e
