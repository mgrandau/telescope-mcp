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
import time
from typing import Any

import cv2
import numpy as np
import zwoasi as asi

from telescope_mcp.drivers.asi_sdk import get_sdk_library_path
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)

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
    Created by ASICameraDriver.open() and should be closed when done.
    """
    
    def __init__(self, camera_id: int, camera: asi.Camera):
        """Create camera instance from opened zwoasi camera.
        
        Queries camera properties and controls from the hardware.
        Should not be called directly - use ASICameraDriver.open().
        
        Args:
            camera_id: Camera ID (0-indexed).
            camera: Opened zwoasi.Camera instance.
        """
        self._camera_id = camera_id
        self._camera = camera
        self._info = camera.get_camera_property()
        self._controls = camera.get_controls()
    
    def get_info(self) -> dict:
        """Get camera information from ASI hardware.
        
        Returns hardware properties queried from the ASI SDK including
        sensor resolution, pixel size, color capability, and features.
        Business context: Camera info is essential for determining capture
        capabilities, calculating field of view, plate solving parameters,
        and GUI display. Sensor specs drive exposure calculations and determine
        whether debayering is needed for color cameras.
        
        Args:
            None. Reads cached info from camera initialization.
        
        Returns:
            Dict with camera properties including:
            - camera_id, name, max_width, max_height
            - pixel_size_um, is_color, bayer_pattern
            - supported_bins, bit_depth, is_usb3
            - has_cooler, has_st4_port
        
        Raises:
            None. Returns cached data from SDK query at open time.
        
        Example:
            >>> info = camera_instance.get_info()
            >>> print(f"{info['name']}: {info['max_width']}x{info['max_height']}")
            >>> if info['is_color']:
            ...     print(f"Color camera with Bayer pattern {info['bayer_pattern']}")
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
        """Get available camera controls from ASI hardware.
        
        Returns control definitions from the ASI SDK including value
        ranges and capabilities for each control.
        Business context: Control definitions enable UI sliders with proper
        ranges, validate user inputs before sending to hardware, and determine
        which features are available (e.g., cooler control, auto-exposure).
        Essential for building adaptive UIs that work across different camera models.
        
        Args:
            None. Returns cached control definitions from SDK.
        
        Returns:
            Dict mapping control name to control info including:
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
        """Set a camera control value on ASI hardware.
        
        Applies the control value to the physical camera via ZWO SDK and reads
        back the actual value set by hardware (which may differ due to clamping
        to valid range or hardware rounding). This is the low-level interface
        to ASI camera control hardware.
        
        Business context: Direct hardware control for ASI cameras enabling real-time
        adjustment of imaging parameters. Used by higher-level Camera class for
        exposure control, gain adjustment, and sensor configuration. Critical for
        automated exposure algorithms, adaptive gain control, and manual optimization
        during imaging sessions. Hardware read-back confirms settings were applied
        successfully.
        
        Args:
            control: Control name from CONTROL_MAP (Gain, Exposure, WB_R, WB_B,
                Offset, Brightness, Gamma, etc.). Must be one of the supported
                ASI camera controls.
            value: Integer value to set. Valid range depends on control and camera
                model. Hardware will clamp to valid range if out of bounds.
            
        Returns:
            Dict with:
            - control: str - Echo of control name
            - value: int - Actual value set by hardware (may differ from requested)
            - is_auto: bool - Whether auto mode is enabled for this control
        
        Raises:
            ValueError: If control name is not in CONTROL_MAP (unsupported control).
            RuntimeError: If SDK fails to apply control (camera disconnected, etc.).
        
        Example:
            >>> instance = ASICameraInstance(camera_obj)
            >>> result = instance.set_control("Gain", 100)
            >>> print(f"Requested 100, hardware set {result['value']}")
            >>> # Set exposure for 5 second frame
            >>> result = instance.set_control("Exposure", 5_000_000)
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
        """Get current value of a camera control from ASI hardware.
        
        Queries the hardware for the current control value and auto status.
        Business context: Real-time control readback enables UI synchronization,
        verifies settings were applied correctly, and monitors read-only controls
        like sensor temperature for cooler management.
        
        Args:
            control: Control name (e.g., "Gain", "Exposure", "Temperature").
            
        Returns:
            Dict with control name, current value, and auto status.
        
        Raises:
            ValueError: If control name is not in CONTROL_MAP.
        
        Example:
            >>> result = camera_instance.get_control("Gain")
            >>> print(f"Current gain: {result['value']}")
            >>> temp = camera_instance.get_control("Temperature")
            >>> print(f"Sensor temp: {temp['value']/10}°C")
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
        """Capture a frame from ASI camera hardware.
        
        Sets exposure, triggers capture, waits for completion, then
        retrieves and encodes the image as JPEG. Uses RAW8 format.
        Business context: Core imaging function for astrophotography. Handles
        hardware timing, buffer management, and JPEG encoding. Exposure times
        range from microseconds (planetary) to minutes (deep sky). JPEG encoding
        enables efficient network transmission for live preview and streaming.
        
        Args:
            exposure_us: Exposure time in microseconds (1000-3600000000 typical).
            
        Returns:
            JPEG-encoded image data as bytes.
        
        Raises:
        
        Example:
            >>> # 5-second exposure for deep sky
            >>> jpeg_data = camera_instance.capture(5_000_000)
            >>> # 1ms exposure for bright planet
            >>> jpeg_data = camera_instance.capture(1000)
        
        Raises:
            RuntimeError: If exposure fails or JPEG encoding fails.
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
        USB cleanup and releases kernel driver locks. No-op if camera already
        closed. Errors during close are logged but not raised.
        
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
        self._camera.close()
        logger.info(f"Closed ASI camera {self._camera_id}")


class ASICameraDriver:
    """ASI Camera Driver for real ZWO hardware.
    
    Wraps the zwoasi library and implements the CameraDriver protocol.
    Handles SDK initialization automatically on first use.
    """
    
    def __init__(self):
        """Create ASI camera driver.
        
        The ASI SDK is initialized lazily on first camera operation,
        not during construction. This allows driver creation even
        when no cameras are connected.
        """
        self._sdk_initialized = False
    
    def _ensure_sdk_initialized(self) -> None:
        """Initialize ASI SDK if not already done.
        
        Loads the ASI SDK library from the bundled location and
        initializes it. Called automatically before camera operations.
        
        Raises:
            RuntimeError: If SDK initialization fails.
        """
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
        """Discover connected ASI cameras via USB enumeration.
        
        Scans USB for connected ZWO cameras, queries basic info
        from each, and returns a summary. Each camera is briefly
        opened to get full properties.
        Business context: Essential for multi-camera setups (guide + imaging),
        auto-discovery in UIs, and device health monitoring. Enables plug-and-play
        camera configuration without manual ID assignment.
        
        Args:
            None. Scans all USB-connected ASI cameras.
        
        Returns:
            Dict mapping camera_id (0-indexed) to camera info dict
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
        """Open an ASI camera for exclusive access.
        
        Establishes connection to the camera hardware and returns
        an instance for control and capture operations.
        Business context: Camera must be opened before any control or capture
        operations. Opening claims exclusive hardware access - only one process
        can open a camera at a time. Essential for preventing conflicts in
        multi-application environments.
        
        Args:
            camera_id: ID of camera to open (0-based index from discovery).
            
        Returns:
            ASICameraInstance for capture and control operations.
            
        Raises:
            RuntimeError: If camera cannot be opened (disconnected, in use).
        
        Example:
            >>> driver = ASICameraDriver()
            >>> cameras = driver.get_connected_cameras()
            >>> if 0 in cameras:
            ...     instance = driver.open(0)
            ...     info = instance.get_info()
        """
        self._ensure_sdk_initialized()
        
        try:
            camera = asi.Camera(camera_id)
            logger.info(f"Opened ASI camera {camera_id}")
            return ASICameraInstance(camera_id, camera)
        except Exception as e:
            logger.error(f"Failed to open camera {camera_id}: {e}")
            raise RuntimeError(f"Cannot open ASI camera {camera_id}: {e}") from e
