"""Camera driver module.

Provides camera control for ASI cameras (real hardware) and digital twin
simulation for development without hardware.
"""

from typing import Protocol

from telescope_mcp.drivers.cameras.asi import (
    ASICameraDriver,
    ASICameraInstance,
)
from telescope_mcp.drivers.cameras.twin import (
    DigitalTwinCameraDriver,
    DigitalTwinConfig,
    ImageSource,
    create_directory_camera,
    create_file_camera,
)


class CameraInstance(Protocol):
    """Protocol for an opened camera instance.
    
    Represents a connection to a specific camera for capture and control.
    Implemented by ASICameraInstance and DigitalTwinCameraInstance.
    """

    def get_info(self) -> dict:
        """Get camera hardware information and capabilities.
        
        Queries the camera for its static properties including model name,
        sensor resolution, pixel size, color/mono designation, and supported
        binning modes. This information is constant for a camera and doesn't
        change between connections.
        
        Business context: Essential for configuring capture parameters, displaying
        camera capabilities to users, and validating requested operations against
        hardware limits. Used by Camera class to populate CameraInfo and by UI
        to show available cameras with their specs.
        
        Implementation details: ASI cameras return properties via SDK calls.
        Digital twin may load from config or infer from sample images. Return
        format should match the camera info structure expected by Camera class.
        
        Args:
            None. Queries the already-opened camera instance.
        
        Returns:
            Dict with camera properties:
            - camera_id: int - Camera ID (0-based)
            - name: str - Model name (e.g., "ZWO ASI183MM Pro")
            - max_width: int - Maximum image width in pixels
            - max_height: int - Maximum image height in pixels
            - is_color: bool - True for color (Bayer), False for mono
            - bayer_pattern: str | None - Bayer pattern if color (e.g., "RGGB")
            - supported_bins: list[int] - Supported binning modes (e.g., [1, 2, 4])
            - pixel_size_um: float - Physical pixel size in micrometers
        
        Raises:
            RuntimeError: If camera hardware fails to respond or disconnected.
            ValueError: If camera returns malformed data.
        
        Example:
            >>> instance = driver.open(0)
            >>> info = instance.get_info()
            >>> print(f"{info['name']}: {info['max_width']}x{info['max_height']}")
            ZWO ASI183MM Pro: 5496x3672
            >>> print(f"Pixel size: {info['pixel_size_um']}µm")
            Pixel size: 2.4µm
        """
        ...

    def get_controls(self) -> dict:
        """Get all available camera controls with current values and ranges.
        
        Queries the camera for its adjustable parameters such as gain, exposure,
        white balance, offset, etc. Returns each control's valid range, current
        value, default, and whether it's currently set to auto. Control availability
        and ranges vary by camera model.
        
        Business context: Enables dynamic UI generation showing available controls
        with valid ranges, preventing invalid value submissions. Essential for
        building adaptive camera control interfaces that work across different
        camera models without hardcoding control lists.
        
        Implementation details: ASI cameras enumerate controls via SDK. Digital twin
        returns simulated controls with reasonable ranges. Control values should be
        cached after first query as they don't change except via set_control().
        Common controls include: Gain, Exposure, Offset, WB_R, WB_B, Brightness,
        BandwidthOverload, HighSpeedMode, Flip.
        
        Args:
            None. Queries the already-opened camera instance.
        
        Returns:
            Dict mapping control names (str) to control info dicts:
            {
                "Gain": {
                    "min": 0,
                    "max": 600,
                    "default": 50,
                    "value": 50,
                    "is_auto": False,
                    "is_writable": True,
                },
                "Exposure": {...},
                ...
            }
        
        Raises:
            RuntimeError: If camera hardware fails to respond or disconnected.
            ValueError: If camera returns malformed control data.
        
        Example:
            >>> instance = driver.open(0)
            >>> controls = instance.get_controls()
            >>> gain_control = controls['Gain']
            >>> print(f"Gain range: {gain_control['min']}-{gain_control['max']}")
            Gain range: 0-600
            >>> print(f"Current: {gain_control['value']}")
            Current: 50
        """
        ...

    def set_control(self, control: str, value: int) -> dict:
        """Set a camera control value and return updated state.
        
        Updates the specified control parameter (gain, exposure, etc.) to the
        given value. The camera hardware applies the setting immediately for
        subsequent captures. Returns the actual value set (may differ from
        requested if clamped to valid range) and current auto status.
        
        Business context: Core interface for adjusting camera settings during
        operation. Enables automated exposure control, gain adjustment for
        different targets, and white balance correction. Used by both manual
        user controls and automated algorithms like auto-exposure.
        
        Args:
            control: Control name (e.g., "Gain", "Exposure", "WB_R", "Offset").
                Must be one of the controls returned by get_controls().
            value: Value to set in control-specific units. For Exposure, this is
                microseconds. For Gain/Offset/etc., this is the control's native
                integer range. Driver may clamp to valid range.
        
        Returns:
            Dict with updated control state:
            {
                "value": int,      # Actual value set (may be clamped)
                "is_auto": bool,   # Whether auto mode is active
            }
        
        Raises:
            ValueError: If control name is invalid or value outside hardware limits.
            RuntimeError: If camera hardware fails to apply setting.
            KeyError: If control name not found in available controls.
        
        Example:
            >>> instance = driver.open(0)
            >>> result = instance.set_control("Gain", 100)
            >>> print(f"Gain set to {result['value']}")
            Gain set to 100
            >>> result = instance.set_control("Exposure", 500_000)
            >>> print(f"Exposure: {result['value']}µs")
            Exposure: 500000µs
        """
        ...

    def get_control(self, control: str) -> dict:
        """Get current value of a specific camera control.
        
        Queries the camera for the current value and auto status of a single
        control parameter. More efficient than calling get_controls() when only
        one control's current value is needed. Used to verify settings after
        changes or monitor controls that may change automatically.
        
        Business context: Enables verification that requested settings were
        applied correctly, monitoring of auto-exposure/gain changes, and
        status display in UIs. Essential for closed-loop control algorithms
        that need to read back current settings.
        
        Implementation details: For ASI cameras, this is a direct SDK query.
        Digital twin returns cached value from most recent set_control(). Consider
        caching results briefly to avoid repeated hardware queries during rapid
        status checks.
        
        Args:
            control: Control name to query (e.g., "Gain", "Exposure").
                Must be one of the controls returned by get_controls().
        
        Returns:
            Dict with control state:
            {
                "value": int,      # Current value in control-specific units
                "is_auto": bool,   # Whether auto mode is active
            }
        
        Raises:
            KeyError: If control name not found in available controls.
            RuntimeError: If camera hardware fails to respond.
            ValueError: If camera returns invalid data for this control.
        
        Example:
            >>> instance = driver.open(0)
            >>> instance.set_control("Gain", 100)
            >>> result = instance.get_control("Gain")
            >>> print(f"Current gain: {result['value']}")
            Current gain: 100
            >>> print(f"Auto: {result['is_auto']}")
            Auto: False
        """
        ...

    def capture(self, exposure_us: int) -> bytes:
        """Capture a single frame with specified exposure time.
        
        Initiates an exposure, waits for completion, retrieves the image data,
        and returns it as JPEG-encoded bytes. This is a blocking operation that
        takes at minimum the exposure duration plus readout time. Current gain
        and other control settings are applied from previous set_control() calls.
        
        Business context: Primary data acquisition interface for both live preview
        and scientific imaging. Used by Camera.capture() which adds overlay logic
        and metadata. Must be efficient for streaming (10-30 fps) yet robust enough
        for long exposures (seconds to minutes) used in astrophotography.
        
        Implementation details: ASI cameras use SDK capture_video_frame() for
        live preview or capture() for still images. Digital twin may return
        pre-loaded images or generate synthetic frames. JPEG compression quality
        should balance file size and image quality (85-95 typical). Consider
        timeout handling for long exposures to detect camera disconnects.
        
        Args:
            exposure_us: Exposure time in microseconds. Range 1-3,600,000,000
                (1µs to 1 hour) typical, actual limits depend on camera model.
                Longer exposures increase sensitivity but also motion blur and noise.
        
        Returns:
            JPEG-encoded image bytes. Image dimensions match camera resolution
            (full frame or binned depending on camera mode). Format is standard
            JPEG suitable for cv2.imdecode() or PIL.Image.open().
        
        Raises:
            ValueError: If exposure_us outside valid range for this camera.
            RuntimeError: If capture fails (camera disconnect, timeout, hardware error).
            TimeoutError: If capture exceeds expected duration (exposure + 5s typical).
        
        Example:
            >>> instance = driver.open(0)
            >>> instance.set_control("Gain", 50)
            >>> image_data = instance.capture(exposure_us=100_000)  # 100ms
            >>> print(f"Captured {len(image_data)} bytes")
            Captured 524288 bytes
            >>> # Decode and display
            >>> img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        """
        ...

    def close(self) -> None:
        """Close the camera and release resources."""
        ...


class CameraDriver(Protocol):
    """Protocol for camera drivers (real or simulated).
    
    Defines the interface for camera discovery and connection.
    Implemented by ASICameraDriver (real hardware) and
    DigitalTwinCameraDriver (simulation).
    
    Drivers can return either:
    - dict[int, dict]: Raw info dicts (legacy, converted by registry)
    - dict[int, CameraInfo]: Structured info (preferred)
    """

    def get_connected_cameras(self) -> dict:
        """Discover and list connected cameras.
        
        Scans for available cameras and returns their basic information.
        The exact return format depends on the driver implementation.
        
        Returns:
            Dict mapping camera_id to camera info (dict or CameraInfo).
        """
        ...

    def open(self, camera_id: int) -> CameraInstance:
        """Open a camera for capture operations.
        
        Establishes connection to the specified camera and returns
        an instance for control and capture.
        
        Args:
            camera_id: ID of camera to open (from get_connected_cameras).
            
        Returns:
            CameraInstance for capture and control operations.
        """
        ...


__all__ = [
    "CameraDriver",
    "CameraInstance",
    "ASICameraDriver",
    "ASICameraInstance",
    "DigitalTwinCameraDriver",
    "DigitalTwinConfig",
    "ImageSource",
    "create_directory_camera",
    "create_file_camera",
]
