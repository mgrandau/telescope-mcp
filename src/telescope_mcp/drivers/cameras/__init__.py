"""Camera driver module.

Provides camera control for ASI cameras (real hardware) and digital twin
simulation for development without hardware.
"""

from typing import Protocol, runtime_checkable

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


@runtime_checkable
class CameraInstance(Protocol):  # pragma: no cover
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
            >>> img = cv2.imdecode(
            ...     np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
            ... )
        """
        ...

    def close(self) -> None:
        """Close camera connection and release hardware resources.

        Closes camera connection releasing exclusive USB handles, driver
        allocations, and memory buffers. After calling, instance is unusable.
        Must call before opening again or from another process.
        Idempotent - safe to call multiple times.

        Business context: Critical for proper resource management in
        long-running telescope systems. USB cameras hold exclusive device
        handles - failing to close prevents other applications from accessing
        camera until system reboot. Essential for camera hot-swapping,
        graceful shutdown, and freeing USB bandwidth in multi-camera systems.

        Args:
            None.

        Returns:
            None. Side effects: USB device closed, internal state cleared.

        Raises:
            None. Should catch all exceptions for reliable cleanup.

        Example:
            >>> instance = driver.open(0)
            >>> instance.capture(100_000)
            >>> instance.close()  # Release USB handle
        """
        ...


@runtime_checkable
class CameraDriver(Protocol):  # pragma: no cover
    """Protocol for camera drivers enabling hardware abstraction and testing.

    Defines the interface for camera discovery and connection, abstracting
    differences between real hardware (ASI SDK) and simulation (digital twin).
    This protocol enables dependency injection for testability and supports
    running the same application code against real cameras or simulated ones.

    Business context: Critical abstraction layer enabling development and testing
    without physical hardware. Allows CI/CD pipelines to run full integration tests
    with simulated cameras, supports offline development when cameras unavailable,
    and enables demonstrations without hardware setup. The protocol pattern ensures
    application code (Camera, CameraController, tools) works identically with both
    ASI hardware and digital twins.

    Implemented by:
    - ASICameraDriver: Real ZWO ASI cameras via SDK
    - DigitalTwinCameraDriver: Simulated cameras for testing

    Factory pattern: Use DriverFactory.create_camera_driver() to obtain the
    appropriate driver based on configuration (hardware vs simulation mode).

    Implementation details: Drivers should be stateless - all camera-specific
    state lives in CameraInstance. get_connected_cameras() may cache results
    briefly to avoid repeated hardware scans. open() should validate camera_id
    and raise appropriate exceptions for invalid IDs.

    Return format flexibility: Drivers can return either raw dicts (legacy) or
    structured CameraInfo objects (preferred). CameraRegistry converts raw dicts
    to CameraInfo for consistency.
    - dict[int, dict]: Raw info dicts (legacy, converted by registry)
    - dict[int, CameraInfo]: Structured info (preferred)

    Raises:
        None at protocol level. Implementations raise:
        - RuntimeError: Hardware initialization failures
        - ValueError: Invalid camera IDs
        - OSError: USB/driver access issues

    Example:
        >>> # Factory pattern for driver selection
        >>> from telescope_mcp.drivers.config import use_hardware, get_factory
        >>> use_hardware()  # or use_digital_twin()
        >>> factory = get_factory()
        >>> driver = factory.create_camera_driver()
        >>>
        >>> # Discovery and connection
        >>> cameras = driver.get_connected_cameras()
        >>> print(f"Found {len(cameras)} cameras")
        >>> instance = driver.open(0)
        >>> info = instance.get_info()
        >>> print(f"Opened {info['name']}")
    """

    def get_connected_cameras(self) -> dict:
        """Discover and list connected cameras with their capabilities.

        Scans for available cameras and returns their basic information without
        opening connections. For ASI drivers, this queries USB devices. For digital
        twins, this returns configured simulated cameras. Results may be cached
        briefly to avoid repeated hardware scans.

        Business context: Entry point for camera discovery in multi-camera systems.
        Enables UI camera selection menus, automatic camera detection on startup,
        and validation that expected cameras are connected before observation sessions.
        Used by CameraRegistry.discover() to populate the camera registry.

        Implementation details: ASI driver calls asi.get_num_cameras() and
        asi.list_cameras() to enumerate USB devices. Digital twin returns cameras
        from configuration. Consider caching results for 1-5 seconds to avoid
        excessive USB scanning when called repeatedly during discovery.

        The exact return format depends on the driver implementation:
        - Legacy: dict[int, dict] with raw camera properties
        - Preferred: dict[int, CameraInfo] with structured info

        Registry automatically converts legacy format to CameraInfo.

        Returns:
            Dict mapping camera_id (int) to camera info. Camera IDs are 0-based
            indices used for all subsequent operations. Empty dict if no cameras
            found. Keys are always integers starting from 0.

        Raises:
            RuntimeError: If driver initialization failed or SDK unavailable.
            OSError: If USB access denied or driver not loaded.

        Example:
            >>> driver = ASICameraDriver()
            >>> cameras = driver.get_connected_cameras()
            >>> for cam_id, info in cameras.items():
            ...     if isinstance(info, dict):
            ...         print(f"Camera {cam_id}: {info['name']}")
            ...     else:
            ...         print(f"Camera {cam_id}: {info.name}")
            Camera 0: ZWO ASI183MM Pro
            Camera 1: ZWO ASI120MM Mini
        """
        ...

    def open(self, camera_id: int) -> CameraInstance:
        """Open a camera for capture and control operations.

        Establishes exclusive connection to the specified camera, making it
        unavailable to other processes. Returns a CameraInstance for capture
        and control. Camera ID must be from get_connected_cameras() results.

        Business context: Required before any camera operations (capture, control
        changes, info queries). Opens USB connection to hardware or initializes
        simulation state. Connection is exclusive - attempting to open same camera
        twice (even from same process) will fail. Essential for resource management
        in multi-camera systems where cameras must be explicitly acquired/released.

        Implementation details: ASI driver calls ASIOpenCamera() and initializes
        camera controls. Digital twin creates simulation state. Connection should
        be closed via instance.close() when done to release hardware. Consider
        using cameras as context managers to ensure cleanup.

        Args:
            camera_id: Camera ID from get_connected_cameras(), typically 0-based
                index. Must be valid and camera must not already be open.

        Returns:
            CameraInstance for capture and control operations. Instance remains
            valid until close() is called or process exits.

        Raises:
            ValueError: If camera_id invalid or not in get_connected_cameras().
            RuntimeError: If camera already open or hardware failure during open.
            OSError: If USB access denied or device disconnected.

        Example:
            >>> driver = ASICameraDriver()
            >>> cameras = driver.get_connected_cameras()
            >>> if 0 in cameras:
            ...     instance = driver.open(0)
            ...     try:
            ...         info = instance.get_info()
            ...         print(f"Opened {info['name']}")
            ...         data = instance.capture(exposure_us=100_000)
            ...     finally:
            ...         instance.close()
            >>>
            >>> # Context manager pattern (if supported)
            >>> with driver.open(0) as instance:
            ...     data = instance.capture(exposure_us=100_000)
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
