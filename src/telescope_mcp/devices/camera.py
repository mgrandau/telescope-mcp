"""Logical camera device with driver injection.

This module provides a hardware-agnostic Camera class that accepts
a driver (real ASI or digital twin) via dependency injection.

Follows SOLID principles:
- Single Responsibility: Camera captures, OverlayRenderer renders
- Open/Closed: New overlay types via new renderers, not Camera changes
- Dependency Inversion: Renderer and Clock are injectable protocols

Example:
    from telescope_mcp.devices import Camera, CameraConfig
    from telescope_mcp.drivers.cameras import ASICameraDriver
    
    driver = ASICameraDriver()
    config = CameraConfig(camera_id=0, name="Main Camera")
    
    with Camera(driver, config) as cam:
        result = cam.capture(exposure_us=100_000)
        print(f"Captured {len(result.image_data)} bytes")
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Protocol

from telescope_mcp.observability import LogContext, get_camera_stats, get_logger

if TYPE_CHECKING:
    from telescope_mcp.drivers.cameras import CameraDriver, CameraInstance

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Protocols (Injectable Dependencies)
# =============================================================================

class Clock(Protocol):
    """Protocol for time functions (injectable for testing).
    
    Defines the interface for time-related operations used by Camera.
    The default implementation uses Python's time module, but this protocol
    allows injecting mock clocks for deterministic testing.
    
    Example:
        class MockClock:
            def __init__(self):
                self._time = 0.0
            
            def monotonic(self) -> float:
                return self._time
            
            def sleep(self, seconds: float) -> None:
                self._time += seconds
        
        camera = Camera(driver, config, clock=MockClock())
    """
    
    def monotonic(self) -> float:
        """Return monotonic time in seconds.
        
        Returns a monotonically increasing time value suitable for measuring
        elapsed time. The reference point is arbitrary but consistent within
        a program run.
        
        Returns:
            Current monotonic time in seconds as a float.
        
        Raises:
            None. Should never fail.
            
        Note:
            Unlike wall-clock time, monotonic time cannot go backwards,
            making it suitable for measuring durations.
        """
        ...
    
    def sleep(self, seconds: float) -> None:
        """Sleep for specified seconds.
        
        Suspends execution for the given duration. Used for rate limiting
        in video streaming to achieve target frame rates.
        
        Args:
            seconds: Duration to sleep in seconds (can be fractional).
        
        Raises:
            None. Should not raise exceptions.
        
        Note:
            Mock implementations may advance internal time instead of
            actually sleeping, enabling fast test execution.
        """
        ...


class SystemClock:
    """Default clock using time module.
    
    Production implementation of the Clock protocol using Python's
    standard time module. This is the default clock injected into
    Camera instances when no custom clock is provided.
    """
    
    def monotonic(self) -> float:
        """Return monotonic time for accurate duration measurement.
        
        Delegates to time.monotonic() which provides a clock that never goes
        backwards, making it suitable for measuring elapsed time. Unlike
        wall-clock time, monotonic time is not affected by system clock
        adjustments (NTP, daylight saving, manual changes).
        
        Business context: Essential for accurate frame timing in video streams
        and performance measurement of camera operations. Using monotonic time
        prevents timing errors when the system clock is adjusted during long
        exposures or streaming sessions. Critical for maintaining consistent
        frame rates in live view.
        
        Implementation details: Returns time.monotonic() directly. The reference
        point (zero) is arbitrary but consistent within a process lifetime.
        Resolution is typically nanoseconds on modern systems. Used by Camera
        class for frame interval calculation and capture duration measurement.
        
        Returns:
            Current monotonic time in seconds as a float. Value is only meaningful
            when compared to other monotonic() calls (for elapsed time calculation).
        
        Raises:
            None. time.monotonic() always succeeds on supported platforms.
        
        Example:
            >>> clock = SystemClock()
            >>> start = clock.monotonic()
            >>> # ... do some work ...
            >>> elapsed = clock.monotonic() - start
            >>> print(f"Operation took {elapsed:.3f}s")
        """
        return time.monotonic()
    
    def sleep(self, seconds: float) -> None:
        """Sleep using time.sleep().
        
        Args:
            seconds: Duration to sleep in seconds.
        
        Raises:
            None. Uses standard library time.sleep().
        """
        time.sleep(seconds)


class OverlayRenderer(Protocol):
    """Protocol for overlay rendering strategies.
    
    Implement this to create new overlay types without modifying Camera.
    Follows the Strategy pattern - swap renderers to change overlay behavior.
    
    Example:
        class CrosshairRenderer:
            def render(self, image_data, config, camera_info):
                # Decode JPEG, draw crosshair, re-encode
                img = decode_jpeg(image_data)
                draw_crosshair(img, config.color, config.opacity)
                return encode_jpeg(img)
        
        camera = Camera(driver, config, renderer=CrosshairRenderer())
    """
    
    def render(
        self, 
        image_data: bytes, 
        config: OverlayConfig,
        camera_info: CameraInfo | None,
    ) -> bytes:
        """Render overlay on image data for live view targeting and alignment.
        
        Takes raw image data and applies the configured overlay. Typical use cases
        include crosshairs for targeting, circles for alignment, grid overlays for
        field of view reference, or custom indicators for platesolving results.
        
        Business context: Essential for visual targeting during telescope operation.
        Overlays guide users in centering objects, verifying alignment, and understanding
        field positioning without modifying captured science data (which uses capture_raw).
        Separation of rendered preview from raw data enables both user-friendly live view
        and accurate data collection.
        
        Implementation details: Renderers should use efficient image processing libraries
        (OpenCV, Pillow) to minimize frame latency. Consider caching decoded images or
        using GPU acceleration for high-resolution cameras. JPEG re-encoding quality should
        balance file size and visual quality (85-95 typically).
        
        Args:
            image_data: Original image as JPEG bytes.
            config: Overlay configuration specifying type, color, opacity, and type-specific
                parameters. Type can be "crosshair", "circles", "grid", "custom", etc.
            camera_info: Camera info for resolution, aspect ratio, and pixel scale. Used to
                scale overlay elements appropriately. May be None if camera info is unavailable
                (renderer should handle gracefully).
            
        Returns:
            Image with overlay applied, as JPEG bytes. Format and dimensions should match
            input image. Quality should be suitable for display but not necessarily archival.
            
        Raises:
            ValueError: If image_data cannot be decoded as JPEG.
            RuntimeError: If rendering fails due to processing error.
            
        Example:
            >>> class CrosshairRenderer:
            ...     def render(self, image_data, config, camera_info):
            ...         # Decode JPEG
            ...         img_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            ...         h, w = img_array.shape[:2]
            ...         
            ...         # Draw crosshair at center
            ...         cv2.line(img_array, (w//2 - 20, h//2), (w//2 + 20, h//2), config.color, 2)
            ...         cv2.line(img_array, (w//2, h//2 - 20), (w//2, h//2 + 20), config.color, 2)
            ...         
            ...         # Re-encode to JPEG
            ...         _, encoded = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
            ...         return encoded.tobytes()
        
        Note:
            Implementations should be efficient as this is called on every frame during
            live streaming (10-30 fps typical). Avoid expensive operations like histogram
            equalization unless specifically requested via config.params.
        """
        ...


class NullRenderer:
    """Default renderer that does nothing (passthrough).
    
    No-op implementation of OverlayRenderer that returns images unchanged.
    Used as the default when no overlay rendering is needed.
    """
    
    def render(
        self, 
        image_data: bytes, 
        config: OverlayConfig,
        camera_info: CameraInfo | None,
    ) -> bytes:
        """Return image data unchanged (no overlay applied).
        
        Passthrough renderer that returns the original image without modifications.
        Used as the default renderer when no overlay functionality is needed or
        when overlays are disabled.
        
        Business context: Enables clean Camera API design by always having a renderer
        present, eliminating null checks. Users can disable overlays without changing
        Camera code. Also serves as a reference implementation showing the minimal
        OverlayRenderer contract.
        
        Args:
            image_data: Original image as JPEG bytes.
            config: Overlay configuration (ignored by this implementation).
            camera_info: Camera info (ignored by this implementation).
            
        Returns:
            The original image_data unchanged. Same bytes object, not a copy.
        
        Raises:
            None. This implementation never fails.
        
        Example:
            >>> renderer = NullRenderer()
            >>> original = b'\xff\xd8\xff\xe0...'  # JPEG bytes
            >>> result = renderer.render(original, config, None)
            >>> result is original  # True - same object returned
            True
        """
        return image_data


class RecoveryStrategy(Protocol):
    """Protocol for camera disconnect recovery.
    
    Implement this to provide custom recovery behavior when
    a camera disconnects unexpectedly. Recovery strategies can
    perform USB reset, device re-enumeration, or other hardware-specific
    operations to restore camera availability.
    
    Example:
        class USBResetRecovery:
            def attempt_recovery(self, camera_id: int) -> bool:
                # Attempt USB device reset
                usb_reset(camera_id)
                time.sleep(2.0)  # Wait for re-enumeration
                return is_camera_available(camera_id)
        
        camera = Camera(driver, config, recovery=USBResetRecovery())
    """
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """Attempt to recover a disconnected camera.
        
        Called when a capture operation fails due to apparent disconnect.
        Implementation should perform hardware-level recovery and check
        if the camera becomes available again.
        
        Args:
            camera_id: ID of the camera to recover (0-indexed).
            
        Returns:
            True if camera is available and ready for reconnection,
            False if recovery failed or camera is not available.
        
        Raises:
            None. Implementations should catch exceptions and return False.
            
        Note:
            This method should be idempotent and safe to call multiple times.
            Recovery may involve blocking operations like USB reset or sleep.
        """
        ...


class NullRecoveryStrategy:
    """No-op recovery strategy (always fails).
    
    Default implementation that does not attempt any recovery.
    Used when no recovery strategy is configured, causing camera
    disconnects to immediately raise CameraDisconnectedError.
    """
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """Return False without attempting any recovery actions.
        
        No-op implementation that immediately fails recovery. Used as the
        default recovery strategy when no specific recovery mechanism is
        configured. Causes camera disconnects to immediately raise
        CameraDisconnectedError without any recovery attempts.
        
        Business context: Provides a safe default that prevents unexpected
        recovery attempts (like USB resets) that might affect other devices.
        Users can opt-in to recovery by injecting a real RecoveryStrategy.
        Follows the Null Object pattern to eliminate null checks in Camera.
        
        Implementation details: Always returns False immediately. No side effects.
        Camera class interprets False return as "recovery failed" and raises
        CameraDisconnectedError. For actual recovery, implement RecoveryStrategy
        with hardware-specific recovery logic (USB reset, driver reload, etc.).
        
        Args:
            camera_id: ID of the camera to recover (0-based index). Ignored by
                this implementation since no recovery is attempted.
            
        Returns:
            Always False, indicating recovery was not attempted and camera is
            not available.
        
        Raises:
            None. This implementation never raises exceptions.
        
        Example:
            >>> strategy = NullRecoveryStrategy()
            >>> success = strategy.attempt_recovery(0)
            >>> print(success)  # Always False
            False
            >>> 
            >>> # This is the default when no recovery specified
            >>> camera = Camera(driver, config)  # Uses NullRecoveryStrategy
            >>> # On disconnect, Camera will immediately raise CameraDisconnectedError
        """
        return False


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class CameraConfig:
    """Configuration for camera initialization.
    
    Attributes:
        camera_id: Hardware camera ID (0, 1, etc.)
        name: Optional friendly name for logging
        default_gain: Default gain value (0-100 typical)
        default_exposure_us: Default exposure in microseconds
    """
    camera_id: int
    name: str | None = None
    default_gain: int = 50
    default_exposure_us: int = 100_000


@dataclass
class CameraInfo:
    """Information about a connected camera.
    
    Returned by Camera.connect() with details about the camera's
    capabilities and current state.
    """
    camera_id: int
    name: str
    max_width: int
    max_height: int
    is_color: bool
    bayer_pattern: str | None
    supported_bins: list[int]
    controls: dict[str, dict]  # control_name -> {min, max, default, value}
    
    @classmethod
    def from_driver_info(cls, info: dict, controls: dict) -> CameraInfo:
        """Create CameraInfo from driver response dictionaries.
        
        Factory method that constructs a CameraInfo instance from the raw
        dictionary responses returned by camera drivers. Handles missing
        keys gracefully with sensible defaults, making it resilient to
        driver variations and incomplete data.
        
        Business context: Bridges the gap between driver implementations
        and the Camera class interface. Different drivers (ASI, digital twin)
        may return slightly different dictionary structures. This factory
        method normalizes them into a consistent CameraInfo structure,
        enabling polymorphic driver usage without Camera code changes.
        Essential for the dependency injection pattern used throughout the
        telescope control system.
        
        Implementation details: Uses dict.get() with defaults for all fields
        to handle missing keys gracefully. Defaults are chosen to be safe
        (non-crashing) but obviously wrong (width=0) so issues are visible.
        The controls dict is passed through unchanged since Camera class
        knows how to handle its structure. Consider validating that required
        fields (width, height) are non-zero if more strict validation is needed.
        
        Args:
            info: Camera information dictionary from driver.get_info().
                Expected keys: camera_id, name, max_width, max_height,
                is_color, bayer_pattern, supported_bins.
                Missing keys get sensible defaults.
            controls: Control values dictionary from driver.get_controls().
                Maps control names to dicts with min, max, default, value.
                Passed through unchanged.
            
        Returns:
            CameraInfo instance populated with driver data and defaults.
            All fields will be populated even if info dict is incomplete.
        
        Raises:
            None. Missing keys result in default values, not exceptions.
        
        Example:
            >>> # Normal case with complete data
            >>> driver_info = camera_instance.get_info()
            >>> driver_controls = camera_instance.get_controls()
            >>> info = CameraInfo.from_driver_info(driver_info, driver_controls)
            >>> print(f"Camera: {info.name}, {info.max_width}x{info.max_height}")
            Camera: ZWO ASI183MM Pro, 5496x3672
            >>> 
            >>> # Handles incomplete data gracefully
            >>> partial_info = {"name": "Test Camera"}
            >>> info = CameraInfo.from_driver_info(partial_info, {})
            >>> print(f"{info.name}: {info.max_width}x{info.max_height}")
            Test Camera: 0x0
        """
        return cls(
            camera_id=info.get("camera_id", 0),
            name=info.get("name", "Unknown"),
            max_width=info.get("max_width", 0),
            max_height=info.get("max_height", 0),
            is_color=info.get("is_color", False),
            bayer_pattern=info.get("bayer_pattern"),
            supported_bins=info.get("supported_bins", [1]),
            controls=controls,
        )


@dataclass
class OverlayConfig:
    """Configuration for image overlays.
    
    Overlays are rendered on top of captured images for live view,
    targeting, alignment, etc. Use apply_overlay=False in CaptureOptions
    to get images without overlays for science/ASDF data.
    
    Attributes:
        enabled: Whether overlay is active
        overlay_type: Type of overlay ("crosshair", "circles", "grid", "custom")
        color: RGB color tuple (0-255 each)
        opacity: Overlay opacity (0.0-1.0)
        params: Type-specific parameters (e.g., line_width, num_circles)
    """
    enabled: bool = False
    overlay_type: str = "none"
    color: tuple[int, int, int] = (255, 0, 0)
    opacity: float = 0.8
    params: dict = field(default_factory=dict)


@dataclass
class CaptureOptions:
    """Options for a single capture operation.
    
    Consolidates all capture parameters into a single object
    for cleaner API and easier extension.
    
    Attributes:
        exposure_us: Exposure time in microseconds (None = use current)
        gain: Gain value (None = use current)
        apply_overlay: Whether to apply configured overlay
        format: Output format ("jpeg" or "raw")
    """
    exposure_us: int | None = None
    gain: int | None = None
    apply_overlay: bool = True
    format: Literal["jpeg", "raw"] = "jpeg"


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class CaptureResult:
    """Result of a frame capture.
    
    Contains the image data and metadata about the capture settings.
    """
    image_data: bytes
    timestamp: datetime
    exposure_us: int
    gain: int
    width: int = 0
    height: int = 0
    format: str = "jpeg"
    metadata: dict = field(default_factory=dict)
    has_overlay: bool = False


@dataclass 
class StreamFrame:
    """A single frame from a video stream.
    
    Attributes:
        image_data: Frame data (JPEG or raw bytes)
        timestamp: When frame was captured
        sequence_number: Frame number in stream
        exposure_us: Exposure time used
        gain: Gain value used
        has_overlay: Whether overlay was applied
    """
    image_data: bytes
    timestamp: datetime
    sequence_number: int
    exposure_us: int
    gain: int
    has_overlay: bool = False


# =============================================================================
# Event Hooks
# =============================================================================

@dataclass
class CameraHooks:
    """Optional callbacks for camera events.
    
    Use hooks for logging, metrics, or custom behavior without
    modifying Camera class.
    
    Attributes:
        on_connect: Called after successful connection
        on_disconnect: Called after disconnection
        on_capture: Called after each capture (before overlay)
        on_stream_frame: Called for each stream frame
        on_error: Called when an error occurs
    """
    on_connect: Callable[[CameraInfo], None] | None = None
    on_disconnect: Callable[[], None] | None = None
    on_capture: Callable[[CaptureResult], None] | None = None
    on_stream_frame: Callable[[StreamFrame], None] | None = None
    on_error: Callable[[Exception], None] | None = None


# =============================================================================
# Exceptions
# =============================================================================

class CameraError(Exception):
    """Base exception for camera operations."""
    pass


class CameraNotConnectedError(CameraError):
    """Raised when operation requires connected camera."""
    pass


class CameraAlreadyConnectedError(CameraError):
    """Raised when connecting to already-connected camera."""
    pass


class CameraDisconnectedError(CameraError):
    """Raised when camera disconnects and recovery fails."""
    pass


# =============================================================================
# Camera Class
# =============================================================================

class Camera:
    """Logical camera device with injected dependencies.
    
    Provides a clean interface for camera operations independent of
    the underlying hardware. Accepts injectable dependencies for
    testability and extensibility.
    
    Injectable Dependencies:
        - driver: Camera hardware driver (required)
        - renderer: Overlay renderer (optional, default: NullRenderer)
        - clock: Time functions (optional, default: SystemClock)
        - hooks: Event callbacks (optional)
    
    Can be used as a context manager for automatic disconnect:
    
        with Camera(driver, config) as cam:
            result = cam.capture()
    
    Or manually:
    
        cam = Camera(driver, config)
        cam.connect()
        try:
            result = cam.capture()
        finally:
            cam.disconnect()
    
    Attributes:
        config: Camera configuration
        is_connected: Whether camera is currently connected
        info: Camera info (None if not connected)
    """
    
    def __init__(
        self, 
        driver: CameraDriver, 
        config: CameraConfig,
        renderer: OverlayRenderer | None = None,
        clock: Clock | None = None,
        hooks: CameraHooks | None = None,
        recovery: RecoveryStrategy | None = None,
    ) -> None:
        """Create camera with injected dependencies.
        
        Initializes a Camera instance with the specified driver and optional
        dependencies. Does not connect to the camera - call connect() or use
        as a context manager to establish the connection.
        
        Args:
            driver: Camera driver implementation (ASICameraDriver or DigitalTwinDriver).
            config: Camera configuration including ID, name, and default settings.
            renderer: Overlay renderer for live view (default: NullRenderer passthrough).
            clock: Time functions for stream timing (default: SystemClock).
            hooks: Optional callbacks for camera events (connect, capture, error).
            recovery: Recovery strategy for disconnect handling (default: NullRecoveryStrategy).
        
        Example:
            from telescope_mcp.drivers.cameras import ASICameraDriver
            
            driver = ASICameraDriver()
            config = CameraConfig(camera_id=0, name="Main", default_gain=50)
            
            # Basic usage
            camera = Camera(driver, config)
            
            # With dependencies
            camera = Camera(
                driver,
                config,
                renderer=CrosshairRenderer(),
                hooks=CameraHooks(on_capture=log_capture),
            )
        """
        self._driver = driver
        self._config = config
        self._renderer = renderer or NullRenderer()
        self._clock = clock or SystemClock()
        self._hooks = hooks or CameraHooks()
        self._recovery = recovery or NullRecoveryStrategy()
        
        self._instance: CameraInstance | None = None
        self._info: CameraInfo | None = None
        self._current_gain: int = config.default_gain
        self._current_exposure_us: int = config.default_exposure_us
        self._overlay: OverlayConfig | None = None
        self._frame_count: int = 0
        self._streaming: bool = False
    
    @property
    def config(self) -> CameraConfig:
        """Get the camera configuration.
        
        Returns the CameraConfig passed during construction, containing
        camera ID, name, and default settings.
        
        Returns:
            The CameraConfig instance for this camera.
        
        Raises:
            None. Config is always available.
        """
        return self._config
    
    @property
    def is_connected(self) -> bool:
        """Check if camera is currently connected.
        
        Returns True when the camera has an active connection established
        via connect() and not yet disconnected via disconnect().
        
        Returns:
            True if connected and ready for capture, False otherwise.
        
        Raises:
            None. Always returns a boolean.
        """
        return self._instance is not None
    
    @property
    def info(self) -> CameraInfo | None:
        """Get camera info (None if not connected).
        
        Returns detailed information about the camera's capabilities
        including resolution, color mode, supported bins, and controls.
        Only available after calling connect().
        
        Returns:
            CameraInfo with camera capabilities, or None if not connected.
        """
        return self._info
    
    @property
    def overlay(self) -> OverlayConfig | None:
        """Get current overlay configuration.
        
        Returns the overlay settings configured via set_overlay().
        When enabled, overlays are applied to captured frames.
        
        Returns:
            OverlayConfig if set, None if no overlay configured.
        
        Raises:
            None. Returns None when no overlay configured.
        """
        return self._overlay
    
    @property
    def is_streaming(self) -> bool:
        """Check if camera is currently streaming.
        
        Returns True while iterating over a stream() generator,
        False otherwise or after stop_stream() is called.
        
        Returns:
            True if actively streaming frames, False otherwise.
        
        Raises:
            None. Always returns a boolean.
        """
        return self._streaming
    
    def connect(self) -> CameraInfo:
        """Connect to camera and return camera info.
        
        Opens a connection to the physical camera via the injected driver,
        queries camera capabilities, and applies default settings (gain, exposure).
        
        Returns:
            CameraInfo with camera capabilities and control values.
            
        Raises:
            CameraAlreadyConnectedError: If already connected to this camera.
            CameraError: If connection fails (camera not found, driver error).
            
        Example:
            camera = Camera(driver, CameraConfig(camera_id=0))
            info = camera.connect()
            print(f"Connected to {info.name}: {info.max_width}x{info.max_height}")
        
        Note:
            After connection, default gain and exposure are applied from config.
            Use the context manager form for automatic disconnect on exit.
        """
        if self._instance is not None:
            raise CameraAlreadyConnectedError(
                f"Camera {self._config.camera_id} is already connected"
            )
        
        camera_id = self._config.camera_id
        logger.info("Connecting to camera", camera_id=camera_id)
        
        try:
            self._instance = self._driver.open(camera_id)
            
            # Get camera info from driver
            driver_info = self._instance.get_info()
            driver_controls = self._instance.get_controls()
            
            self._info = CameraInfo.from_driver_info(driver_info, driver_controls)
            
            # Apply default settings
            self._instance.set_control("Gain", self._config.default_gain)
            self._instance.set_control("Exposure", self._config.default_exposure_us)
            self._current_gain = self._config.default_gain
            self._current_exposure_us = self._config.default_exposure_us
            
            # Fire hook
            if self._hooks.on_connect:
                self._hooks.on_connect(self._info)
            
            logger.info(
                "Camera connected",
                camera_id=camera_id,
                name=self._info.name,
                resolution=f"{self._info.max_width}x{self._info.max_height}",
            )
            return self._info
            
        except Exception as e:
            self._instance = None
            self._info = None
            logger.error(
                "Camera connection failed",
                camera_id=camera_id,
                error=str(e),
            )
            if self._hooks.on_error:
                self._hooks.on_error(e)
            raise CameraError(f"Failed to connect to camera {camera_id}: {e}") from e
    
    def disconnect(self) -> None:
        """Disconnect from camera and release resources.
        
        Closes the camera connection and releases hardware resources.
        Safe to call even if not connected (no-op in that case).
        Fires the on_disconnect hook if configured.
        
        Raises:
            None. Errors during disconnect are logged but not raised.
        
        Note:
            Any errors during disconnect are logged but not raised.
            The camera is marked as disconnected regardless of errors.
        """
        if self._instance is not None:
            camera_id = self._config.camera_id
            logger.info("Disconnecting camera", camera_id=camera_id)
            try:
                self._instance.close()
                logger.debug("Camera disconnected", camera_id=camera_id)
            except Exception as e:
                logger.warning(
                    "Error during camera disconnect",
                    camera_id=camera_id,
                    error=str(e),
                )
            finally:
                self._instance = None
                self._info = None
                
                # Fire hook
                if self._hooks.on_disconnect:
                    self._hooks.on_disconnect()
    
    def capture(self, options: CaptureOptions | None = None) -> CaptureResult:
        """Capture a frame with specified options.
        
        Takes a single exposure and returns the image data with metadata.
        Supports exposure/gain override and optional overlay application.
        
        Args:
            options: Capture options (exposure, gain, overlay, format).
                If None, uses current settings with overlay applied.
            
        Returns:
            CaptureResult with JPEG image data and capture metadata.
            
        Raises:
            CameraNotConnectedError: If camera is not connected.
            CameraError: If capture fails.
            CameraDisconnectedError: If camera disconnects and recovery fails.
            
        Example:
            # Default capture with overlay
            result = camera.capture()
            
            # Custom settings, no overlay (for science data)
            result = camera.capture(CaptureOptions(
                exposure_us=500_000,
                gain=80,
                apply_overlay=False,
            ))
        """
        opts = options or CaptureOptions()
        result = self._capture_internal(opts.exposure_us, opts.gain)
        
        # Fire hook (before overlay)
        if self._hooks.on_capture:
            self._hooks.on_capture(result)
        
        # Apply overlay if requested and configured
        if opts.apply_overlay and self._overlay and self._overlay.enabled:
            result = self._apply_overlay(result)
        
        return result
    
    def capture_raw(
        self,
        exposure_us: int | None = None,
        gain: int | None = None,
    ) -> CaptureResult:
        """Capture a frame WITHOUT overlay (for ASDF/science data).
        
        Convenience method for capturing frames intended for data analysis
        or storage in ASDF sessions. Equivalent to calling capture() with
        apply_overlay=False.
        
        Args:
            exposure_us: Exposure time in microseconds, or None to use current.
            gain: Gain value (0-100 typical), or None to use current.
            
        Returns:
            CaptureResult with raw JPEG image data (no overlay applied).
            
        Raises:
            CameraNotConnectedError: If camera is not connected.
            CameraError: If capture fails.
            
        Example:
            # Capture for ASDF session storage
            result = camera.capture_raw(exposure_us=1_000_000, gain=50)
            session.add_frame(result.image_data, result.metadata)
        """
        return self.capture(CaptureOptions(
            exposure_us=exposure_us,
            gain=gain,
            apply_overlay=False,
        ))
    
    def _capture_internal(
        self,
        exposure_us: int | None = None,
        gain: int | None = None,
    ) -> CaptureResult:
        """Internal capture without overlay logic.
        
        Core capture implementation that handles settings application,
        driver communication, statistics recording, and automatic recovery
        on disconnect. Called by public capture methods. Not intended for
        external use - call capture() or capture_quick() instead.
        
        Args:
            exposure_us: Exposure time in microseconds, or None for current.
            gain: Gain value, or None for current.
            
        Returns:
            CaptureResult with raw image data (overlay not applied).
            
        Raises:
            CameraNotConnectedError: If camera is not connected.
            CameraDisconnectedError: If disconnect recovery fails.
        
        Note:
            Records capture statistics via observability module.
            On failure, attempts recovery via injected RecoveryStrategy.
        """
        if self._instance is None:
            raise CameraNotConnectedError("Camera is not connected")
        
        camera_id = self._config.camera_id
        stats = get_camera_stats()
        start_time = self._clock.monotonic()
        
        # Apply overrides if provided
        effective_exposure = exposure_us if exposure_us is not None else self._current_exposure_us
        effective_gain = gain if gain is not None else self._current_gain
        
        # Update settings if changed
        if effective_gain != self._current_gain:
            self.set_control("Gain", effective_gain)
        if effective_exposure != self._current_exposure_us:
            self.set_control("Exposure", effective_exposure)
        
        try:
            image_data = self._instance.capture(effective_exposure)
            duration_ms = (self._clock.monotonic() - start_time) * 1000
            
            # Record successful capture stats
            stats.record_capture(
                camera_id=camera_id,
                duration_ms=duration_ms,
                success=True,
            )
            
            logger.debug(
                "Frame captured",
                camera_id=camera_id,
                exposure_us=effective_exposure,
                gain=effective_gain,
                size_bytes=len(image_data),
                duration_ms=round(duration_ms, 1),
            )
            
            return CaptureResult(
                image_data=image_data,
                timestamp=datetime.now(timezone.utc),
                exposure_us=effective_exposure,
                gain=effective_gain,
                width=self._info.max_width if self._info else 0,
                height=self._info.max_height if self._info else 0,
                format="jpeg",
                metadata={
                    "camera_id": camera_id,
                    "camera_name": self._config.name or self._info.name if self._info else "Unknown",
                    "capture_duration_ms": round(duration_ms, 1),
                },
                has_overlay=False,
            )
        except Exception as e:
            duration_ms = (self._clock.monotonic() - start_time) * 1000
            
            # Record failed capture stats
            stats.record_capture(
                camera_id=camera_id,
                duration_ms=duration_ms,
                success=False,
                error_type=type(e).__name__,
            )
            
            logger.warning(
                "Capture failed, attempting recovery",
                camera_id=camera_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            
            # Attempt recovery from disconnect
            return self._recover_and_capture(
                exposure_us=effective_exposure,
                gain=effective_gain,
                original_error=e,
            )
    
    def _recover_and_capture(
        self,
        exposure_us: int,
        gain: int,
        original_error: Exception,
    ) -> CaptureResult:
        """Attempt to recover from camera disconnect and retry capture.
        
        Called when a capture operation fails. Uses the injected RecoveryStrategy
        to attempt hardware-level recovery (e.g., USB reset), then reconnects
        and retries the capture. This is an internal method not intended for
        external use.
        
        Args:
            exposure_us: Exposure time in microseconds to use on retry.
            gain: Gain value to use on retry.
            original_error: The exception that triggered recovery attempt.
            
        Returns:
            CaptureResult if recovery and retry succeed.
            
        Raises:
            CameraDisconnectedError: If recovery fails or retry fails.
        
        Note:
            Recovery metadata is added to successful results for debugging.
            Failed recovery fires the on_error hook if configured.
        """
        camera_id = self._config.camera_id
        stats = get_camera_stats()
        
        # Clear stale instance
        self._instance = None
        self._info = None
        
        logger.info("Attempting camera recovery", camera_id=camera_id)
        
        # Attempt recovery via injected strategy
        if not self._recovery.attempt_recovery(camera_id):
            logger.error(
                "Camera recovery failed",
                camera_id=camera_id,
                original_error=str(original_error),
            )
            stats.record_capture(
                camera_id=camera_id,
                duration_ms=0,
                success=False,
                error_type="recovery_failed",
            )
            if self._hooks.on_error:
                self._hooks.on_error(original_error)
            raise CameraDisconnectedError(
                f"Camera {camera_id} disconnected and recovery failed"
            ) from original_error
        
        # Camera is available - reconnect
        try:
            self.connect()
            logger.info("Camera recovered successfully", camera_id=camera_id)
            
            # Retry capture
            start_time = self._clock.monotonic()
            image_data = self._instance.capture(exposure_us)
            duration_ms = (self._clock.monotonic() - start_time) * 1000
            
            # Record recovered capture
            stats.record_capture(
                camera_id=camera_id,
                duration_ms=duration_ms,
                success=True,
            )
            
            return CaptureResult(
                image_data=image_data,
                timestamp=datetime.now(timezone.utc),
                exposure_us=exposure_us,
                gain=gain,
                width=self._info.max_width if self._info else 0,
                height=self._info.max_height if self._info else 0,
                format="jpeg",
                metadata={
                    "camera_id": camera_id,
                    "camera_name": self._config.name or self._info.name if self._info else "Unknown",
                    "recovered": True,
                    "capture_duration_ms": round(duration_ms, 1),
                },
                has_overlay=False,
            )
        except Exception as e:
            if self._hooks.on_error:
                self._hooks.on_error(e)
            raise CameraDisconnectedError(
                f"Camera {self._config.camera_id} recovery failed on reconnect"
            ) from e
    
    def set_overlay(self, config: OverlayConfig | None) -> None:
        """Set or clear overlay configuration.
        
        Configures the overlay to be applied to captured frames when
        CaptureOptions.apply_overlay is True. Overlays are rendered
        by the injected OverlayRenderer.
        
        Args:
            config: Overlay configuration specifying type, color, opacity.
                Pass None to disable overlays entirely.
        
        Raises:
            None. Always succeeds.
        
        Example:
            # Enable crosshair overlay for targeting
            camera.set_overlay(OverlayConfig(
                enabled=True,
                overlay_type="crosshair",
                color=(0, 255, 0),
                opacity=0.9,
            ))
            
            # Disable overlay
            camera.set_overlay(None)
        """
        self._overlay = config
    
    def stream(
        self,
        options: CaptureOptions | None = None,
        max_fps: float = 30.0,
    ) -> Iterator[StreamFrame]:
        """Yield continuous frames for live view and monitoring.
        
        Creates a generator that continuously captures frames at up to max_fps
        rate. Designed for live preview displays, real-time alignment, focus
        verification, and object tracking. Uses the injected Clock for precise
        frame timing and rate limiting. Streaming continues until stop_stream()
        is called or the generator is closed.
        
        Business context: Essential for interactive telescope operation where
        users need real-time feedback for focusing, alignment, and targeting.
        Enables "eyepiece view" functionality in digital telescopes. Used by
        web dashboards (MJPEG streams), desktop GUIs, and automated focusing
        algorithms. Frame rate balances responsiveness (higher FPS) against
        bandwidth and CPU usage (lower FPS).
        
        Implementation details: Uses Python generator (yield) for memory
        efficiency - only one frame in memory at a time. Rate limiting uses
        injected clock's sleep() for testability. Captures via self.capture()
        which applies settings, overlay, and fires hooks. The actual frame rate
        may be lower than max_fps if exposure time + processing time exceeds
        the frame interval. For 30 fps with overlays, keep exposure under 25ms.
        Generator cleanup (finally block) ensures _streaming flag is reset.
        
        Args:
            options: Capture options applied to each frame. If None, uses current
                camera settings with overlay enabled. Set apply_overlay=False for
                faster streaming without overlay rendering.
            max_fps: Maximum frames per second (1-120 typical). Actual rate is
                limited by min(max_fps, 1/(exposure_time + processing_time)).
                Common values: 30 fps for smooth live view, 10-15 fps for
                bandwidth-limited situations, 5 fps for long exposures.
            
        Yields:
            StreamFrame with image data (JPEG bytes), timestamp, sequence number
            (starting at 0), exposure_us, gain, and has_overlay flag. Frames are
            yielded continuously until stop_stream() called or generator closed.
            
        Raises:
            CameraNotConnectedError: If camera is not connected when streaming starts.
            CameraError: If capture fails during streaming. Generator exits on error.
        
        Example:
            >>> # Basic streaming for live view
            >>> camera.connect()
            >>> for frame in camera.stream(max_fps=15):
            ...     display_frame(frame.image_data)
            ...     if user_clicked_stop:
            ...         camera.stop_stream()
            ...         break
            >>> 
            >>> # Stream without overlay for faster frame rate
            >>> opts = CaptureOptions(exposure_us=10_000, apply_overlay=False)
            >>> for frame in camera.stream(opts, max_fps=30):
            ...     if frame.sequence_number >= 100:  # Capture 100 frames
            ...         camera.stop_stream()
            ...     analyze_frame(frame.image_data)
        
        Note:
            Frame rate is limited by max_fps and actual capture time.
            The on_stream_frame hook is called for each frame if configured.
            Use stop_stream() for clean shutdown rather than breaking the loop
            abruptly, as it ensures the streaming flag is cleared properly.
        """
        if self._instance is None:
            raise CameraNotConnectedError("Camera is not connected")
        
        opts = options or CaptureOptions()
        min_interval = 1.0 / max_fps
        self._streaming = True
        self._frame_count = 0
        
        try:
            while self._streaming:
                start = self._clock.monotonic()
                
                # Capture frame
                result = self.capture(opts)
                
                frame = StreamFrame(
                    image_data=result.image_data,
                    timestamp=result.timestamp,
                    sequence_number=self._frame_count,
                    exposure_us=result.exposure_us,
                    gain=result.gain,
                    has_overlay=result.has_overlay,
                )
                
                # Fire hook
                if self._hooks.on_stream_frame:
                    self._hooks.on_stream_frame(frame)
                
                yield frame
                
                self._frame_count += 1
                
                # Rate limiting using injected clock
                elapsed = self._clock.monotonic() - start
                if elapsed < min_interval:
                    self._clock.sleep(min_interval - elapsed)
        finally:
            self._streaming = False
    
    def stop_stream(self) -> None:
        """Signal the stream to stop after current frame.
        
        Sets the internal streaming flag to False, causing the stream()
        generator to exit cleanly after completing the current frame.
        Safe to call from another thread or callback.
        
        Raises:
            None. Always succeeds, even if not streaming.
        
        Note:
            Does not immediately interrupt capture - the current frame
            will complete before the generator exits.
        """
        self._streaming = False
    
    def _apply_overlay(self, result: CaptureResult) -> CaptureResult:
        """Apply configured overlay using injected renderer.
        
        Delegates overlay rendering to the injected OverlayRenderer and
        returns a new CaptureResult with the rendered image. Original
        result is not modified.
        
        Args:
            result: Original capture result without overlay.
            
        Returns:
            New CaptureResult with overlay applied and has_overlay=True.
            Metadata includes the overlay_type.
        
        Note:
            Returns original result unchanged if no overlay is configured.
        """
        if not self._overlay:
            return result
        
        # Delegate to injected renderer
        rendered_data = self._renderer.render(
            result.image_data,
            self._overlay,
            self._info,
        )
        
        return CaptureResult(
            image_data=rendered_data,
            timestamp=result.timestamp,
            exposure_us=result.exposure_us,
            gain=result.gain,
            width=result.width,
            height=result.height,
            format=result.format,
            metadata={**result.metadata, "overlay_type": self._overlay.overlay_type},
            has_overlay=True,
        )
    
    def set_control(self, name: str, value: int) -> None:
        """Set a camera control value.
        
        Updates a camera hardware control such as gain, exposure, or white
        balance. Common controls like Gain and Exposure are tracked internally
        for state management and used as defaults in capture operations.
        
        Args:
            name: Control name (e.g., "Gain", "Exposure", "WB_R", "WB_B").
                Control names are camera-specific but common ones include
                Gain (0-100), Exposure (microseconds), WB_R/WB_B (white balance).
            value: Integer value to set. Valid range depends on the specific
                control - check camera.info.controls for min/max values.
            
        Raises:
            CameraNotConnectedError: If camera is not connected.
            CameraError: If driver rejects the control name or value.
            
        Example:
            camera.set_control("Gain", 80)
            camera.set_control("Exposure", 500_000)  # 500ms
            camera.set_control("WB_R", 52)  # Red white balance
        """
        if self._instance is None:
            raise CameraNotConnectedError("Camera is not connected")
        
        try:
            self._instance.set_control(name, value)
            
            # Track common controls
            if name == "Gain":
                self._current_gain = value
            elif name == "Exposure":
                self._current_exposure_us = value
                
        except Exception as e:
            if self._hooks.on_error:
                self._hooks.on_error(e)
            raise CameraError(f"Failed to set {name}={value}: {e}") from e
    
    def get_control(self, name: str) -> int:
        """Get current value of a camera control.
        
        Queries the camera hardware for the current value of a control.
        Use camera.info.controls for control ranges and defaults.
        
        Args:
            name: Control name (e.g., "Gain", "Exposure").
            
        Returns:
            Current integer value of the control.
            
        Raises:
            CameraNotConnectedError: If camera is not connected.
            CameraError: If control name is invalid or query fails.
            
        Example:
            current_gain = camera.get_control("Gain")
            current_exposure = camera.get_control("Exposure")
        """
        if self._instance is None:
            raise CameraNotConnectedError("Camera is not connected")
        
        try:
            result = self._instance.get_control(name)
            return result.get("value", 0)
        except Exception as e:
            if self._hooks.on_error:
                self._hooks.on_error(e)
            raise CameraError(f"Failed to get {name}: {e}") from e
    
    # Context manager support
    
    def __enter__(self) -> Camera:
        """Enter context manager, connect to camera.
        
        Establishes camera connection on context entry, enabling the
        'with' statement pattern for automatic resource management.
        
        Returns:
            Self (the Camera instance) for use in the with block.
        """
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, disconnect from camera.
        
        Ensures camera disconnection on context exit, regardless of
        whether an exception occurred. Does not suppress exceptions.
        
        Args:
            exc_type: Exception type if an exception was raised, else None.
            exc_val: Exception value if an exception was raised, else None.
            exc_tb: Traceback if an exception was raised, else None.
        """
        self.disconnect()
    
    def __repr__(self) -> str:
        """Return string representation of camera.
        
        Returns:
            String like '<Camera(name, connected)>' or '<Camera(name, disconnected)>'.
        
        Raises:
            None. Always returns a valid string.
        """
        status = "connected" if self.is_connected else "disconnected"
        name = self._config.name or f"camera_{self._config.camera_id}"
        return f"<Camera({name}, {status})>"
