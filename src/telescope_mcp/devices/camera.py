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
        """Monotonic time for accurate duration measurement immune to clock adjustments.
        
        Returns monotonically increasing time value suitable for measuring elapsed time (durations,
        intervals, timeouts). Reference point arbitrary but consistent within process. Value never
        decreases, unaffected by system clock adjustments (NTP sync, daylight saving, manual changes).
        
        Business context: Essential for accurate frame timing in video streams and performance
        measurement of camera operations. System clock adjustments during long exposures or
        multi-hour streaming sessions would cause negative durations or incorrect frame intervals
        if using wall-clock time. Monotonic time prevents these errors, ensuring consistent frame
        rates in live view and accurate capture duration logging. Critical for time-sensitive
        telescope operations (exposures timed to millisecond precision).
        
        Implementation details: Standard implementation returns time.monotonic(). Reference point
        (zero) platform-specific, typically system boot time or process start. Resolution typically
        nanoseconds on modern systems. Used by Camera for elapsed time calculations:
        `elapsed = clock.monotonic() - start`. Mock clocks return internal time variable for
        deterministic testing without actual wall-clock time passing.
        
        Returns:
            Current monotonic time in seconds as float. Value only meaningful when compared to
            other monotonic() calls to calculate elapsed time. Absolute value has no significance.
            Precision typically microseconds (6 decimal places).
        
        Raises:
            None. Should never fail on supported platforms. SystemClock implementation uses
            time.monotonic() which always succeeds.
        
        Example:
            >>> clock = SystemClock()
            >>> start = clock.monotonic()
            >>> # ... capture frame (100ms exposure) ...
            >>> elapsed = clock.monotonic() - start
            >>> print(f\"Capture took {elapsed:.3f}s\")  # Output: "Capture took 0.102s"
            
        Note:
            Unlike wall-clock time, monotonic time immune to: NTP synchronization, daylight saving
            transitions, manual clock adjustments, leap seconds. Perfect for measuring durations but
            meaningless as absolute timestamp (use datetime.now() for timestamps).
        """
        ...
    
    def sleep(self, seconds: float) -> None:
        """Sleep for specified duration without busy-waiting.
        
        Suspends execution for given duration, yielding CPU to other threads/processes. Used for
        frame rate limiting in video streaming to achieve target FPS without consuming 100% CPU
        in tight capture loops. Mock implementations advance internal time without actual sleep
        for fast test execution (100x+ speedup).
        
        Business context: Video streaming must maintain consistent frame rates (30fps, 15fps, etc.)
        for smooth live preview. Without sleep, Camera would capture as fast as possible (~100+fps),
        saturating CPU/USB bandwidth, generating unnecessary frames, causing UI lag. Sleep maintains
        target rate while keeping system responsive. Essential for long-duration live view sessions
        during telescope alignment, focusing, or target acquisition where thermal issues could arise
        from sustained high CPU usage.
        
        Implementation details: SystemClock uses time.sleep() (suspends thread). Mock clocks
        increment internal time variable without blocking, enabling tests to simulate hours of
        streaming in milliseconds. Camera calculates sleep duration: `frame_interval - capture_time`.
        Short sleeps (<10ms) have reduced accuracy due to OS scheduler quantum. Fractional seconds
        supported (e.g., 0.033 for 30fps). Interruptible by signals (Ctrl+C).
        
        Args:
            seconds: Duration to sleep in seconds as float (fractional values supported). Typical
                range 0.001-1.0 for streaming (1ms-1s). Zero/negative values return immediately
                (no-op). For 30fps: 0.033s, 15fps: 0.067s, 10fps: 0.100s.
            
        Raises:
            None under normal operation. SystemClock.sleep() uses time.sleep() which may raise
            KeyboardInterrupt on Ctrl+C (intended for interrupting streaming). Mock clocks never
            raise.
        
        Example:
            >>> clock = SystemClock()
            >>> clock.sleep(0.1)  # Sleep 100ms for frame rate limiting
            
            >>> # Mock clock for fast testing:
            >>> mock_clock = MockClock()
            >>> mock_clock.sleep(3600.0)  # Instant - no actual wait
            >>> assert mock_clock.monotonic() == 3600.0  # Time advanced
            
        Note:
            Mock clock pattern enables deterministic testing: tests run at full speed while simulating
            real-time behavior. Critical for CI/CD where waiting real time would slow test suites.
            Production always uses SystemClock with real sleep.
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
        """Sleep for specified seconds without busy-waiting.
        
        Delegates to time.sleep() which suspends thread execution, yielding CPU to other
        threads/processes. Used by Camera for frame rate control in video streaming, preventing
        busy loops that would consume 100% CPU.
        
        Business context: Video streaming must maintain consistent frame rates (e.g., 30fps =
        33.3ms between frames) for smooth live view. Sleep between captures prevents CPU saturation
        that would interfere with image processing, UI responsiveness, and other system processes.
        Critical for long-duration live preview sessions where thermal throttling could occur.
        
        Implementation details: Uses time.sleep() which suspends the calling thread. Resolution
        platform-dependent (typically 1-15ms on Linux/Windows). Camera calculates target sleep
        by subtracting capture duration from frame interval. Short sleeps (<10ms) may be
        inaccurate due to OS scheduler granularity. Interruptible by signals on Unix systems.
        
        Args:
            seconds: Duration to sleep in seconds (float, can be fractional). Typical range
                0.001-1.0 for frame rate control (1ms-1s). Zero or negative values return
                immediately (time.sleep handles gracefully).
            
        Raises:
            None. time.sleep() doesn't raise on normal usage. May raise KeyboardInterrupt on
            Ctrl+C but that's intended behavior for interrupting streaming.
        
        Example:
            >>> clock = SystemClock()
            >>> start = clock.monotonic()
            >>> clock.sleep(0.1)  # Sleep 100ms
            >>> elapsed = clock.monotonic() - start
            >>> assert 0.095 <= elapsed <= 0.110  # Approximate due to scheduling
        
        Note:
            Mock clocks override this for testing, advancing internal time without actually sleeping,
            enabling tests to run 100x faster. Production code always uses real sleep via time.sleep().
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
        """Hardware-level recovery from camera disconnect (USB reset, driver reload).
        
        Called when capture fails due to apparent disconnect. Implementation performs
        hardware-level recovery operations (USB device reset, driver reload, device re-enumeration,
        vendor-specific recovery commands), waits for device stabilization, checks if camera
        becomes available again. Must be blocking until recovery completes or fails.
        
        Business context: USB cameras in long-running observatory sessions experience transient
        disconnects from power glitches, USB enumeration failures, driver hangs, cable issues.
        Hardware recovery (especially USB reset) often restores camera without manual intervention,
        preventing observation failures in remote/unattended operations. Strategy pattern allows
        custom recovery per camera type (ZWO vs QHY vs QHYCCD have different reset mechanisms).
        
        Implementation details: Typical implementations call OS-level USB reset (Linux: ioctl
        USBDEVFS_RESET, Windows: device restart), wait 2-5 seconds for re-enumeration, query
        camera presence via driver's get_connected_cameras(). Should be idempotent - safe to
        call multiple times. Should catch all exceptions and return False (Camera expects boolean,
        not exception). May log recovery attempts internally. Blocking operation OK - called from
        capture flow which already blocks on exposure.
        
        Args:
            camera_id: 0-based camera ID to recover. Used to identify specific USB device for
                reset. For multi-camera systems, must target correct device.
            
        Returns:
            True if camera available and ready for reconnection (driver.open() should succeed).
            False if recovery failed, camera not available, or exceptions occurred during recovery.
            Camera class interprets True as "retry connect", False as "raise CameraDisconnectedError".
        
        Raises:
            None. Implementations must catch all exceptions and return False. Raising would crash
            Camera's recovery flow - always return False on error.
        
        Example:
            >>> class USBResetRecovery:
            ...     def attempt_recovery(self, camera_id: int) -> bool:
            ...         try:
            ...             usb_reset_device(camera_id)  # OS-specific USB reset
            ...             time.sleep(3.0)  # Wait for re-enumeration
            ...             return camera_id in get_available_camera_ids()
            ...         except Exception:
            ...             return False  # Never raise
            
        Note:
            Should be idempotent and safe to call repeatedly. May take 2-10 seconds (USB reset +
            device enumeration). NullRecoveryStrategy always returns False (no recovery attempt).
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
        
        Returns:
            None. Initializes internal state without connecting to hardware.
        
        Raises:
            None. Construction never fails - connection errors occur in connect().
        
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
        """Get camera configuration for ID, name, and default settings access.
        
        Returns CameraConfig passed during construction with camera ID, friendly name,
        and default gain/exposure. Read-only view - changes don't affect camera behavior
        (use set_control for runtime changes). Always available even when disconnected.
        
        Business context: Enables logging and debugging by providing camera identity and
        defaults. Used by diagnostics to show which camera is which in multi-camera setups,
        by automated routines to restore defaults after experiments. The camera_id is
        particularly important for driver operations and hardware identification.
        
        Returns:
            CameraConfig with camera_id (int), name (str|None), default_gain (int),
            default_exposure_us (int). Actual current settings may differ - use get_control().
        
        Raises:
            None. Config always available from construction.
        
        Example:
            >>> camera = Camera(driver, CameraConfig(camera_id=0, name="Main", default_gain=50))
            >>> cfg = camera.config
            >>> print(f"{cfg.name} (ID {cfg.camera_id}): gain={cfg.default_gain}")
            Main (ID 0): gain=50
        """
        return self._config
    
    @property
    def is_connected(self) -> bool:
        """Check if camera is currently connected and ready for operations.
        
        Returns True when camera has active connection via connect() and not yet
        disconnected. Indicates readiness for capture, control changes, info queries.
        Fast local check, no hardware queries.
        
        Business context: Critical for conditional operations in multi-camera systems
        where cameras may connect/disconnect dynamically. Used by UI to enable/disable
        controls, by scripts to verify readiness before capture sequences, by recovery
        logic to detect disconnections. Prevents errors from operations on disconnected cameras.
        
        Returns:
            True if connected and ready for capture(), set_control(), get_control(). False
            if not yet connected or after disconnect(). False doesn't mean hardware unavailable,
            just that this instance hasn't established connection.
        
        Raises:
            None. Always returns boolean, never raises.
        
        Example:
            >>> camera = Camera(driver, config)
            >>> print(camera.is_connected)  # False - not connected yet
            False
            >>> camera.connect()
            >>> print(camera.is_connected)  # True
            True
            >>> if camera.is_connected:
            ...     result = camera.capture()
        """
        return self._instance is not None
    
    @property
    def info(self) -> CameraInfo | None:
        """Get camera capabilities and hardware information (None if not connected).
        
        Returns detailed camera info including resolution, color/mono designation, binning
        modes, available controls with ranges. Queried from hardware during connect() and
        cached. Only available after connect(), returns None if not connected.
        
        Business context: Essential for configuring captures within hardware limits,
        displaying camera specs to users, validating requested settings. Used by UI to show
        properties, by scripts to adapt to available cameras (color vs mono), by validation
        to ensure exposure/gain within control ranges. Critical for multi-camera systems.
        
        Returns:
            CameraInfo with camera_id, name (e.g. "ZWO ASI183MM Pro"), max_width/height,
            is_color, bayer_pattern (str|None), supported_bins (list[int]), controls (dict).
            None if not connected.
        
        Raises:
            None. Returns None when not connected, never raises.
        
        Example:
            >>> camera.connect()
            >>> info = camera.info
            >>> if info:
            ...     print(f"{info.name}: {info.max_width}x{info.max_height}")
            ...     print(f"Color: {info.is_color}, Bins: {info.supported_bins}")
            ZWO ASI183MM Pro: 5496x3672
            Color: False, Bins: [1, 2, 4]
        """
        return self._info
    
    @property
    def overlay(self) -> OverlayConfig | None:
        """Get current overlay configuration for visual targeting and alignment.
        
        Returns overlay settings configured via set_overlay(). When enabled and capture
        options specify apply_overlay=True, overlays (crosshairs, grids, circles) are
        rendered via OverlayRenderer. Purely visual - raw captures unaffected.
        
        Business context: Enables visual targeting aids during live view and alignment
        without modifying science data. Users enable crosshairs for centering, grid overlays
        for field reference, circles for plate solving. Overlay state persists across captures.
        Essential for interactive control where visual feedback guides adjustments.
        
        Returns:
            OverlayConfig with enabled (bool), overlay_type (str), color (tuple), opacity (float),
            params (dict) if configured. None if no overlay or after set_overlay(None).
        
        Raises:
            None. Returns None when no overlay, never raises.
        
        Example:
            >>> camera.set_overlay(OverlayConfig(enabled=True, overlay_type="crosshair", color=(0,255,0)))
            >>> overlay = camera.overlay
            >>> if overlay:
            ...     print(f"Overlay: {overlay.overlay_type}")
            Overlay: crosshair
        """
        return self._overlay
    
    @property
    def is_streaming(self) -> bool:
        """Check if camera is currently streaming continuous frames.
        
        Returns True while iterating over stream() generator, False otherwise or after
        stop_stream(). Indicates continuous capture mode for live view/monitoring.
        Independent of connection state - must be both connected AND streaming.
        
        Business context: Enables coordinated control in multi-camera systems. UIs use this
        to show/hide stop buttons, scripts check before starting new streams (preventing
        conflicts), diagnostics identify active streams. Prevents accidentally starting
        multiple concurrent streams causing hardware contention.
        
        Returns:
            True if stream() generator actively running and yielding frames. False if not
            streaming, generator exited, or stop_stream() called. False doesn't mean disconnected.
        
        Raises:
            None. Always returns boolean, never raises.
        
        Example:
            >>> camera.connect()
            >>> print(camera.is_streaming)  # False
            False
            >>> for frame in camera.stream(max_fps=15):
            ...     if camera.is_streaming:
            ...         display(frame.image_data)
            ...     break
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
        """Disconnect from camera and release hardware resources safely.
        
        Closes connection, releases hardware (USB), resets internal state. Safe to call
        even if not connected (no-op). Errors logged but not raised, ensuring cleanup
        completes. Fires on_disconnect hook if configured.
        
        Business context: Essential for resource management in multi-camera systems where
        cameras must be explicitly released. USB cameras allow only one connection - disconnect()
        makes camera available to other processes. Used when switching cameras, during error
        recovery, at session end. Critical for long-running services where leaks would exhaust cameras.
        
        Args:
            None.
        
        Returns:
            None. Side effects: USB device closed, internal state cleared.
        
        Raises:
            None. Errors logged and suppressed. This ensures disconnect() can be called in
            cleanup code (finally blocks, __exit__) without propagating exceptions that mask
            original errors.
        
        Example:
            >>> camera.connect()
            >>> camera.capture()
            >>> camera.disconnect()  # Release USB handle
        
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
        """Internal capture implementation handling settings, driver calls, and recovery.
        
        Core capture handling settings application, driver communication, performance stats
        recording, automatic recovery on disconnect. Called by public methods (capture(),
        capture_raw(), capture_quick()). Not for external use - call public methods which add
        overlay rendering and cleaner interfaces.
        
        Business context: Centralizes all capture logic ensuring consistent behavior. Handles
        complex flow: applying settings, calling driver, measuring performance, recording metrics,
        detecting failures, attempting recovery. By keeping internal, public methods stay simple
        while complex logic is tested once. Critical for reliable production operation where camera
        disconnects (USB issues, power) must be handled gracefully without crashing.
        
        Implementation details: Uses monotonic clock for accurate duration measurement. Updates
        _current_gain/_current_exposure_us for state tracking. Records success/failure to observability.
        On exception, calls _recover_and_capture() using injected RecoveryStrategy. CaptureResult
        includes raw JPEG with has_overlay=False - caller applies overlay if needed.
        
        Args:
            exposure_us: Exposure time in microseconds. None uses _current_exposure_us. Typical
                range 1-3,600,000,000 (1µs to 1 hour) but actual limits per camera model.
            gain: Gain value. None uses _current_gain. Typical 0-600 but varies by model.
            
        Returns:
            CaptureResult with raw JPEG, timestamp (UTC), effective exposure/gain, width/height,
            format="jpeg", metadata (camera_id/name/duration_ms), has_overlay=False.
            
        Raises:
            CameraNotConnectedError: If not connected (is_connected is False). Call connect() first.
            CameraDisconnectedError: If capture fails AND recovery fails. Original error chained.
        
        Example:
            >>> result = camera._capture_internal(exposure_us=100_000, gain=50)
            >>> print(f\"Captured {len(result.image_data)} bytes in {result.metadata['capture_duration_ms']}ms\")
        
        Note:
            Records stats via get_camera_stats().record_capture() for monitoring. Success logs at
            DEBUG, failures at WARNING. Recovery attempt logged at INFO/ERROR.
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
        """Hardware recovery and retry after disconnect during capture.
        
        Handles camera disconnect recovery using injected RecoveryStrategy (hardware-level operations:
        USB reset, driver reload, device re-enumeration), reconnects camera, retries capture with
        same settings. Marks result metadata showing recovery occurred. Chains original error if
        recovery or reconnect fails.
        
        Business context: USB cameras in long-running observatory operations experience transient
        issues (power glitches, USB enumeration failures, driver hangs). Automated recovery prevents
        observation session failures requiring manual intervention. Critical for unattended operation
        (remote observatories, multi-hour imaging sessions). RecoveryStrategy pattern allows custom
        recovery (e.g., different USB reset methods, vendor-specific commands).
        
        Implementation details: Calls recovery_strategy.attempt_recovery() (may take seconds for
        USB reset), reconnects via connect(), retries _instance.capture(). Clears stale instance
        before recovery. Metadata includes "recovered_from_error": original_error type name. Fires
        on_camera_disconnected() hook after successful recovery for notification/logging. Records
        recovery stats via get_camera_stats(). Fires on_error hook on recovery failure.
        
        Args:
            exposure_us: Exposure time in microseconds for retry capture. Should match failed
                capture settings for consistency. Typical range 1-3,600,000,000 depending on camera.
            gain: Gain value for retry capture. Should match failed settings. Typical 0-600.
            original_error: Exception that triggered recovery (typically CameraError or USB error).
                Chained in CameraDisconnectedError if recovery fails via `from original_error`.
            
        Returns:
            CaptureResult with successfully captured image after recovery. Metadata includes
            "recovered_from_error" key with original error type name, indicating frame captured
            post-recovery (important for data provenance and quality assessment).
            
        Raises:
            CameraDisconnectedError: If recovery fails (recovery_strategy returns False) OR
                reconnect fails (connect() raises) OR retry capture fails. Original error chained
                for debugging. Hook on_error fired before raising.
        
        Example:
            >>> try:
            ...     result = camera._capture_internal(exposure_us=100_000, gain=50)
            ... except Exception as e:
            ...     result = camera._recover_and_capture(exposure_us=100_000, gain=50, original_error=e)
            >>> assert "recovered_from_error" in result.metadata
            >>> print(f"Recovered from {result.metadata['recovered_from_error']}")
        
        Note:
            Logs recovery at INFO (start/success) and ERROR (failure). Recovery takes 2-10 seconds
            (USB reset, device enumeration). Idempotent - safe to call multiple times. Clears
            _instance and _info before recovery to force clean reconnection. Stats recorded with
            error_type="recovery_failed" on failure for monitoring.
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
        """Set or clear overlay configuration for visual targeting aids.
        
        Configures overlay applied to frames when CaptureOptions.apply_overlay is True.
        Overlays rendered by injected OverlayRenderer (crosshairs, grids, circles). Purely
        visual - raw captures via capture_raw() unaffected. Persists across captures.
        
        Business context: Enables interactive targeting in telescope control. Users toggle
        crosshairs when centering objects, add grids for field reference, draw circles around
        plate-solved stars. Separation from raw data ensures science frames remain unmodified
        while previews get visual aids. Essential for manual operation where visual feedback
        guides adjustments.
        
        Args:
            config: OverlayConfig with enabled (bool), overlay_type ("crosshair", "grid", "circles"),
                color (RGB tuple), opacity (0.0-1.0), params (dict). Pass None to disable.
        
        Raises:
            None. Always succeeds. Invalid config fails during capture() rendering.
        
        Example:
            >>> camera.set_overlay(OverlayConfig(
            ...     enabled=True, overlay_type="crosshair",
            ...     color=(0, 255, 0), opacity=0.9
            ... ))
            >>> result = camera.capture()  # Has crosshair
            >>> camera.set_overlay(None)  # Disable
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
        """Signal stream to stop gracefully after current frame.
        
        Sets internal streaming flag to False, causing stream() generator to exit cleanly
        after completing current frame. Doesn't abort capture in progress. Safe from another
        thread, callback, or event loop. Idempotent - safe multiple times or when not streaming.
        
        Business context: Essential for user-initiated stream termination in interactive telescope
        control. Users click "Stop" in dashboards/GUIs, triggering this from UI threads while stream()
        runs in background. Scripts use this to stop after N frames or when conditions change. Graceful
        shutdown ensures current frame completes, preventing corruption and allowing cleanup.
        
        Args:
            None.
        
        Returns:
            None. Side effects: _streaming flag set False, causing stream() to exit.
        
        Raises:
            None. Always succeeds. Calling when not streaming is harmless (no-op).
        
        Example:
            >>> for frame in camera.stream(max_fps=15):
            ...     display(frame.image_data)
            ...     if user_clicked_stop:
            ...         camera.stop_stream()
            ...         break
        
        Note:
            Prefer calling stop_stream() before breaking from loop to ensure clean shutdown.
            Thread-safe because boolean assignment is atomic.
        
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
        
        Business context: Separates overlay rendering from capture logic, enabling different
        overlay strategies (crosshairs, grids, platesolve markers) via dependency injection.
        Called by capture() when overlay enabled, not by capture_raw() ensuring science data
        remains unmodified.
        
        Args:
            result: Original capture result without overlay.
            
        Returns:
            New CaptureResult with overlay applied and has_overlay=True.
            Metadata includes the overlay_type.
        
        Raises:
            None explicitly. Renderer errors propagate to caller.
        
        Example:
            >>> result = camera._capture_internal(100_000, 50)
            >>> overlaid = camera._apply_overlay(result)  # If overlay configured
        
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
        """Set camera control value for gain, exposure, and other parameters.
        
        Updates camera hardware control (gain, exposure, white balance, offset, bandwidth).
        Common controls like Gain and Exposure tracked internally for state management and
        used as defaults when not explicitly specified. Changes take effect immediately and
        persist until changed or disconnect.
        
        Business context: Core interface for adjusting camera settings during operation. Enables
        adaptive exposure as sky conditions change (moonrise, clouds), gain adjustment for target
        brightness (galaxies vs planets), white balance for color cameras. Used by automated
        algorithms (auto-exposure, auto-focus) and manual UI controls. Critical for astrophotography
        where optimal settings vary by target, conditions, and imaging goals.
        
        Common controls (ZWO ASI): Gain (0-600, amplification), Exposure (1-3,600,000,000 microseconds),
        Offset (black level), WB_R/WB_B (white balance), Gamma, Brightness, BandwidthOverload (USB %),
        HighSpeedMode, Flip.
        
        Args:
            name: Control name (e.g., "Gain", "Exposure", "WB_R", "Offset"). Must be in
                camera.info.controls. Case-sensitive, camera-model-specific.
            value: Integer value. Range depends on control - check camera.info.controls[name]["min"/"max"].
                Driver typically clamps to valid range. For Exposure, value is microseconds.
            
        Raises:
            CameraNotConnectedError: If not connected. Call connect() first.
            CameraError: If driver rejects control name (invalid) or value (hardware error).
            
        Example:
            >>> camera.set_control("Gain", 80)  # Bright target
            >>> camera.set_control("Exposure", 5_000_000)  # 5 second frame
            >>> camera.set_control("WB_R", 52)  # Red balance for color camera
        
        Note:
            Gain and Exposure tracked internally for defaults. Other controls not tracked - use
            get_control() to read back. Some controls interact (high speed affects exposure limits).
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
        
        Args:
            None. Context managers don't take arguments.
        
        Returns:
            Self (the Camera instance) for use in the with block.
        
        Raises:
            CameraError: If connection fails (propagated from connect()).
        
        Example:
            >>> with Camera(driver, config) as camera:
            ...     result = camera.capture()
            # Camera automatically disconnected on exit
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
        
        Returns:
            None. Context manager returns None (doesn't suppress exceptions).
        
        Raises:
            None. disconnect() never raises (errors logged and suppressed).
        
        Example:
            >>> with Camera(driver, config) as camera:
            ...     raise RuntimeError("Test error")
            # Camera disconnected even though exception raised
        """
        self.disconnect()
    
    def __repr__(self) -> str:
        """Return string representation of camera for debugging.
        
        Business context: Useful for logging, debugging, and REPL inspection showing camera
        state at a glance (ID, name, connection status).
        
        Args:
            None.
        
        Returns:
            String like "<Camera(id=0, name='Main', connected=True)>".
        
        Example:
            >>> camera = Camera(driver, CameraConfig(camera_id=0, name="Main"))
            >>> print(camera)  # <Camera(id=0, name='Main', connected=False)>
        
        Returns:
            String like '<Camera(name, connected)>' or '<Camera(name, disconnected)>'.
        
        Raises:
            None. Always returns a valid string.
        """
        status = "connected" if self.is_connected else "disconnected"
        name = self._config.name or f"camera_{self._config.camera_id}"
        return f"<Camera({name}, {status})>"
