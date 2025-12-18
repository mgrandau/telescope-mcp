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
    """Protocol for time functions (injectable for testing)."""
    
    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        ...
    
    def sleep(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        ...


class SystemClock:
    """Default clock using time module."""
    
    def monotonic(self) -> float:
        return time.monotonic()
    
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class OverlayRenderer(Protocol):
    """Protocol for overlay rendering strategies.
    
    Implement this to create new overlay types without modifying Camera.
    """
    
    def render(
        self, 
        image_data: bytes, 
        config: OverlayConfig,
        camera_info: CameraInfo | None,
    ) -> bytes:
        """Render overlay on image data.
        
        Args:
            image_data: Original image (JPEG bytes)
            config: Overlay configuration
            camera_info: Camera info for resolution, etc.
            
        Returns:
            Image with overlay applied (JPEG bytes)
        """
        ...


class NullRenderer:
    """Default renderer that does nothing (passthrough)."""
    
    def render(
        self, 
        image_data: bytes, 
        config: OverlayConfig,
        camera_info: CameraInfo | None,
    ) -> bytes:
        return image_data


class RecoveryStrategy(Protocol):
    """Protocol for camera disconnect recovery.
    
    Implement this to provide custom recovery behavior when
    a camera disconnects unexpectedly.
    """
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """Attempt to recover a disconnected camera.
        
        Args:
            camera_id: ID of the camera to recover
            
        Returns:
            True if camera is available after recovery attempt
        """
        ...


class NullRecoveryStrategy:
    """No-op recovery strategy (always fails)."""
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """Always returns False (no recovery attempted)."""
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
        """Create CameraInfo from driver response dicts."""
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
        
        Args:
            driver: Camera driver (ASI or digital twin)
            config: Camera configuration including ID and defaults
            renderer: Overlay renderer (default: NullRenderer)
            clock: Time functions for stream timing (default: SystemClock)
            hooks: Optional event callbacks
            recovery: Recovery strategy for disconnect handling (default: NullRecoveryStrategy)
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
        """Get camera configuration."""
        return self._config
    
    @property
    def is_connected(self) -> bool:
        """Check if camera is currently connected."""
        return self._instance is not None
    
    @property
    def info(self) -> CameraInfo | None:
        """Get camera info (None if not connected)."""
        return self._info
    
    @property
    def overlay(self) -> OverlayConfig | None:
        """Get current overlay configuration."""
        return self._overlay
    
    @property
    def is_streaming(self) -> bool:
        """Check if camera is currently streaming."""
        return self._streaming
    
    def connect(self) -> CameraInfo:
        """Connect to camera and return camera info.
        
        Returns:
            CameraInfo with camera capabilities and controls
            
        Raises:
            CameraAlreadyConnectedError: If already connected
            CameraError: If connection fails
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
        
        Safe to call even if not connected (no-op).
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
        
        Args:
            options: Capture options (exposure, gain, overlay, format)
                    If None, uses defaults with overlay applied.
            
        Returns:
            CaptureResult with image data and metadata
            
        Raises:
            CameraNotConnectedError: If not connected
            CameraError: If capture fails
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
        
        Convenience method equivalent to:
            capture(CaptureOptions(exposure_us=..., gain=..., apply_overlay=False))
        
        Args:
            exposure_us: Override exposure time (microseconds), or None for current
            gain: Override gain value, or None for current
            
        Returns:
            CaptureResult with raw image data (no overlay)
            
        Raises:
            CameraNotConnectedError: If not connected
            CameraError: If capture fails
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
        
        Includes automatic recovery on disconnect.
        Records statistics for observability.
        
        Args:
            exposure_us: Override exposure time (microseconds), or None for current
            gain: Override gain value, or None for current
            
        Returns:
            CaptureResult with raw image data
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
        
        Args:
            exposure_us: Exposure time to use on retry
            gain: Gain value to use on retry
            original_error: The exception that triggered recovery
            
        Returns:
            CaptureResult if recovery successful
            
        Raises:
            CameraDisconnectedError: If recovery fails
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
        
        Args:
            config: Overlay configuration, or None to disable
        """
        self._overlay = config
    
    def stream(
        self,
        options: CaptureOptions | None = None,
        max_fps: float = 30.0,
    ) -> Iterator[StreamFrame]:
        """Yield continuous frames for live view.
        
        Creates a generator that captures frames at up to max_fps.
        Overlay is applied based on options. Use stop_stream() to
        signal the stream to stop.
        
        Args:
            options: Capture options for each frame
            max_fps: Maximum frames per second (actual may be lower)
            
        Yields:
            StreamFrame with image data and metadata
            
        Raises:
            CameraNotConnectedError: If not connected
            CameraError: If capture fails
            
        Example:
            for frame in camera.stream(max_fps=15):
                process(frame)
                if should_stop:
                    camera.stop_stream()
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
        """Signal the stream to stop after current frame."""
        self._streaming = False
    
    def _apply_overlay(self, result: CaptureResult) -> CaptureResult:
        """Apply configured overlay using injected renderer.
        
        Args:
            result: Original capture result
            
        Returns:
            New CaptureResult with overlay applied
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
        
        Args:
            name: Control name (e.g., "Gain", "Exposure", "WB_R")
            value: Control value to set
            
        Raises:
            CameraNotConnectedError: If not connected
            CameraError: If setting control fails
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
        
        Args:
            name: Control name (e.g., "Gain", "Exposure")
            
        Returns:
            Current control value
            
        Raises:
            CameraNotConnectedError: If not connected
            CameraError: If getting control fails
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
        """Enter context manager, connect to camera."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, disconnect from camera."""
        self.disconnect()
    
    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        name = self._config.name or f"camera_{self._config.camera_id}"
        return f"<Camera({name}, {status})>"
