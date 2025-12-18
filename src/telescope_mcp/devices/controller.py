"""Camera controller for multi-camera coordination.

This module provides CameraController for orchestrating synchronized
captures across multiple cameras, such as alignment captures where
a short exposure must be centered within a long exposure.

Example:
    controller = CameraController(
        cameras={"finder": finder_cam, "main": main_cam},
    )
    
    result = controller.sync_capture(SyncCaptureConfig(
        primary="finder",
        secondary="main",
        primary_exposure_us=176_000_000,  # 176 seconds
        secondary_exposure_us=312_000,     # 312 ms
    ))
    
    # result.secondary_frame captured at midpoint of primary
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from telescope_mcp.devices.camera import Camera, CaptureResult

# Import Clock from camera module to avoid duplication
from telescope_mcp.devices.camera import Clock, SystemClock


@dataclass
class SyncCaptureConfig:
    """Configuration for synchronized capture.
    
    Attributes:
        primary: Name of primary camera (long exposure)
        secondary: Name of secondary camera (short exposure)
        primary_exposure_us: Primary exposure time in microseconds
        secondary_exposure_us: Secondary exposure time in microseconds
        primary_gain: Optional gain override for primary
        secondary_gain: Optional gain override for secondary
    """
    primary: str
    secondary: str
    primary_exposure_us: int
    secondary_exposure_us: int
    primary_gain: int | None = None
    secondary_gain: int | None = None


@dataclass
class SyncCaptureResult:
    """Result of synchronized capture.
    
    Attributes:
        primary_frame: Capture result from primary (long exposure)
        secondary_frame: Capture result from secondary (centered in primary)
        primary_start: When primary exposure started
        secondary_start: When secondary exposure started
        ideal_secondary_start_us: When secondary should have started (relative to primary)
        actual_secondary_start_us: When secondary actually started (relative to primary)
        timing_error_us: Difference between ideal and actual timing
    """
    primary_frame: CaptureResult
    secondary_frame: CaptureResult
    primary_start: datetime
    secondary_start: datetime
    ideal_secondary_start_us: int
    actual_secondary_start_us: int
    timing_error_us: int
    
    @property
    def timing_error_ms(self) -> float:
        """Timing error in milliseconds.
        
        Converts timing_error_us to milliseconds for easier reading.
        
        Returns:
            Timing error in milliseconds as float.
        """
        return self.timing_error_us / 1000.0


class CameraControllerError(Exception):
    """Base exception for controller operations."""
    pass


class CameraNotFoundError(CameraControllerError):
    """Raised when referenced camera doesn't exist."""
    pass


class SyncCaptureError(CameraControllerError):
    """Raised when synchronized capture fails."""
    pass


class CameraController:
    """Orchestrates multiple cameras for synchronized operations.
    
    Manages a collection of cameras and coordinates operations that
    require precise timing across multiple devices.
    
    Thread Safety:
        Sync capture uses threads internally but the controller
        itself is not thread-safe for concurrent operations.
    
    Example:
        controller = CameraController({
            "finder": finder_camera,
            "main": main_camera,
        })
        
        # Synchronized capture for alignment
        result = controller.sync_capture(SyncCaptureConfig(
            primary="finder",
            secondary="main", 
            primary_exposure_us=176_000_000,  # 176s finder exposure
            secondary_exposure_us=312_000,     # 312ms main exposure
        ))
        
        print(f"Timing error: {result.timing_error_ms:.1f}ms")
    """
    
    def __init__(
        self,
        cameras: dict[str, Camera] | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Create controller with cameras and optional clock.
        
        Initializes the controller with a collection of named cameras.
        The clock can be injected for deterministic timing in tests.
        
        Args:
            cameras: Dict mapping camera names to Camera instances.
            clock: Clock implementation for timing (default: SystemClock).
        
        Example:
            controller = CameraController({
                "finder": finder_camera,
                "main": main_camera,
            })
        """
        self._cameras: dict[str, Camera] = cameras or {}
        self._clock = clock or SystemClock()
    
    def add_camera(self, name: str, camera: Camera) -> None:
        """Add a camera to the controller.
        
        Registers a camera under the given name for use in sync operations.
        Camera should be connected before adding.
        
        Args:
            name: Unique name for the camera (e.g., "finder", "main").
            camera: Camera instance, typically already connected.
        
        Example:
            controller.add_camera("guide", guide_camera)
        """
        self._cameras[name] = camera
    
    def remove_camera(self, name: str) -> Camera | None:
        """Remove and return a camera from the controller.
        
        Unregisters the camera but does not disconnect it. The caller
        is responsible for disconnecting if needed.
        
        Args:
            name: Name of camera to remove.
            
        Returns:
            Removed Camera instance, or None if name not found.
        
        Example:
            camera = controller.remove_camera("guide")
            if camera:
                camera.disconnect()
        """
        return self._cameras.pop(name, None)
    
    def get_camera(self, name: str) -> Camera:
        """Get a camera by name.
        
        Retrieves a registered camera for direct access. Used internally
        by sync_capture and can be used by callers.
        
        Args:
            name: Camera name (e.g., "finder", "main").
            
        Returns:
            Camera instance.
            
        Raises:
            CameraNotFoundError: If name not registered.
        
        Example:
            camera = controller.get_camera("main")
            result = camera.capture()
        """
        if name not in self._cameras:
            raise CameraNotFoundError(f"Camera '{name}' not found")
        return self._cameras[name]
    
    @property
    def camera_names(self) -> list[str]:
        """List of registered camera names for iteration and UI display.
        
        Returns the names of all cameras currently managed by this controller.
        Useful for building UIs, validating camera availability, and iterating
        over all cameras for batch operations.
        
        Business context: Enables dynamic UI generation showing available cameras,
        validation of camera references in configurations, and bulk operations
        across all cameras. Essential for multi-camera setups where camera
        availability may change at runtime.
        
        Returns:
            List of camera names (strings) in registration order. Empty list if
            no cameras registered. Names correspond to keys used in register()
            and get_camera().
        
        Raises:
            None. Always succeeds, returning empty list if no cameras.
        
        Example:
            >>> controller.register("main", main_camera)
            >>> controller.register("finder", finder_camera)
            >>> print(controller.camera_names)
            ['main', 'finder']
            >>> for name in controller.camera_names:
            ...     camera = controller.get_camera(name)
            ...     print(f"{name}: {camera.is_connected}")
        """
        return list(self._cameras.keys())
    
    def calculate_sync_timing(
        self,
        primary_exposure_us: int,
        secondary_exposure_us: int,
    ) -> int:
        """Calculate delay before starting secondary capture for temporal centering.
        
        Determines the optimal delay to start the secondary camera exposure so that
        it's temporally centered within the primary camera's exposure. This ensures
        both cameras capture the same moment in time at their respective exposure
        midpoints, critical for accurate alignment and tracking.
        
        Formula: delay = (primary / 2) - (secondary / 2)
        
        Business context: Essential for synchronized multi-camera astrophotography
        where a long-exposure main camera is guided by a shorter-exposure finder
        camera. By centering exposures, tracking corrections measured in the finder
        frame correspond to the main frame's temporal midpoint, minimizing field
        rotation and drift artifacts. Used in guided imaging to ensure guide
        corrections represent the science exposure's actual position.
        
        Implementation details: Uses integer division for microsecond precision.
        For very long primary exposures (>30 minutes), secondary should be centered
        to avoid accumulation of tracking errors. The delay can be quite long
        (tens of seconds) for typical long-exposure scenarios.
        
        Args:
            primary_exposure_us: Primary camera exposure time in microseconds.
                Typically the main imaging camera with long exposure (1s-10min).
            secondary_exposure_us: Secondary camera exposure time in microseconds.
                Typically the guide/finder camera with short exposure (100ms-1s).
            
        Returns:
            Delay in microseconds to wait after starting primary before starting
            secondary. Ensures secondary's temporal midpoint aligns with primary's.
            Result may be 0 if secondary exposure equals or exceeds primary (no delay).
        
        Raises:
            None. Always returns a valid delay value.
        
        Example:
            >>> # Typical astrophotography: 176s main, 312ms guide
            >>> delay = controller.calculate_sync_timing(176_000_000, 312_000)
            >>> print(f"Wait {delay / 1_000_000:.1f}s before starting guide")
            Wait 87.8s before starting guide
            >>> 
            >>> # Verify centering
            >>> primary_mid = 176_000_000 / 2  # 88s
            >>> secondary_start = delay / 1_000_000  # 87.844s
            >>> secondary_mid = secondary_start + (312_000 / 2 / 1_000_000)  # 88.0s
            >>> print(f"Midpoints: primary={primary_mid:.1f}s, secondary={secondary_mid:.1f}s")
            Midpoints: primary=88.0s, secondary=88.0s
        """
        primary_midpoint = primary_exposure_us // 2
        secondary_half = secondary_exposure_us // 2
        return primary_midpoint - secondary_half
    
    def sync_capture(self, config: SyncCaptureConfig) -> SyncCaptureResult:
        """Capture with secondary centered in primary exposure.
        
        Coordinates two cameras to capture simultaneously, with the secondary
        exposure centered within the primary exposure. Useful for alignment
        where a short main camera exposure must be taken during a long finder
        exposure to ensure both see the same sky state.
        
        Timeline for 176s primary, 312ms secondary:
        
            Primary:   |================176s================|
            Secondary:              |312ms|
                       ^            ^
                       t=0          t=87.844s (midpoint - 156ms)
        
        The secondary exposure starts at:
            (primary_exposure / 2) - (secondary_exposure / 2)
        
        So the CENTER of secondary aligns with CENTER of primary.
        
        Args:
            config: Sync capture configuration with camera names and timings.
            
        Returns:
            SyncCaptureResult with both frames and timing measurements.
            
        Raises:
            CameraNotFoundError: If camera name not registered.
            SyncCaptureError: If either capture fails.
        
        Example:
            result = controller.sync_capture(SyncCaptureConfig(
                primary="finder",
                secondary="main",
                primary_exposure_us=176_000_000,
                secondary_exposure_us=312_000,
            ))
            print(f"Timing error: {result.timing_error_ms:.1f}ms")
        """
        primary_cam = self.get_camera(config.primary)
        secondary_cam = self.get_camera(config.secondary)
        
        # Calculate when to start secondary (in microseconds)
        delay_us = self.calculate_sync_timing(
            config.primary_exposure_us,
            config.secondary_exposure_us,
        )
        delay_seconds = delay_us / 1_000_000.0
        
        # Results storage
        primary_result: CaptureResult | None = None
        secondary_result: CaptureResult | None = None
        primary_error: Exception | None = None
        secondary_error: Exception | None = None
        
        # Timing tracking
        primary_start_mono: float = 0.0
        secondary_start_mono: float = 0.0
        primary_start_wall: datetime | None = None
        secondary_start_wall: datetime | None = None
        
        def capture_primary() -> None:
            nonlocal primary_result, primary_error, primary_start_mono, primary_start_wall
            try:
                primary_start_mono = self._clock.monotonic()
                primary_start_wall = datetime.now(timezone.utc)
                primary_result = primary_cam.capture_raw(
                    exposure_us=config.primary_exposure_us,
                    gain=config.primary_gain,
                )
            except Exception as e:
                primary_error = e
        
        def capture_secondary() -> None:
            nonlocal secondary_result, secondary_error, secondary_start_mono, secondary_start_wall
            try:
                # Wait for the calculated delay
                self._clock.sleep(delay_seconds)
                
                secondary_start_mono = self._clock.monotonic()
                secondary_start_wall = datetime.now(timezone.utc)
                secondary_result = secondary_cam.capture_raw(
                    exposure_us=config.secondary_exposure_us,
                    gain=config.secondary_gain,
                )
            except Exception as e:
                secondary_error = e
        
        # Start both captures - primary immediately, secondary after delay
        with ThreadPoolExecutor(max_workers=2) as executor:
            primary_future = executor.submit(capture_primary)
            secondary_future = executor.submit(capture_secondary)
            
            # Wait for both to complete
            primary_future.result()
            secondary_future.result()
        
        # Check for errors
        if primary_error:
            raise SyncCaptureError(f"Primary capture failed: {primary_error}") from primary_error
        if secondary_error:
            raise SyncCaptureError(f"Secondary capture failed: {secondary_error}") from secondary_error
        if primary_result is None or secondary_result is None:
            raise SyncCaptureError("Capture returned None")
        
        # Calculate actual timing
        actual_delay_seconds = secondary_start_mono - primary_start_mono
        actual_delay_us = int(actual_delay_seconds * 1_000_000)
        timing_error_us = actual_delay_us - delay_us
        
        return SyncCaptureResult(
            primary_frame=primary_result,
            secondary_frame=secondary_result,
            primary_start=primary_start_wall,  # type: ignore
            secondary_start=secondary_start_wall,  # type: ignore
            ideal_secondary_start_us=delay_us,
            actual_secondary_start_us=actual_delay_us,
            timing_error_us=timing_error_us,
        )
