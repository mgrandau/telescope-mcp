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
        """Timing error in milliseconds."""
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
        
        Args:
            cameras: Dict mapping names to Camera instances
            clock: Clock implementation (for testing), defaults to SystemClock
        """
        self._cameras: dict[str, Camera] = cameras or {}
        self._clock = clock or SystemClock()
    
    def add_camera(self, name: str, camera: Camera) -> None:
        """Add a camera to the controller.
        
        Args:
            name: Unique name for the camera
            camera: Camera instance (should be connected)
        """
        self._cameras[name] = camera
    
    def remove_camera(self, name: str) -> Camera | None:
        """Remove and return a camera from the controller.
        
        Args:
            name: Name of camera to remove
            
        Returns:
            Removed camera or None if not found
        """
        return self._cameras.pop(name, None)
    
    def get_camera(self, name: str) -> Camera:
        """Get a camera by name.
        
        Args:
            name: Camera name
            
        Returns:
            Camera instance
            
        Raises:
            CameraNotFoundError: If camera doesn't exist
        """
        if name not in self._cameras:
            raise CameraNotFoundError(f"Camera '{name}' not found")
        return self._cameras[name]
    
    @property
    def camera_names(self) -> list[str]:
        """List of registered camera names."""
        return list(self._cameras.keys())
    
    def calculate_sync_timing(
        self,
        primary_exposure_us: int,
        secondary_exposure_us: int,
    ) -> int:
        """Calculate delay before starting secondary capture.
        
        To center the secondary exposure within the primary:
        
            delay = (primary / 2) - (secondary / 2)
        
        Args:
            primary_exposure_us: Primary exposure in microseconds
            secondary_exposure_us: Secondary exposure in microseconds
            
        Returns:
            Delay in microseconds before starting secondary
        """
        primary_midpoint = primary_exposure_us // 2
        secondary_half = secondary_exposure_us // 2
        return primary_midpoint - secondary_half
    
    def sync_capture(self, config: SyncCaptureConfig) -> SyncCaptureResult:
        """Capture with secondary centered in primary exposure.
        
        Timeline for 176s primary, 312ms secondary:
        
            Primary:   |================176s================|
            Secondary:              |312ms|
                       ^            ^
                       t=0          t=87.844s (midpoint - 156ms)
        
        The secondary exposure starts at:
            (primary_exposure / 2) - (secondary_exposure / 2)
        
        So the CENTER of secondary aligns with CENTER of primary.
        
        Args:
            config: Sync capture configuration
            
        Returns:
            SyncCaptureResult with both frames and timing info
            
        Raises:
            CameraNotFoundError: If camera doesn't exist
            SyncCaptureError: If capture fails
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
