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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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
        ideal_secondary_start_us: When secondary should have started
            (relative to primary)
        actual_secondary_start_us: When secondary actually started
            (relative to primary)
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
        """Timing error in milliseconds (convenience accessor).

        Converts timing_error_us (microseconds) to milliseconds for easier
        reading in logs, UI displays, and diagnostics. Positive values indicate
        secondary started late, negative early. Magnitude indicates
        synchronization quality.

        Business context: Critical metric for assessing dual-camera
        synchronization quality in telescope operations. Platesolving workflows
        require finder and main camera captures taken at same moment (within
        ~100ms) to ensure both frames show same sky position. Large timing
        errors (>500ms) may cause platesolve failures if telescope drifted
        between captures. Used in quality monitoring ("are captures
        synchronized well enough?"), performance tuning (adjusting clock
        resolution, thread priorities), and troubleshooting ("why did alignment
        fail?"). Logged to session metadata for post-observation analysis.

        Implementation details: Simple conversion: timing_error_us / 1000.0.
        Returns float with sub-millisecond precision (e.g., 1.234ms). Sign
        preserved: positive = secondary late, negative = secondary early.
        Typical values: 0-50ms (good synchronization), 50-200ms (acceptable),
        >200ms (poor - may affect platesolving). Zero indicates perfect timing
        (rare - usually 1-20ms due to thread scheduling jitter).

        Returns:
            Timing error in milliseconds as float. Range typically -500.0 to
            +500.0 (extreme cases), commonly -50.0 to +50.0 for well-tuned
            systems. Precision to 0.001ms (1 microsecond).

        Example:
            >>> result = controller.sync_capture(config)
            >>> if abs(result.timing_error_ms) > 100:
            ...     logger.warning(f"Poor sync: {result.timing_error_ms:.1f}ms error")
            >>> print(f"Timing: {result.timing_error_ms:+.2f}ms")  # +1.23ms or -2.45ms
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
        Camera should be connected before adding. The name becomes the key for
        subsequent get_camera() calls and config references.

        Business context: Enables runtime camera registration for flexible
        multi-camera setups. Supports dynamic camera addition as hardware is
        connected, hot-swapping cameras without controller restart, and
        descriptive naming that clarifies role (e.g., "main_imaging",
        "finder_guide", "autoguider"). Essential for observatory setups where
        camera configuration may change per observation session or target.

        Implementation details: Does not validate camera state or connectivity -
        caller should ensure camera is connected and configured before
        registration. Duplicate names will overwrite previous registration
        without warning. Consider validating camera.is_connected before adding
        in production code.

        Args:
            name: Unique name for the camera (e.g., "finder", "main",
                "autoguider"). Used as key in sync_capture config. Should be
                descriptive and match names used elsewhere in the system for
                consistency.
            camera: Camera instance, typically already connected via
                Camera.connect(). Should have driver initialized and be ready
                for capture operations.

        Returns:
            None. Camera is registered and immediately available for operations.

        Raises:
            None. Duplicate names silently overwrite. Consider adding validation.

        Example:
            >>> # Basic registration
            >>> controller.add_camera("guide", guide_camera)
            >>> controller.add_camera("main", main_camera)
            >>>
            >>> # Dynamic setup for multi-camera observatory
            >>> registry = get_registry()
            >>> for camera_id in registry.discovered_camera_ids:
            ...     camera = registry.get(camera_id)
            ...     name = f"camera_{camera_id}"
            ...     controller.add_camera(name, camera)
            >>> print(f"Registered {len(controller.camera_names)} cameras")
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

        Business context: Enables dynamic UI generation showing available
        cameras, validation of camera references in configurations, and bulk
        operations across all cameras. Essential for multi-camera setups where
        camera availability may change at runtime.

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

        Determines the optimal delay to start the secondary camera exposure so
        that it's temporally centered within the primary camera's exposure. This
        ensures both cameras capture the same moment in time at their respective
        exposure midpoints, critical for accurate alignment and tracking.

        Formula: delay = (primary / 2) - (secondary / 2)

        Business context: Essential for synchronized multi-camera astrophotography
        where a long-exposure main camera is guided by a shorter-exposure finder
        camera. By centering exposures, tracking corrections measured in the
        finder frame correspond to the main frame's temporal midpoint, minimizing
        field rotation and drift artifacts. Used in guided imaging to ensure
        guide corrections represent the science exposure's actual position.

        Implementation details: Uses integer division for microsecond precision.
        For very long primary exposures (>30 minutes), secondary should be
        centered to avoid accumulation of tracking errors. The delay can be
        quite long (tens of seconds) for typical long-exposure scenarios.

        Args:
            primary_exposure_us: Primary camera exposure time in microseconds.
                Typically the main imaging camera with long exposure (1s-10min).
            secondary_exposure_us: Secondary camera exposure time in
                microseconds. Typically the guide/finder camera with short
                exposure (100ms-1s).

        Returns:
            Delay in microseconds to wait after starting primary before starting
            secondary. Ensures secondary's temporal midpoint aligns with
            primary's. Result may be 0 if secondary exposure equals or exceeds
            primary (no delay).

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
            >>> secondary_mid = secondary_start + (312_000 / 2 / 1_000_000)
            >>> print(f"Midpoints: primary={primary_mid:.1f}s, "
            ...       f"secondary={secondary_mid:.1f}s")
            Midpoints: primary=88.0s, secondary=88.0s
        """
        primary_midpoint = primary_exposure_us // 2
        secondary_half = secondary_exposure_us // 2
        return primary_midpoint - secondary_half

    def sync_capture(self, config: SyncCaptureConfig) -> SyncCaptureResult:
        """Capture with secondary centered in primary exposure for temporal alignment.

        Coordinates two cameras to capture simultaneously, with the secondary
        exposure centered within the primary exposure. Ensures both cameras
        capture the same temporal moment at their respective exposure midpoints,
        critical for accurate alignment and tracking in astrophotography.

        Business context: Essential for synchronized multi-camera astrophotography
        where precise temporal alignment is required. Primary use case is plate
        solving for telescope alignment: a long-exposure finder camera captures
        sufficient stars for plate solving (determining telescope pointing),
        while a short-exposure main camera captures a quick frame at the same
        moment for alignment verification. This ensures the plate solve result
        accurately represents where the main camera was pointing, accounting for
        atmospheric refraction, mount flexure, and tracking errors that vary
        over time.

        Additional use cases:
        - Autoguiding: Guide camera long exposure centered on science camera
          exposure
        - Multi-wavelength imaging: Synchronized captures across different
          filters
        - Stereo imaging: Synchronized captures for 3D reconstruction
        - Event capture: Recording transient events across multiple cameras

        Timeline for 176s primary, 312ms secondary:

            Primary:   |================176s================|
            Secondary:              |312ms|
                       ^            ^
                       t=0          t=87.844s (midpoint - 156ms)

        The secondary exposure starts at:
            (primary_exposure / 2) - (secondary_exposure / 2)

        So the CENTER of secondary aligns with CENTER of primary, ensuring both
        cameras see the sky at the same moment despite different exposure
        durations.

        Implementation details: Uses ThreadPoolExecutor for concurrent camera
        control. Primary capture starts immediately, secondary waits for
        calculated delay before starting. Both captures block until complete.
        Timing measurements use monotonic clock to avoid wall clock
        discontinuities. Timing error is typically <10ms on modern systems but
        may increase under heavy system load.

        Args:
            config: Sync capture configuration specifying:
                - Camera names (must be registered via add_camera)
                - Exposure times in microseconds
                - Optional gain overrides for each camera

        Returns:
            SyncCaptureResult containing:
                - Both captured frames with metadata
                - Timing measurements (ideal vs actual delay)
                - Timing error for quality assessment
            Use timing_error_ms < 50ms as "good" threshold for most
            applications.

        Raises:
            CameraNotFoundError: If camera name not registered via add_camera.
            SyncCaptureError: If either capture fails due to hardware error,
                timeout, or camera disconnect. Original exception chained.

        Example:
            >>> # Typical plate solving scenario
            >>> result = controller.sync_capture(SyncCaptureConfig(
            ...     primary="finder",      # Long exposure for many stars
            ...     secondary="main",      # Quick frame for alignment
            ...     primary_exposure_us=176_000_000,  # 176 seconds
            ...     secondary_exposure_us=312_000,    # 312 ms
            ...     primary_gain=50,       # Lower gain for finder
            ...     secondary_gain=100,    # Higher gain for quick main frame
            ... ))
            >>> print(f"Timing error: {result.timing_error_ms:.1f}ms")
            Timing error: 3.2ms
            >>>
            >>> # Check timing quality
            >>> if result.timing_error_ms < 50:
            ...     print("Excellent temporal alignment")
            ...     # Use finder frame for plate solving
            ...     # Use main frame for alignment verification
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
            """Thread worker capturing primary camera frame (finder/guide).

            Executes in dedicated thread capturing primary camera frame
            immediately without delay. Records start timestamps (monotonic for
            timing calculations, wall-clock for metadata). Catches all
            exceptions storing in primary_error for main thread handling. Uses
            nonlocal to communicate results back to sync_capture scope.

            Business context: Primary camera typically finder/guide (short
            exposure, wide field) capturing stars for platesolving while
            secondary (main imager) captures verification frame. Primary starts
            immediately establishing timing reference point. Threading enables
            overlap: while primary exposing (e.g., 2s finder exposure),
            secondary thread waits calculated delay then starts (e.g., 1.9s
            wait then 0.2s main exposure). Without threading, captures would be
            sequential losing synchronization.

            Implementation: Records self._clock.monotonic() for high-precision
            timing, datetime.now(UTC) for human-readable timestamps. Calls
            primary_cam.capture_raw() (no overlay) with configured
            exposure/gain. Exceptions stored in primary_error (checked by main
            thread after join). Nonlocal variables modified: primary_result
            (CaptureResult), primary_error (Exception), primary_start_mono
            (float), primary_start_wall (datetime). Thread executor ensures
            clean cleanup even on exception.
            """
            nonlocal \
                primary_result, \
                primary_error, \
                primary_start_mono, \
                primary_start_wall
            try:
                primary_start_mono = self._clock.monotonic()
                primary_start_wall = datetime.now(UTC)
                primary_result = primary_cam.capture_raw(
                    exposure_us=config.primary_exposure_us,
                    gain=config.primary_gain,
                )
            except Exception as e:
                primary_error = e

        def capture_secondary() -> None:
            """Thread worker capturing secondary camera frame with delay.

            Executes in dedicated thread sleeping for calculated delay (temporal
            centering: makes exposure midpoints align), then capturing secondary
            camera frame. Records start timestamps. Catches all exceptions
            storing in secondary_error. Uses nonlocal for result communication.

            Business context: Secondary camera typically main imager (long
            exposure, narrow field) capturing verification frame at moment
            finder solves. Delay ensures both cameras capture same sky position
            despite different exposure times. Example: finder 2s exposure
            (midpoint 1s), main 0.2s exposure needs 0.9s delay so midpoint
            aligns at 1s. Critical for platesolving accuracy - misaligned
            captures show different sky positions due to telescope
            drift/tracking errors, causing solve failures or inaccurate
            alignment models.

            Implementation: Sleeps delay_seconds (calculated by
            calculate_sync_timing) using self._clock.sleep(), then records
            timestamps and calls secondary_cam.capture_raw(). Sleep blocks
            thread without busy-waiting. Monotonic clock used for timing
            precision (immune to system clock adjustments). Exceptions stored
            in secondary_error (checked after join). Nonlocal variables:
            secondary_result (CaptureResult), secondary_error (Exception),
            secondary_start_mono (float), secondary_start_wall (datetime).
            Thread executor handles cleanup.
            """
            nonlocal \
                secondary_result, \
                secondary_error, \
                secondary_start_mono, \
                secondary_start_wall
            try:
                # Wait for the calculated delay
                self._clock.sleep(delay_seconds)

                secondary_start_mono = self._clock.monotonic()
                secondary_start_wall = datetime.now(UTC)
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
            raise SyncCaptureError(
                f"Primary capture failed: {primary_error}"
            ) from primary_error
        if secondary_error:
            raise SyncCaptureError(
                f"Secondary capture failed: {secondary_error}"
            ) from secondary_error
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
