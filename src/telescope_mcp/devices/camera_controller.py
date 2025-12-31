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

__all__ = [
    # Constants
    "TIMING_THRESHOLD_GOOD_MS",
    "TIMING_THRESHOLD_ACCEPTABLE_MS",
    "TIMING_THRESHOLD_POOR_MS",
    # Dataclasses
    "SyncCaptureConfig",
    "SyncCaptureResult",
    # Exceptions
    "CameraControllerError",
    "CameraNotFoundError",
    "SyncCaptureError",
    # Main class
    "CameraController",
]


# --- Timing Thresholds ---
# Named constants for synchronization quality assessment
TIMING_THRESHOLD_GOOD_MS: float = 50.0
"""Timing error below this indicates excellent synchronization."""

TIMING_THRESHOLD_ACCEPTABLE_MS: float = 200.0
"""Timing error below this is acceptable for most applications."""

TIMING_THRESHOLD_POOR_MS: float = 500.0
"""Timing error above this may cause platesolve failures."""


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
        """Timing error in milliseconds.

        Converts timing_error_us to milliseconds for logs and diagnostics.
        Positive = secondary late, negative = secondary early.

        Business context: Platesolving requires both images to show stars in
        nearly identical positions. Timing errors > 500ms cause noticeable
        star drift due to Earth's rotation, degrading alignment accuracy.

        Quality thresholds:
            - < TIMING_THRESHOLD_GOOD_MS (50ms): Excellent sync
            - < TIMING_THRESHOLD_ACCEPTABLE_MS (200ms): Acceptable
            - > TIMING_THRESHOLD_POOR_MS (500ms): May cause platesolve failures

        Args:
            self: SyncCaptureResult instance containing timing_error_us.

        Returns:
            Timing error in milliseconds as float.

        Raises:
            No exceptions raised; pure arithmetic conversion.

        Example:
            >>> if abs(result.timing_error_ms) > TIMING_THRESHOLD_GOOD_MS:
            ...     logger.warning(f"Poor sync: {result.timing_error_ms:.1f}ms")
        """
        return self.timing_error_us / 1000.0


class CameraControllerError(Exception):
    """Base exception for controller operations."""

    pass


class CameraNotFoundError(CameraControllerError):
    """Raised when referenced camera doesn't exist.

    Attributes:
        camera_name: The name that was not found.
    """

    def __init__(self, camera_name: str) -> None:
        """Initialize with the camera name that wasn't found.

        Stores the camera name as a structured attribute for programmatic
        error handling, enabling callers to suggest corrections or list
        available cameras.

        Business context: In multi-camera setups, typos in camera names are
        common. Exposing camera_name allows UI to show "Did you mean 'finder'?"
        suggestions or list registered cameras.

        Args:
            camera_name: The name that was looked up but not registered.
                Stored in self.camera_name for later access.

        Returns:
            None. Exception instance ready to be raised.

        Raises:
            No exceptions raised during initialization.

        Example:
            >>> try:
            ...     controller.get_camera("funder")  # typo
            ... except CameraNotFoundError as e:
            ...     print(f"'{e.camera_name}' not found")
        """
        self.camera_name = camera_name
        super().__init__(f"Camera '{camera_name}' not found")


class SyncCaptureError(CameraControllerError):
    """Raised when synchronized capture fails.

    Attributes:
        camera_role: Which camera failed ('primary', 'secondary', or None).
        original_error: The underlying exception if available.
    """

    def __init__(
        self,
        message: str,
        *,
        camera_role: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize with failure details.

        Creates structured exception with context for programmatic handling.
        Attributes enable targeted recovery (e.g., retry specific camera).

        Business context: During long sync captures (minutes), failures need
        clear attribution. Knowing which camera failed allows retry logic
        to attempt recovery on just that device, preserving the other's frame.

        Args:
            message: Human-readable error description for logs and display.
            camera_role: 'primary', 'secondary', or None if not camera-specific.
                Enables targeted retry logic.
            original_error: The underlying exception that caused the failure.
                Enables detailed debugging and exception chaining.

        Returns:
            None. Exception instance ready to be raised.

        Raises:
            No exceptions raised during initialization.

        Example:
            >>> try:
            ...     result = controller.sync_capture(config)
            ... except SyncCaptureError as e:
            ...     if e.camera_role == "secondary":
            ...         # Retry secondary only
            ...         pass
        """
        self.camera_role = camera_role
        self.original_error = original_error
        super().__init__(message)


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

        Initializes the controller with an optional initial set of cameras
        and a clock abstraction for timing operations.

        Business context: The clock parameter enables deterministic testing
        by injecting a mock clock. Production uses SystemClock for real
        timing; tests use MockClock for instant, predictable execution.

        Args:
            cameras: Dict mapping camera names to Camera instances.
                Names should be descriptive ("finder", "main", "guide").
                Cameras should typically be connected before adding.
            clock: Clock implementation for timing (default: SystemClock).
                Inject MockClock for testing sync_capture timing logic.

        Returns:
            None. Controller ready for add_camera() or sync_capture().

        Raises:
            No exceptions raised. Empty cameras dict is valid initial state.

        Example:
            >>> # Production usage
            >>> controller = CameraController({
            ...     "finder": finder_camera,
            ...     "main": main_camera,
            ... })
            >>> # Testing with mock clock
            >>> controller = CameraController(clock=MockClock())
        """
        self._cameras: dict[str, Camera] = cameras or {}
        self._clock = clock or SystemClock()

    def add_camera(
        self,
        name: str,
        camera: Camera,
        *,
        overwrite: bool = False,
    ) -> None:
        """Add a camera to the controller.

        Registers a camera under the given name for use in sync operations.
        Camera should be connected before adding.

        Args:
            name: Unique name for the camera (e.g., "finder", "main").
            camera: Camera instance, typically already connected.
            overwrite: If True, replace existing camera with same name.
                If False (default), raise ValueError on duplicate.

        Returns:
            None. Camera registered and immediately available.

        Raises:
            ValueError: If name already registered and overwrite=False.

        Example:
            >>> controller.add_camera("guide", guide_camera)
            >>> controller.add_camera("guide", new_camera, overwrite=True)
        """
        if name in self._cameras and not overwrite:
            raise ValueError(
                f"Camera '{name}' already registered. Use overwrite=True to replace."
            )
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
            CameraNotFoundError: If name not registered. The exception's
                camera_name attribute contains the requested name.

        Example:
            camera = controller.get_camera("main")
            result = camera.capture()
        """
        if name not in self._cameras:
            raise CameraNotFoundError(name)
        return self._cameras[name]

    @property
    def camera_names(self) -> list[str]:
        """List of registered camera names.

        Business context: Used by UI and diagnostics to enumerate available
        cameras without exposing internal dict. Order preserved for consistent
        display and iteration in status reports.

        Args:
            self: CameraController instance.

        Returns:
            List of camera names in registration order. Empty if none registered.

        Raises:
            No exceptions raised; returns empty list if no cameras.

        Example:
            >>> for name in controller.camera_names:
            ...     print(f"{name}: {controller.get_camera(name).is_connected}")
        """
        return list(self._cameras.keys())

    def calculate_sync_timing(
        self,
        primary_exposure_us: int,
        secondary_exposure_us: int,
    ) -> int:
        """Calculate delay before starting secondary capture for temporal centering.

        Determines the optimal delay to start the secondary camera exposure so
        that it's temporally centered within the primary camera's exposure.

        Formula: delay = (primary / 2) - (secondary / 2)

        Args:
            primary_exposure_us: Primary camera exposure time in microseconds.
                Must be positive.
            secondary_exposure_us: Secondary camera exposure time in
                microseconds. Must be positive.

        Returns:
            Delay in microseconds to wait after starting primary before starting
            secondary. Returns 0 if secondary >= primary (no delay needed).

        Raises:
            ValueError: If either exposure time is not positive.

        Example:
            >>> delay = controller.calculate_sync_timing(176_000_000, 312_000)
            >>> print(f"Wait {delay / 1_000_000:.1f}s before starting guide")
            Wait 87.8s before starting guide
        """
        if primary_exposure_us <= 0:
            raise ValueError(
                f"primary_exposure_us must be positive, got {primary_exposure_us}"
            )
        if secondary_exposure_us <= 0:
            raise ValueError(
                f"secondary_exposure_us must be positive, got {secondary_exposure_us}"
            )

        # If secondary is longer than primary, no delay needed
        if secondary_exposure_us >= primary_exposure_us:
            return 0

        primary_midpoint = primary_exposure_us // 2
        secondary_half = secondary_exposure_us // 2
        return primary_midpoint - secondary_half

    def sync_capture(self, config: SyncCaptureConfig) -> SyncCaptureResult:
        """Capture with secondary centered in primary exposure.

        Coordinates two cameras so secondary exposure is temporally centered
        within primary exposure. Both cameras capture the same moment at their
        respective exposure midpoints.

        Timeline for 176s primary, 312ms secondary::

            Primary:   |================176s================|
            Secondary:              |312ms|
                       ^            ^
                       t=0          t=87.844s

        Args:
            config: Sync capture configuration with camera names, exposure
                times (microseconds), and optional gain overrides.

        Returns:
            SyncCaptureResult with both frames, timing measurements, and
            timing_error_ms for quality assessment (< 50ms is good).

        Raises:
            CameraNotFoundError: If camera name not registered.
            SyncCaptureError: If capture fails. Check camera_role and
                original_error attributes for details.

        Example:
            >>> result = controller.sync_capture(SyncCaptureConfig(
            ...     primary="finder",
            ...     secondary="main",
            ...     primary_exposure_us=176_000_000,
            ...     secondary_exposure_us=312_000,
            ... ))
            >>> if abs(result.timing_error_ms) < TIMING_THRESHOLD_GOOD_MS:
            ...     print("Excellent sync")
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
            """Thread worker for primary camera capture (starts immediately).

            Records timestamps and captures primary frame. Exceptions stored
            in primary_error for main thread handling via nonlocal.

            Business context: Primary camera (typically long-exposure finder)
            starts immediately while secondary waits. Timestamps enable
            timing error calculation for platesolve quality assessment.

            Implementation:
                1. Record monotonic timestamp (for delay calculation)
                2. Record wall-clock timestamp (for metadata/logs)
                3. Call capture_raw with configured exposure/gain
                4. Store result or exception via nonlocal

            Args:
                Uses enclosing scope: primary_cam, config, self._clock.
                Modifies via nonlocal: primary_result, primary_error,
                    primary_start_mono, primary_start_wall.

            Returns:
                None. Results stored in enclosing scope variables.

            Raises:
                No exceptions raised; all caught and stored in primary_error.
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
            """Thread worker for secondary camera capture (waits for delay).

            Sleeps for calculated delay to center exposures, then captures.
            Exceptions stored in secondary_error for main thread via nonlocal.

            Business context: Secondary camera (typically fast guide/main)
            waits until its exposure would be temporally centered within
            the primary's exposure, minimizing star position drift.

            Implementation:
                1. Sleep for calculated delay_seconds
                2. Record monotonic timestamp (for actual delay measurement)
                3. Record wall-clock timestamp (for metadata/logs)
                4. Call capture_raw with configured exposure/gain
                5. Store result or exception via nonlocal

            Args:
                Uses enclosing scope: secondary_cam, config, delay_seconds,
                    self._clock.
                Modifies via nonlocal: secondary_result, secondary_error,
                    secondary_start_mono, secondary_start_wall.

            Returns:
                None. Results stored in enclosing scope variables.

            Raises:
                No exceptions raised; all caught and stored in secondary_error.
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

        # Check for errors (use structured exceptions with context)
        if primary_error:
            raise SyncCaptureError(
                f"Primary capture failed: {primary_error}",
                camera_role="primary",
                original_error=primary_error,
            ) from primary_error
        if secondary_error:
            raise SyncCaptureError(
                f"Secondary capture failed: {secondary_error}",
                camera_role="secondary",
                original_error=secondary_error,
            ) from secondary_error
        if primary_result is None or secondary_result is None:
            raise SyncCaptureError(
                "Capture returned None unexpectedly",
                camera_role="primary" if primary_result is None else "secondary",
            )

        # Timing is guaranteed non-None here: set before capture_raw in try block,
        # and any exception would have been caught and raised above
        assert primary_start_wall is not None  # for type checker
        assert secondary_start_wall is not None  # for type checker

        # Calculate actual timing
        actual_delay_seconds = secondary_start_mono - primary_start_mono
        actual_delay_us = int(actual_delay_seconds * 1_000_000)
        timing_error_us = actual_delay_us - delay_us

        return SyncCaptureResult(
            primary_frame=primary_result,
            secondary_frame=secondary_result,
            primary_start=primary_start_wall,
            secondary_start=secondary_start_wall,
            ideal_secondary_start_us=delay_us,
            actual_secondary_start_us=actual_delay_us,
            timing_error_us=timing_error_us,
        )
