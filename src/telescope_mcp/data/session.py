"""Session management for telescope data storage.

A session is a time-bounded period of telescope activity that produces
an ASDF file with complete provenance.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import asdf
import numpy as np
from numpy.typing import NDArray

from telescope_mcp.observability import get_logger

logger = get_logger(__name__)


class SessionType(str, Enum):
    """Types of telescope sessions."""

    OBSERVATION = "observation"  # Scientific data collection
    ALIGNMENT = "alignment"  # Calibration procedures
    EXPERIMENT = "experiment"  # Testing and development
    MAINTENANCE = "maintenance"  # System checks
    IDLE = "idle"  # Background system logs


class LogLevel(str, Enum):
    """Log levels for session logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Session:
    """A telescope session that collects data and writes to ASDF.

    A session represents a time-bounded period of telescope activity.
    All logs, frames, telemetry, and calibration data are collected
    in-memory and written to an ASDF file when the session closes.

    Example:
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=Path("/data/telescope"),
            target="M31",
        )
        session.log(LogLevel.INFO, "Starting observation")
        session.add_frame("main", frame_array, settings={"gain": 200})
        path = session.close()
    """

    def __init__(
        self,
        session_type: SessionType,
        data_dir: Path,
        *,
        target: str | None = None,
        purpose: str | None = None,
        location: dict[str, float] | None = None,
        auto_rotate: bool = False,
        rotate_interval_hours: int = 1,
    ) -> None:
        """Initialize a new session.

        Creates a new session with a unique ID based on timestamp and type.
        Session data is buffered in memory until close() writes ASDF.

        Args:
            session_type: Type of session (observation, alignment, etc.).
            data_dir: Base directory for ASDF file storage.
            target: Target object name (e.g., "M31") for observations.
            purpose: Purpose description for alignment/experiment sessions.
            location: Observer location dict with lat, lon, alt keys.
            auto_rotate: Whether to auto-rotate idle sessions.
            rotate_interval_hours: Hours between auto-rotations.

        Returns:
            None. Session initialized and ready for logging.

        Raises:
            No exceptions raised during initialization.

        Example:
            session = Session(
                SessionType.OBSERVATION,
                Path("/data/telescope"),
                target="M42",
                location={"lat": 34.05, "lon": -118.25, "alt": 100.0},
            )
        """
        self.session_type = session_type
        self.data_dir = Path(data_dir)
        self.target = target
        self.purpose = purpose
        self.location = location or {}
        self.auto_rotate = auto_rotate
        self.rotate_interval_hours = rotate_interval_hours

        # Generate session identity
        self.start_time = datetime.now(UTC)
        self.session_id = self._generate_session_id()

        # In-memory data buffers
        self._logs: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._cameras: dict[str, dict[str, Any]] = {}
        self._telemetry: dict[str, list[dict[str, Any]]] = {
            "mount_position": [],
            "temperature": [],
            "focus_position": [],
        }
        self._calibration: dict[str, Any] = {
            "dark_frames": [],
            "flat_frames": [],
            "plate_solve_results": [],
        }

        # Metrics counters
        self._frames_captured = 0
        self._error_count = 0
        self._warning_count = 0

        # State
        self._closed = False
        self._end_time: datetime | None = None

        logger.info(
            "Session started: %s (%s)", self.session_id, self.session_type.value
        )

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for ASDF filename.

        Creates an ID from session type, optional target, and timestamp.
        Format: '{type}_{target}_{YYYYMMDD_HHMMSS}' or '{type}_{YYYYMMDD_HHMMSS}'.
        Target name is slugified (lowercase, spaces to underscores, max 20 chars).

        Business context: Session IDs serve as both unique identifiers and
        human-readable filenames. Including target enables easy identification
        when browsing ASDF files. Timestamp ensures uniqueness even for
        same-target observations.

        Args:
            No arguments. Uses self.session_type, self.target, self.start_time.

        Returns:
            str: Unique session ID like 'observation_m31_20251214_210000'
                or 'alignment_20251214_203000'.

        Raises:
            No exceptions raised.

        Example:
            >>> session = Session(SessionType.OBSERVATION, Path("/data"), target="M31")
            >>> session.session_id
            'observation_m31_20251214_210000'
        """
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        if self.target:
            # observation_m31_20251214_210000
            target_slug = self.target.lower().replace(" ", "_")[:20]
            return f"{self.session_type.value}_{target_slug}_{timestamp}"
        else:
            # alignment_20251214_203000 or idle_20251214_180000
            return f"{self.session_type.value}_{timestamp}"

    def log(
        self,
        level: LogLevel | str,
        message: str,
        source: str = "telescope_mcp",
        **context: Any,
    ) -> None:
        """Log a message to the session and console.

        Implements dual-write: logs go to both console (real-time)
        and the in-memory buffer (for ASDF). Also tracks error/warning counts.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: Human-readable log message.
            source: Source component name for filtering.
            **context: Additional context as key-value pairs.

        Raises:
            RuntimeError: If session is already closed.

        Example:
            session.log(
                LogLevel.INFO, "Exposure started", camera="main", exposure_us=500000
            )
        """
        if self._closed:
            raise RuntimeError("Cannot log to a closed session")

        if isinstance(level, str):
            level = LogLevel(level.upper())

        log_entry = {
            "time": datetime.now(UTC).isoformat(),
            "level": level.value,
            "source": source,
            "message": message,
            "context": context if context else None,
        }
        self._logs.append(log_entry)

        # Dual-write to console
        log_func = getattr(logger, level.value.lower(), logger.info)
        if context:
            log_func("%s | %s", message, context)
        else:
            log_func(message)

        # Track error/warning counts
        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            self._error_count += 1
        elif level == LogLevel.WARNING:
            self._warning_count += 1

    def add_event(self, event: str, **details: Any) -> None:
        """Record a significant event to the session.

        Events mark important occurrences like tracking_lost, slew_complete,
        or cloud_detected. Timestamped and stored in ASDF for analysis.

        Args:
            event: Event name/type identifier.
            **details: Event-specific details as key-value pairs.

        Returns:
            None.

        Raises:
            RuntimeError: If session is already closed.

        Example:
            session.add_event("tracking_lost", reason="clouds", duration_sec=30)
        """
        if self._closed:
            raise RuntimeError("Cannot add event to a closed session")

        self._events.append(
            {
                "time": datetime.now(UTC).isoformat(),
                "event": event,
                "details": details if details else None,
            }
        )
        logger.info("Event: %s | %s", event, details)

    def add_frame(
        self,
        camera: str,
        frame: NDArray[np.uint8 | np.uint16],
        *,
        camera_info: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """Add a captured frame to the session.

        Frames are grouped by camera identifier and stored with their
        capture settings. Increments the frames_captured counter.

        Args:
            camera: Camera identifier ("main", "finder", etc.).
            frame: Image data as numpy array (uint8 or uint16).
            camera_info: Camera metadata (model, sensor_size, etc.). Optional.
            settings: Capture settings (gain, exposure_us, etc.). Optional.

        Returns:
            None.

        Raises:
            RuntimeError: If session is already closed.

        Example:
            session.add_frame(
                "main",
                image_array,
                camera_info={"name": "ASI482MC"},
                settings={"exposure_us": 500000, "gain": 50},
            )
        """
        if self._closed:
            raise RuntimeError("Cannot add frame to a closed session")

        if camera not in self._cameras:
            self._cameras[camera] = {
                "info": camera_info or {},
                "settings": settings or {},
                "frames": [],
            }
        elif camera_info:
            self._cameras[camera]["info"].update(camera_info)
        elif settings:
            self._cameras[camera]["settings"].update(settings)

        self._cameras[camera]["frames"].append(frame)
        self._frames_captured += 1

    def add_telemetry(
        self,
        telemetry_type: str,
        **data: Any,
    ) -> None:
        """Add a telemetry data point.

        Records time-series sensor data like mount position, temperature,
        or focus readings. Each entry is timestamped automatically.

        Args:
            telemetry_type: Type of telemetry (mount_position, temperature, etc.).
            **data: Telemetry values (e.g., ra=12.5, dec=45.2).

        Returns:
            None.

        Raises:
            RuntimeError: If session is already closed.

        Example:
            session.add_telemetry("mount_position", ra=12.5, dec=45.2, alt=60.0)
        """
        if self._closed:
            raise RuntimeError("Cannot add telemetry to a closed session")

        if telemetry_type not in self._telemetry:
            self._telemetry[telemetry_type] = []

        entry = {"time": datetime.now(UTC).isoformat(), **data}
        self._telemetry[telemetry_type].append(entry)

    def add_calibration(self, calibration_type: str, data: Any) -> None:
        """Add calibration data to the session.

        Stores calibration results like dark frames, flat fields, or
        plate solve results for later processing. Data is appended to
        the appropriate calibration list.

        Args:
            calibration_type: Type (dark_frames, flat_frames, plate_solve_results).
            data: Calibration data (frame array or result dict).

        Returns:
            None.

        Raises:
            RuntimeError: If session is already closed.

        Example:
            session.add_calibration("dark_frames", dark_array)
            session.add_calibration("plate_solve_results", {"ra": 12.5, "dec": 45.2})
        """
        if self._closed:
            raise RuntimeError("Cannot add calibration to a closed session")

        if calibration_type not in self._calibration:
            self._calibration[calibration_type] = []

        if isinstance(self._calibration[calibration_type], list):
            self._calibration[calibration_type].append(data)
        else:
            self._calibration[calibration_type] = data

    def _build_asdf_tree(self) -> dict[str, Any]:
        """Build the ASDF tree structure from session data.

        Assembles all session data (meta, cameras, telemetry, calibration,
        observability) into the hierarchical structure expected by ASDF.
        Sets end_time and calculates duration.

        Business context: ASDF files store telescope observation data in a
        standardized format. This method structures in-memory data for
        serialization, enabling post-session analysis with astropy tools.

        Args:
            No arguments. Uses internal state (_logs, _events, _cameras, etc.).

        Returns:
            dict: Complete ASDF tree with keys: meta, cameras, telemetry,
                calibration, observability. Ready for asdf.AsdfFile().

        Raises:
            No exceptions raised.

        Example:
            >>> # Called internally by close()
            >>> tree = session._build_asdf_tree()
            >>> tree['meta']['session_type']
            'observation'
        """
        self._end_time = datetime.now(UTC)
        duration_seconds = (self._end_time - self.start_time).total_seconds()

        return {
            "meta": {
                "session_type": self.session_type.value,
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self._end_time.isoformat(),
                "target": self.target,
                "purpose": self.purpose,
                "location": self.location if self.location else None,
            },
            "cameras": self._cameras,
            "telemetry": self._telemetry,
            "calibration": self._calibration,
            "observability": {
                "logs": self._logs,
                "events": self._events,
                "metrics": {
                    "frames_captured": self._frames_captured,
                    "errors": self._error_count,
                    "warnings": self._warning_count,
                    "duration_seconds": duration_seconds,
                },
            },
        }

    def _get_output_path(self) -> Path:
        """Get the output path for the ASDF file.

        Organizes files by date: data_dir/YYYY/MM/DD/session_id.asdf.
        Creates directories if they don't exist.

        Business context: Date-based directory structure enables easy
        browsing and archival of observation data. Automatic directory
        creation simplifies deployment without manual setup.

        Args:
            No arguments. Uses self.data_dir, self.start_time, self.session_id.

        Returns:
            Path: Full path where ASDF file will be written, e.g.,
                '/data/telescope/2025/12/14/observation_m31_20251214_210000.asdf'.

        Raises:
            No exceptions raised. Creates directories as needed.

        Example:
            >>> path = session._get_output_path()
            >>> str(path)
            '/data/telescope/2025/12/14/observation_m31_20251214_210000.asdf'
        """
        # Organize by date: data_dir/YYYY/MM/DD/session_id.asdf
        date_path = self.start_time.strftime("%Y/%m/%d")
        output_dir = self.data_dir / date_path
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.session_id}.asdf"
        return output_dir / filename

    def close(self) -> Path:
        """Close the session and write ASDF file.

        Logs a closing message, builds the ASDF tree from all accumulated
        data, and writes to disk. Session becomes read-only after close.

        Args:
            None. Operates on session's internal state.

        Returns:
            Path to the written ASDF file.

        Raises:
            RuntimeError: If session is already closed.

        Example:
            path = session.close()
            print(f"Session saved to {path}")
        """
        if self._closed:
            raise RuntimeError("Session already closed")

        # Log before setting closed flag
        self.log(LogLevel.INFO, "Session closing", session_id=self.session_id)
        self._closed = True

        # Build ASDF tree and write
        tree = self._build_asdf_tree()
        output_path = self._get_output_path()

        af = asdf.AsdfFile(tree)
        af.write_to(output_path)

        logger.info("Session written: %s", output_path)
        return output_path

    @property
    def is_closed(self) -> bool:
        """Check if session is closed (written to disk, immutable).

        Returns True if session closed via close() method (written to ASDF
        file, cannot accept new data). All add_* methods (add_frame,
        add_event, add_telemetry, add_calibration) raise
        RuntimeError on closed sessions. False indicates active session
        accepting data.

        Business context: Essential for preventing data corruption in
        long-running telescope systems where session objects may persist in
        memory after closing. Guards against accidental writes to finalized
        session files (ASDF format doesn't support appending). Used in error
        handling ("why did add_frame fail?"), workflow validation ("is this
        session still active?"), and session lifecycle management (close old
        session before starting new one). Critical for data integrity in
        multi-hour observations where hundreds of frames accumulated -
        accidentally reopening closed session would corrupt ASDF file.

        Implementation details: Returns self._closed boolean set by close()
        method. Initially False (set in __init__), becomes True after close()
        writes tree to ASDF and calls af.close(). Immutable state - once
        closed, stays closed (no reopen mechanism). Checked by all add_*
        methods before modifying tree data structure. Zero-cost operation
        (simple attribute access).

        Args:
            None. Property access pattern (not a method call).

        Returns:
            True if session closed (immutable, written to disk). False if
            active (accepting data).

        Raises:
            None. Always returns boolean - never raises exceptions.

        Example:
            >>> session = Session(SessionType.OBSERVATION, name="M31")
            >>> session.add_frame(frame_data, metadata)
            >>> assert session.is_closed == False  # Still active
            >>> session.close()
            >>> assert session.is_closed == True  # Now immutable
            >>> try:
            ...     session.add_frame(more_data, metadata)  # Fails
            ... except RuntimeError as e:
            ...     print(f"Cannot add to closed session: {e}")
        """
        return self._closed

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds (elapsed time since start).

        Returns elapsed time since session start (start_time). For active
        sessions, calculates duration to current moment (datetime.now). For
        closed sessions, uses finalized end_time from close() method. Useful
        for monitoring, logging, and session metadata.

        Business context: Critical metric for observation session monitoring
        and analysis. Long sessions (>2 hours) may indicate successful
        deep-sky imaging campaigns or stuck workflows needing intervention.
        Short sessions (<5 minutes) may indicate setup issues or failed
        observations. Used in dashboards ("current session: 1h 23m running"),
        alerting ("session exceeded 4 hour limit"), and post-observation
        analysis ("total observing time tonight: 6.5h"). Essential for
        calculating efficiency metrics (frames per hour, duty cycle,
        overhead time).

        Implementation details: Computes (end - start_time).total_seconds()
        where end is self._end_time (if closed) or datetime.now(UTC) (if
        active). Uses UTC to avoid daylight saving issues during overnight
        sessions. Returns float with subsecond precision (e.g., 3661.234 =
        1h 1m 1.234s). Active session duration continuously increases (call
        multiple times for progress tracking). Closed session duration fixed
        (snapshot at close() time). Typical values: 300-14400 seconds
        (5min-4h) for astronomy sessions.

        Args:
            None. Property access pattern (not a method call).

        Returns:
            Duration in seconds as float. Range typically 0.0 (just started)
            to 14400.0 (4 hours). Sub-second precision available (e.g.,
            1234.567 seconds). Always non-negative.

        Raises:
            None. Always computes valid duration - never raises exceptions.

        Example:
            >>> session = Session(SessionType.OBSERVATION, name="M31")
            >>> time.sleep(5)
            >>> print(f"Session running {session.duration_seconds:.1f}s")  # ~5.0s
            >>> session.add_frame(frame_data, metadata)  # Add more frames...
            >>> session.close()
            >>> print(f"Session completed in {session.duration_seconds:.1f}s")
            >>> # Format for human-readable display
            >>> hours, remainder = divmod(session.duration_seconds, 3600)
            >>> minutes, seconds = divmod(remainder, 60)
            >>> print(f"Duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        """
        end = self._end_time or datetime.now(UTC)
        return (end - self.start_time).total_seconds()
