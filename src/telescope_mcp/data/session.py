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


# Maximum length for target name in session ID (for filename safety)
MAX_TARGET_SLUG_LENGTH = 20


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
            auto_rotate: Reserved for SessionManager idle rotation.
            rotate_interval_hours: Reserved for SessionManager idle rotation.

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

        Format: '{type}_{target}_{YYYYMMDD_HHMMSS}' or '{type}_{YYYYMMDD_HHMMSS}'.
        Target is slugified (lowercase, spaces to underscores, max 20 chars).

        Business context: Session IDs appear in filenames, logs, and UI.
        Human-readable format aids debugging and file organization. Timestamp
        ensures uniqueness even for same-target observations.

        Args:
            None. Uses self.session_type, self.target, self.start_time.

        Returns:
            Unique session ID like 'observation_m31_20251214_210000'.

        Raises:
            None. Always succeeds.

        Example:
            >>> session = Session(SessionType.OBSERVATION, Path("/data"), target="M31")
            >>> session.session_id  # Uses _generate_session_id internally
            'observation_m31_20251231_120000'
        """
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        if self.target:
            # observation_m31_20251214_210000
            target_slug = self.target.lower().replace(" ", "_")[:MAX_TARGET_SLUG_LENGTH]
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

        if not isinstance(level, LogLevel):
            level = LogLevel(str(level).upper())

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

        self._calibration[calibration_type].append(data)

    def _build_asdf_tree(self) -> dict[str, Any]:
        """Build the ASDF tree structure from session data.

        Assembles meta, cameras, telemetry, calibration, and observability
        into the hierarchical structure for ASDF serialization. Sets end_time
        and calculates final duration.

        Business context: ASDF is the standard format for astronomical data.
        The tree structure organizes all session data (frames, telemetry,
        logs) for reproducibility and analysis with standard astronomy tools.

        Args:
            None. Uses session's internal state.

        Returns:
            Complete ASDF tree dict with keys: meta, cameras, telemetry,
            calibration, observability.

        Raises:
            None. Always succeeds (empty lists for missing data).

        Example:
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
        Creates directories if needed.

        Business context: Date-based directory structure enables easy browsing
        and cleanup of old data. Matches common astronomy data organization
        patterns and simplifies backup/archive workflows.

        Args:
            None. Uses self.data_dir, self.start_time, self.session_id.

        Returns:
            Full path for the ASDF file, e.g.,
            '/data/2025/12/31/observation_m31_20251231_210000.asdf'.

        Raises:
            OSError: If directory creation fails (permissions, disk full).

        Example:
            >>> session = Session(SessionType.OBSERVATION, Path("/data"), target="M31")
            >>> session._get_output_path()
            PosixPath('/data/2025/12/31/observation_m31_20251231_120000.asdf')
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

        Returns True after close() has been called. Closed sessions
        reject all add_* methods with RuntimeError.

        Business context: Session immutability after close ensures ASDF
        files are complete and consistent. Tools check is_closed before
        attempting writes to provide clear error messages rather than
        silent failures.

        Args:
            None. Property accessor.

        Returns:
            True if closed, False if still active.

        Raises:
            None. Read-only property always succeeds.

        Example:
            >>> session = Session(SessionType.OBSERVATION, Path("/data"), target="M31")
            >>> session.is_closed
            False
            >>> session.close()
            >>> session.is_closed
            True
        """
        return self._closed

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds.

        For active sessions, returns time since start. For closed sessions,
        returns the final duration (start to close time).

        Business context: Duration tracking is essential for observation
        planning and post-session analysis. Displayed in dashboard UI,
        recorded in ASDF metadata, and used for session rotation decisions.

        Args:
            None. Property accessor.

        Returns:
            Duration in seconds as float (e.g., 3661.5 = ~1 hour).

        Raises:
            None. Always succeeds.

        Example:
            >>> session = Session(SessionType.OBSERVATION, Path("/data"), target="M31")
            >>> # ... do work ...
            >>> print(f"Running for {session.duration_seconds:.1f}s")
        """
        end = self._end_time or datetime.now(UTC)
        return (end - self.start_time).total_seconds()
