"""Session manager for telescope data storage.

The SessionManager ensures there's always an active session to receive logs,
implementing the "always have a session" pattern from the architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from telescope_mcp.data.session import LogLevel, Session, SessionType
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages telescope session lifecycle with auto-idle.

    The SessionManager guarantees there's always an active session.
    When no explicit session is running, an idle session captures
    all logs and events. This solves the "where do logs go when
    there's no observation?" problem.

    Example:
        # Initialize (starts idle session automatically)
        sessions = SessionManager(data_dir=Path("/data/telescope"))

        # Logs go to idle session
        sessions.log(LogLevel.INFO, "Server started")

        # Start observation (closes idle session)
        sessions.start_session(SessionType.OBSERVATION, target="M31")
        sessions.log("INFO", "Observing M31")

        # End observation (returns to idle)
        path = sessions.end_session()

        # Logs go to new idle session
        sessions.log("INFO", "Waiting for next command")
    """

    def __init__(
        self,
        data_dir: Path | str,
        *,
        location: dict[str, float] | None = None,
        auto_rotate_idle: bool = True,
        idle_rotate_hours: int = 1,
    ) -> None:
        """Initialize the session manager.

        Creates the data directory if needed and starts an idle session.
        All telescope operations should go through the SessionManager to
        ensure logs and data are captured.

        Business context: SessionManager is the central coordinator for
        observation data persistence. It ensures continuous logging via
        idle sessions and proper ASDF file organization.

        Args:
            data_dir: Base directory for ASDF file storage.
            location: Default observer location {lat, lon, alt} where lat/lon
                are in degrees and alt is meters above sea level.
            auto_rotate_idle: Whether to auto-rotate idle sessions periodically.
            idle_rotate_hours: Hours between idle session rotations.

        Returns:
            None. Manager initialized with idle session active.

        Raises:
            No exceptions raised. Directory created if missing.

        Example:
            sessions = SessionManager(
                data_dir=Path("/data/telescope"),
                location={"lat": 34.05, "lon": -118.25, "alt": 100.0},
            )
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.location = location
        self.auto_rotate_idle = auto_rotate_idle
        self.idle_rotate_hours = idle_rotate_hours

        self._active_session: Session | None = None
        self._ensure_idle_session()

        logger.info("SessionManager initialized: %s", self.data_dir)

    def _ensure_idle_session(self) -> None:
        """Create idle session if no session is active.

        Maintains the "always have a session" invariant. Called after session
        end and during operations that require a session.
        """
        if self._active_session is None:
            self._active_session = Session(
                session_type=SessionType.IDLE,
                data_dir=self.data_dir,
                location=self.location,
                auto_rotate=self.auto_rotate_idle,
                rotate_interval_hours=self.idle_rotate_hours,
            )

    def start_session(
        self,
        session_type: SessionType | str,
        *,
        target: str | None = None,
        purpose: str | None = None,
        location: dict[str, float] | None = None,
    ) -> Session:
        """Start a new session, closing any existing one.

        Ends the current session (saving to ASDF) and creates a new one
        of the specified type. Use for observations, alignments, or experiments.

        Args:
            session_type: Type of session (observation, alignment, experiment,
                maintenance, idle).
            target: Target object name for observations (e.g., "M31", "Jupiter").
            purpose: Description for alignment/experiment sessions.
            location: Override default observer location.

        Returns:
            The newly created Session instance.

        Raises:
            ValueError: If session_type string is invalid.

        Example:
            session = sessions.start_session(
                SessionType.OBSERVATION,
                target="M42",
                purpose="Orion Nebula imaging",
            )
        """
        if not isinstance(session_type, SessionType):
            session_type = SessionType(str(session_type).lower())

        # Close existing session
        if self._active_session is not None:
            self._active_session.close()

        # Create new session
        self._active_session = Session(
            session_type=session_type,
            data_dir=self.data_dir,
            target=target,
            purpose=purpose,
            location=location or self.location,
        )

        logger.info(
            "Started %s session: %s",
            session_type.value,
            target or purpose or self._active_session.session_id,
        )

        return self._active_session

    def end_session(self) -> Path:
        """End current session and return to idle.

        Closes the active session, writes ASDF file to disk, and starts
        a new idle session to capture subsequent logs. The returned path
        points to the saved session data.

        Args:
            None. Operates on the currently active session.

        Returns:
            Path to the written ASDF file.

        Raises:
            RuntimeError: If no active session exists.

        Example:
            asdf_path = sessions.end_session()
            print(f"Session saved to {asdf_path}")
        """
        if self._active_session is None:
            raise RuntimeError("No active session to end")

        if self._active_session.session_type == SessionType.IDLE:
            logger.warning("Ending idle session explicitly")

        path = self._active_session.close()
        self._active_session = None
        self._ensure_idle_session()

        return path

    def log(
        self,
        level: LogLevel | str,
        message: str,
        source: str = "telescope_mcp",
        **context: Any,
    ) -> None:
        """Log to current session.

        Adds a structured log entry to the active session. Ensures idle
        session exists if needed. Context kwargs become searchable metadata.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: Human-readable log message.
            source: Source component identifier. Defaults to "telescope_mcp".
            **context: Additional context as key-value pairs.

        Returns:
            None.

        Raises:
            None. Automatically creates idle session if needed.

        Example:
            sessions.log(
                "INFO",
                "Starting exposure",
                source="camera",
                camera_id=0,
                exposure_us=500000,
            )
        """
        self._ensure_idle_session()
        assert self._active_session is not None
        self._active_session.log(level, message, source, **context)

    def add_event(self, event: str, **details: Any) -> None:
        """Record an event to current session.

        Events are timestamped markers for significant occurrences like
        slewing, focusing, or capturing. Different from logs in that
        events are structured data meant for programmatic analysis.

        Args:
            event: Event name/type (e.g., "slew_start", "focus_complete").
            **details: Event-specific details as key-value pairs.

        Returns:
            None.

        Raises:
            None. Automatically creates idle session if needed.

        Example:
            sessions.add_event(
                "slew_complete",
                target="M31",
                ra=10.684,
                dec=41.269,
                duration_sec=15.3,
            )
        """
        self._ensure_idle_session()
        assert self._active_session is not None
        self._active_session.add_event(event, **details)

    def add_frame(
        self,
        camera: str,
        frame: NDArray[np.uint8 | np.uint16],
        *,
        camera_info: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """Add a captured frame to current session.

        Stores image data with metadata in the session's ASDF file.
        Frames are organized by camera identifier for multi-camera setups.

        Args:
            camera: Camera identifier (e.g., "finder", "main").
            frame: Image data as numpy array (uint8 or uint16).
            camera_info: Camera metadata (resolution, pixel size, etc.). Optional.
            settings: Capture settings (exposure, gain, etc.). Optional.

        Returns:
            None.

        Raises:
            None. Automatically creates idle session if needed.

        Example:
            sessions.add_frame(
                "main",
                image_array,
                camera_info={"name": "ASI482MC", "width": 1920},
                settings={"exposure_us": 500000, "gain": 50},
            )
        """
        self._ensure_idle_session()
        assert self._active_session is not None
        self._active_session.add_frame(
            camera, frame, camera_info=camera_info, settings=settings
        )

    def add_telemetry(self, telemetry_type: str, **data: Any) -> None:
        """Add telemetry data to current session.

        Records time-series sensor data like temperature, humidity, or
        position readings. Useful for correlating environmental conditions
        with image quality.

        Args:
            telemetry_type: Type of telemetry (e.g., "environment", "motor").
            **data: Telemetry values as key-value pairs.

        Returns:
            None.

        Raises:
            None. Automatically creates idle session if needed.

        Example:
            sessions.add_telemetry(
                "environment",
                temperature_c=18.5,
                humidity_pct=65.0,
                pressure_hpa=1013.25,
            )
        """
        self._ensure_idle_session()
        assert self._active_session is not None
        self._active_session.add_telemetry(telemetry_type, **data)

    def add_calibration(self, calibration_type: str, data: Any) -> None:
        """Add calibration data to current session.

        Stores calibration results like dark frames, flat fields, or
        alignment matrices for later processing.

        Args:
            calibration_type: Type of calibration (e.g., "dark", "flat", "alignment").
            data: Calibration data (numpy array, dict, or other serializable).

        Returns:
            None.

        Raises:
            None. Automatically creates idle session if needed.

        Example:
            sessions.add_calibration("dark_frame", dark_array)
            sessions.add_calibration("alignment", {"rotation": 1.5, "scale": 0.98})
        """
        self._ensure_idle_session()
        assert self._active_session is not None
        self._active_session.add_calibration(calibration_type, data)

    @property
    def active_session(self) -> Session | None:
        """Get currently active session for direct access.

        Returns Session object for direct method access (add_frame, add_event,
        add_telemetry, close). May be idle session if no observation in progress.
        None only after shutdown() called.

        Most applications should use SessionManager.add_frame() etc. instead
        (simpler, auto-creates session). Use this for session metadata access
        (session_id, duration_seconds) or advanced use cases.

        Returns:
            Active Session instance, or None after shutdown().

        Example:
            >>> manager.start_session(SessionType.OBSERVATION, target="M31")
            >>> session = manager.active_session
            >>> print(f"{session.session_id}: {session.duration_seconds:.1f}s")
        """
        return self._active_session

    @property
    def active_session_type(self) -> SessionType | None:
        """Get the type of the currently active session.

        Used for context-aware behavior where different session types have
        different requirements (e.g., OBSERVATION needs calibration frames).

        Returns:
            SessionType enum (OBSERVATION, ALIGNMENT, EXPERIMENT, MAINTENANCE,
            or IDLE) if session active, None otherwise.

        Example:
            >>> manager.start_session(SessionType.OBSERVATION, target="M31")
            >>> if manager.active_session_type == SessionType.OBSERVATION:
            ...     print("Taking science frames")
        """
        if self._active_session:
            return self._active_session.session_type
        return None

    @property
    def active_session_id(self) -> str | None:
        """Get ID of currently active session.

        Returns session ID string in format '{type}_{target}_{YYYYMMDD_HHMMSS}'
        (e.g., 'observation_m31_20251218_203045'). Used for file naming,
        logging correlation, and UI display.

        Returns:
            Session ID string, or None after shutdown().

        Example:
            >>> manager.start_session(SessionType.OBSERVATION, target="M31")
            >>> print(f"Session: {manager.active_session_id}")
            observation_m31_20251231_004500
        """
        if self._active_session:
            return self._active_session.session_id
        return None

    def shutdown(self) -> Path | None:
        """Shutdown the session manager, closing any active session.

        Logs a shutdown message to the active session, closes it (writing
        ASDF to disk), and clears the session reference. Does not start
        a new idle session. Call at application exit.

        Args:
            None. Operates on the currently active session.

        Returns:
            Path to final ASDF file, or None if no session was active.

        Raises:
            None. Safe to call even if no session active.

        Example:
            path = sessions.shutdown()
            if path:
                print(f"Final session saved to {path}")
        """
        if self._active_session is not None:
            self._active_session.log(LogLevel.INFO, "SessionManager shutting down")
            path = self._active_session.close()
            self._active_session = None
            return path
        return None
