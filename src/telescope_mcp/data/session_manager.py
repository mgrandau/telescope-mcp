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
        sessions.log("INFO", "Server started")

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

        Args:
            data_dir: Base directory for ASDF file storage
            location: Default observer location {lat, lon, alt}
            auto_rotate_idle: Whether to auto-rotate idle sessions
            idle_rotate_hours: Hours between idle session rotations
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
        """Create idle session if nothing else is active."""
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

        Args:
            session_type: Type of session to start
            target: Target object (for observations)
            purpose: Purpose description (for alignment/experiment)
            location: Override default location

        Returns:
            The newly created session
        """
        if isinstance(session_type, str):
            session_type = SessionType(session_type.lower())

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

        return self._active_session

    def end_session(self) -> Path:
        """End current session and return to idle.

        Returns:
            Path to the written ASDF file
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
        """Log to current session (always exists).

        Args:
            level: Log level
            message: Log message
            source: Source component
            **context: Additional context
        """
        self._ensure_idle_session()
        self._active_session.log(level, message, source, **context)  # type: ignore[union-attr]

    def add_event(self, event: str, **details: Any) -> None:
        """Record an event to current session.

        Args:
            event: Event name
            **details: Event details
        """
        self._ensure_idle_session()
        self._active_session.add_event(event, **details)  # type: ignore[union-attr]

    def add_frame(
        self,
        camera: str,
        frame: NDArray[np.uint8 | np.uint16],
        *,
        camera_info: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """Add a frame to current session.

        Args:
            camera: Camera identifier
            frame: Image data
            camera_info: Camera metadata
            settings: Capture settings
        """
        self._ensure_idle_session()
        self._active_session.add_frame(  # type: ignore[union-attr]
            camera, frame, camera_info=camera_info, settings=settings
        )

    def add_telemetry(self, telemetry_type: str, **data: Any) -> None:
        """Add telemetry to current session.

        Args:
            telemetry_type: Type of telemetry
            **data: Telemetry data
        """
        self._ensure_idle_session()
        self._active_session.add_telemetry(telemetry_type, **data)  # type: ignore[union-attr]

    def add_calibration(self, calibration_type: str, data: Any) -> None:
        """Add calibration data to current session.

        Args:
            calibration_type: Type of calibration
            data: Calibration data
        """
        self._ensure_idle_session()
        self._active_session.add_calibration(calibration_type, data)  # type: ignore[union-attr]

    @property
    def active_session(self) -> Session | None:
        """Get the currently active session."""
        return self._active_session

    @property
    def active_session_type(self) -> SessionType | None:
        """Get the type of the currently active session."""
        if self._active_session:
            return self._active_session.session_type
        return None

    @property
    def active_session_id(self) -> str | None:
        """Get the ID of the currently active session."""
        if self._active_session:
            return self._active_session.session_id
        return None

    def shutdown(self) -> Path | None:
        """Shutdown the session manager, closing any active session.

        Returns:
            Path to final ASDF file, or None if no session was active
        """
        if self._active_session is not None:
            self._active_session.log(
                LogLevel.INFO, "SessionManager shutting down"
            )
            path = self._active_session.close()
            self._active_session = None
            return path
        return None
