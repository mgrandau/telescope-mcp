"""Data storage module using ASDF as single source of truth.

This module implements session-based data storage where every telescope
activity (observation, alignment, experiment, maintenance, idle) produces
an ASDF file with complete provenance.

Example:
    from pathlib import Path

    import numpy as np

    from telescope_mcp.data import LogLevel, SessionManager, SessionType

    # Initialize (starts idle session automatically)
    sessions = SessionManager(data_dir=Path("/data/telescope"))

    # Logs go to idle session (string or LogLevel enum)
    sessions.log(LogLevel.INFO, "Server started")

    # Start observation
    sessions.start_session(SessionType.OBSERVATION, target="M31")
    frame_array: np.ndarray = np.zeros((1920, 1080), dtype=np.uint16)
    sessions.add_frame("main", frame_array, settings={"gain": 200})

    # End observation → writes ASDF file
    path = sessions.end_session()
"""

from telescope_mcp.data.session import LogLevel, Session, SessionType
from telescope_mcp.data.session_manager import SessionManager

__all__ = [
    "LogLevel",
    "Session",
    "SessionManager",
    "SessionType",
]
