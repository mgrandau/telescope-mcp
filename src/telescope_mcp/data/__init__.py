"""Data storage module using ASDF as single source of truth.

This module implements session-based data storage where every telescope
activity (observation, alignment, experiment, maintenance, idle) produces
an ASDF file with complete provenance.

Example:
    from telescope_mcp.data import SessionManager, SessionType

    # Initialize (starts idle session automatically)
    sessions = SessionManager(data_dir="/data/telescope")

    # Logs go to idle session
    sessions.log("INFO", "Server started")

    # Start observation
    sessions.start_session(SessionType.OBSERVATION, target="M31")
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
