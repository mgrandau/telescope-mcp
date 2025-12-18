"""Tests for telescope_mcp.data module."""

from __future__ import annotations

from pathlib import Path

import asdf
import numpy as np
import pytest

from telescope_mcp.data import LogLevel, Session, SessionManager, SessionType


class TestSessionType:
    """Tests for SessionType enum."""

    def test_all_session_types_exist(self) -> None:
        """All documented session types should exist."""
        assert SessionType.OBSERVATION.value == "observation"
        assert SessionType.ALIGNMENT.value == "alignment"
        assert SessionType.EXPERIMENT.value == "experiment"
        assert SessionType.MAINTENANCE.value == "maintenance"
        assert SessionType.IDLE.value == "idle"

    def test_session_type_from_string(self) -> None:
        """SessionType can be created from string."""
        assert SessionType("observation") == SessionType.OBSERVATION
        assert SessionType("idle") == SessionType.IDLE


class TestSession:
    """Tests for Session class."""

    def test_session_creates_with_required_fields(self, tmp_path: Path) -> None:
        """Session initializes with required fields."""
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=tmp_path,
            target="M31",
        )

        assert session.session_type == SessionType.OBSERVATION
        assert session.target == "M31"
        assert session.session_id.startswith("observation_m31_")
        assert not session.is_closed

    def test_session_id_format_observation(self, tmp_path: Path) -> None:
        """Observation session ID includes target."""
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=tmp_path,
            target="Andromeda Galaxy",
        )
        # Should be: observation_andromeda_galaxy_YYYYMMDD_HHMMSS
        assert "observation_andromeda_galaxy_" in session.session_id

    def test_session_id_format_idle(self, tmp_path: Path) -> None:
        """Idle session ID has no target."""
        session = Session(
            session_type=SessionType.IDLE,
            data_dir=tmp_path,
        )
        # Should be: idle_YYYYMMDD_HHMMSS
        assert session.session_id.startswith("idle_")
        assert len(session.session_id.split("_")) == 3  # idle_date_time

    def test_session_log_stores_entries(self, tmp_path: Path) -> None:
        """Logging adds entries to internal buffer."""
        session = Session(
            session_type=SessionType.IDLE,
            data_dir=tmp_path,
        )

        session.log(LogLevel.INFO, "Test message", key="value")
        session.log("WARNING", "Warning message")

        assert len(session._logs) == 2
        assert session._logs[0]["message"] == "Test message"
        assert session._logs[0]["level"] == "INFO"
        assert session._logs[0]["context"] == {"key": "value"}
        assert session._logs[1]["level"] == "WARNING"

    def test_session_tracks_error_count(self, tmp_path: Path) -> None:
        """Session tracks error and warning counts."""
        session = Session(
            session_type=SessionType.IDLE,
            data_dir=tmp_path,
        )

        session.log(LogLevel.INFO, "Info")
        session.log(LogLevel.WARNING, "Warning 1")
        session.log(LogLevel.WARNING, "Warning 2")
        session.log(LogLevel.ERROR, "Error")
        session.log(LogLevel.CRITICAL, "Critical")

        assert session._warning_count == 2
        assert session._error_count == 2  # ERROR + CRITICAL

    def test_session_add_frame(self, tmp_path: Path) -> None:
        """Session stores frames with metadata."""
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=tmp_path,
            target="M31",
        )

        frame = np.random.randint(0, 65535, (1080, 1920), dtype=np.uint16)
        session.add_frame(
            "main",
            frame,
            camera_info={"model": "ASI482MC"},
            settings={"gain": 200, "exposure_us": 312000},
        )

        assert "main" in session._cameras
        assert len(session._cameras["main"]["frames"]) == 1
        assert session._cameras["main"]["info"]["model"] == "ASI482MC"
        assert session._frames_captured == 1

    def test_session_add_telemetry(self, tmp_path: Path) -> None:
        """Session stores telemetry data points."""
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=tmp_path,
            target="M31",
        )

        session.add_telemetry("mount_position", ra=12.5, dec=45.2, alt=60.0, az=180.0)
        session.add_telemetry("temperature", sensor="ambient", value=15.5)

        assert len(session._telemetry["mount_position"]) == 1
        assert session._telemetry["mount_position"][0]["ra"] == 12.5
        assert len(session._telemetry["temperature"]) == 1

    def test_session_add_event(self, tmp_path: Path) -> None:
        """Session stores events."""
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=tmp_path,
            target="M31",
        )

        session.add_event("tracking_lost", reason="wind gust")
        session.add_event("cloud_detected", coverage=0.3)

        assert len(session._events) == 2
        assert session._events[0]["event"] == "tracking_lost"

    def test_session_close_writes_asdf(self, tmp_path: Path) -> None:
        """Closing session writes ASDF file."""
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=tmp_path,
            target="M31",
        )

        session.log(LogLevel.INFO, "Test")
        frame = np.zeros((100, 100), dtype=np.uint16)
        session.add_frame("main", frame)

        path = session.close()

        assert path.exists()
        assert path.suffix == ".asdf"
        assert session.is_closed

    def test_session_asdf_structure(self, tmp_path: Path) -> None:
        """ASDF file has correct structure."""
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=tmp_path,
            target="M31",
            location={"lat": 45.5, "lon": -122.6, "alt": 100},
        )

        session.log(LogLevel.INFO, "Starting observation")
        frame = np.zeros((100, 100), dtype=np.uint16)
        session.add_frame("main", frame, settings={"gain": 200})
        session.add_telemetry("temperature", sensor="ccd", value=0.5)
        session.add_event("guiding_started")

        path = session.close()

        # Read back and verify structure
        with asdf.open(path) as af:
            assert "meta" in af.tree
            assert af["meta"]["session_type"] == "observation"
            assert af["meta"]["target"] == "M31"
            assert af["meta"]["location"]["lat"] == 45.5

            assert "cameras" in af.tree
            assert "main" in af["cameras"]
            assert len(af["cameras"]["main"]["frames"]) == 1

            assert "telemetry" in af.tree
            assert len(af["telemetry"]["temperature"]) == 1

            assert "observability" in af.tree
            assert len(af["observability"]["logs"]) >= 1
            assert len(af["observability"]["events"]) == 1
            assert af["observability"]["metrics"]["frames_captured"] == 1

    def test_session_file_organization(self, tmp_path: Path) -> None:
        """ASDF files are organized by date."""
        session = Session(
            session_type=SessionType.IDLE,
            data_dir=tmp_path,
        )

        path = session.close()

        # Should be: data_dir/YYYY/MM/DD/session_id.asdf
        parts = path.relative_to(tmp_path).parts
        assert len(parts) == 4  # YYYY/MM/DD/file.asdf
        assert parts[0].isdigit() and len(parts[0]) == 4  # Year
        assert parts[1].isdigit() and len(parts[1]) == 2  # Month
        assert parts[2].isdigit() and len(parts[2]) == 2  # Day

    def test_session_cannot_operate_after_close(self, tmp_path: Path) -> None:
        """Closed session rejects operations."""
        session = Session(
            session_type=SessionType.IDLE,
            data_dir=tmp_path,
        )
        session.close()

        with pytest.raises(RuntimeError, match="closed"):
            session.log(LogLevel.INFO, "Should fail")

        with pytest.raises(RuntimeError, match="closed"):
            session.add_frame("main", np.zeros((10, 10), dtype=np.uint8))

        with pytest.raises(RuntimeError, match="closed"):
            session.close()


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_manager_starts_with_idle_session(self, tmp_path: Path) -> None:
        """SessionManager starts with an idle session."""
        manager = SessionManager(data_dir=tmp_path)

        assert manager.active_session is not None
        assert manager.active_session_type == SessionType.IDLE

    def test_manager_log_goes_to_active_session(self, tmp_path: Path) -> None:
        """Logging through manager goes to active session."""
        manager = SessionManager(data_dir=tmp_path)

        manager.log("INFO", "Test message")

        assert len(manager.active_session._logs) == 1  # type: ignore

    def test_manager_start_session_closes_previous(self, tmp_path: Path) -> None:
        """Starting a session closes the previous one."""
        manager = SessionManager(data_dir=tmp_path)
        initial_id = manager.active_session_id

        manager.start_session(SessionType.OBSERVATION, target="M31")

        assert manager.active_session_type == SessionType.OBSERVATION
        assert manager.active_session_id != initial_id

        # Previous idle session should have been written
        asdf_files = list(tmp_path.rglob("*.asdf"))
        assert len(asdf_files) == 1
        assert "idle_" in asdf_files[0].name

    def test_manager_end_session_returns_to_idle(self, tmp_path: Path) -> None:
        """Ending a session returns to idle."""
        manager = SessionManager(data_dir=tmp_path)

        manager.start_session(SessionType.OBSERVATION, target="M31")
        path = manager.end_session()

        assert path.exists()
        assert "observation_m31_" in path.name
        assert manager.active_session_type == SessionType.IDLE

    def test_manager_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow from docs."""
        manager = SessionManager(
            data_dir=tmp_path,
            location={"lat": 45.5, "lon": -122.6, "alt": 100},
        )

        # Logs to idle session
        manager.log("INFO", "Server started")

        # Start alignment
        manager.start_session(SessionType.ALIGNMENT, purpose="ra_dec_calibration")
        manager.log("INFO", "Beginning plate solve")
        manager.add_calibration("plate_solve_results", {"ra": 12.5, "dec": 45.2})
        _alignment_path = manager.end_session()  # noqa: F841

        # Back to idle
        manager.log("INFO", "Waiting for observation command")

        # Start observation
        manager.start_session(SessionType.OBSERVATION, target="M31")
        manager.log("INFO", "Starting M31 observation")
        frame = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        manager.add_frame("main", frame, settings={"gain": 200})
        obs_path = manager.end_session()

        # Shutdown
        _final_path = manager.shutdown()  # noqa: F841

        # Verify files created
        asdf_files = list(tmp_path.rglob("*.asdf"))
        # Note: Multiple idle sessions in same second may share filename
        # At minimum: initial idle + alignment + observation = 3
        assert len(asdf_files) >= 3

        # Verify observation file
        with asdf.open(obs_path) as af:
            assert af["meta"]["target"] == "M31"
            assert len(af["cameras"]["main"]["frames"]) == 1

    def test_manager_accepts_string_session_type(self, tmp_path: Path) -> None:
        """SessionManager accepts string session types."""
        manager = SessionManager(data_dir=tmp_path)

        manager.start_session("observation", target="M31")
        assert manager.active_session_type == SessionType.OBSERVATION

        manager.start_session("alignment", purpose="focus")
        assert manager.active_session_type == SessionType.ALIGNMENT

    def test_manager_shutdown_returns_none_if_no_session(self, tmp_path: Path) -> None:
        """Shutdown returns None if no active session."""
        manager = SessionManager(data_dir=tmp_path)
        manager.shutdown()  # Closes idle session

        # Second shutdown with no session
        result = manager.shutdown()
        assert result is None
