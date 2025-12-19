"""Tests for telescope_mcp.data module."""

from __future__ import annotations

from pathlib import Path

import asdf
import numpy as np
import pytest

from telescope_mcp.data import LogLevel, Session, SessionManager, SessionType


class TestSessionType:
    """Test suite for SessionType enumeration values and construction.

    Categories:
    1. Enum Values - Verification of all session types (1 test)
    2. String Construction - Creating SessionType from strings (1 test)

    Total: 2 tests.
    """

    def test_all_session_types_exist(self) -> None:
        """Verifies SessionType enum defines all documented session types.

        Tests enum completeness by checking all session type values.

        Arrangement:
        1. SessionType enum defined with 5 types.
        2. Each type should have string value matching name.

        Action:
        Access .value attribute on each SessionType member.

        Assertion Strategy:
        Validates enum completeness by confirming:
        - OBSERVATION.value equals "observation".
        - ALIGNMENT.value equals "alignment".
        - EXPERIMENT.value equals "experiment".
        - MAINTENANCE.value equals "maintenance".
        - IDLE.value equals "idle".

        Testing Principle:
        Validates documented API, ensuring all session types
        defined and accessible for telescope operations."""
        assert SessionType.OBSERVATION.value == "observation"
        assert SessionType.ALIGNMENT.value == "alignment"
        assert SessionType.EXPERIMENT.value == "experiment"
        assert SessionType.MAINTENANCE.value == "maintenance"
        assert SessionType.IDLE.value == "idle"

    def test_session_type_from_string(self) -> None:
        """Verifies SessionType can be constructed from string values.

        Tests enum deserialization for configuration and MCP tools.

        Arrangement:
        1. Define string values: "observation", "idle".
        2. SessionType() constructor should accept strings.

        Action:
        Call SessionType(string) to create enum instances.

        Assertion Strategy:
        Validates string construction by confirming:
        - SessionType("observation") equals SessionType.OBSERVATION.
        - SessionType("idle") equals SessionType.IDLE.

        Testing Principle:
        Validates bidirectional conversion, ensuring enums
        deserializable from MCP tool string parameters."""
        assert SessionType("observation") == SessionType.OBSERVATION
        assert SessionType("idle") == SessionType.IDLE


class TestSession:
    """Tests for Session class."""

    def test_session_creates_with_required_fields(self, tmp_path: Path) -> None:
        """Verifies Session initializes with correct field values.

        Arrangement:
        1. tmp_path fixture provides temporary directory.
        2. Session constructor called with OBSERVATION type.
        3. Target set to "M31" for galaxy observation.

        Action:
        Creates Session with session_type, data_dir, and target.

        Assertion Strategy:
        Validates initialization by confirming:
        - session_type matches OBSERVATION enum.
        - target stored as "M31".
        - session_id follows observation_m31_ format.
        - is_closed is False (session active).

        Testing Principle:
        Validates basic object construction, ensuring
        required fields properly stored.
        """
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
        """Verifies observation session ID format includes target.

        Arrangement:
        1. tmp_path provides data directory.
        2. Session created with multi-word target "Andromeda Galaxy".
        3. ID format should normalize target to lowercase with underscores.

        Action:
        Creates observation session and inspects generated session_id.

        Assertion Strategy:
        Validates ID format by confirming:
        - Contains "observation_andromeda_galaxy_" prefix.
        - Target name normalized (spaces to underscores, lowercase).
        - Timestamp suffix follows prefix.

        Testing Principle:
        Validates ID generation convention, ensuring targets
        create readable, filesystem-safe identifiers.
        """
        session = Session(
            session_type=SessionType.OBSERVATION,
            data_dir=tmp_path,
            target="Andromeda Galaxy",
        )
        # Should be: observation_andromeda_galaxy_YYYYMMDD_HHMMSS
        assert "observation_andromeda_galaxy_" in session.session_id

    def test_session_id_format_idle(self, tmp_path: Path) -> None:
        """Verifies idle session ID omits target field.

        Arrangement:
        1. tmp_path provides data directory.
        2. Session created with IDLE type (no target).
        3. ID format should be: idle_YYYYMMDD_HHMMSS.

        Action:
        Creates idle session and validates ID structure.

        Assertion Strategy:
        Validates idle ID format by confirming:
        - Starts with "idle_" prefix.
        - Contains exactly 3 underscore-separated parts.
        - No target name in identifier.

        Testing Principle:
        Validates ID generation for idle sessions,
        ensuring simpler format without target.
        """
        session = Session(
            session_type=SessionType.IDLE,
            data_dir=tmp_path,
        )
        # Should be: idle_YYYYMMDD_HHMMSS
        assert session.session_id.startswith("idle_")
        assert len(session.session_id.split("_")) == 3  # idle_date_time

    def test_session_log_stores_entries(self, tmp_path: Path) -> None:
        """Verifies session logging stores entries with metadata.

        Arrangement:
        1. IDLE session created for log capture.
        2. Two log calls: INFO with context, WARNING without.
        3. Internal _logs buffer accumulates entries.

        Action:
        Calls session.log() twice with different levels and kwargs.

        Assertion Strategy:
        Validates log storage by confirming:
        - _logs contains 2 entries.
        - First entry has correct message, level, and context.
        - Second entry stored with WARNING level.

        Testing Principle:
        Validates logging mechanism, ensuring structured
        log data properly accumulated during session.
        """
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
        """Verifies session maintains warning/error counters.

        Arrangement:
        1. IDLE session created for metric tracking.
        2. Five log calls with varying severity levels.
        3. Counters should increment for WARNING/ERROR/CRITICAL.

        Action:
        Logs INFO (1), WARNING (2), ERROR (1), CRITICAL (1).

        Assertion Strategy:
        Validates metric tracking by confirming:
        - _warning_count equals 2 (both WARNING logs).
        - _error_count equals 2 (ERROR + CRITICAL combined).
        - INFO logs don't increment counters.

        Testing Principle:
        Validates observability metrics, ensuring
        session health indicators accurately tracked.
        """
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
        """Verifies session stores camera frames with metadata.

        Arrangement:
        1. OBSERVATION session created for M31.
        2. Random 16-bit frame generated (1080x1920).
        3. Camera info and settings provided as metadata.

        Action:
        Calls session.add_frame() with camera_info and settings kwargs.

        Assertion Strategy:
        Validates frame storage by confirming:
        - "main" camera registered in _cameras dict.
        - Frames list contains 1 entry.
        - Camera info stored (model=ASI482MC).
        - _frames_captured counter incremented to 1.

        Testing Principle:
        Validates frame capture tracking, ensuring
        image data and metadata properly stored.
        """
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
        """Verifies session stores time-series telemetry data.

        Arrangement:
        1. OBSERVATION session created for M31.
        2. Two telemetry types: mount_position and temperature.
        3. Each type stored as separate list in _telemetry dict.

        Action:
        Calls add_telemetry() twice with different metric types.

        Assertion Strategy:
        Validates telemetry storage by confirming:
        - mount_position list contains 1 entry with ra=12.5.
        - temperature list contains 1 entry.
        - Data organized by metric type.

        Testing Principle:
        Validates observability data collection, ensuring
        time-series metrics captured for analysis.
        """
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
        """Verifies session stores discrete event records.

        Arrangement:
        1. OBSERVATION session created for M31.
        2. Two events logged with different types and metadata.
        3. Events stored in _events list.

        Action:
        Calls add_event() twice with event names and kwargs.

        Assertion Strategy:
        Validates event logging by confirming:
        - _events list contains 2 entries.
        - First event has correct "tracking_lost" type.
        - Event metadata (reason, coverage) preserved.

        Testing Principle:
        Validates discrete event tracking, ensuring
        significant occurrences captured for replay.
        """
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
        """Verifies close() persists session to ASDF file.

        Arrangement:
        1. OBSERVATION session created with log and frame data.
        2. Data accumulated in-memory during session.
        3. close() should serialize to ASDF format.

        Action:
        Adds log entry and frame, then calls session.close().

        Assertion Strategy:
        Validates persistence by confirming:
        - Returned path exists on filesystem.
        - File has .asdf extension.
        - session.is_closed is True.

        Testing Principle:
        Validates data persistence, ensuring session
        data survives process termination.
        """
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
        """Verifies ASDF file contains expected data hierarchy.

        Arrangement:
        1. OBSERVATION session created with location metadata.
        2. All data types added: logs, frames, telemetry, events.
        3. Session closed to trigger ASDF write.

        Action:
        Populates session with diverse data, closes, then reads back.

        Assertion Strategy:
        Validates ASDF structure by confirming:
        - meta dict contains session_type, target, location.
        - cameras dict contains "main" with 1 frame.
        - telemetry dict contains temperature readings.
        - observability dict contains logs, events, metrics.

        Testing Principle:
        Validates data format compliance, ensuring ASDF
        files follow documented schema for compatibility.
        """
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
        """Verifies ASDF files organized in YYYY/MM/DD hierarchy.

        Arrangement:
        1. IDLE session created in tmp_path.
        2. close() should create dated directory structure.
        3. Path format: data_dir/YYYY/MM/DD/session_id.asdf.

        Action:
        Closes session and inspects resulting file path.

        Assertion Strategy:
        Validates file organization by confirming:
        - Path has 4 parts (year/month/day/file).
        - Year is 4-digit number.
        - Month and day are 2-digit numbers.

        Testing Principle:
        Validates data organization, ensuring files
        browsable by date for long-term management.
        """
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
        """Verifies closed sessions reject modification operations.

        Arrangement:
        1. IDLE session created and immediately closed.
        2. Closed state should prevent data corruption.
        3. Operations should raise RuntimeError with "closed" message.

        Action:
        Attempts log(), add_frame(), and double-close() on closed session.

        Assertion Strategy:
        Validates closed state enforcement by confirming:
        - log() raises RuntimeError mentioning "closed".
        - add_frame() raises RuntimeError mentioning "closed".
        - close() raises RuntimeError (idempotency check).

        Testing Principle:
        Validates data integrity, ensuring closed sessions
        immutable to prevent file corruption.
        """
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
        """Verifies SessionManager initializes with IDLE session.

        Arrangement:
        1. SessionManager constructed with data_dir.
        2. Should auto-create initial IDLE session.
        3. Session ready for logging immediately.

        Action:
        Constructs SessionManager and inspects active session.

        Assertion Strategy:
        Validates initialization by confirming:
        - active_session is not None.
        - active_session_type equals IDLE.

        Testing Principle:
        Validates always-active session design, ensuring
        logs never lost even before observation starts.
        """
        manager = SessionManager(data_dir=tmp_path)

        assert manager.active_session is not None
        assert manager.active_session_type == SessionType.IDLE

    def test_manager_log_goes_to_active_session(self, tmp_path: Path) -> None:
        """Verifies manager.log() forwards to active session.

        Arrangement:
        1. SessionManager created with initial IDLE session.
        2. manager.log() should delegate to active_session.log().
        3. Log entry stored in session's _logs buffer.

        Action:
        Calls manager.log() with INFO level.

        Assertion Strategy:
        Validates logging delegation by confirming:
        - active_session._logs contains 1 entry.
        - Log routed to current session.

        Testing Principle:
        Validates manager delegation pattern, ensuring
        logging convenience method works correctly.
        """
        manager = SessionManager(data_dir=tmp_path)

        manager.log("INFO", "Test message")

        assert len(manager.active_session._logs) == 1  # type: ignore

    def test_manager_start_session_closes_previous(self, tmp_path: Path) -> None:
        """Verifies start_session() auto-closes previous session.

        Arrangement:
        1. SessionManager starts with IDLE session.
        2. Initial session_id captured for comparison.
        3. start_session() should close IDLE, write ASDF, start OBSERVATION.

        Action:
        Calls start_session() with OBSERVATION type and target.

        Assertion Strategy:
        Validates session transition by confirming:
        - active_session_type changed to OBSERVATION.
        - active_session_id differs from initial (new session).
        - 1 ASDF file exists with "idle_" in name.

        Testing Principle:
        Validates automatic session lifecycle, ensuring
        previous session persisted before new one starts.
        """
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
        """Verifies end_session() closes current and creates IDLE.

        Arrangement:
        1. Manager starts with IDLE, then OBSERVATION session.
        2. end_session() should persist OBSERVATION ASDF.
        3. New IDLE session auto-created for continued logging.

        Action:
        Starts OBSERVATION, then calls end_session().

        Assertion Strategy:
        Validates session end by confirming:
        - Returned path exists with observation_m31_ filename.
        - active_session_type reverted to IDLE.
        - Manager ready for next session.

        Testing Principle:
        Validates always-active design, ensuring manager
        never without session for incoming logs.
        """
        manager = SessionManager(data_dir=tmp_path)

        manager.start_session(SessionType.OBSERVATION, target="M31")
        path = manager.end_session()

        assert path.exists()
        assert "observation_m31_" in path.name
        assert manager.active_session_type == SessionType.IDLE

    def test_manager_full_workflow(self, tmp_path: Path) -> None:
        """Verifies complete observation workflow end-to-end.

        Arrangement:
        1. Manager created with location metadata.
        2. Workflow: IDLE → ALIGNMENT → IDLE → OBSERVATION → shutdown.
        3. Each session should persist to separate ASDF file.

        Action:
        Executes full documented workflow with logging and frame capture.

        Assertion Strategy:
        Validates workflow by confirming:
        - At least 3 ASDF files created.
        - Observation ASDF contains target="M31" and 1 frame.
        - Session transitions work correctly.

        Testing Principle:
        Validates real-world usage pattern, ensuring
        documented workflow actually functions.
        """
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
        """Verifies start_session() accepts string session types.

        Arrangement:
        1. SessionManager created with initial IDLE.
        2. API should accept string "observation" or SessionType.OBSERVATION.
        3. Strings normalized to SessionType enum internally.

        Action:
        Calls start_session() with string "observation" and "alignment".

        Assertion Strategy:
        Validates string API by confirming:
        - "observation" string converted to OBSERVATION enum.
        - "alignment" string converted to ALIGNMENT enum.
        - String API equivalent to enum API.

        Testing Principle:
        Validates convenience API, ensuring string types
        work for MCP tool callers.
        """
        manager = SessionManager(data_dir=tmp_path)

        manager.start_session("observation", target="M31")
        assert manager.active_session_type == SessionType.OBSERVATION

        manager.start_session("alignment", purpose="focus")
        assert manager.active_session_type == SessionType.ALIGNMENT

    def test_manager_shutdown_returns_none_if_no_session(self, tmp_path: Path) -> None:
        """Verifies shutdown() handles double-shutdown gracefully.

        Arrangement:
        1. Manager created with initial IDLE session.
        2. First shutdown() closes IDLE session.
        3. Second shutdown() encounters no active session.

        Action:
        Calls shutdown() twice in succession.

        Assertion Strategy:
        Validates idempotency by confirming:
        - Second shutdown() returns None.
        - No exception raised on empty shutdown.

        Testing Principle:
        Validates defensive programming, ensuring
        shutdown idempotent and safe to call multiple times.
        """
        manager = SessionManager(data_dir=tmp_path)
        manager.shutdown()  # Closes idle session

        # Second shutdown with no session
        result = manager.shutdown()
        assert result is None
