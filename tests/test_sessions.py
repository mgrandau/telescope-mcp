"""Tests for session MCP tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telescope_mcp.drivers import config
from telescope_mcp.tools import sessions


@pytest.fixture(autouse=True)
def reset_config(tmp_path: Path):
    """Pytest fixture resetting global config state for test isolation.

    Clears factory and session manager globals, configures fresh
    DriverConfig with tmp_path, ensures clean state per test.

    Arrangement:
    1. Resets config._factory and config._session_manager to None.
    2. Creates new DriverConfig with tmp_path for isolated storage.
    3. Configures system via config.configure(cfg).
    4. Yields for test execution.
    5. Cleanup: shuts down session manager, clears globals.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None (generator fixture, yields for test execution).

    Raises:
        None.

    Example:
        # Automatic fixture, applied to all tests in module
        def test_something():
            # Fresh config with isolated tmp_path
            pass

    Testing Principle:
        Ensures test isolation by resetting global state,
        preventing config leakage between tests.
    """
    # Reset globals
    config._factory = None
    config._session_manager = None

    # Configure with temp data dir
    cfg = config.DriverConfig(data_dir=tmp_path)
    config.configure(cfg)

    yield

    # Cleanup: shutdown session manager
    if config._session_manager is not None:
        config._session_manager.shutdown()
    config._factory = None
    config._session_manager = None


class TestSessionTools:
    """Test suite for session MCP tool functions.

    Categories:
    1. Session Info Query - get_session_info (1 test)
    2. Session Lifecycle - start/end sessions (5 tests)
    3. Session Data - logging and events (2 tests)
    4. Configuration - data directory (2 tests)
    5. Integration - full workflow (1 test)

    Total: 11 tests (includes 2 already documented).
    """

    @pytest.mark.asyncio
    async def test_get_session_info_returns_idle(self) -> None:
        """Verifies _get_session_info() returns IDLE session by default.

        Tests default session state after initialization.

        Arrangement:
        1. reset_config fixture initializes SessionManager.
        2. Manager auto-creates initial IDLE session.
        3. No observation started yet.

        Action:
        Calls await sessions._get_session_info() MCP tool.

        Assertion Strategy:
        Validates default session by confirming:
        - Result contains 1 text item.
        - JSON data["session_type"] equals "idle".
        - data["is_idle"] is True.

        Testing Principle:
        Validates always-active session design, ensuring
        MCP tool returns IDLE before observations start."""
        result = await sessions._get_session_info()
        assert len(result) == 1

        data = json.loads(result[0].text)
        assert data["session_type"] == "idle"
        assert data["is_idle"] is True

    @pytest.mark.asyncio
    async def test_start_observation_session(self) -> None:
        """Verifies _start_session() creates observation session with target.

        Tests observation session creation via MCP tool.

        Arrangement:
        1. Initial IDLE session active.
        2. Target specified as "M31" (Andromeda Galaxy).
        3. Purpose omitted (not required for observations).

        Action:
        Calls _start_session("observation", target="M31", purpose=None).

        Assertion Strategy:
        Validates session creation by confirming:
        - Result has 1 text item.
        - data["status"] equals "started".
        - data["session_type"] equals "observation".
        - data["target"] equals "M31".
        - data["session_id"] contains "m31" (normalized target).

        Testing Principle:
        Validates MCP tool API, ensuring observation sessions
        include target in metadata and session ID."""
        result = await sessions._start_session(
            "observation", target="M31", purpose=None
        )
        assert len(result) == 1

        data = json.loads(result[0].text)
        assert data["status"] == "started"
        assert data["session_type"] == "observation"
        assert data["target"] == "M31"
        assert "m31" in data["session_id"].lower()

    @pytest.mark.asyncio
    async def test_start_alignment_session(self) -> None:
        """Verifies _start_session() creates alignment session with purpose.

        Tests alignment session creation with purpose field.

        Arrangement:
        1. Initial IDLE session active.
        2. Purpose specified as "ra_dec_calibration".
        3. Target omitted (not required for alignment).

        Action:
        Calls _start_session("alignment", target=None, purpose="ra_dec_calibration").

        Assertion Strategy:
        Validates alignment session by confirming:
        - data["session_type"] equals "alignment".
        - data["purpose"] equals "ra_dec_calibration".

        Testing Principle:
        Validates session type handling, ensuring alignment
        sessions support purpose instead of target."""
        result = await sessions._start_session(
            "alignment", target=None, purpose="ra_dec_calibration"
        )

        data = json.loads(result[0].text)
        assert data["session_type"] == "alignment"
        assert data["purpose"] == "ra_dec_calibration"

    @pytest.mark.asyncio
    async def test_cannot_start_idle_session(self) -> None:
        """Verifies _start_session() rejects manual IDLE session creation.

        Tests IDLE session restriction (auto-created only).

        Arrangement:
        1. IDLE sessions managed automatically by SessionManager.
        2. Manual creation should be rejected.
        3. _start_session() validates session_type.

        Action:
        Attempts _start_session("idle", target=None, purpose=None).

        Assertion Strategy:
        Validates restriction by confirming:
        - result[0].text contains "Cannot manually start an idle session".

        Testing Principle:
        Validates input validation, ensuring IDLE sessions
        not creatable via MCP tool (system-managed only)."""
        result = await sessions._start_session("idle", target=None, purpose=None)

        assert "Cannot manually start an idle session" in result[0].text

    @pytest.mark.asyncio
    async def test_invalid_session_type(self) -> None:
        """Verifies _start_session() rejects unknown session types.

        Tests session type validation.

        Arrangement:
        1. Valid session types: observation, alignment, experiment, maintenance.
        2. "invalid" not in SessionType enum.
        3. _start_session() should validate type.

        Action:
        Attempts _start_session("invalid", target=None, purpose=None).

        Assertion Strategy:
        Validates type checking by confirming:
        - result[0].text contains "Invalid session type".

        Testing Principle:
        Validates input validation, ensuring MCP tool
        rejects malformed session type values."""
        result = await sessions._start_session("invalid", target=None, purpose=None)

        assert "Invalid session type" in result[0].text

    @pytest.mark.asyncio
    async def test_end_session_returns_to_idle(self, tmp_path: Path) -> None:
        """Verifies _end_session() closes observation and returns IDLE.

        Arrangement:
        1. Start observation session for M31.
        2. Active session should be OBSERVATION type.
        3. _end_session() should persist and create IDLE.

        Action:
        Calls _end_session() after starting observation.

        Assertion Strategy:
        Validates session ending by confirming:
        - Result status is "ended".
        - Result session_type is "observation".
        - file_path ends with .asdf.
        - Subsequent get_session_info shows IDLE type.

        Testing Principle:
        Validates MCP tool workflow, ensuring end_session
        properly transitions state.
        """
        # Start observation
        await sessions._start_session("observation", target="M31", purpose=None)

        # End it
        result = await sessions._end_session()
        data = json.loads(result[0].text)

        assert data["status"] == "ended"
        assert data["session_type"] == "observation"
        assert data["file_path"].endswith(".asdf")

        # Check we're back to idle
        info = await sessions._get_session_info()
        info_data = json.loads(info[0].text)
        assert info_data["session_type"] == "idle"

    @pytest.mark.asyncio
    async def test_end_idle_session_rejected(self) -> None:
        """Verifies _end_session() rejects ending IDLE session.

        Tests IDLE session protection from manual ending.

        Arrangement:
        1. Only IDLE session active (no observation started).
        2. IDLE sessions system-managed, not user-endable.
        3. _end_session() should reject.

        Action:
        Calls _end_session() with only IDLE active.

        Assertion Strategy:
        Validates protection by confirming:
        - result[0].text contains "No active session to end".

        Testing Principle:
        Validates business logic, ensuring IDLE sessions
        cannot be manually ended (auto-managed)."""
        result = await sessions._end_session()

        assert "No active session to end" in result[0].text

    @pytest.mark.asyncio
    async def test_session_log(self) -> None:
        """Verifies _session_log() records log entries to active session.

        Tests session logging via MCP tool.

        Arrangement:
        1. IDLE session active by default.
        2. _session_log() should append to session's log buffer.
        3. Log entry includes level, message, source.

        Action:
        Calls _session_log("INFO", "Test message", "test").

        Assertion Strategy:
        Validates logging by confirming:
        - data["status"] equals "logged".
        - data["level"] equals "INFO".
        - data["message"] equals "Test message".

        Testing Principle:
        Validates session data collection, ensuring MCP
        tool properly forwards logs to session manager."""
        result = await sessions._session_log("INFO", "Test message", "test")
        data = json.loads(result[0].text)

        assert data["status"] == "logged"
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"

    @pytest.mark.asyncio
    async def test_session_event(self) -> None:
        """Verifies _session_event() records discrete events to session.

        Tests event recording via MCP tool.

        Arrangement:
        1. IDLE session active.
        2. Event type: "tracking_lost" with details dict.
        3. _session_event() should store event metadata.

        Action:
        Calls _session_event("tracking_lost", {"reason": "wind"}).

        Assertion Strategy:
        Validates event recording by confirming:
        - data["status"] equals "recorded".
        - data["event"] equals "tracking_lost".
        - data["details"]["reason"] equals "wind".

        Testing Principle:
        Validates event tracking, ensuring MCP tool
        captures significant occurrences with metadata."""
        result = await sessions._session_event("tracking_lost", {"reason": "wind"})
        data = json.loads(result[0].text)

        assert data["status"] == "recorded"
        assert data["event"] == "tracking_lost"
        assert data["details"]["reason"] == "wind"

    @pytest.mark.asyncio
    async def test_get_data_dir(self, tmp_path: Path) -> None:
        """Verifies _get_data_dir() returns configured directory.

        Arrangement:
        1. reset_config fixture sets data_dir to tmp_path.
        2. _get_data_dir() should return current config value.
        3. Result in JSON format.

        Action:
        Calls _get_data_dir() MCP tool.

        Assertion Strategy:
        Validates directory retrieval by confirming:
        - data["data_dir"] equals str(tmp_path).

        Testing Principle:
        Validates configuration query, ensuring MCP tool
        provides access to data directory setting.
        """
        result = await sessions._get_data_dir()
        data = json.loads(result[0].text)

        assert data["data_dir"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_set_data_dir(self, tmp_path: Path) -> None:
        """Verifies _set_data_dir() updates data directory.

        Arrangement:
        1. Initial data_dir set by reset_config fixture.
        2. new_dir path created under tmp_path.
        3. _set_data_dir() should update configuration.

        Action:
        Calls _set_data_dir() with new directory path.

        Assertion Strategy:
        Validates directory update by confirming:
        - Result status is "updated".
        - data["data_dir"] equals str(new_dir).

        Testing Principle:
        Validates configuration mutation, ensuring MCP tool
        can change data directory at runtime.
        """
        new_dir = tmp_path / "new_data"

        result = await sessions._set_data_dir(str(new_dir))
        data = json.loads(result[0].text)

        assert data["status"] == "updated"
        assert data["data_dir"] == str(new_dir)

    @pytest.mark.asyncio
    async def test_full_workflow(self, tmp_path: Path) -> None:
        """Verifies complete session workflow via MCP tools.

        Arrangement:
        1. All MCP session tools available.
        2. Workflow: IDLE → OBSERVATION → log/event → end → IDLE.
        3. All operations should succeed.

        Action:
        Executes full documented workflow using MCP tools.

        Assertion Strategy:
        Validates end-to-end workflow by confirming:
        - All tool calls succeed (no exceptions).
        - Session transitions work correctly.
        - Data persisted to ASDF file.

        Testing Principle:
        Validates integration, ensuring MCP tools compose
        correctly for real-world usage.
        """
        # Log to idle
        await sessions._session_log("INFO", "Server started", "server")

        # Start observation
        start_result = await sessions._start_session(
            "observation", target="M31", purpose=None
        )
        start_data = json.loads(start_result[0].text)
        assert start_data["status"] == "started"

        # Log and record event
        await sessions._session_log("INFO", "Observing M31", "camera")
        await sessions._session_event("guiding_started", {"mode": "auto"})

        # Check metrics
        info = await sessions._get_session_info()
        info_data = json.loads(info[0].text)
        assert info_data["metrics"]["log_entries"] >= 1
        assert info_data["metrics"]["events"] == 1

        # End session
        end_result = await sessions._end_session()
        end_data = json.loads(end_result[0].text)

        # Verify file was created
        file_path = Path(end_data["file_path"])
        assert file_path.exists()
        assert file_path.suffix == ".asdf"
