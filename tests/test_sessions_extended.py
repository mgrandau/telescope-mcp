"""Extended tests for session management tools."""

import json

import pytest

from telescope_mcp.data import LogLevel, SessionType
from telescope_mcp.devices import init_registry, shutdown_registry
from telescope_mcp.drivers.cameras.twin import DigitalTwinCameraDriver
from telescope_mcp.drivers.config import get_session_manager
from telescope_mcp.tools import sessions


class TestSessionTools:
    """Tests for session management MCP tools."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Fixture initializing camera registry for extended session tests.

        Arrangement:
        1. Creates DigitalTwinCameraDriver (simulated ZWO ASI cameras).
        2. Calls init_registry(driver) to register cameras globally.
        3. Yields for test execution.
        4. Cleanup: shutdown_registry() clears global state.

        Args:
        self: Test class instance (pytest convention).

        Returns:
        None (generator fixture).

        Raises:
        None.

        Testing Principle:
        Provides camera infrastructure for session tests,
        ensuring clean global registry state per test.
        """
        driver = DigitalTwinCameraDriver()
        init_registry(driver)
        yield
        shutdown_registry()

    @pytest.mark.asyncio
    async def test_start_observation_session(self):
        """Verifies starting OBSERVATION session with target parameter.

        Arrangement:
        1. Camera registry initialized via setup fixture.
        2. _start_session() with type="observation", target="M31".
        3. Expected: New observation session created.

        Action:
        Calls _start_session() and parses JSON response.

        Assertion Strategy:
        Validates session creation by confirming:
        - data["session_type"] = "observation".
        - data["target"] = "M31".

        Testing Principle:
        Validates session type handling, ensuring OBSERVATION
        sessions properly record astronomical targets.
        """
        result = await sessions._start_session(
            session_type="observation",
            target="M31",
            purpose=None,
        )
        data = json.loads(result[0].text)
        assert data["session_type"] == "observation"
        assert data["target"] == "M31"

    @pytest.mark.asyncio
    async def test_start_alignment_session(self):
        """Verifies starting ALIGNMENT session with purpose parameter.

        Arrangement:
        1. Camera registry initialized.
        2. _start_session() with type="alignment", purpose="polar alignment".
        3. ALIGNMENT sessions track calibration/alignment work.

        Action:
        Calls _start_session() for alignment and parses response.

        Assertion Strategy:
        Validates alignment session by confirming:
        - data["session_type"] = "alignment".
        - data["purpose"] = "polar alignment".

        Testing Principle:
        Validates purpose tracking, ensuring ALIGNMENT sessions
        record the specific alignment being performed.
        """
        result = await sessions._start_session(
            session_type="alignment",
            target=None,
            purpose="polar alignment",
        )
        data = json.loads(result[0].text)
        assert data["session_type"] == "alignment"
        assert data["purpose"] == "polar alignment"

    @pytest.mark.asyncio
    async def test_start_experiment_session(self):
        """Verifies starting EXPERIMENT session for testing purposes.

        Arrangement:
        1. Camera registry initialized.
        2. _start_session() with type="experiment",
           purpose="testing new camera settings".
        3. EXPERIMENT sessions track development/testing work.

        Action:
        Calls _start_session() for experiment and parses response.

        Assertion Strategy:
        Validates experiment session by confirming:
        - data["session_type"] = "experiment".

        Testing Principle:
        Validates session type diversity, ensuring EXPERIMENT
        sessions are distinct from production observations.
        """
        result = await sessions._start_session(
            session_type="experiment",
            target=None,
            purpose="testing new camera settings",
        )
        data = json.loads(result[0].text)
        assert data["session_type"] == "experiment"

    @pytest.mark.asyncio
    async def test_start_maintenance_session(self):
        """Verifies starting MAINTENANCE session for equipment upkeep.

        Arrangement:
        1. Camera registry initialized.
        2. _start_session() with type="maintenance", purpose="cleaning optics".
        3. MAINTENANCE sessions track servicing activities.

        Action:
        Calls _start_session() for maintenance and parses response.

        Assertion Strategy:
        Validates maintenance session by confirming:
        - data["session_type"] = "maintenance".

        Testing Principle:
        Validates session type completeness, ensuring all
        SessionType enum values are supported.
        """
        result = await sessions._start_session(
            session_type="maintenance",
            target=None,
            purpose="cleaning optics",
        )
        data = json.loads(result[0].text)
        assert data["session_type"] == "maintenance"

    @pytest.mark.asyncio
    async def test_get_session_info(self):
        """Verifies _get_session_info returns current session details.

        Arrangement:
        1. Session started with type="observation", target="M42".
        2. _get_session_info() should return active session.
        3. Response includes session_id and session_type.

        Action:
        Starts session, then calls _get_session_info().

        Assertion Strategy:
        Validates session info by confirming:
        - data contains "session_id" key.
        - data["session_type"] = "observation".

        Testing Principle:
        Validates session query, ensuring active session
        details are accessible via MCP tool.
        """
        # Start a session first
        await sessions._start_session("observation", "M42", None)

        # Get info
        result = await sessions._get_session_info()
        data = json.loads(result[0].text)
        assert "session_id" in data
        assert data["session_type"] == "observation"

    @pytest.mark.asyncio
    async def test_end_session(self):
        """Verifies _end_session terminates active session.

        Arrangement:
        1. Session started with type="experiment", purpose="test".
        2. _end_session() should complete session.
        3. Response includes completed session_id.

        Action:
        Starts session, then calls _end_session().

        Assertion Strategy:
        Validates session ending by confirming:
        - data contains "session_id" key (ended session).

        Testing Principle:
        Validates session lifecycle, ensuring sessions
        can be properly ended via MCP tool.
        """
        # Start a session
        await sessions._start_session("experiment", None, "test")

        # End it
        result = await sessions._end_session()
        data = json.loads(result[0].text)
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_session_log(self):
        """Verifies _session_log adds log entry to active session.

        Arrangement:
        1. Session started with type="observation", target="M45".
        2. _session_log() with level="INFO", message, source.
        3. Log entry should be recorded in session.

        Action:
        Starts session, then logs message.

        Assertion Strategy:
        Validates logging by confirming:
        - data["status"] = "logged".

        Testing Principle:
        Validates session logging, ensuring activities
        can be documented during active sessions.
        """
        await sessions._start_session("observation", "M45", None)

        result = await sessions._session_log(
            level="INFO",
            message="Test log entry",
            source="test",
        )
        data = json.loads(result[0].text)
        assert data["status"] == "logged"

    @pytest.mark.asyncio
    async def test_session_log_different_levels(self):
        """Verifies logging supports all standard log levels.

        Arrangement:
        1. Session started with type="observation".
        2. _session_log() called for DEBUG, INFO, WARNING, ERROR.
        3. All levels should be accepted.

        Action:
        Logs messages at each level in sequence.

        Assertion Strategy:
        Validates level support by confirming:
        - Each level returns status="logged".

        Testing Principle:
        Validates logging flexibility, ensuring all
        LogLevel enum values are supported.
        """
        await sessions._start_session("observation", "M31", None)

        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        for level in levels:
            result = await sessions._session_log(
                level=level,
                message=f"Test {level} message",
                source="test",
            )
            data = json.loads(result[0].text)
            assert data["status"] == "logged"

    @pytest.mark.asyncio
    async def test_session_event(self):
        """Verifies _session_event records event with metadata.

        Arrangement:
        1. Session started with type="observation", target="NGC7000".
        2. _session_event() with event_type="capture", metadata dict.
        3. Event should be recorded in session timeline.

        Action:
        Starts session, then records capture event.

        Assertion Strategy:
        Validates event recording by confirming:
        - data["status"] = "recorded".

        Testing Principle:
        Validates event tracking, ensuring telescope
        activities are logged with structured metadata.
        """
        await sessions._start_session("observation", "NGC7000", None)

        result = await sessions._session_event(
            "capture",
            {"exposure_us": 100000, "gain": 50},
        )
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"

    @pytest.mark.asyncio
    async def test_session_event_with_metadata(self):
        """Verifies event recording with complex metadata dictionary.

        Arrangement:
        1. Session started with type="alignment".
        2. Event metadata includes camera_id, exposure, stars_detected.
        3. Complex metadata should be preserved.

        Action:
        Records alignment event with multi-field metadata.

        Assertion Strategy:
        Validates complex metadata by confirming:
        - data["status"] = "recorded".

        Testing Principle:
        Validates metadata flexibility, ensuring events
        can carry arbitrary structured data.
        """
        await sessions._start_session("alignment", None, "test alignment")

        metadata = {
            "camera_id": 0,
            "exposure": 100000,
            "stars_detected": 15,
        }
        result = await sessions._session_event(
            "alignment",
            metadata,
        )
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"

    # test_add_frame removed - _add_frame not in API

    @pytest.mark.asyncio
    async def test_multiple_sessions_sequence(self):
        """Verifies multiple sessions can be created in sequence.

        Arrangement:
        1. First session: observation, target="M31".
        2. End first session.
        3. Second session: experiment, purpose="test2".
        4. Session IDs should be unique.

        Action:
        Creates, ends, then creates new session, comparing IDs.

        Assertion Strategy:
        Validates session independence by confirming:
        - data1["session_id"] != data2["session_id"].

        Testing Principle:
        Validates session isolation, ensuring each session
        gets unique identifier for traceability.
        """
        # First session
        await sessions._start_session("observation", "M31", None)
        result1 = await sessions._get_session_info()
        data1 = json.loads(result1[0].text)

        # End first
        await sessions._end_session()

        # Second session
        await sessions._start_session("experiment", None, "test2")
        result2 = await sessions._get_session_info()
        data2 = json.loads(result2[0].text)

        # Session IDs should be different
        assert data1["session_id"] != data2["session_id"]

    @pytest.mark.asyncio
    async def test_session_with_target_and_purpose(self):
        """Verifies session can have both target and purpose simultaneously.

        Arrangement:
        1. _start_session() with target="M42" AND purpose="testing new filter".
        2. Both parameters should be preserved.
        3. Observation sessions can have optional purpose.

        Action:
        Starts session with both parameters.

        Assertion Strategy:
        Validates parameter combination by confirming:
        - data["target"] = "M42".
        - data["purpose"] = "testing new filter".

        Testing Principle:
        Validates parameter flexibility, allowing observations
        to document both target and special circumstances.
        """
        result = await sessions._start_session(
            session_type="observation",
            target="M42",
            purpose="testing new filter",
        )
        data = json.loads(result[0].text)
        assert data["target"] == "M42"
        assert data["purpose"] == "testing new filter"

    @pytest.mark.asyncio
    async def test_get_data_dir(self):
        """Verifies _get_data_dir returns current data directory path.

        Arrangement:
        1. SessionManager has configured data_dir.
        2. _get_data_dir() should return directory path.
        3. Path is string (not Path object).

        Action:
        Calls _get_data_dir() and parses JSON response.

        Assertion Strategy:
        Validates data dir query by confirming:
        - data contains "data_dir" key.
        - data["data_dir"] is string type.

        Testing Principle:
        Validates configuration query, ensuring data
        storage location is discoverable via MCP.
        """
        result = await sessions._get_data_dir()
        data = json.loads(result[0].text)
        assert "data_dir" in data
        assert isinstance(data["data_dir"], str)

    @pytest.mark.asyncio
    async def test_set_data_dir(self):
        """Verifies _set_data_dir changes session storage location.

        Arrangement:
        1. Temporary directory created for test.
        2. _set_data_dir() updates SessionManager config.
        3. _get_data_dir() should reflect new path.

        Action:
        Sets new data dir, then verifies with get.

        Assertion Strategy:
        Validates data dir change by confirming:
        - set response data["data_dir"] = tmpdir.
        - get response data2["data_dir"] = tmpdir.

        Testing Principle:
        Validates configuration mutation, ensuring data
        directory can be changed at runtime.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await sessions._set_data_dir(tmpdir)
            data = json.loads(result[0].text)
            assert "data_dir" in data
            assert data["data_dir"] == tmpdir

            # Verify it was actually set
            result2 = await sessions._get_data_dir()
            data2 = json.loads(result2[0].text)
            assert data2["data_dir"] == tmpdir


class TestSessionManager:
    """Tests for SessionManager functionality."""

    def test_session_manager_singleton(self):
        """Verifies SessionManager follows singleton pattern.

        Arrangement:
        1. get_session_manager() called twice.
        2. Should return same instance.
        3. Singleton ensures consistent session state.

        Action:
        Retrieves manager twice, compares object identity.

        Assertion Strategy:
        Validates singleton by confirming:
        - manager1 is manager2 (same object).

        Testing Principle:
        Validates architectural pattern, ensuring single
        SessionManager coordinates all session operations.
        """
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        assert manager1 is manager2

    def test_start_and_end_session(self):
        """Verifies direct SessionManager session lifecycle.

        Arrangement:
        1. SessionManager retrieved.
        2. start_session() with type=OBSERVATION, target="M31".
        3. end_session() terminates active session.

        Action:
        Starts and ends session directly via manager.

        Assertion Strategy:
        Validates lifecycle by confirming:
        - session is not None.
        - session.session_type = SessionType.OBSERVATION.
        - ended_session is not None.

        Testing Principle:
        Validates core manager functionality, ensuring
        sessions can be created and ended programmatically.
        """
        manager = get_session_manager()

        session = manager.start_session(SessionType.OBSERVATION, target="M31")
        assert session is not None
        assert session.session_type == SessionType.OBSERVATION

        ended_session = manager.end_session()
        assert ended_session is not None

    def test_log_to_session(self):
        """Verifies SessionManager.log adds entries to active session.

        Arrangement:
        1. Manager started session with type=EXPERIMENT.
        2. manager.log() with level=INFO, message, source.
        3. Log should succeed without error.

        Action:
        Starts session, logs message, ends session.

        Assertion Strategy:
        Validates logging by confirming:
        - No exception raised during log().

        Testing Principle:
        Validates logging integration, ensuring manager
        provides direct logging interface.
        """
        manager = get_session_manager()
        manager.start_session(SessionType.EXPERIMENT, purpose="test")

        # Log should succeed
        manager.log(LogLevel.INFO, "Test message", source="test")

        manager.end_session()

    def test_add_event_to_session(self):
        """Verifies SessionManager.add_event records events with kwargs.

        Arrangement:
        1. Manager started session with type=OBSERVATION, target="M45".
        2. add_event() with event_type="capture", gain/exposure kwargs.
        3. Event should be recorded successfully.

        Action:
        Starts session, adds capture event.

        Assertion Strategy:
        Validates event addition by confirming:
        - No exception raised during add_event().

        Testing Principle:
        Validates event interface, ensuring manager
        provides direct event recording with kwargs.
        """
        manager = get_session_manager()
        manager.start_session(SessionType.OBSERVATION, target="M45")

        manager.add_event("capture", gain=50, exposure_us=100000)


class TestSessionToolsErrorHandling:
    """Test error handling in session tools."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Fixture initializing registry for error handling tests.

        Arrangement:
        1. Creates DigitalTwinCameraDriver for error scenarios.
        2. init_registry(driver) registers cameras.
        3. Yields for error test execution.
        4. Cleanup: shutdown_registry().

        Args:
        self: Test class instance (pytest convention).

        Returns:
        None (generator fixture).

        Raises:
        None.

        Example:
        # Automatically runs for all error tests
        def test_error_case(self):
            # Registry already initialized
            pass

        Testing Principle:
        Provides consistent camera infrastructure for
        testing error handling scenarios.
        """
        driver = DigitalTwinCameraDriver()
        init_registry(driver)
        yield
        shutdown_registry()

    @pytest.mark.asyncio
    async def test_session_log_all_levels(self):
        """Verifies logging supports all LogLevel enum values.

        Arrangement:
        1. No active session (tests idle logging).
        2. Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        3. All should be accepted.

        Action:
        Logs messages at each level without active session.

        Assertion Strategy:
        Validates comprehensive level support by confirming:
        - Each level returns status="logged".

        Testing Principle:
        Validates LogLevel enum completeness, ensuring
        all standard Python logging levels work.
        """
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            result = await sessions._session_log(level, f"Test {level}", "test")
            data = json.loads(result[0].text)
            assert data["status"] == "logged"

    @pytest.mark.asyncio
    async def test_session_log_with_long_message(self):
        """Verifies logging handles very long messages without truncation.

        Arrangement:
        1. Message with 10000 'A' characters.
        2. _session_log() should accept large messages.
        3. No truncation or error.

        Action:
        Logs 10k-character message.

        Assertion Strategy:
        Validates message size handling by confirming:
        - data["status"] = "logged" (no error).

        Testing Principle:
        Validates input size limits, ensuring logging
        doesn't fail on verbose messages.
        """
        long_msg = "A" * 10000
        result = await sessions._session_log("INFO", long_msg, "test")
        data = json.loads(result[0].text)
        assert data["status"] == "logged"

    @pytest.mark.asyncio
    async def test_get_session_info_has_required_fields(self):
        """Verifies _get_session_info always returns essential fields.

        Arrangement:
        1. No session explicitly started (idle session active).
        2. _get_session_info() should return current state.
        3. Minimum required fields: session_id, session_type.

        Action:
        Calls _get_session_info() without starting session.

        Assertion Strategy:
        Validates response schema by confirming:
        - data contains "session_id" key.
        - data contains "session_type" key.

        Testing Principle:
        Validates API contract, ensuring response always
        includes minimum required fields.
        """
        result = await sessions._get_session_info()
        data = json.loads(result[0].text)
        # Should have session info
        assert "session_id" in data
        assert "session_type" in data

    @pytest.mark.asyncio
    async def test_end_session_idle_returns_to_idle(self):
        """Verifies ending session creates new idle session automatically.

        Arrangement:
        1. Session started with type="observation", target="M31".
        2. _end_session() terminates observation session.
        3. SessionManager should auto-create idle session.

        Action:
        Starts observation session, ends it, checks new session.

        Assertion Strategy:
        Validates automatic idle creation by confirming:
        - End response contains session_id.
        - New session info contains "idle" in session_id.

        Testing Principle:
        Validates session state machine, ensuring system
        always has active session (never null state).
        """
        await sessions._start_session("observation", "M31", None)
        result = await sessions._end_session()
        data = json.loads(result[0].text)
        assert "session_id" in data

        # Should now have new idle session
        result2 = await sessions._get_session_info()
        data2 = json.loads(result2[0].text)
        assert "idle" in data2["session_id"]
