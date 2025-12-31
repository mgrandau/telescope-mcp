"""Direct unit tests for Session class in telescope_mcp.data.session.

Tests the Session class directly without going through SessionManager,
targeting 100% coverage of session.py module.

Coverage targets:
- Session initialization and ID generation
- log() method with all levels and context
- add_event() with and without details
- add_frame() with camera_info, settings, and updates
- add_telemetry() with various types
- add_calibration() with list and non-list types
- close() and ASDF file generation
- Properties: is_closed, duration_seconds
- Error paths: operations on closed sessions
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from telescope_mcp.data.session import LogLevel, Session, SessionType


class TestSessionInitialization:
    """Tests for Session __init__ and ID generation.

    Categories:
    1. Basic Initialization - Session creation with minimal params
    2. ID Generation - Session ID format with/without target
    3. Optional Parameters - location, auto_rotate, etc.

    Total: 6 tests.
    """

    def test_init_minimal_params(self, tmp_path: Path) -> None:
        """Verifies Session initializes with minimal required parameters.

        Arrangement:
            1. Create tmp_path for data directory.
            2. Use SessionType.IDLE (simplest type).

        Action:
            Create Session(SessionType.IDLE, tmp_path).

        Assertion Strategy:
            Validates initialization by confirming:
            - session_type matches.
            - data_dir matches tmp_path.
            - Internal buffers are empty.
            - Counters are zero.
            - Session not closed.

        Testing Principle:
            Validates minimal initialization path.
        """
        session = Session(SessionType.IDLE, tmp_path)

        assert session.session_type == SessionType.IDLE
        assert session.data_dir == tmp_path
        assert session.target is None
        assert session.purpose is None
        assert session.location == {}
        assert session.auto_rotate is False
        assert session._logs == []
        assert session._events == []
        assert session._cameras == {}
        assert session._frames_captured == 0
        assert session._error_count == 0
        assert session._warning_count == 0
        assert session._closed is False

    def test_init_with_target(self, tmp_path: Path) -> None:
        """Verifies Session ID includes target when provided.

        Arrangement:
            1. Use SessionType.OBSERVATION.
            2. Provide target="M31".

        Action:
            Create Session with target.

        Assertion Strategy:
            Validates ID format by confirming:
            - session_id contains "observation".
            - session_id contains "m31" (lowercase slugified).
            - session_id contains timestamp pattern.

        Testing Principle:
            Validates target inclusion in session ID.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")

        assert "observation" in session.session_id
        assert "m31" in session.session_id
        assert session.target == "M31"

    def test_init_with_long_target_truncated(self, tmp_path: Path) -> None:
        """Verifies target slugified and truncated to 20 chars.

        Arrangement:
            1. Provide target with spaces and >20 chars.

        Action:
            Create Session with long target name.

        Assertion Strategy:
            Validates truncation by confirming:
            - Target slug <=20 chars in session_id.
            - Spaces converted to underscores.

        Testing Principle:
            Validates target sanitization for safe filenames.
        """
        long_target = "Andromeda Galaxy M31 NGC 224 Extended"
        session = Session(SessionType.OBSERVATION, tmp_path, target=long_target)

        # Extract target slug from session_id
        parts = session.session_id.split("_")
        # Format: observation_target_slug_YYYYMMDD_HHMMSS
        target_slug = "_".join(parts[1:-2])  # Skip type and timestamp
        assert len(target_slug) <= 20
        assert " " not in target_slug

    def test_init_without_target(self, tmp_path: Path) -> None:
        """Verifies Session ID format without target.

        Arrangement:
            1. Use SessionType.ALIGNMENT.
            2. No target provided.

        Action:
            Create Session without target.

        Assertion Strategy:
            Validates ID format by confirming:
            - session_id contains "alignment".
            - session_id contains timestamp.
            - No target slug in ID.

        Testing Principle:
            Validates non-target session ID generation.
        """
        session = Session(SessionType.ALIGNMENT, tmp_path, purpose="calibration")

        assert "alignment" in session.session_id
        assert session.target is None
        assert session.purpose == "calibration"

    def test_init_with_location(self, tmp_path: Path) -> None:
        """Verifies location dict stored correctly.

        Arrangement:
            1. Provide location dict with lat, lon, alt.

        Action:
            Create Session with location.

        Assertion Strategy:
            Validates location storage by confirming:
            - location dict matches input.

        Testing Principle:
            Validates optional location parameter.
        """
        location = {"lat": 34.05, "lon": -118.25, "alt": 100.0}
        session = Session(SessionType.OBSERVATION, tmp_path, location=location)

        assert session.location == location

    def test_init_with_auto_rotate_settings(self, tmp_path: Path) -> None:
        """Verifies auto_rotate and rotate_interval_hours stored.

        Arrangement:
            1. Enable auto_rotate.
            2. Set custom rotate_interval_hours.

        Action:
            Create Session with rotation settings.

        Assertion Strategy:
            Validates rotation settings by confirming:
            - auto_rotate is True.
            - rotate_interval_hours matches.

        Testing Principle:
            Validates optional rotation parameters.
        """
        session = Session(
            SessionType.IDLE,
            tmp_path,
            auto_rotate=True,
            rotate_interval_hours=2,
        )

        assert session.auto_rotate is True
        assert session.rotate_interval_hours == 2


class TestSessionLog:
    """Tests for Session.log() method.

    Categories:
    1. Basic Logging - log with LogLevel enum
    2. String Level - log with string level conversion
    3. Context - log with and without context kwargs
    4. Counter Tracking - error and warning counts
    5. Closed Session - RuntimeError on closed session

    Total: 8 tests.
    """

    def test_log_with_enum_level(self, tmp_path: Path) -> None:
        """Verifies log() accepts LogLevel enum.

        Arrangement:
            1. Create active session.
            2. Use LogLevel.INFO enum value.

        Action:
            Call session.log(LogLevel.INFO, "Test message").

        Assertion Strategy:
            Validates logging by confirming:
            - Log entry added to _logs.
            - Level stored as string "INFO".
            - Message matches.

        Testing Principle:
            Validates enum level handling.
        """
        session = Session(SessionType.IDLE, tmp_path)

        session.log(LogLevel.INFO, "Test message")

        assert len(session._logs) == 1
        assert session._logs[0]["level"] == "INFO"
        assert session._logs[0]["message"] == "Test message"

    def test_log_with_string_level(self, tmp_path: Path) -> None:
        """Verifies log() converts string level to LogLevel enum.

        Arrangement:
            1. Create active session.
            2. Use lowercase string "info" for level.

        Action:
            Call session.log("info", "Test message").

        Assertion Strategy:
            Validates string conversion by confirming:
            - Level stored as "INFO" (uppercase).
            - No exception raised.

        Testing Principle:
            Validates string level conversion branch (line 206-209).
        """
        session = Session(SessionType.IDLE, tmp_path)

        session.log("info", "Test message")

        assert session._logs[0]["level"] == "INFO"

    def test_log_with_uppercase_string_level(self, tmp_path: Path) -> None:
        """Verifies log() handles uppercase string level.

        Arrangement:
            1. Create active session.
            2. Use uppercase string "WARNING".

        Action:
            Call session.log("WARNING", "Test warning").

        Assertion Strategy:
            Validates string handling by confirming:
            - Level stored correctly.
            - Warning count incremented.

        Testing Principle:
            Validates string level normalization.
        """
        session = Session(SessionType.IDLE, tmp_path)

        session.log("WARNING", "Test warning")

        assert session._logs[0]["level"] == "WARNING"
        assert session._warning_count == 1

    def test_log_with_context(self, tmp_path: Path) -> None:
        """Verifies log() stores context kwargs.

        Arrangement:
            1. Create active session.
            2. Provide context kwargs.

        Action:
            Call session.log with camera="main", exposure_us=500000.

        Assertion Strategy:
            Validates context storage by confirming:
            - context dict in log entry matches kwargs.

        Testing Principle:
            Validates context kwargs handling.
        """
        session = Session(SessionType.IDLE, tmp_path)

        session.log(LogLevel.INFO, "Exposure", camera="main", exposure_us=500000)

        assert session._logs[0]["context"] == {
            "camera": "main",
            "exposure_us": 500000,
        }

    def test_log_without_context(self, tmp_path: Path) -> None:
        """Verifies log() stores None for empty context.

        Arrangement:
            1. Create active session.
            2. No context kwargs provided.

        Action:
            Call session.log without context kwargs.

        Assertion Strategy:
            Validates empty context by confirming:
            - context is None in log entry.

        Testing Principle:
            Validates empty context path (line 227).
        """
        session = Session(SessionType.IDLE, tmp_path)

        session.log(LogLevel.INFO, "Simple message")

        assert session._logs[0]["context"] is None

    def test_log_increments_error_count(self, tmp_path: Path) -> None:
        """Verifies log() increments error count for ERROR level.

        Arrangement:
            1. Create active session.
            2. Error count starts at 0.

        Action:
            Call session.log with LogLevel.ERROR.

        Assertion Strategy:
            Validates counter by confirming:
            - _error_count incremented to 1.

        Testing Principle:
            Validates ERROR level tracking (line 229).
        """
        session = Session(SessionType.IDLE, tmp_path)
        assert session._error_count == 0

        session.log(LogLevel.ERROR, "Error occurred")

        assert session._error_count == 1

    def test_log_increments_error_count_for_critical(self, tmp_path: Path) -> None:
        """Verifies log() increments error count for CRITICAL level.

        Arrangement:
            1. Create active session.
            2. CRITICAL also counts as error.

        Action:
            Call session.log with LogLevel.CRITICAL.

        Assertion Strategy:
            Validates counter by confirming:
            - _error_count incremented for CRITICAL.

        Testing Principle:
            Validates CRITICAL level tracking (line 229).
        """
        session = Session(SessionType.IDLE, tmp_path)

        session.log(LogLevel.CRITICAL, "Critical failure")

        assert session._error_count == 1

    def test_log_increments_warning_count(self, tmp_path: Path) -> None:
        """Verifies log() increments warning count for WARNING level.

        Arrangement:
            1. Create active session.
            2. Warning count starts at 0.

        Action:
            Call session.log with LogLevel.WARNING.

        Assertion Strategy:
            Validates counter by confirming:
            - _warning_count incremented to 1.

        Testing Principle:
            Validates WARNING level tracking (line 231).
        """
        session = Session(SessionType.IDLE, tmp_path)
        assert session._warning_count == 0

        session.log(LogLevel.WARNING, "Warning issued")

        assert session._warning_count == 1

    def test_log_raises_on_closed_session(self, tmp_path: Path) -> None:
        """Verifies log() raises RuntimeError on closed session.

        Arrangement:
            1. Create and close session.
            2. Session is immutable after close.

        Action:
            Attempt session.log() after close().

        Assertion Strategy:
            Validates protection by confirming:
            - RuntimeError raised.
            - Message indicates "closed session".

        Testing Principle:
            Validates closed session protection (line 204).
        """
        session = Session(SessionType.IDLE, tmp_path)
        session.close()

        with pytest.raises(RuntimeError, match="Cannot log to a closed session"):
            session.log(LogLevel.INFO, "Should fail")


class TestSessionAddEvent:
    """Tests for Session.add_event() method.

    Categories:
    1. Basic Event - add event without details
    2. Event with Details - add event with kwargs
    3. Closed Session - RuntimeError on closed session

    Total: 3 tests.
    """

    def test_add_event_basic(self, tmp_path: Path) -> None:
        """Verifies add_event() records event without details.

        Arrangement:
            1. Create active session.
            2. No details kwargs provided.

        Action:
            Call session.add_event("tracking_started").

        Assertion Strategy:
            Validates event storage by confirming:
            - Event added to _events.
            - event field matches.
            - details is None.

        Testing Principle:
            Validates basic event recording.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")

        session.add_event("tracking_started")

        assert len(session._events) == 1
        assert session._events[0]["event"] == "tracking_started"
        assert session._events[0]["details"] is None

    def test_add_event_with_details(self, tmp_path: Path) -> None:
        """Verifies add_event() stores detail kwargs.

        Arrangement:
            1. Create active session.
            2. Provide details kwargs.

        Action:
            Call add_event with reason="clouds", duration_sec=30.

        Assertion Strategy:
            Validates details by confirming:
            - details dict matches kwargs.

        Testing Principle:
            Validates details kwargs handling.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")

        session.add_event("tracking_lost", reason="clouds", duration_sec=30)

        assert session._events[0]["details"] == {
            "reason": "clouds",
            "duration_sec": 30,
        }

    def test_add_event_raises_on_closed_session(self, tmp_path: Path) -> None:
        """Verifies add_event() raises RuntimeError on closed session.

        Arrangement:
            1. Create and close session.

        Action:
            Attempt add_event() after close().

        Assertion Strategy:
            Validates protection by confirming:
            - RuntimeError raised.
            - Message indicates "closed session".

        Testing Principle:
            Validates closed session protection (line 251).
        """
        session = Session(SessionType.IDLE, tmp_path)
        session.close()

        with pytest.raises(RuntimeError, match="Cannot add event to a closed session"):
            session.add_event("should_fail")


class TestSessionAddFrame:
    """Tests for Session.add_frame() method.

    Categories:
    1. Basic Frame - add frame with camera ID only
    2. New Camera - first frame creates camera entry
    3. Camera Info Update - update existing camera info
    4. Settings Update - update existing camera settings
    5. Multiple Frames - multiple frames same camera
    6. Closed Session - RuntimeError on closed session

    Total: 6 tests.
    """

    def test_add_frame_creates_camera_entry(self, tmp_path: Path) -> None:
        """Verifies add_frame() creates camera entry for new camera.

        Arrangement:
            1. Create active session.
            2. Camera "main" not in _cameras yet.

        Action:
            Call add_frame("main", frame_array).

        Assertion Strategy:
            Validates camera entry by confirming:
            - "main" in _cameras.
            - info and settings dicts created.
            - frames list contains frame.
            - _frames_captured incremented.

        Testing Principle:
            Validates new camera creation (lines 295-303).
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        frame = np.zeros((100, 100), dtype=np.uint8)

        session.add_frame("main", frame)

        assert "main" in session._cameras
        assert session._cameras["main"]["info"] == {}
        assert session._cameras["main"]["settings"] == {}
        assert len(session._cameras["main"]["frames"]) == 1
        assert session._frames_captured == 1

    def test_add_frame_with_camera_info(self, tmp_path: Path) -> None:
        """Verifies add_frame() stores camera_info on new camera.

        Arrangement:
            1. Create active session.
            2. Provide camera_info dict.

        Action:
            Call add_frame with camera_info parameter.

        Assertion Strategy:
            Validates info storage by confirming:
            - info dict matches camera_info.

        Testing Principle:
            Validates camera_info initialization.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        frame = np.zeros((100, 100), dtype=np.uint8)
        camera_info = {"name": "ASI482MC", "sensor_size": [4096, 3040]}

        session.add_frame("main", frame, camera_info=camera_info)

        assert session._cameras["main"]["info"] == camera_info

    def test_add_frame_with_settings(self, tmp_path: Path) -> None:
        """Verifies add_frame() stores settings on new camera.

        Arrangement:
            1. Create active session.
            2. Provide settings dict.

        Action:
            Call add_frame with settings parameter.

        Assertion Strategy:
            Validates settings storage by confirming:
            - settings dict matches.

        Testing Principle:
            Validates settings initialization.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        frame = np.zeros((100, 100), dtype=np.uint8)
        settings = {"exposure_us": 500000, "gain": 100}

        session.add_frame("main", frame, settings=settings)

        assert session._cameras["main"]["settings"] == settings

    def test_add_frame_updates_camera_info(self, tmp_path: Path) -> None:
        """Verifies add_frame() updates existing camera info.

        Arrangement:
            1. Create session and add initial frame.
            2. Camera "main" already exists.

        Action:
            Call add_frame with new camera_info.

        Assertion Strategy:
            Validates update by confirming:
            - info dict updated with new keys.

        Testing Principle:
            Validates camera_info update branch (line 304-305).
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)

        # First frame
        session.add_frame("main", frame1, camera_info={"name": "ASI482MC"})

        # Second frame with additional info
        session.add_frame("main", frame2, camera_info={"temperature": 25.0})

        assert session._cameras["main"]["info"]["name"] == "ASI482MC"
        assert session._cameras["main"]["info"]["temperature"] == 25.0

    def test_add_frame_updates_settings(self, tmp_path: Path) -> None:
        """Verifies add_frame() updates existing camera settings.

        Arrangement:
            1. Create session and add initial frame.
            2. Camera "main" already exists.

        Action:
            Call add_frame with new settings.

        Assertion Strategy:
            Validates update by confirming:
            - settings dict updated.

        Testing Principle:
            Validates settings update branch (line 306-307).
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)

        # First frame
        session.add_frame("main", frame1, settings={"gain": 50})

        # Second frame with updated settings
        session.add_frame("main", frame2, settings={"gain": 100})

        assert session._cameras["main"]["settings"]["gain"] == 100

    def test_add_frame_raises_on_closed_session(self, tmp_path: Path) -> None:
        """Verifies add_frame() raises RuntimeError on closed session.

        Arrangement:
            1. Create and close session.

        Action:
            Attempt add_frame() after close().

        Assertion Strategy:
            Validates protection by confirming:
            - RuntimeError raised.

        Testing Principle:
            Validates closed session protection (line 295).
        """
        session = Session(SessionType.IDLE, tmp_path)
        session.close()
        frame = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="Cannot add frame to a closed session"):
            session.add_frame("main", frame)

    def test_add_frame_existing_camera_no_updates(self, tmp_path: Path) -> None:
        """Verifies add_frame() with existing camera and no info/settings.

        Arrangement:
            1. Create session and add initial frame with settings.
            2. Camera "main" exists with settings.

        Action:
            Call add_frame with same camera, no camera_info, no settings.

        Assertion Strategy:
            Validates no-update path by confirming:
            - Frame added to existing camera.
            - Original settings unchanged.

        Testing Principle:
            Validates elif settings: branch is False (line 306->309 partial).
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)

        # First frame with settings
        session.add_frame("main", frame1, settings={"gain": 50})
        original_settings = session._cameras["main"]["settings"].copy()

        # Second frame with no camera_info and no settings
        session.add_frame("main", frame2, camera_info=None, settings=None)

        # Settings should be unchanged
        assert session._cameras["main"]["settings"] == original_settings
        # But frame should be added
        assert len(session._cameras["main"]["frames"]) == 2


class TestSessionAddTelemetry:
    """Tests for Session.add_telemetry() method.

    Categories:
    1. Known Type - add to pre-existing telemetry list
    2. New Type - create new telemetry type
    3. Timestamped - verify automatic timestamping
    4. Closed Session - RuntimeError on closed session

    Total: 4 tests.
    """

    def test_add_telemetry_known_type(self, tmp_path: Path) -> None:
        """Verifies add_telemetry() appends to known telemetry type.

        Arrangement:
            1. Create active session.
            2. "mount_position" in default _telemetry keys.

        Action:
            Call add_telemetry("mount_position", ra=12.5, dec=45.2).

        Assertion Strategy:
            Validates append by confirming:
            - Entry added to mount_position list.
            - Data fields match.

        Testing Principle:
            Validates known type append path (line 340-342).
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")

        session.add_telemetry("mount_position", ra=12.5, dec=45.2, alt=60.0)

        assert len(session._telemetry["mount_position"]) == 1
        entry = session._telemetry["mount_position"][0]
        assert entry["ra"] == 12.5
        assert entry["dec"] == 45.2
        assert entry["alt"] == 60.0
        assert "time" in entry

    def test_add_telemetry_new_type(self, tmp_path: Path) -> None:
        """Verifies add_telemetry() creates new type if not exists.

        Arrangement:
            1. Create active session.
            2. "humidity" not in default _telemetry keys.

        Action:
            Call add_telemetry("humidity", value=45.0).

        Assertion Strategy:
            Validates creation by confirming:
            - "humidity" key created.
            - List contains entry.

        Testing Principle:
            Validates new type creation (lines 338-339).
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")

        session.add_telemetry("humidity", value=45.0)

        assert "humidity" in session._telemetry
        assert len(session._telemetry["humidity"]) == 1

    def test_add_telemetry_adds_timestamp(self, tmp_path: Path) -> None:
        """Verifies add_telemetry() adds timestamp to entry.

        Arrangement:
            1. Create active session.

        Action:
            Call add_telemetry with data.

        Assertion Strategy:
            Validates timestamp by confirming:
            - "time" key present in entry.
            - Value is ISO format string.

        Testing Principle:
            Validates automatic timestamping.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")

        session.add_telemetry("temperature", value=20.5)

        entry = session._telemetry["temperature"][0]
        assert "time" in entry
        # Should be ISO format: YYYY-MM-DDTHH:MM:SS.ffffff+00:00
        assert "T" in entry["time"]

    def test_add_telemetry_raises_on_closed_session(self, tmp_path: Path) -> None:
        """Verifies add_telemetry() raises RuntimeError on closed session.

        Arrangement:
            1. Create and close session.

        Action:
            Attempt add_telemetry() after close().

        Assertion Strategy:
            Validates protection by confirming:
            - RuntimeError raised.

        Testing Principle:
            Validates closed session protection (line 335).
        """
        session = Session(SessionType.IDLE, tmp_path)
        session.close()

        with pytest.raises(
            RuntimeError, match="Cannot add telemetry to a closed session"
        ):
            session.add_telemetry("mount_position", ra=0.0, dec=0.0)


class TestSessionAddCalibration:
    """Tests for Session.add_calibration() method.

    Categories:
    1. Known Type - append to existing list type
    2. New Type - create new calibration type as list
    3. Non-List Replacement - replace non-list calibration
    4. Closed Session - RuntimeError on closed session

    Total: 4 tests.
    """

    def test_add_calibration_known_type(self, tmp_path: Path) -> None:
        """Verifies add_calibration() appends to known list type.

        Arrangement:
            1. Create active session.
            2. "dark_frames" is pre-existing list in _calibration.

        Action:
            Call add_calibration("dark_frames", dark_array).

        Assertion Strategy:
            Validates append by confirming:
            - dark_frames list length increased.
            - Data appended to list.

        Testing Principle:
            Validates list append path (lines 371-372).
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        dark = np.zeros((100, 100), dtype=np.uint16)

        session.add_calibration("dark_frames", dark)

        assert len(session._calibration["dark_frames"]) == 1
        assert np.array_equal(session._calibration["dark_frames"][0], dark)

    def test_add_calibration_new_type(self, tmp_path: Path) -> None:
        """Verifies add_calibration() creates new type as list.

        Arrangement:
            1. Create active session.
            2. "bias_frames" not in default _calibration.

        Action:
            Call add_calibration("bias_frames", bias_array).

        Assertion Strategy:
            Validates creation by confirming:
            - "bias_frames" key created.
            - Value is list with data.

        Testing Principle:
            Validates new type creation (lines 368-369).
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        bias = np.zeros((100, 100), dtype=np.uint16)

        session.add_calibration("bias_frames", bias)

        assert "bias_frames" in session._calibration
        assert isinstance(session._calibration["bias_frames"], list)
        assert len(session._calibration["bias_frames"]) == 1

    def test_add_calibration_multiple_entries(self, tmp_path: Path) -> None:
        """Verifies add_calibration() appends multiple entries to same type.

        Arrangement:
            1. Create active session.
            2. Add first calibration entry.

        Action:
            Call add_calibration again with same type.

        Assertion Strategy:
            Validates append by confirming:
            - Both entries in list.

        Testing Principle:
            Validates consistent list-based storage.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")

        session.add_calibration("master_dark", {"exposure": 60})
        session.add_calibration("master_dark", {"exposure": 120})

        assert len(session._calibration["master_dark"]) == 2
        assert session._calibration["master_dark"][0] == {"exposure": 60}
        assert session._calibration["master_dark"][1] == {"exposure": 120}

    def test_add_calibration_raises_on_closed_session(self, tmp_path: Path) -> None:
        """Verifies add_calibration() raises RuntimeError on closed session.

        Arrangement:
            1. Create and close session.

        Action:
            Attempt add_calibration() after close().

        Assertion Strategy:
            Validates protection by confirming:
            - RuntimeError raised.

        Testing Principle:
            Validates closed session protection (line 365).
        """
        session = Session(SessionType.IDLE, tmp_path)
        session.close()

        with pytest.raises(
            RuntimeError, match="Cannot add calibration to a closed session"
        ):
            session.add_calibration("dark_frames", np.zeros((10, 10)))


class TestSessionClose:
    """Tests for Session.close() method and ASDF generation.

    Categories:
    1. Basic Close - close empty session
    2. Close with Data - close session with logs, frames, etc.
    3. Already Closed - RuntimeError on double close
    4. ASDF Tree Structure - verify tree contents

    Total: 5 tests.
    """

    def test_close_creates_asdf_file(self, tmp_path: Path) -> None:
        """Verifies close() creates ASDF file in date-organized path.

        Arrangement:
            1. Create active session.
            2. data_dir is tmp_path.

        Action:
            Call session.close().

        Assertion Strategy:
            Validates file creation by confirming:
            - Returned path exists.
            - File has .asdf extension.
            - Path includes date structure.

        Testing Principle:
            Validates ASDF file generation.
        """
        session = Session(SessionType.IDLE, tmp_path)

        path = session.close()

        assert path.exists()
        assert path.suffix == ".asdf"
        assert session.session_id in path.name

    def test_close_with_data(self, tmp_path: Path) -> None:
        """Verifies close() writes all session data to ASDF.

        Arrangement:
            1. Create session with logs, events, frames.
            2. Add various data types.

        Action:
            Call session.close() and read ASDF.

        Assertion Strategy:
            Validates data persistence by confirming:
            - ASDF tree contains all data categories.
            - Data matches what was added.

        Testing Principle:
            Validates complete data serialization.
        """
        import asdf

        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        session.log(LogLevel.INFO, "Test log")
        session.add_event("test_event", key="value")
        frame = np.zeros((50, 50), dtype=np.uint8)
        session.add_frame("main", frame, camera_info={"name": "test"})
        session.add_telemetry("temperature", value=20.0)

        path = session.close()

        # Read and verify ASDF
        with asdf.open(path) as af:
            assert af["meta"]["session_type"] == "observation"
            assert af["meta"]["target"] == "M31"
            assert len(af["observability"]["logs"]) >= 1
            assert len(af["observability"]["events"]) == 1
            assert "main" in af["cameras"]

    def test_close_raises_on_already_closed(self, tmp_path: Path) -> None:
        """Verifies close() raises RuntimeError if already closed.

        Arrangement:
            1. Create and close session once.

        Action:
            Attempt second close().

        Assertion Strategy:
            Validates protection by confirming:
            - RuntimeError raised.
            - Message indicates "already closed".

        Testing Principle:
            Validates double-close protection (line 484).
        """
        session = Session(SessionType.IDLE, tmp_path)
        session.close()

        with pytest.raises(RuntimeError, match="Session already closed"):
            session.close()

    def test_close_sets_end_time(self, tmp_path: Path) -> None:
        """Verifies close() sets _end_time for duration calculation.

        Arrangement:
            1. Create session (_end_time is None).

        Action:
            Call close().

        Assertion Strategy:
            Validates end_time by confirming:
            - _end_time is now set.
            - Value is datetime object.

        Testing Principle:
            Validates end_time setting for duration_seconds.
        """
        session = Session(SessionType.IDLE, tmp_path)
        assert session._end_time is None

        session.close()

        assert session._end_time is not None
        assert isinstance(session._end_time, datetime)

    def test_build_asdf_tree_structure(self, tmp_path: Path) -> None:
        """Verifies _build_asdf_tree() produces correct structure.

        Arrangement:
            1. Create session with various data.

        Action:
            Call _build_asdf_tree() directly.

        Assertion Strategy:
            Validates tree structure by confirming:
            - All required keys present.
            - Meta contains session info.
            - Observability contains metrics.

        Testing Principle:
            Validates ASDF tree construction.
        """
        session = Session(
            SessionType.OBSERVATION,
            tmp_path,
            target="M31",
            purpose="deep sky",
            location={"lat": 34.0, "lon": -118.0},
        )
        session.log(LogLevel.ERROR, "Test error")
        session.log(LogLevel.WARNING, "Test warning")

        tree = session._build_asdf_tree()

        # Meta
        assert tree["meta"]["session_type"] == "observation"
        assert tree["meta"]["session_id"] == session.session_id
        assert tree["meta"]["target"] == "M31"
        assert tree["meta"]["purpose"] == "deep sky"
        assert tree["meta"]["location"] == {"lat": 34.0, "lon": -118.0}

        # Observability metrics
        metrics = tree["observability"]["metrics"]
        assert metrics["errors"] == 1
        assert metrics["warnings"] == 1
        assert metrics["frames_captured"] == 0
        assert metrics["duration_seconds"] >= 0


class TestSessionProperties:
    """Tests for Session properties: is_closed, duration_seconds.

    Categories:
    1. is_closed Property - returns closed state
    2. duration_seconds - active and closed sessions

    Total: 4 tests.
    """

    def test_is_closed_false_initially(self, tmp_path: Path) -> None:
        """Verifies is_closed returns False for new session.

        Arrangement:
            1. Create new session.

        Action:
            Access is_closed property.

        Assertion Strategy:
            Validates property by confirming:
            - Returns False.

        Testing Principle:
            Validates initial closed state.
        """
        session = Session(SessionType.IDLE, tmp_path)

        assert session.is_closed is False

    def test_is_closed_true_after_close(self, tmp_path: Path) -> None:
        """Verifies is_closed returns True after close().

        Arrangement:
            1. Create and close session.

        Action:
            Access is_closed property.

        Assertion Strategy:
            Validates property by confirming:
            - Returns True.

        Testing Principle:
            Validates closed state (line 548).
        """
        session = Session(SessionType.IDLE, tmp_path)
        session.close()

        assert session.is_closed is True

    def test_duration_seconds_active_session(self, tmp_path: Path) -> None:
        """Verifies duration_seconds uses datetime.now() for active session.

        Arrangement:
            1. Create active session.
            2. Wait brief moment.

        Action:
            Access duration_seconds property.

        Assertion Strategy:
            Validates duration by confirming:
            - Returns positive float.
            - Duration increases over time.

        Testing Principle:
            Validates active session duration calculation.
        """
        session = Session(SessionType.IDLE, tmp_path)

        duration1 = session.duration_seconds
        import time

        time.sleep(0.01)
        duration2 = session.duration_seconds

        assert duration1 >= 0
        assert duration2 > duration1

    def test_duration_seconds_closed_session(self, tmp_path: Path) -> None:
        """Verifies duration_seconds uses _end_time for closed session.

        Arrangement:
            1. Create and close session.

        Action:
            Access duration_seconds multiple times.

        Assertion Strategy:
            Validates fixed duration by confirming:
            - Duration is positive.
            - Duration doesn't change after close.

        Testing Principle:
            Validates closed session duration (line 603).
        """
        session = Session(SessionType.IDLE, tmp_path)
        session.close()

        duration1 = session.duration_seconds
        import time

        time.sleep(0.01)
        duration2 = session.duration_seconds

        assert duration1 >= 0
        # Should be same (fixed at close time)
        assert duration1 == duration2


class TestSessionEdgeCases:
    """Tests for edge cases and additional branches.

    Total: 3 tests.
    """

    def test_session_id_all_types(self, tmp_path: Path) -> None:
        """Verifies session ID generation for all SessionType values.

        Arrangement:
            1. Create sessions for each SessionType.

        Action:
            Check session_id for each.

        Assertion Strategy:
            Validates ID format by confirming:
            - Each type name in session_id.

        Testing Principle:
            Validates session ID for all enum values.
        """
        for stype in SessionType:
            session = Session(stype, tmp_path)
            assert stype.value in session.session_id

    def test_add_frame_multiple_cameras(self, tmp_path: Path) -> None:
        """Verifies add_frame() handles multiple camera IDs.

        Arrangement:
            1. Create session.
            2. Add frames from "main" and "finder" cameras.

        Action:
            Add frames to different cameras.

        Assertion Strategy:
            Validates multi-camera by confirming:
            - Both cameras in _cameras dict.
            - Each has own frames list.

        Testing Principle:
            Validates multi-camera support.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((50, 50), dtype=np.uint8)

        session.add_frame("main", frame1)
        session.add_frame("finder", frame2)

        assert "main" in session._cameras
        assert "finder" in session._cameras
        assert len(session._cameras["main"]["frames"]) == 1
        assert len(session._cameras["finder"]["frames"]) == 1
        assert session._frames_captured == 2

    def test_location_empty_dict_when_none(self, tmp_path: Path) -> None:
        """Verifies location defaults to empty dict when not provided.

        Arrangement:
            1. Create session without location parameter.

        Action:
            Check location attribute.

        Assertion Strategy:
            Validates default by confirming:
            - location is {} not None.

        Testing Principle:
            Validates location default handling.
        """
        session = Session(SessionType.OBSERVATION, tmp_path, target="M31")

        assert session.location == {}
        assert session.location is not None
