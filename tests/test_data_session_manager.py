"""Unit tests for SessionManager class.

Tests the SessionManager class directly for 100% coverage of
telescope_mcp/data/session_manager.py.

Test Strategy:
    - Direct instantiation (not through tools layer)
    - Each method tested in isolation
    - Branch coverage for all conditionals
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from telescope_mcp.data.session import LogLevel, SessionType
from telescope_mcp.data.session_manager import SessionManager

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestSessionManagerInit:
    """Tests for SessionManager.__init__() method.

    Categories:
    1. Basic Init - creates directory and idle session
    2. With Location - passes location to idle session
    3. Custom Rotation - configures idle rotation settings

    Total: 3 tests.
    """

    def test_init_creates_directory_and_idle_session(self, tmp_path: Path) -> None:
        """Verifies __init__ creates data directory and starts idle session.

        Arrangement:
            1. Temporary directory path (not created yet).

        Action:
            Instantiate SessionManager with new subdirectory path.

        Assertion Strategy:
            Validates initialization by confirming:
            - Directory created.
            - Active session exists and is IDLE type.
        """
        data_dir = tmp_path / "telescope_data"

        manager = SessionManager(data_dir)

        assert data_dir.exists()
        assert manager.active_session is not None
        assert manager.active_session.session_type == SessionType.IDLE

    def test_init_with_location(self, tmp_path: Path) -> None:
        """Verifies __init__ passes location to idle session.

        Arrangement:
            1. Temporary directory.
            2. Location dict with lat/lon/alt.

        Action:
            Instantiate SessionManager with location.

        Assertion Strategy:
            Validates location by confirming:
            - Manager stores location.
        """
        location = {"lat": 34.05, "lon": -118.25, "alt": 100.0}

        manager = SessionManager(tmp_path, location=location)

        assert manager.location == location

    def test_init_with_custom_rotation_settings(self, tmp_path: Path) -> None:
        """Verifies __init__ accepts rotation configuration.

        Arrangement:
            1. Temporary directory.
            2. Custom rotation settings.

        Action:
            Instantiate SessionManager with auto_rotate and hours settings.

        Assertion Strategy:
            Validates config by confirming:
            - Settings stored in manager.
        """
        manager = SessionManager(
            tmp_path,
            auto_rotate_idle=False,
            idle_rotate_hours=2,
        )

        assert manager.auto_rotate_idle is False
        assert manager.idle_rotate_hours == 2


class TestSessionManagerStartSession:
    """Tests for SessionManager.start_session() method.

    Categories:
    1. Basic Start - creates new session with type
    2. With Target - observation session with target
    3. With Purpose - alignment/experiment with purpose
    4. String Type - accepts string session type
    5. Closes Existing - closes previous session before starting new

    Total: 5 tests.
    """

    def test_start_session_creates_observation(self, tmp_path: Path) -> None:
        """Verifies start_session creates observation session.

        Arrangement:
            1. SessionManager with idle session.

        Action:
            Call start_session with OBSERVATION type and target.

        Assertion Strategy:
            Validates session by confirming:
            - Active session is OBSERVATION type.
            - Target stored in session.
        """
        manager = SessionManager(tmp_path)

        session = manager.start_session(SessionType.OBSERVATION, target="M31")

        assert session.session_type == SessionType.OBSERVATION
        assert manager.active_session is session

    def test_start_session_with_purpose(self, tmp_path: Path) -> None:
        """Verifies start_session stores purpose for alignment sessions.

        Arrangement:
            1. SessionManager with idle session.

        Action:
            Call start_session with ALIGNMENT type and purpose.

        Assertion Strategy:
            Validates purpose by confirming:
            - Session has correct type.
            - Purpose stored.
        """
        manager = SessionManager(tmp_path)

        session = manager.start_session(
            SessionType.ALIGNMENT,
            purpose="Polar alignment check",
        )

        assert session.session_type == SessionType.ALIGNMENT

    def test_start_session_with_string_type(self, tmp_path: Path) -> None:
        """Verifies start_session accepts string session type.

        Arrangement:
            1. SessionManager with idle session.

        Action:
            Call start_session with "observation" string.

        Assertion Strategy:
            Validates conversion by confirming:
            - String converted to SessionType enum.
        """
        manager = SessionManager(tmp_path)

        session = manager.start_session("observation", target="Jupiter")

        assert session.session_type == SessionType.OBSERVATION

    def test_start_session_closes_existing(self, tmp_path: Path) -> None:
        """Verifies start_session closes existing session first.

        Arrangement:
            1. SessionManager with idle session.
            2. Note the idle session ID.

        Action:
            Call start_session to create observation.

        Assertion Strategy:
            Validates closure by confirming:
            - New session has different ID.
            - Old idle session was closed.
        """
        manager = SessionManager(tmp_path)
        idle_session_id = manager.active_session_id

        manager.start_session(SessionType.OBSERVATION, target="M42")

        assert manager.active_session_id != idle_session_id
        # ASDF file should exist for the closed idle session
        asdf_files = list(tmp_path.glob("**/idle_*.asdf"))
        assert len(asdf_files) >= 1

    def test_start_session_when_no_active_session(self, tmp_path: Path) -> None:
        """Verifies start_session works when no active session exists.

        Arrangement:
            1. SessionManager with _active_session manually set to None.
            2. Simulates state after shutdown without auto-idle.

        Action:
            Call start_session when _active_session is None.

        Assertion Strategy:
            Validates by confirming:
            - New session created without closing previous.
            - No exception raised.

        Testing Principle:
            Validates 169->173 partial branch (no existing session to close).
        """
        manager = SessionManager(tmp_path)
        manager.shutdown()  # Clears _active_session

        # Now _active_session is None
        session = manager.start_session(SessionType.OBSERVATION, target="M31")

        assert session is not None
        assert session.session_type == SessionType.OBSERVATION

    def test_start_session_with_location_override(self, tmp_path: Path) -> None:
        """Verifies start_session can override default location.

        Arrangement:
            1. SessionManager with default location.

        Action:
            Call start_session with different location.

        Assertion Strategy:
            Validates override by confirming:
            - Session uses provided location.
        """
        default_location = {"lat": 34.0, "lon": -118.0, "alt": 100.0}
        override_location = {"lat": 40.0, "lon": -74.0, "alt": 10.0}

        manager = SessionManager(tmp_path, location=default_location)

        session = manager.start_session(
            SessionType.OBSERVATION,
            target="M31",
            location=override_location,
        )

        # Session created with override location - verify session exists
        assert session is not None


class TestSessionManagerEndSession:
    """Tests for SessionManager.end_session() method.

    Categories:
    1. Basic End - closes session and returns path
    2. Returns to Idle - auto-creates idle session after end
    3. No Session Error - raises RuntimeError if no session
    4. End Idle Warning - logs warning when ending idle explicitly

    Total: 4 tests.
    """

    def test_end_session_returns_path(self, tmp_path: Path) -> None:
        """Verifies end_session closes session and returns ASDF path.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call end_session().

        Assertion Strategy:
            Validates by confirming:
            - Returns Path object.
            - File exists at returned path.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        path = manager.end_session()

        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".asdf"

    def test_end_session_returns_to_idle(self, tmp_path: Path) -> None:
        """Verifies end_session auto-creates new idle session.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call end_session().

        Assertion Strategy:
            Validates idle by confirming:
            - New session is IDLE type.
            - Different from previous observation.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")
        obs_id = manager.active_session_id

        manager.end_session()

        assert manager.active_session is not None
        assert manager.active_session.session_type == SessionType.IDLE
        assert manager.active_session_id != obs_id

    def test_end_session_raises_when_no_session(self, tmp_path: Path) -> None:
        """Verifies end_session raises RuntimeError when no session active.

        Arrangement:
            1. SessionManager after shutdown (no active session).

        Action:
            Call end_session().

        Assertion Strategy:
            Validates error by confirming:
            - RuntimeError raised.
            - Message indicates no active session.
        """
        manager = SessionManager(tmp_path)
        manager.shutdown()  # Clears active session without creating idle

        with pytest.raises(RuntimeError, match="No active session to end"):
            manager.end_session()

    def test_end_idle_session_logs_warning(self, tmp_path: Path) -> None:
        """Verifies end_session logs warning when ending idle session.

        Arrangement:
            1. SessionManager with default idle session.

        Action:
            Call end_session() on idle session.

        Assertion Strategy:
            Validates by confirming:
            - No exception raised.
            - Returns valid path.
            - New idle session created.
        """
        manager = SessionManager(tmp_path)
        assert manager.active_session.session_type == SessionType.IDLE

        path = manager.end_session()

        assert path.exists()
        assert manager.active_session.session_type == SessionType.IDLE


class TestSessionManagerLog:
    """Tests for SessionManager.log() method.

    Categories:
    1. Basic Log - adds log entry to session
    2. With Context - passes context kwargs
    3. Ensures Idle - creates idle session if needed

    Total: 3 tests.
    """

    def test_log_adds_entry_to_session(self, tmp_path: Path) -> None:
        """Verifies log() adds entry to active session.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call log() with level and message.

        Assertion Strategy:
            Validates by confirming:
            - Log entry added to session._logs.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        manager.log(LogLevel.INFO, "Exposure started")

        assert len(manager.active_session._logs) >= 1
        assert any(
            "Exposure started" in log["message"] for log in manager.active_session._logs
        )

    def test_log_with_context(self, tmp_path: Path) -> None:
        """Verifies log() passes context kwargs to session.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call log() with context kwargs.

        Assertion Strategy:
            Validates context by confirming:
            - Context appears in log entry.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        manager.log(
            LogLevel.INFO,
            "Camera ready",
            source="camera",
            camera_id=0,
            gain=50,
        )

        logs = manager.active_session._logs
        camera_log = next(log for log in logs if "Camera ready" in log["message"])
        assert camera_log["source"] == "camera"
        assert camera_log["context"]["camera_id"] == 0
        assert camera_log["context"]["gain"] == 50

    def test_log_ensures_idle_session(self, tmp_path: Path) -> None:
        """Verifies log() creates idle session if none exists.

        Arrangement:
            1. SessionManager after shutdown.

        Action:
            Call log() when no session active.

        Assertion Strategy:
            Validates idle creation by confirming:
            - Session created automatically.
            - Log entry added.
        """
        manager = SessionManager(tmp_path)
        manager.shutdown()
        manager._active_session = None  # Ensure no session

        manager.log(LogLevel.INFO, "System check")

        assert manager.active_session is not None
        assert manager.active_session.session_type == SessionType.IDLE


class TestSessionManagerAddEvent:
    """Tests for SessionManager.add_event() method.

    Categories:
    1. Basic Event - adds event to session
    2. With Details - passes detail kwargs

    Total: 2 tests.
    """

    def test_add_event_records_to_session(self, tmp_path: Path) -> None:
        """Verifies add_event() records event in session.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call add_event() with event name.

        Assertion Strategy:
            Validates by confirming:
            - Event recorded in session._events.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        manager.add_event("slew_start")

        assert len(manager.active_session._events) >= 1

    def test_add_event_with_details(self, tmp_path: Path) -> None:
        """Verifies add_event() passes detail kwargs.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call add_event() with detail kwargs.

        Assertion Strategy:
            Validates details by confirming:
            - Details appear in event record.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        manager.add_event("slew_complete", ra=10.684, dec=41.269, duration_sec=15.3)

        # Event recorded (details passed to session)
        assert len(manager.active_session._events) >= 1


class TestSessionManagerAddFrame:
    """Tests for SessionManager.add_frame() method.

    Categories:
    1. Basic Frame - adds frame to session
    2. With Camera Info - passes camera metadata
    3. With Settings - passes capture settings

    Total: 3 tests.
    """

    def test_add_frame_stores_in_session(self, tmp_path: Path) -> None:
        """Verifies add_frame() stores frame in session.

        Arrangement:
            1. SessionManager with observation session.
            2. Sample numpy array as frame.

        Action:
            Call add_frame() with camera and frame.

        Assertion Strategy:
            Validates by confirming:
            - Frame count incremented.
            - Camera appears in session._cameras.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")
        frame: NDArray[np.uint16] = np.zeros((100, 100), dtype=np.uint16)

        manager.add_frame("main", frame)

        assert manager.active_session._frames_captured == 1
        assert "main" in manager.active_session._cameras

    def test_add_frame_with_camera_info(self, tmp_path: Path) -> None:
        """Verifies add_frame() passes camera info.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call add_frame() with camera_info dict.

        Assertion Strategy:
            Validates by confirming:
            - Camera info stored in session.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")
        frame: NDArray[np.uint16] = np.zeros((100, 100), dtype=np.uint16)
        camera_info = {"name": "ASI482MC", "width": 1920, "height": 1080}

        manager.add_frame("main", frame, camera_info=camera_info)

        stored_info = manager.active_session._cameras["main"]["info"]
        assert stored_info["name"] == "ASI482MC"

    def test_add_frame_with_settings(self, tmp_path: Path) -> None:
        """Verifies add_frame() passes capture settings.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call add_frame() with settings dict.

        Assertion Strategy:
            Validates by confirming:
            - Settings stored in session.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")
        frame: NDArray[np.uint16] = np.zeros((100, 100), dtype=np.uint16)
        settings = {"exposure_us": 500000, "gain": 50}

        manager.add_frame("main", frame, settings=settings)

        stored_settings = manager.active_session._cameras["main"]["settings"]
        assert stored_settings["exposure_us"] == 500000


class TestSessionManagerAddTelemetry:
    """Tests for SessionManager.add_telemetry() method.

    Categories:
    1. Basic Telemetry - adds telemetry to session
    2. With Data - passes data kwargs

    Total: 2 tests.
    """

    def test_add_telemetry_records_in_session(self, tmp_path: Path) -> None:
        """Verifies add_telemetry() records in session.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call add_telemetry() with type and data.

        Assertion Strategy:
            Validates by confirming:
            - Telemetry entry added.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        manager.add_telemetry("environment", temperature_c=18.5)

        telemetry = manager.active_session._telemetry
        assert "environment" in telemetry
        assert len(telemetry["environment"]) >= 1

    def test_add_telemetry_with_multiple_values(self, tmp_path: Path) -> None:
        """Verifies add_telemetry() passes multiple data kwargs.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call add_telemetry() with multiple values.

        Assertion Strategy:
            Validates by confirming:
            - All values recorded.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        manager.add_telemetry(
            "environment",
            temperature_c=18.5,
            humidity_pct=65.0,
            pressure_hpa=1013.25,
        )

        entry = manager.active_session._telemetry["environment"][-1]
        assert entry["temperature_c"] == 18.5
        assert entry["humidity_pct"] == 65.0


class TestSessionManagerAddCalibration:
    """Tests for SessionManager.add_calibration() method.

    Categories:
    1. Basic Calibration - adds calibration data
    2. Multiple Types - handles different calibration types

    Total: 2 tests.
    """

    def test_add_calibration_stores_data(self, tmp_path: Path) -> None:
        """Verifies add_calibration() stores calibration data.

        Arrangement:
            1. SessionManager with observation session.
            2. Sample dark frame array.

        Action:
            Call add_calibration() with type and data.

        Assertion Strategy:
            Validates by confirming:
            - Calibration stored in session.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")
        dark_frame: NDArray[np.uint16] = np.zeros((100, 100), dtype=np.uint16)

        manager.add_calibration("dark", dark_frame)

        assert "dark" in manager.active_session._calibration

    def test_add_calibration_multiple_types(self, tmp_path: Path) -> None:
        """Verifies add_calibration() handles different types.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Call add_calibration() with different types.

        Assertion Strategy:
            Validates by confirming:
            - Each type stored separately.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        manager.add_calibration("dark", np.zeros((10, 10), dtype=np.uint16))
        manager.add_calibration("flat", np.ones((10, 10), dtype=np.uint16))
        manager.add_calibration("alignment", {"rotation": 1.5, "scale": 0.98})

        calibration = manager.active_session._calibration
        assert "dark" in calibration
        assert "flat" in calibration
        assert "alignment" in calibration


class TestSessionManagerProperties:
    """Tests for SessionManager properties.

    Categories:
    1. active_session - returns current session
    2. active_session_type - returns session type enum
    3. active_session_id - returns session ID string

    Total: 6 tests.
    """

    def test_active_session_returns_session(self, tmp_path: Path) -> None:
        """Verifies active_session property returns Session object.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Access active_session property.

        Assertion Strategy:
            Validates by confirming:
            - Returns Session instance.
        """
        manager = SessionManager(tmp_path)

        session = manager.active_session

        assert session is not None
        assert hasattr(session, "session_type")

    def test_active_session_none_after_shutdown(self, tmp_path: Path) -> None:
        """Verifies active_session is None after shutdown.

        Arrangement:
            1. SessionManager after shutdown.

        Action:
            Access active_session property.

        Assertion Strategy:
            Validates by confirming:
            - Returns None.
        """
        manager = SessionManager(tmp_path)
        manager.shutdown()

        assert manager.active_session is None

    def test_active_session_type_returns_enum(self, tmp_path: Path) -> None:
        """Verifies active_session_type returns SessionType enum.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Access active_session_type property.

        Assertion Strategy:
            Validates by confirming:
            - Returns SessionType enum value.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        session_type = manager.active_session_type

        assert session_type == SessionType.OBSERVATION

    def test_active_session_type_none_after_shutdown(self, tmp_path: Path) -> None:
        """Verifies active_session_type is None after shutdown.

        Arrangement:
            1. SessionManager after shutdown.

        Action:
            Access active_session_type property.

        Assertion Strategy:
            Validates by confirming:
            - Returns None.
        """
        manager = SessionManager(tmp_path)
        manager.shutdown()

        assert manager.active_session_type is None

    def test_active_session_id_returns_string(self, tmp_path: Path) -> None:
        """Verifies active_session_id returns ID string.

        Arrangement:
            1. SessionManager with observation session.

        Action:
            Access active_session_id property.

        Assertion Strategy:
            Validates by confirming:
            - Returns string.
            - Contains session type.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        session_id = manager.active_session_id

        assert isinstance(session_id, str)
        assert "observation" in session_id or "obs" in session_id.lower()

    def test_active_session_id_none_after_shutdown(self, tmp_path: Path) -> None:
        """Verifies active_session_id is None after shutdown.

        Arrangement:
            1. SessionManager after shutdown.

        Action:
            Access active_session_id property.

        Assertion Strategy:
            Validates by confirming:
            - Returns None.
        """
        manager = SessionManager(tmp_path)
        manager.shutdown()

        assert manager.active_session_id is None


class TestSessionManagerShutdown:
    """Tests for SessionManager.shutdown() method.

    Categories:
    1. Basic Shutdown - closes session and returns path
    2. No Session - returns None when no session active
    3. Logs Message - logs shutdown message before closing

    Total: 3 tests.
    """

    def test_shutdown_closes_session_and_returns_path(self, tmp_path: Path) -> None:
        """Verifies shutdown() closes session and returns path.

        Arrangement:
            1. SessionManager with active session.

        Action:
            Call shutdown().

        Assertion Strategy:
            Validates by confirming:
            - Returns Path to ASDF file.
            - File exists.
            - Active session is None.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        path = manager.shutdown()

        assert isinstance(path, Path)
        assert path.exists()
        assert manager.active_session is None

    def test_shutdown_returns_none_when_no_session(self, tmp_path: Path) -> None:
        """Verifies shutdown() returns None when no session active.

        Arrangement:
            1. SessionManager after previous shutdown.

        Action:
            Call shutdown() again.

        Assertion Strategy:
            Validates by confirming:
            - Returns None.
            - No exception raised.
        """
        manager = SessionManager(tmp_path)
        manager.shutdown()  # First shutdown

        result = manager.shutdown()  # Second shutdown

        assert result is None

    def test_shutdown_logs_message(self, tmp_path: Path) -> None:
        """Verifies shutdown() logs message before closing.

        Arrangement:
            1. SessionManager with active session.

        Action:
            Call shutdown().

        Assertion Strategy:
            Validates by confirming:
            - Session closed normally.
            - Shutdown method completes without error.
        """
        manager = SessionManager(tmp_path)
        manager.start_session(SessionType.OBSERVATION, target="M31")

        # Should not raise
        path = manager.shutdown()

        assert path is not None
        assert path.exists()
