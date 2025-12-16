"""Tests for session MCP tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telescope_mcp.tools import sessions
from telescope_mcp.drivers import config


@pytest.fixture(autouse=True)
def reset_config(tmp_path: Path):
    """Reset config and use temp directory for each test."""
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
    """Tests for session MCP tools."""

    @pytest.mark.asyncio
    async def test_get_session_info_returns_idle(self) -> None:
        """Default session is idle."""
        result = await sessions._get_session_info()
        assert len(result) == 1
        
        data = json.loads(result[0].text)
        assert data["session_type"] == "idle"
        assert data["is_idle"] is True

    @pytest.mark.asyncio
    async def test_start_observation_session(self) -> None:
        """Can start an observation session."""
        result = await sessions._start_session("observation", target="M31", purpose=None)
        assert len(result) == 1
        
        data = json.loads(result[0].text)
        assert data["status"] == "started"
        assert data["session_type"] == "observation"
        assert data["target"] == "M31"
        assert "m31" in data["session_id"].lower()

    @pytest.mark.asyncio
    async def test_start_alignment_session(self) -> None:
        """Can start an alignment session."""
        result = await sessions._start_session(
            "alignment", target=None, purpose="ra_dec_calibration"
        )
        
        data = json.loads(result[0].text)
        assert data["session_type"] == "alignment"
        assert data["purpose"] == "ra_dec_calibration"

    @pytest.mark.asyncio
    async def test_cannot_start_idle_session(self) -> None:
        """Cannot manually start an idle session."""
        result = await sessions._start_session("idle", target=None, purpose=None)
        
        assert "Cannot manually start an idle session" in result[0].text

    @pytest.mark.asyncio
    async def test_invalid_session_type(self) -> None:
        """Invalid session type returns error."""
        result = await sessions._start_session("invalid", target=None, purpose=None)
        
        assert "Invalid session type" in result[0].text

    @pytest.mark.asyncio
    async def test_end_session_returns_to_idle(self, tmp_path: Path) -> None:
        """Ending a session returns to idle."""
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
        """Cannot end an idle session."""
        result = await sessions._end_session()
        
        assert "No active session to end" in result[0].text

    @pytest.mark.asyncio
    async def test_session_log(self) -> None:
        """Can log to session."""
        result = await sessions._session_log("INFO", "Test message", "test")
        data = json.loads(result[0].text)
        
        assert data["status"] == "logged"
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"

    @pytest.mark.asyncio
    async def test_session_event(self) -> None:
        """Can record events."""
        result = await sessions._session_event("tracking_lost", {"reason": "wind"})
        data = json.loads(result[0].text)
        
        assert data["status"] == "recorded"
        assert data["event"] == "tracking_lost"
        assert data["details"]["reason"] == "wind"

    @pytest.mark.asyncio
    async def test_get_data_dir(self, tmp_path: Path) -> None:
        """Can get data directory."""
        result = await sessions._get_data_dir()
        data = json.loads(result[0].text)
        
        assert data["data_dir"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_set_data_dir(self, tmp_path: Path) -> None:
        """Can set data directory."""
        new_dir = tmp_path / "new_data"
        
        result = await sessions._set_data_dir(str(new_dir))
        data = json.loads(result[0].text)
        
        assert data["status"] == "updated"
        assert data["data_dir"] == str(new_dir)

    @pytest.mark.asyncio
    async def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete session workflow."""
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
