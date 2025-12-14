"""Tests for MCP tools."""

import pytest
from telescope_mcp.tools import cameras, motors, position


class TestCameraTools:
    """Camera tool tests."""

    @pytest.mark.asyncio
    async def test_list_cameras_stub(self):
        """Test list_cameras returns stub response."""
        result = await cameras._list_cameras()
        assert len(result) == 1
        assert "not yet implemented" in result[0].text.lower()


class TestMotorTools:
    """Motor tool tests."""

    @pytest.mark.asyncio
    async def test_move_altitude_stub(self):
        """Test move_altitude returns stub response."""
        result = await motors.move_altitude(100, 50)
        assert len(result) == 1
        assert "not yet implemented" in result[0].text.lower()


class TestPositionTools:
    """Position tool tests."""

    @pytest.mark.asyncio
    async def test_get_position_stub(self):
        """Test get_position returns stub response."""
        result = await position.get_position()
        assert len(result) == 1
        assert "not yet implemented" in result[0].text.lower()
