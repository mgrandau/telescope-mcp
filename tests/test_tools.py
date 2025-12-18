"""Tests for MCP tools."""

import pytest

from telescope_mcp.devices import init_registry, shutdown_registry
from telescope_mcp.drivers.cameras.twin import DigitalTwinCameraDriver
from telescope_mcp.tools import cameras, motors, position


class TestCameraTools:
    """Camera tool tests."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Set up camera registry with digital twin for each test."""
        driver = DigitalTwinCameraDriver()
        init_registry(driver)
        yield
        shutdown_registry()

    @pytest.mark.asyncio
    async def test_list_cameras(self):
        """Test list_cameras returns digital twin cameras."""
        result = await cameras._list_cameras()
        assert len(result) == 1
        # Should return JSON with camera list
        text = result[0].text
        assert "cameras" in text
        assert "ASI120MC-S" in text or "count" in text


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
