"""Tests for driver stubs."""

import pytest
from telescope_mcp.drivers.motors import StubMotorController, MotorType, MotorStatus
from telescope_mcp.drivers.sensors import StubPositionSensor, TelescopePosition


class TestStubMotorController:
    """Tests for the stub motor controller."""

    def test_initial_position(self):
        """Motors start at position 0."""
        controller = StubMotorController()
        status = controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 0

    def test_move_updates_position(self):
        """Move updates position correctly."""
        controller = StubMotorController()
        controller.move(MotorType.ALTITUDE, 100)
        status = controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 100

    def test_home_resets_position(self):
        """Home resets position to 0."""
        controller = StubMotorController()
        controller.move(MotorType.AZIMUTH, 500)
        controller.home(MotorType.AZIMUTH)
        status = controller.get_status(MotorType.AZIMUTH)
        assert status.position_steps == 0


class TestStubPositionSensor:
    """Tests for the stub position sensor."""

    def test_initial_position(self):
        """Sensor has default position."""
        sensor = StubPositionSensor()
        pos = sensor.read()
        assert pos.altitude == 45.0
        assert pos.azimuth == 180.0

    def test_calibrate_updates_position(self):
        """Calibrate updates reported position."""
        sensor = StubPositionSensor()
        sensor.calibrate(30.0, 90.0)
        pos = sensor.read()
        assert pos.altitude == 30.0
        assert pos.azimuth == 90.0
