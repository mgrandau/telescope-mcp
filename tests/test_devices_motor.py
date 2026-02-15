"""Unit tests for Motor device abstraction.

Tests the high-level async Motor interface including:
- set_home() — zeros both axes (Issue #4)
- zero_position() — zeros single axis
- stop() — sends stop to driver

Example:
    Run all motor device tests:
        pdm run pytest tests/test_devices_motor.py -v
"""

from __future__ import annotations

import pytest

from telescope_mcp.devices.motor import Motor
from telescope_mcp.drivers.motors import (
    DigitalTwinMotorConfig,
    DigitalTwinMotorDriver,
    MotorType,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def driver() -> DigitalTwinMotorDriver:
    """Create a digital twin motor driver with no timing simulation.

    Returns:
        DigitalTwinMotorDriver with timing disabled for fast tests.
    """
    config = DigitalTwinMotorConfig(simulate_timing=False)
    return DigitalTwinMotorDriver(config)


@pytest.fixture
async def motor(driver: DigitalTwinMotorDriver) -> Motor:
    """Create and connect a Motor device for testing.

    Args:
        driver: Digital twin driver fixture.

    Yields:
        Connected Motor device ready for operations.
    """
    m = Motor(driver)
    await m.connect()
    yield m
    await m.disconnect()


# =============================================================================
# Test: set_home (Issue #4)
# =============================================================================


class TestSetHome:
    """Tests for Motor.set_home() — the Set Home feature from Issue #4."""

    @pytest.mark.asyncio
    async def test_set_home_zeros_both_axes(self, motor: Motor) -> None:
        """set_home zeros both altitude and azimuth positions.

        Business context: User presses Set Home on dashboard. Both axes
        should read 0 afterward without any physical movement.
        """
        await motor.move_to(MotorType.ALTITUDE, 70000)
        await motor.move_to(MotorType.AZIMUTH, 30000)

        await motor.set_home()

        alt_status = motor.get_status(MotorType.ALTITUDE)
        az_status = motor.get_status(MotorType.AZIMUTH)
        assert alt_status.position_steps == 0
        assert az_status.position_steps == 0

    @pytest.mark.asyncio
    async def test_set_home_when_already_zero(self, motor: Motor) -> None:
        """set_home when positions are already zero is a safe no-op."""
        await motor.set_home()

        alt_status = motor.get_status(MotorType.ALTITUDE)
        az_status = motor.get_status(MotorType.AZIMUTH)
        assert alt_status.position_steps == 0
        assert az_status.position_steps == 0

    @pytest.mark.asyncio
    async def test_set_home_when_disconnected_raises(
        self, driver: DigitalTwinMotorDriver
    ) -> None:
        """set_home before connect raises RuntimeError."""
        m = Motor(driver)
        with pytest.raises(RuntimeError, match="not connected"):
            await m.set_home()


# =============================================================================
# Test: zero_position (single axis)
# =============================================================================


class TestZeroPosition:
    """Tests for Motor.zero_position() — single axis zeroing."""

    @pytest.mark.asyncio
    async def test_zero_altitude(self, motor: Motor) -> None:
        """zero_position zeros only the specified altitude axis."""
        await motor.move_to(MotorType.ALTITUDE, 50000)
        await motor.move_to(MotorType.AZIMUTH, 30000)

        await motor.zero_position(MotorType.ALTITUDE)

        assert motor.get_status(MotorType.ALTITUDE).position_steps == 0
        assert motor.get_status(MotorType.AZIMUTH).position_steps == 30000

    @pytest.mark.asyncio
    async def test_zero_azimuth(self, motor: Motor) -> None:
        """zero_position zeros only the specified azimuth axis."""
        await motor.move_to(MotorType.ALTITUDE, 50000)
        await motor.move_to(MotorType.AZIMUTH, 30000)

        await motor.zero_position(MotorType.AZIMUTH)

        assert motor.get_status(MotorType.ALTITUDE).position_steps == 50000
        assert motor.get_status(MotorType.AZIMUTH).position_steps == 0

    @pytest.mark.asyncio
    async def test_zero_when_disconnected_raises(
        self, driver: DigitalTwinMotorDriver
    ) -> None:
        """zero_position before connect raises RuntimeError."""
        m = Motor(driver)
        with pytest.raises(RuntimeError, match="not connected"):
            await m.zero_position(MotorType.ALTITUDE)


# =============================================================================
# Test: stop()
# =============================================================================


class TestStop:
    """Tests for Motor.stop() — sends stop to driver."""

    @pytest.mark.asyncio
    async def test_stop_single_axis(self, motor: Motor) -> None:
        """Stop single axis completes without error."""
        await motor.stop(MotorType.ALTITUDE)

    @pytest.mark.asyncio
    async def test_stop_all(self, motor: Motor) -> None:
        """Emergency stop (None) completes without error."""
        await motor.stop()

    @pytest.mark.asyncio
    async def test_stop_when_disconnected_raises(
        self, driver: DigitalTwinMotorDriver
    ) -> None:
        """Stop before connect raises RuntimeError."""
        m = Motor(driver)
        with pytest.raises(RuntimeError, match="not connected"):
            await m.stop()
