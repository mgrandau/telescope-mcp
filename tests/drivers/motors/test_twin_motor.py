"""Unit tests for digital twin motor driver.

Tests DigitalTwinMotorInstance and DigitalTwinMotorDriver including:
- zero_position (Set Home feature, Issue #4)
- Interruptible stop via threading.Event
- Basic movement and position tracking
- Driver lifecycle

Example:
    Run all digital twin motor tests:
        pdm run pytest tests/drivers/motors/test_twin.py -v
"""

from __future__ import annotations

import threading
import time

import pytest

from telescope_mcp.drivers.motors import (
    DigitalTwinMotorConfig,
    DigitalTwinMotorDriver,
    DigitalTwinMotorInstance,
    MotorType,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config_no_timing() -> DigitalTwinMotorConfig:
    """Config with timing simulation disabled for fast tests.

    Returns:
        DigitalTwinMotorConfig with simulate_timing=False.
    """
    return DigitalTwinMotorConfig(simulate_timing=False)


@pytest.fixture
def config_with_timing() -> DigitalTwinMotorConfig:
    """Config with timing simulation enabled for stop interruptibility tests.

    Uses fast slew speed so moves take a known duration.

    Returns:
        DigitalTwinMotorConfig with simulate_timing=True, slow speed.
    """
    return DigitalTwinMotorConfig(
        simulate_timing=True,
        altitude_slew_speed=1000,  # Slow enough that moves take measurable time
        altitude_accel_time=0.0,  # No accel ramp for predictable timing
        azimuth_slew_speed=1000,
        azimuth_accel_time=0.0,
    )


@pytest.fixture
def controller(config_no_timing: DigitalTwinMotorConfig) -> DigitalTwinMotorInstance:
    """Create a digital twin controller with timing disabled.

    Args:
        config_no_timing: Config fixture with no timing simulation.

    Returns:
        Ready-to-use DigitalTwinMotorInstance.
    """
    return DigitalTwinMotorInstance(config_no_timing)


@pytest.fixture
def timed_controller(
    config_with_timing: DigitalTwinMotorConfig,
) -> DigitalTwinMotorInstance:
    """Create a digital twin controller with timing enabled.

    Args:
        config_with_timing: Config fixture with timing simulation.

    Returns:
        DigitalTwinMotorInstance with realistic move delays.
    """
    return DigitalTwinMotorInstance(config_with_timing)


# =============================================================================
# Test: zero_position (Set Home - Issue #4)
# =============================================================================


class TestZeroPosition:
    """Tests for zero_position() — the core Set Home functionality."""

    def test_zero_altitude_from_nonzero(
        self, controller: DigitalTwinMotorInstance
    ) -> None:
        """Zero altitude position after moving to a non-zero position.

        Verifies that zero_position sets the counter to 0 without
        physical movement.
        """
        controller.move(MotorType.ALTITUDE, -50000)
        assert controller.get_status(MotorType.ALTITUDE).position_steps == -50000

        controller.zero_position(MotorType.ALTITUDE)
        assert controller.get_status(MotorType.ALTITUDE).position_steps == 0

    def test_zero_azimuth_from_nonzero(
        self, controller: DigitalTwinMotorInstance
    ) -> None:
        """Zero azimuth position after moving to a non-zero position."""
        controller.move(MotorType.AZIMUTH, 50000)
        assert controller.get_status(MotorType.AZIMUTH).position_steps == 50000

        controller.zero_position(MotorType.AZIMUTH)
        assert controller.get_status(MotorType.AZIMUTH).position_steps == 0

    def test_zero_both_axes(self, controller: DigitalTwinMotorInstance) -> None:
        """Zero both axes simulating full Set Home operation."""
        controller.move(MotorType.ALTITUDE, -50000)
        controller.move(MotorType.AZIMUTH, 30000)

        controller.zero_position(MotorType.ALTITUDE)
        controller.zero_position(MotorType.AZIMUTH)

        assert controller.get_status(MotorType.ALTITUDE).position_steps == 0
        assert controller.get_status(MotorType.AZIMUTH).position_steps == 0

    def test_zero_already_zero(self, controller: DigitalTwinMotorInstance) -> None:
        """Zero position when already at zero is a no-op."""
        controller.zero_position(MotorType.ALTITUDE)
        assert controller.get_status(MotorType.ALTITUDE).position_steps == 0

    def test_zero_clears_stall_flag(self, controller: DigitalTwinMotorInstance) -> None:
        """Zero position clears any stall state on the axis."""
        # Force stall via move_until_stall
        controller.move_until_stall(MotorType.ALTITUDE, direction=1)
        status = controller.get_status(MotorType.ALTITUDE)
        assert status.stalled is True

        controller.zero_position(MotorType.ALTITUDE)
        status = controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == 0
        assert status.stalled is False

    def test_zero_does_not_affect_other_axis(
        self, controller: DigitalTwinMotorInstance
    ) -> None:
        """Zeroing one axis does not change the other axis position."""
        controller.move(MotorType.ALTITUDE, -50000)
        controller.move(MotorType.AZIMUTH, 30000)

        controller.zero_position(MotorType.ALTITUDE)

        assert controller.get_status(MotorType.ALTITUDE).position_steps == 0
        assert controller.get_status(MotorType.AZIMUTH).position_steps == 30000

    def test_zero_when_closed_raises(
        self, controller: DigitalTwinMotorInstance
    ) -> None:
        """Zero position on closed controller raises RuntimeError."""
        controller.close()
        with pytest.raises(RuntimeError, match="closed"):
            controller.zero_position(MotorType.ALTITUDE)


# =============================================================================
# Test: Interruptible Stop
# =============================================================================


class TestInterruptibleStop:
    """Tests for stop() interrupting active moves via threading.Event."""

    def test_stop_interrupts_move(
        self, timed_controller: DigitalTwinMotorInstance
    ) -> None:
        """Stop interrupts a blocking move and returns early.

        Starts a long move in a background thread, calls stop() from the
        main thread, and verifies the move returns much faster than the
        full move duration would take.
        """
        move_completed = threading.Event()
        move_start_time = [0.0]
        move_end_time = [0.0]

        def do_move() -> None:
            """Execute a long move in background thread."""
            move_start_time[0] = time.monotonic()
            # -90000 steps at 1000 steps/sec = 90 seconds (would take forever)
            timed_controller.move(MotorType.ALTITUDE, -90000)
            move_end_time[0] = time.monotonic()
            move_completed.set()

        t = threading.Thread(target=do_move)
        t.start()

        # Give the move a moment to start blocking
        time.sleep(0.1)

        # Send stop — should interrupt the Event.wait()
        timed_controller.stop(MotorType.ALTITUDE)

        # Move should complete almost immediately
        move_completed.wait(timeout=2.0)
        assert move_completed.is_set(), "Move did not return after stop"

        # Verify it was fast (much less than the 100s full move)
        elapsed = move_end_time[0] - move_start_time[0]
        assert elapsed < 2.0, f"Move took {elapsed:.2f}s, expected < 2s"

        t.join(timeout=2.0)

    def test_stop_all_interrupts_move(
        self, timed_controller: DigitalTwinMotorInstance
    ) -> None:
        """Emergency stop (motor=None) also interrupts active moves."""
        move_completed = threading.Event()

        def do_move() -> None:
            timed_controller.move(MotorType.AZIMUTH, 100000)
            move_completed.set()

        t = threading.Thread(target=do_move)
        t.start()
        time.sleep(0.1)

        # Emergency stop all
        timed_controller.stop()

        move_completed.wait(timeout=2.0)
        assert move_completed.is_set(), "Move did not return after emergency stop"
        t.join(timeout=2.0)

    def test_stop_preserves_position_on_interrupt(
        self, timed_controller: DigitalTwinMotorInstance
    ) -> None:
        """When stop interrupts a move, position stays at pre-move value.

        The move didn't complete, so position should not reflect the target.
        """
        # Move to known position first
        timed_controller._config.simulate_timing = False
        timed_controller.move(MotorType.ALTITUDE, -50000)
        timed_controller._config.simulate_timing = True

        move_completed = threading.Event()

        def do_move() -> None:
            timed_controller.move(MotorType.ALTITUDE, 4000)
            move_completed.set()

        t = threading.Thread(target=do_move)
        t.start()
        time.sleep(0.1)

        timed_controller.stop(MotorType.ALTITUDE)
        move_completed.wait(timeout=2.0)

        # Position should be at -50000 (pre-move), not 4000 (target)
        status = timed_controller.get_status(MotorType.ALTITUDE)
        assert status.position_steps == -50000
        t.join(timeout=2.0)

    def test_stop_without_active_move(
        self, controller: DigitalTwinMotorInstance
    ) -> None:
        """Stop when no move is active is a safe no-op."""
        controller.stop(MotorType.ALTITUDE)
        controller.stop()  # Emergency stop all
        # No error, no crash

    def test_stop_clears_moving_flag(
        self, controller: DigitalTwinMotorInstance
    ) -> None:
        """Stop clears the is_moving flag for the specified motor."""
        controller.stop(MotorType.ALTITUDE)
        assert controller.get_status(MotorType.ALTITUDE).is_moving is False


# =============================================================================
# Test: Basic Movement (no timing)
# =============================================================================


class TestMovement:
    """Tests for basic move/move_relative operations."""

    def test_move_absolute(self, controller: DigitalTwinMotorInstance) -> None:
        """Move to absolute position updates position correctly."""
        controller.move(MotorType.ALTITUDE, -50000)
        assert controller.get_status(MotorType.ALTITUDE).position_steps == -50000

    def test_move_relative(self, controller: DigitalTwinMotorInstance) -> None:
        """Relative move adds to current position."""
        controller.move(MotorType.ALTITUDE, -40000)
        controller.move_relative(MotorType.ALTITUDE, 10000)
        assert controller.get_status(MotorType.ALTITUDE).position_steps == -30000

    def test_move_relative_negative(self, controller: DigitalTwinMotorInstance) -> None:
        """Relative move with negative steps subtracts from position."""
        controller.move(MotorType.ALTITUDE, -40000)
        controller.move_relative(MotorType.ALTITUDE, -10000)
        assert controller.get_status(MotorType.ALTITUDE).position_steps == -50000

    def test_move_exceeds_max_raises(
        self, controller: DigitalTwinMotorInstance
    ) -> None:
        """Move beyond maximum limit raises ValueError."""
        with pytest.raises(ValueError, match="steps must be"):
            controller.move(MotorType.ALTITUDE, 999999)

    def test_move_below_min_raises(self, controller: DigitalTwinMotorInstance) -> None:
        """Move below minimum limit raises ValueError."""
        with pytest.raises(ValueError, match="steps must be"):
            controller.move(MotorType.ALTITUDE, -100000)

    def test_move_when_closed_raises(
        self, controller: DigitalTwinMotorInstance
    ) -> None:
        """Move on closed controller raises RuntimeError."""
        controller.close()
        with pytest.raises(RuntimeError, match="closed"):
            controller.move(MotorType.ALTITUDE, 1000)


# =============================================================================
# Test: Driver Lifecycle
# =============================================================================


class TestDriverLifecycle:
    """Tests for DigitalTwinMotorDriver open/close lifecycle."""

    def test_open_returns_instance(self) -> None:
        """Opening driver returns a DigitalTwinMotorInstance."""
        driver = DigitalTwinMotorDriver()
        instance = driver.open()
        assert isinstance(instance, DigitalTwinMotorInstance)
        assert instance.is_open
        driver.close()

    def test_context_manager(self) -> None:
        """Driver works as context manager."""
        with DigitalTwinMotorDriver() as controller:
            assert isinstance(controller, DigitalTwinMotorInstance)
            assert controller.is_open
        # Controller should be closed after exiting context

    def test_double_open_raises(self) -> None:
        """Opening driver twice raises RuntimeError."""
        driver = DigitalTwinMotorDriver()
        driver.open()
        with pytest.raises(RuntimeError, match="already open"):
            driver.open()
        driver.close()

    def test_get_available_controllers(self) -> None:
        """get_available_controllers returns single digital twin entry."""
        driver = DigitalTwinMotorDriver()
        controllers = driver.get_available_controllers()
        assert len(controllers) == 1
        assert controllers[0]["type"] == "digital_twin"

    def test_get_info(self, controller: DigitalTwinMotorInstance) -> None:
        """Controller get_info returns digital_twin metadata."""
        info = controller.get_info()
        assert info["type"] == "digital_twin"
        assert info["port"] == "simulated"
