"""Digital twin motor driver for testing without hardware.

Provides simulated motor controller responses for development and testing.
Supports configurable slew speeds, acceleration, position limits, and
realistic timing simulation.

Hardware Simulation:
    Simulates Teensy + AMIS stepper motor controller behavior:
    - NEMA 23 stepper for altitude (0-140000 steps = 0°-90°)
    - NEMA 17 stepper for azimuth (belt-driven, ~270° range)

Example:
    from telescope_mcp.drivers.motors import DigitalTwinMotorDriver

    driver = DigitalTwinMotorDriver()
    controller = driver.open()

    controller.move(MotorType.ALTITUDE, 70000)  # Move to 45°
    status = controller.get_status(MotorType.ALTITUDE)
    print(f"Position: {status.position_steps} steps")

    driver.close()
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import TracebackType

from telescope_mcp.drivers.motors.types import (
    AvailableMotorController,
    MotorInfo,
    MotorInstance,
    MotorStatus,
    MotorType,
)
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)

__all__ = [
    "DigitalTwinMotorConfig",
    "DigitalTwinMotorInstance",
    "DigitalTwinMotorDriver",
]

# Default motor configuration matching real hardware
# Steps-per-degree is a physical constant of the motor/gearing/microstep config.
# Position 0 = zenith.  Positive → past zenith, Negative → toward horizon.
DEFAULT_ALTITUDE_STEPS_PER_DEGREE = 140000 / 90.0  # ~1555.56 (hardware constant)
DEFAULT_ALTITUDE_MAX_STEPS = int(
    3 * DEFAULT_ALTITUDE_STEPS_PER_DEGREE
)  # +3° past zenith
DEFAULT_ALTITUDE_MIN_STEPS = int(
    -60 * DEFAULT_ALTITUDE_STEPS_PER_DEGREE
)  # -60° toward horizon
DEFAULT_AZIMUTH_STEPS_PER_DEGREE = 110000 / 135.0  # ~814.81 (hardware constant)
DEFAULT_AZIMUTH_MAX_STEPS = int(
    190 * DEFAULT_AZIMUTH_STEPS_PER_DEGREE
)  # +190° from home
DEFAULT_AZIMUTH_MIN_STEPS = 0  # Home position

# Timing simulation defaults — per-axis, matching burned-in stepper_amis settings
# Altitude (Axis 0): 17HS24-2104S, 128 microsteps, 25600 μsteps/rev
DEFAULT_ALTITUDE_SLEW_SPEED = 1200.0  # μsteps/sec (stepper_amis velocity)
DEFAULT_ALTITUDE_ACCEL_TIME = 0.2  # seconds to reach velocity
# Azimuth (Axis 1): 23HS41-1804S, 64 microsteps, 12800 μsteps/rev
DEFAULT_AZIMUTH_SLEW_SPEED = 1100.0  # μsteps/sec (stepper_amis velocity)
DEFAULT_AZIMUTH_ACCEL_TIME = 0.1  # seconds to reach velocity
DEFAULT_SIMULATE_TIMING = True  # Whether to add realistic delays


@dataclass
class DigitalTwinMotorConfig:
    """Configuration for digital twin motor controller behavior.

    Attributes:
        altitude_min_steps: Minimum allowed altitude position.
        altitude_max_steps: Maximum allowed altitude position.
        azimuth_min_steps: Minimum allowed azimuth position (CCW limit).
        azimuth_max_steps: Maximum allowed azimuth position (CW limit).
        altitude_home_steps: Home position for altitude axis.
        azimuth_home_steps: Home position for azimuth axis.
        altitude_steps_per_degree: Steps per degree for altitude.
        azimuth_steps_per_degree: Steps per degree for azimuth.
        altitude_slew_speed: Altitude max velocity in μsteps/sec.
        altitude_accel_time: Altitude acceleration ramp time in seconds.
        azimuth_slew_speed: Azimuth max velocity in μsteps/sec.
        azimuth_accel_time: Azimuth acceleration ramp time in seconds.
        simulate_timing: If True, add realistic movement delays.
        position_noise_steps: Random noise added to positions (0=none).
    """

    # Position limits
    altitude_min_steps: int = DEFAULT_ALTITUDE_MIN_STEPS
    altitude_max_steps: int = DEFAULT_ALTITUDE_MAX_STEPS
    azimuth_min_steps: int = DEFAULT_AZIMUTH_MIN_STEPS
    azimuth_max_steps: int = DEFAULT_AZIMUTH_MAX_STEPS

    # Home positions
    altitude_home_steps: int = 0  # Zenith
    azimuth_home_steps: int = 0  # Center

    # Conversion factors
    altitude_steps_per_degree: float = DEFAULT_ALTITUDE_STEPS_PER_DEGREE
    azimuth_steps_per_degree: float = DEFAULT_AZIMUTH_STEPS_PER_DEGREE

    # Timing simulation — per-axis, matching stepper_amis hardware
    altitude_slew_speed: float = DEFAULT_ALTITUDE_SLEW_SPEED
    altitude_accel_time: float = DEFAULT_ALTITUDE_ACCEL_TIME
    azimuth_slew_speed: float = DEFAULT_AZIMUTH_SLEW_SPEED
    azimuth_accel_time: float = DEFAULT_AZIMUTH_ACCEL_TIME
    simulate_timing: bool = DEFAULT_SIMULATE_TIMING

    # Noise simulation (for testing error handling)
    position_noise_steps: int = 0

    def __repr__(self) -> str:
        """Return concise config representation for logging.

        Creates a human-readable string showing key configuration values.
        Useful for log messages and interactive debugging sessions.

        Returns:
            str: Formatted string showing limits and timing settings.

        Example:
            >>> config = DigitalTwinMotorConfig()
            >>> print(config)
            DigitalTwinMotorConfig(alt=[0, 140000], az=[-110000, 110000], timing=True)
        """
        return (
            f"DigitalTwinMotorConfig("
            f"alt=[{self.altitude_min_steps}, {self.altitude_max_steps}], "
            f"az=[{self.azimuth_min_steps}, {self.azimuth_max_steps}], "
            f"timing={self.simulate_timing})"
        )


class DigitalTwinMotorInstance:
    """Simulated motor controller instance for testing.

    Provides motor control with configurable timing simulation, position
    limits, and realistic behavior. Simulates Teensy + AMIS board behavior.

    Thread Safety:
        Uses threading lock for position updates. Safe for concurrent
        status queries but moves should be serialized.
    """

    def __init__(self, config: DigitalTwinMotorConfig) -> None:
        """Initialize simulated motor controller with configuration.

        Creates a digital twin motor controller that simulates stepper motor
        behavior including configurable timing, limits, and positions.

        Business context: Digital twin enables development and testing
        without physical Teensy + AMIS hardware. Simulates realistic motor
        behavior including slew times for algorithm testing.

        Args:
            config: Motor behavior configuration including limits, timing,
                and conversion factors.

        Returns:
            None. Instance ready for move(), get_status(), etc.

        Raises:
            No exceptions raised.

        Example:
            >>> config = DigitalTwinMotorConfig(simulate_timing=False)
            >>> instance = DigitalTwinMotorInstance(config)
            >>> instance.move(MotorType.ALTITUDE, 70000)
        """
        self._config = config
        self._is_open = True

        # Current positions
        self._positions = {
            MotorType.ALTITUDE: 0,
            MotorType.AZIMUTH: 0,
        }

        # Movement state
        self._is_moving = {
            MotorType.ALTITUDE: False,
            MotorType.AZIMUTH: False,
        }
        self._target_positions = {
            MotorType.ALTITUDE: 0,
            MotorType.AZIMUTH: 0,
        }
        self._current_speeds = {
            MotorType.ALTITUDE: 0,
            MotorType.AZIMUTH: 0,
        }

        # Stall detection state
        self._stalled = {
            MotorType.ALTITUDE: False,
            MotorType.AZIMUTH: False,
        }

        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        logger.debug(
            "Digital twin motor controller initialized",
            config=repr(config),
        )

    def get_info(self) -> MotorInfo:
        """Get motor controller hardware information.

        Returns metadata describing this digital twin controller including
        type, name, and conversion factors. Used by Motor device layer.

        Returns:
            MotorInfo TypedDict with controller details.

        Example:
            >>> info = controller.get_info()
            >>> print(f"Type: {info['type']}")
            Type: digital_twin
        """
        return MotorInfo(
            type="digital_twin",
            name="Digital Twin Motor Controller",
            port="simulated",
            altitude_steps_per_degree=self._config.altitude_steps_per_degree,
            azimuth_steps_per_degree=self._config.azimuth_steps_per_degree,
        )

    def _get_limits(self, motor: MotorType) -> tuple[int, int]:
        """Get position limits for specified motor.

        Args:
            motor: Which motor to get limits for.

        Returns:
            Tuple of (min_steps, max_steps).
        """
        if motor == MotorType.ALTITUDE:
            return self._config.altitude_min_steps, self._config.altitude_max_steps
        else:
            return self._config.azimuth_min_steps, self._config.azimuth_max_steps

    def _get_home_position(self, motor: MotorType) -> int:
        """Get home position for specified motor.

        Args:
            motor: Which motor to get home position for.

        Returns:
            Home position in steps.
        """
        if motor == MotorType.ALTITUDE:
            return self._config.altitude_home_steps
        else:
            return self._config.azimuth_home_steps

    def _simulate_move_time(self, motor: MotorType, steps: int, speed: int) -> float:
        """Calculate simulated move time based on distance and speed.

        Uses trapezoidal velocity profile with acceleration ramps.
        Per-axis speed/accel matching real stepper_amis hardware.

        Args:
            motor: Which motor axis (for per-axis speed/accel lookup).
            steps: Number of steps to move (absolute value used).
            speed: Speed percentage 1-100.

        Returns:
            Simulated move time in seconds.
        """
        if not self._config.simulate_timing:
            return 0.0

        distance = abs(steps)
        if distance == 0:
            return 0.0

        # Per-axis speed and acceleration from config
        if motor == MotorType.ALTITUDE:
            max_speed = self._config.altitude_slew_speed
            accel_time = self._config.altitude_accel_time
        else:
            max_speed = self._config.azimuth_slew_speed
            accel_time = self._config.azimuth_accel_time

        # Scale speed by percentage
        effective_speed = max_speed * (speed / 100.0)

        # Simple model: acceleration + cruise + deceleration
        accel_distance = effective_speed * accel_time / 2  # Distance during accel

        if distance < 2 * accel_distance:
            # Short move - triangular profile
            move_time: float = 2 * (distance / effective_speed) ** 0.5
        else:
            # Long move - trapezoidal profile
            cruise_distance = distance - 2 * accel_distance
            cruise_time = cruise_distance / effective_speed
            move_time = 2 * accel_time + cruise_time

        return move_time

    def move(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Move motor to absolute step position.

        Commands the specified motor to move to an absolute position. If
        timing simulation is enabled, blocks for realistic move duration.

        Business context: Core telescope pointing operation. Digital twin
        simulates realistic slew times to test UI responsiveness and
        timeout handling.

        Args:
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            steps: Absolute target position in steps.
            speed: Speed percentage 1-100. Default 100.

        Returns:
            None. Blocks until simulated move completes.

        Raises:
            ValueError: If steps outside motor's valid range or speed invalid.
            RuntimeError: If controller is closed.

        Example:
            >>> controller.move(MotorType.ALTITUDE, 70000)  # 45° altitude
            >>> controller.move(MotorType.AZIMUTH, 35000, speed=50)
        """
        if not self._is_open:
            raise RuntimeError("Controller is closed")

        if not 1 <= speed <= 100:
            raise ValueError(f"Speed must be 1-100, got {speed}")

        min_steps, max_steps = self._get_limits(motor)
        if not min_steps <= steps <= max_steps:
            raise ValueError(
                f"{motor.value} steps must be {min_steps}-{max_steps}, got {steps}"
            )

        with self._lock:
            current_pos = self._positions[motor]
            distance = steps - current_pos

            self._is_moving[motor] = True
            self._target_positions[motor] = steps
            self._current_speeds[motor] = speed

        # Simulate move time (interruptible via _stop_event)
        move_time = self._simulate_move_time(motor, distance, speed)
        self._stop_event.clear()
        if move_time > 0:
            logger.debug(
                "Simulating move",
                motor=motor.value,
                from_steps=current_pos,
                to_steps=steps,
                duration_sec=f"{move_time:.2f}",
            )
            interrupted = self._stop_event.wait(timeout=move_time)
            if interrupted:
                # Stop was called during move — position stays unchanged
                with self._lock:
                    self._is_moving[motor] = False
                    self._current_speeds[motor] = 0
                logger.info(
                    "Move interrupted by stop",
                    motor=motor.value,
                    position=current_pos,
                )
                return

        with self._lock:
            self._positions[motor] = steps
            self._is_moving[motor] = False
            self._current_speeds[motor] = 0

        logger.debug(
            "Move complete",
            motor=motor.value,
            position=steps,
        )

    def move_relative(self, motor: MotorType, steps: int, speed: int = 100) -> None:
        """Move motor by relative step offset from current position.

        Commands the specified motor to move by a relative number of steps.
        Validates that resulting position stays within motor limits.

        Args:
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            steps: Number of steps to move from current position.
            speed: Speed percentage 1-100. Default 100.

        Returns:
            None. Blocks until simulated move completes.

        Raises:
            ValueError: If resulting position would exceed motor limits.
            RuntimeError: If controller is closed.

        Example:
            >>> controller.move_relative(MotorType.ALTITUDE, 1000)  # Nudge
            >>> controller.move_relative(MotorType.AZIMUTH, -5000)
        """
        if not self._is_open:
            raise RuntimeError("Controller is closed")

        with self._lock:
            current_pos = self._positions[motor]

        new_position = current_pos + steps
        min_steps, max_steps = self._get_limits(motor)

        if not min_steps <= new_position <= max_steps:
            raise ValueError(
                f"{motor.value} move would exceed limits: "
                f"{current_pos} + {steps} = {new_position}"
            )

        self.move(motor, new_position, speed)

    def stop(self, motor: MotorType | None = None) -> None:
        """Stop motor(s) immediately by interrupting any active move.

        Sets the stop event to wake any blocking move() call, causing it
        to return early without updating position. This provides true
        interruptible stop for the digital twin.

        Thread Safety:
            The _stop_event.set() call is thread-safe and interrupts any
            thread blocked in Event.wait() inside move(). No lock needed
            for the interrupt itself.

        Args:
            motor: Motor to stop, or None for emergency stop all.

        Returns:
            None. Any in-progress moves will return immediately.

        Example:
            >>> controller.stop(MotorType.ALTITUDE)
            >>> controller.stop()  # Emergency stop all
        """
        # Signal any blocking move() to wake up and abort
        self._stop_event.set()

        if motor is not None:
            logger.info("Stop requested", motor=motor.value)
            with self._lock:
                self._is_moving[motor] = False
                self._current_speeds[motor] = 0
        else:
            logger.info("Emergency stop all requested")
            with self._lock:
                for m in MotorType:
                    self._is_moving[m] = False
                    self._current_speeds[m] = 0

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get current motor status including position and movement state.

        Returns the current believed position and movement state for the
        specified motor.

        Args:
            motor: Which motor to query (ALTITUDE or AZIMUTH).

        Returns:
            MotorStatus dataclass with position, is_moving, speed, stalled, at_limit.

        Example:
            >>> status = controller.get_status(MotorType.ALTITUDE)
            >>> print(f"Position: {status.position_steps}")
            >>> print(f"Moving: {status.is_moving}")
        """
        with self._lock:
            at_limit = self._check_at_limit(motor)
            return MotorStatus(
                motor=motor,
                is_moving=self._is_moving[motor],
                position_steps=self._positions[motor],
                speed=self._current_speeds[motor],
                target_steps=self._target_positions[motor]
                if self._is_moving[motor]
                else None,
                stalled=self._stalled[motor],
                at_limit=at_limit,
            )

    def _check_at_limit(self, motor: MotorType) -> str | None:
        """Check if motor is at a position limit (internal, no lock).

        Args:
            motor: Which motor to check.

        Returns:
            'min' if at minimum limit, 'max' if at maximum limit, None otherwise.
        """
        min_steps, max_steps = self._get_limits(motor)
        pos = self._positions[motor]
        if pos <= min_steps:
            return "min"
        elif pos >= max_steps:
            return "max"
        return None

    def is_at_limit(self, motor: MotorType) -> str | None:
        """Check if motor is at a position limit.

        Useful for detecting when motor has reached its mechanical range
        or when approaching limits during homing sequences.

        Args:
            motor: Which motor to check (ALTITUDE or AZIMUTH).

        Returns:
            'min' if at minimum limit, 'max' if at maximum limit,
            None if within normal operating range.

        Example:
            >>> limit = controller.is_at_limit(MotorType.AZIMUTH)
            >>> if limit == 'min':
            ...     print("At CCW limit")
        """
        with self._lock:
            return self._check_at_limit(motor)

    def home(self, motor: MotorType) -> None:
        """Move motor to its configured home position.

        Moves the specified motor to its predefined home position using
        normal move semantics.

        Args:
            motor: Which motor to home.

        Returns:
            None. Blocks until home position reached.

        Raises:
            RuntimeError: If controller is closed.

        Example:
            >>> controller.home(MotorType.ALTITUDE)  # Park at zenith
        """
        home_pos = self._get_home_position(motor)
        logger.info("Homing motor", motor=motor.value, target=home_pos)
        self.move(motor, home_pos)
        logger.info("Motor homed", motor=motor.value)

    def home_all(self) -> None:
        """Home both altitude and azimuth motors to safe positions.

        Sequentially moves both motors to their configured home positions.
        Altitude homes first, then azimuth.

        Returns:
            None. Blocks until both motors reach home.

        Example:
            >>> controller.home_all()  # Park telescope safely
        """
        logger.info("Homing all motors")
        self.home(MotorType.ALTITUDE)
        self.home(MotorType.AZIMUTH)
        logger.info("All motors homed")

    def zero_position(self, motor: MotorType) -> None:
        """Zero the position counter at current physical location.

        Sets the specified motor's internal position counter to 0 without
        any simulated movement. Used to establish the current telescope
        position as the reference origin.

        Business context: Called when user presses 'Set Home' on dashboard.
        Records current physical position as (0,0) reference for the
        observing session.

        Args:
            motor: Which motor to zero (ALTITUDE or AZIMUTH).

        Returns:
            None. Position counter set to 0 immediately.

        Raises:
            RuntimeError: If controller is closed.

        Example:
            >>> controller.zero_position(MotorType.ALTITUDE)
            >>> status = controller.get_status(MotorType.ALTITUDE)
            >>> assert status.position_steps == 0
        """
        if not self._is_open:
            raise RuntimeError("Controller is closed")

        with self._lock:
            old_pos = self._positions[motor]
            self._positions[motor] = 0
            self._stalled[motor] = False

        logger.info(
            "Position zeroed",
            motor=motor.value,
            old_position=old_pos,
        )

    def move_until_stall(
        self,
        motor: MotorType,
        direction: int,
        speed: int = 20,
        step_size: int = 100,
    ) -> int:
        """Move motor slowly until stall/slip is detected.

        Moves motor incrementally in the specified direction at low speed
        until the motor stalls (reaches limit and cannot continue). Used
        for homing sequences to find mechanical end stops without damage.

        Business context: Homing procedure requires finding mechanical limits
        without damaging belt or mount. Moving slowly allows detecting slip
        when belt reaches end of travel. Backing off 2-3 steps from detected
        limit establishes safe operating boundary.

        For digital twin: Simulates stall when position reaches configured
        limits. Real hardware would detect motor current spike or encoder
        stall.

        Args:
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            direction: Direction to move. Negative = CCW/down, Positive = CW/up.
            speed: Speed percentage 1-100. Default 20 for slow homing.
            step_size: Steps per increment. Default 100.

        Returns:
            Final position in steps when stall detected.

        Raises:
            RuntimeError: If controller not connected.
            ValueError: If direction is 0.

        Example:
            >>> # Find CCW limit
            >>> ccw_limit = controller.move_until_stall(
            ...     MotorType.AZIMUTH, direction=-1, speed=20
            ... )
            >>> print(f"CCW limit at {ccw_limit} steps")
        """
        if not self._is_open:
            raise RuntimeError("Controller is closed")

        if direction == 0:
            raise ValueError("Direction must be non-zero")

        # Normalize direction to -1 or +1
        dir_sign = -1 if direction < 0 else 1
        actual_step = step_size * dir_sign

        min_steps, max_steps = self._get_limits(motor)
        logger.info(
            "Starting move_until_stall",
            motor=motor.value,
            direction="CCW" if dir_sign < 0 else "CW",
            speed=speed,
            step_size=step_size,
        )

        # Clear any previous stall state
        with self._lock:
            self._stalled[motor] = False

        while True:
            with self._lock:
                current_pos = self._positions[motor]

            # Check if we're at the limit
            target_pos = current_pos + actual_step

            # Would this move exceed limits?
            if target_pos < min_steps or target_pos > max_steps:
                # Stall detected - we've hit the limit
                with self._lock:
                    self._stalled[motor] = True
                    # Clamp to limit
                    final_pos = max(min_steps, min(max_steps, target_pos))
                    self._positions[motor] = final_pos

                logger.info(
                    "Stall detected at limit",
                    motor=motor.value,
                    position=final_pos,
                    limit="min" if final_pos <= min_steps else "max",
                )
                return final_pos

            # Move one step
            try:
                self.move_relative(motor, actual_step, speed)
            except ValueError:
                # Hit limit during move
                with self._lock:
                    self._stalled[motor] = True
                    return self._positions[motor]

    def set_position(self, motor: MotorType, steps: int) -> None:
        """Set the simulated motor position without movement (for testing).

        Directly sets the internal position counter. Used for test setup
        to establish known starting positions.

        Business context: Integration tests need deterministic motor state.
        This method allows tests to simulate "already moved" conditions
        without waiting for simulated slew times.

        Args:
            motor: Which motor to set position for.
            steps: Position in steps to set.

        Returns:
            None.

        Example:
            >>> controller.set_position(MotorType.ALTITUDE, 70000)
            >>> status = controller.get_status(MotorType.ALTITUDE)
            >>> assert status.position_steps == 70000
        """
        min_steps, max_steps = self._get_limits(motor)
        if not min_steps <= steps <= max_steps:
            raise ValueError(
                f"{motor.value} position must be {min_steps}-{max_steps}, got {steps}"
            )

        with self._lock:
            self._positions[motor] = steps

        logger.debug("Position set", motor=motor.value, steps=steps)

    def close(self) -> None:
        """Close the simulated motor controller connection.

        Marks controller as closed. Subsequent operations will raise
        RuntimeError. Safe to call multiple times.

        Returns:
            None.

        Example:
            >>> controller.close()
        """
        self._is_open = False
        logger.debug("Digital twin motor controller closed")

    @property
    def is_open(self) -> bool:
        """Return True if controller connection is open.

        Returns:
            bool: True if connected, False if closed.

        Example:
            >>> if controller.is_open:
            ...     controller.move(MotorType.ALTITUDE, 1000)
        """
        return self._is_open


class DigitalTwinMotorDriver:
    """Digital twin motor driver for testing without hardware.

    Creates simulated motor controller instances with configurable behavior
    for development and testing.

    Example:
        driver = DigitalTwinMotorDriver()
        controller = driver.open()
        controller.move(MotorType.ALTITUDE, 70000)
        driver.close()
    """

    def __init__(self, config: DigitalTwinMotorConfig | None = None) -> None:
        """Initialize driver with optional configuration.

        Sets up driver with simulation configuration. Default config
        provides realistic limits and timing for testing.

        Business context: Driver manages digital twin motor controller
        lifecycle. Configuration allows customizing simulation behavior
        for different test scenarios.

        Args:
            config: Motor behavior configuration. Uses defaults if None.

        Returns:
            None. Driver ready for get_available_controllers() or open().

        Raises:
            No exceptions raised.

        Example:
            >>> driver = DigitalTwinMotorDriver()
            >>> controller = driver.open()
        """
        self._config = config or DigitalTwinMotorConfig()
        self._instance: DigitalTwinMotorInstance | None = None

    def _ensure_not_open(self) -> None:
        """Ensure no controller is currently open before opening a new one.

        Validates that the driver doesn't have an active controller instance.
        Prevents resource leaks by enforcing single-instance semantics.

        Raises:
            RuntimeError: If a controller instance is already open.

        Example:
            >>> driver.open()
            >>> driver._ensure_not_open()  # Raises RuntimeError
        """
        if self._instance is not None and self._instance.is_open:
            raise RuntimeError("Controller already open")

    def get_available_controllers(self) -> list[AvailableMotorController]:
        """List available motor controllers from this driver.

        For the digital twin driver, always returns a single simulated
        controller. This matches the MotorDriver protocol interface.

        Business context: The digital twin enables development and testing
        without physical hardware. By implementing the same interface as
        real drivers, it allows full system testing including device
        enumeration workflows.

        Returns:
            list[AvailableMotorController]: Single-element list with:
                - id: Always 0
                - type: Always 'digital_twin'
                - name: Human-readable name
                - port: Always 'simulated'

        Example:
            >>> driver = DigitalTwinMotorDriver()
            >>> controllers = driver.get_available_controllers()
            >>> print(f"Found {len(controllers)} controller(s)")
            Found 1 controller(s)
        """
        return [
            AvailableMotorController(
                id=0,
                type="digital_twin",
                name="Digital Twin Motor Controller",
                port="simulated",
                description="Simulated Teensy + AMIS motor controller",
            )
        ]

    def open(self, controller_id: int | str = 0) -> MotorInstance:
        """Open a simulated digital twin motor controller instance.

        Creates a new DigitalTwinMotorInstance with the configured
        simulation parameters. Only one instance can be open at a time.

        Business context: Enables development and testing without physical
        hardware. The digital twin simulates realistic motor behavior
        including slew times.

        Args:
            controller_id: Controller identifier. Ignored for digital twin
                since only one simulated controller exists.

        Returns:
            DigitalTwinMotorInstance: Simulated controller ready for use.

        Raises:
            RuntimeError: If a controller is already open.

        Example:
            >>> driver = DigitalTwinMotorDriver()
            >>> controller = driver.open()
            >>> controller.move(MotorType.ALTITUDE, 70000)
            >>> driver.close()
        """
        self._ensure_not_open()

        self._instance = DigitalTwinMotorInstance(self._config)
        logger.debug("Digital twin motor controller opened")
        return self._instance

    def close(self) -> None:
        """Close the current simulated motor controller instance.

        Closes underlying DigitalTwinMotorInstance if open. Safe to call
        when no instance is open.

        Returns:
            None.

        Example:
            >>> driver.close()
        """
        if self._instance is not None:
            self._instance.close()
            self._instance = None

    def __enter__(self) -> DigitalTwinMotorInstance:
        """Enter context manager, opening simulated controller.

        Enables "with driver:" syntax for automatic cleanup.

        Returns:
            DigitalTwinMotorInstance: Open controller ready for use.

        Raises:
            RuntimeError: If a controller is already open.

        Example:
            >>> with DigitalTwinMotorDriver() as controller:
            ...     controller.move(MotorType.ALTITUDE, 70000)
        """
        instance = self.open()
        assert isinstance(instance, DigitalTwinMotorInstance)
        return instance

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing simulated controller.

        Called automatically when exiting a `with` block.

        Args:
            exc_type: Exception type if raised, None otherwise.
            exc_val: Exception instance if raised, None otherwise.
            exc_tb: Traceback if raised, None otherwise.

        Returns:
            None. Exceptions are not suppressed.

        Example:
            >>> with DigitalTwinMotorDriver() as controller:
            ...     controller.move(MotorType.ALTITUDE, 70000)
            # Controller automatically closed
        """
        self.close()
