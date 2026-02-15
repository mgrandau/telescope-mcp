"""Motor device abstraction for telescope mount control.

Provides high-level async interface to stepper motors that control telescope
altitude and azimuth axes. Supports driver injection for testing with DigitalTwin
or real hardware with SerialMotorDriver.

Key Components:
- Motor: High-level device abstraction with driver injection
- MotorConfig: Configuration for motor behavior
- DeviceMotorInfo: Device-layer motor metadata

Example:
    import asyncio
    from telescope_mcp.devices.motor import Motor
    from telescope_mcp.drivers.motors import DigitalTwinMotorDriver

    async def main():
        driver = DigitalTwinMotorDriver()
        motor = Motor(driver)

        await motor.connect()

        # Move to absolute position
        await motor.move_to(MotorType.ALTITUDE, steps=70000)

        # Move by relative offset
        await motor.move_by(MotorType.AZIMUTH, steps=1000)

        # Get status
        status = motor.get_status(MotorType.ALTITUDE)
        print(f"Position: {status.position_steps}")

        # Home all axes
        await motor.home_all()

        await motor.disconnect()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    from telescope_mcp.drivers.motors.types import (
        AvailableMotorController,
        MotorDriver,
        MotorInstance,
        MotorStatus,
        MotorType,
    )

logger = get_logger(__name__)

__all__ = [
    "Motor",
    "MotorConfig",
    "DeviceMotorInfo",
    "MotorDeviceStatus",
]


@dataclass
class MotorConfig:
    """Configuration for Motor device.

    Attributes:
        controller_id: Default controller to connect to (index or port path).
        default_speed: Default speed percentage for moves (1-100).
    """

    controller_id: int | str = 0
    default_speed: int = 100


class MotorDeviceStatus(TypedDict, total=False):
    """Status from Motor device layer.

    Attributes:
        connected: Whether motor controller is currently connected.
        type: Controller type string (e.g., "digital_twin", "serial").
        name: Human-readable controller name.
        connect_time: ISO timestamp when connection was established.
        altitude_position: Current altitude position in steps.
        azimuth_position: Current azimuth position in steps.
        altitude_moving: Whether altitude motor is moving.
        azimuth_moving: Whether azimuth motor is moving.
        is_open: Whether underlying driver connection is open.
        error: Error message if controller is in error state.
    """

    connected: bool
    type: str | None
    name: str | None
    connect_time: str | None
    altitude_position: int
    azimuth_position: int
    altitude_moving: bool
    azimuth_moving: bool
    is_open: bool
    error: str | None


@dataclass
class DeviceMotorInfo:
    """Device-layer motor controller metadata.

    Attributes:
        type: Controller type string (e.g., "digital_twin", "serial").
        name: Human-readable controller name.
        port: Connection port/path (e.g., "/dev/ttyACM0").
        firmware: Firmware version (if available).
        altitude_steps_per_degree: Conversion factor for altitude axis.
        azimuth_steps_per_degree: Conversion factor for azimuth axis.
        extra: Additional driver-specific metadata.
    """

    type: str
    name: str
    port: str | None = None
    firmware: str | None = None
    altitude_steps_per_degree: float = 1555.56  # 140000/90
    azimuth_steps_per_degree: float = 814.81  # 110000/135
    extra: dict[str, object] = field(default_factory=dict)


class Motor:
    """High-level async abstraction for telescope motor controller.

    Provides async interface to motor operations with driver injection
    for testing. Mirrors the Sensor device pattern for consistency.

    Example:
        async with Motor(driver) as motor:
            await motor.move_to(MotorType.ALTITUDE, 70000)
            status = motor.get_status(MotorType.ALTITUDE)
    """

    __slots__ = (
        "_driver",
        "_config",
        "_instance",
        "_info",
        "_connected",
        "_connect_time",
    )

    def __init__(
        self,
        driver: MotorDriver,
        config: MotorConfig | None = None,
    ) -> None:
        """Initialize Motor with driver and optional configuration.

        Creates motor device abstraction with specified driver for
        hardware communication. Does not connect to controller - call
        connect() separately.

        Args:
            driver: Motor driver for hardware communication.
            config: Optional configuration. Uses defaults if None.

        Returns:
            None

        Raises:
            None

        Example:
            >>> from telescope_mcp.drivers.motors import DigitalTwinMotorDriver
            >>> driver = DigitalTwinMotorDriver()
            >>> motor = Motor(driver)
            >>> await motor.connect()
        """
        self._driver = driver
        self._config = config or MotorConfig()
        self._instance: MotorInstance | None = None
        self._info: DeviceMotorInfo | None = None
        self._connected = False
        self._connect_time: datetime | None = None
        logger.info("Motor device initialized")

    @property
    def connected(self) -> bool:
        """Check whether motor controller is currently connected.

        Returns True only if both the connection flag is set and a valid
        controller instance exists.

        Returns:
            True if connected and instance available, False otherwise.

        Example:
            >>> if motor.connected:
            ...     await motor.move_to(MotorType.ALTITUDE, 70000)
        """
        return self._connected and self._instance is not None

    @property
    def info(self) -> DeviceMotorInfo | None:
        """Get detailed information about connected motor controller.

        Returns controller capabilities, conversion factors, and metadata.
        None if not connected.

        Returns:
            DeviceMotorInfo with controller type, name, conversion factors.
            None if motor not connected.

        Example:
            >>> await motor.connect()
            >>> print(f"Controller: {motor.info.name}")
        """
        return self._info

    def get_available_controllers(self) -> list[AvailableMotorController]:
        """Enumerate all controllers available through the configured driver.

        Discovers all motor controllers that can be opened with this driver.

        Returns:
            List of AvailableMotorController dicts with keys like 'id', 'name'.
            Empty list if no controllers found.

        Example:
            >>> motor = Motor(driver)
            >>> available = motor.get_available_controllers()
            >>> for c in available:
            ...     print(f"{c['id']}: {c['name']}")
        """
        return self._driver.get_available_controllers()

    async def connect(self, controller_id: int | str | None = None) -> None:
        """Connect to a motor controller and initialize for operation.

        Opens connection to specified controller (or default from config),
        retrieves device information. After successful connection,
        motor is ready for move operations.

        Args:
            controller_id: Controller ID to connect to. Can be int (index) or
                str (port path like "/dev/ttyACM0"). If None, uses
                default from MotorConfig.

        Returns:
            None

        Raises:
            RuntimeError: If already connected or connection fails.

        Example:
            >>> motor = Motor(driver)
            >>> await motor.connect()  # Uses default from config
            >>> # Or connect to specific controller:
            >>> await motor.connect("/dev/ttyACM0")
        """
        if self._connected:
            raise RuntimeError("Motor already connected. Call disconnect() first.")

        target_id = (
            controller_id if controller_id is not None else self._config.controller_id
        )
        logger.info("Connecting to motor controller", controller_id=target_id)

        try:
            loop = asyncio.get_running_loop()
            self._instance = await loop.run_in_executor(
                None, self._driver.open, target_id
            )
            self._connected = True
            self._connect_time = datetime.now(UTC)

            raw_info = self._instance.get_info()
            self._info = DeviceMotorInfo(
                type=str(raw_info.get("type", "unknown")),
                name=str(raw_info.get("name", "Unknown Controller")),
                port=str(raw_info.get("port")) if raw_info.get("port") else None,
                firmware=str(raw_info.get("firmware"))
                if raw_info.get("firmware")
                else None,
                altitude_steps_per_degree=float(
                    raw_info.get("altitude_steps_per_degree", 1555.56)
                ),
                azimuth_steps_per_degree=float(
                    raw_info.get("azimuth_steps_per_degree", 814.81)
                ),
                extra={
                    k: v
                    for k, v in raw_info.items()
                    if k
                    not in (
                        "type",
                        "name",
                        "port",
                        "firmware",
                        "altitude_steps_per_degree",
                        "azimuth_steps_per_degree",
                    )
                },
            )

            logger.info(
                "Motor controller connected",
                type=self._info.type,
                name=self._info.name,
            )

        except Exception as e:
            self._connected = False
            self._instance = None
            available = self._driver.get_available_controllers()
            available_msg = (
                ", ".join(c.get("name", str(c.get("id"))) for c in available) or "none"
            )
            msg = f"Failed to connect to motor controller {target_id}: {e}. "
            msg += f"Available: {available_msg}"
            raise RuntimeError(msg) from e

    async def disconnect(self) -> None:
        """Disconnect from motor controller and release hardware resources.

        Closes the controller connection and resets all state. Safe to call
        even if controller not connected.

        Returns:
            None

        Raises:
            None. Exceptions from driver.close() are logged as warnings.

        Example:
            >>> await motor.connect()
            >>> await motor.move_to(MotorType.ALTITUDE, 70000)
            >>> await motor.disconnect()
        """
        if self._instance is not None:
            try:
                self._instance.close()
            except Exception as e:
                logger.warning("Error closing motor controller", error=str(e))
            self._instance = None

        self._connected = False
        self._info = None
        self._connect_time = None
        logger.info("Motor controller disconnected")

    async def move_to(
        self,
        motor: MotorType,
        steps: int,
        speed: int | None = None,
    ) -> None:
        """Move motor to absolute step position.

        Commands the specified motor to move to an absolute position.
        Blocks until move completes.

        Args:
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            steps: Absolute target position in steps.
            speed: Speed percentage 1-100. Uses default from config if None.

        Returns:
            None. Blocks until move completes.

        Raises:
            RuntimeError: If motor not connected.
            ValueError: If steps outside motor's valid range.

        Example:
            >>> await motor.move_to(MotorType.ALTITUDE, 70000)  # 45° up
            >>> await motor.move_to(MotorType.AZIMUTH, 35000, speed=50)
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Motor not connected. Call connect() first.")

        effective_speed = speed if speed is not None else self._config.default_speed

        logger.info(
            "Moving motor",
            motor=motor.value,
            target_steps=steps,
            speed=effective_speed,
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._instance.move, motor, steps, effective_speed
        )

        logger.info("Move complete", motor=motor.value, position=steps)

    async def move_by(
        self,
        motor: MotorType,
        steps: int,
        speed: int | None = None,
    ) -> None:
        """Move motor by relative step offset from current position.

        Commands the specified motor to move by a relative number of steps.
        Blocks until move completes.

        Args:
            motor: Which motor to move (ALTITUDE or AZIMUTH).
            steps: Number of steps to move from current position.
            speed: Speed percentage 1-100. Uses default from config if None.

        Returns:
            None. Blocks until move completes.

        Raises:
            RuntimeError: If motor not connected.
            ValueError: If resulting position would exceed motor limits.

        Example:
            >>> await motor.move_by(MotorType.ALTITUDE, 1000)  # Nudge down
            >>> await motor.move_by(MotorType.AZIMUTH, -5000)  # Rotate CW
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Motor not connected. Call connect() first.")

        effective_speed = speed if speed is not None else self._config.default_speed

        logger.info(
            "Moving motor relative",
            motor=motor.value,
            steps=steps,
            speed=effective_speed,
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._instance.move_relative, motor, steps, effective_speed
        )

        logger.info("Relative move complete", motor=motor.value)

    async def stop(self, motor: MotorType | None = None) -> None:
        """Stop motor(s) immediately.

        Sends stop request to halt motor movement. Essential for safety.

        Args:
            motor: Motor to stop, or None for emergency stop all.

        Returns:
            None.

        Raises:
            RuntimeError: If motor not connected.

        Example:
            >>> await motor.stop(MotorType.ALTITUDE)
            >>> await motor.stop()  # Emergency stop all
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Motor not connected. Call connect() first.")

        logger.info("Stopping motor", motor=motor.value if motor else "all")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._instance.stop, motor)

    def get_status(self, motor: MotorType) -> MotorStatus:
        """Get current motor status synchronously.

        Returns the current position and movement state for the
        specified motor.

        Args:
            motor: Which motor to query (ALTITUDE or AZIMUTH).

        Returns:
            MotorStatus dataclass with position, is_moving, speed.

        Raises:
            RuntimeError: If motor not connected.

        Example:
            >>> status = motor.get_status(MotorType.ALTITUDE)
            >>> print(f"Position: {status.position_steps}")
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Motor not connected. Call connect() first.")

        return self._instance.get_status(motor)

    async def home(self, motor: MotorType) -> None:
        """Move motor to its configured home position.

        Moves the specified motor to its predefined home position.
        Blocks until home position reached.

        Args:
            motor: Which motor to home.

        Returns:
            None. Blocks until home position reached.

        Raises:
            RuntimeError: If motor not connected.

        Example:
            >>> await motor.home(MotorType.ALTITUDE)  # Park at zenith
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Motor not connected. Call connect() first.")

        logger.info("Homing motor", motor=motor.value)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._instance.home, motor)

        logger.info("Motor homed", motor=motor.value)

    async def home_all(self) -> None:
        """Home both altitude and azimuth motors to safe positions.

        Sequentially moves both motors to their configured home positions.
        Altitude homes first, then azimuth.

        Returns:
            None. Blocks until both motors reach home.

        Raises:
            RuntimeError: If motor not connected.

        Example:
            >>> await motor.home_all()  # Park telescope safely
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Motor not connected. Call connect() first.")

        logger.info("Homing all motors")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._instance.home_all)

        logger.info("All motors homed")

    async def zero_position(self, motor: MotorType) -> None:
        """Zero the position counter for a single motor axis.

        Sets the motor's internal position counter to 0 without any
        physical movement. Used to establish the current physical
        position as the reference origin.

        Args:
            motor: Which motor to zero (ALTITUDE or AZIMUTH).

        Returns:
            None. Position counter set to 0 immediately.

        Raises:
            RuntimeError: If motor not connected.

        Example:
            >>> await motor.zero_position(MotorType.ALTITUDE)
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Motor not connected. Call connect() first.")

        logger.info("Zeroing position", motor=motor.value)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._instance.zero_position, motor)

        logger.info("Position zeroed", motor=motor.value)

    async def set_home(self) -> None:
        """Zero both motor axes at current position (Set Home).

        Sets altitude and azimuth position counters to 0 without any
        physical movement. Called when user presses 'Set Home' on the
        dashboard to establish the current telescope position as (0,0)
        reference for the observing session.

        Business context: At the start of an observing session, the operator
        physically positions the telescope to a known reference point (e.g.,
        level, pointed north), then calls set_home to record that position
        as the zero reference. All subsequent position readouts will be
        relative to this home.

        Returns:
            None. Both axes read position 0 after this call.

        Raises:
            RuntimeError: If motor not connected.

        Example:
            >>> # Operator positions telescope manually, then:
            >>> await motor.set_home()
            >>> # Both axes now read 0
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Motor not connected. Call connect() first.")

        from telescope_mcp.drivers.motors.types import MotorType

        logger.info("Setting home - zeroing both axes")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._instance.zero_position, MotorType.ALTITUDE
        )
        await loop.run_in_executor(
            None, self._instance.zero_position, MotorType.AZIMUTH
        )

        logger.info("Home set - both axes zeroed")

    def get_device_status(self) -> MotorDeviceStatus:
        """Get comprehensive motor controller status for debugging.

        Returns a dictionary with connection state, device information,
        and positions of both motors.

        Returns:
            MotorDeviceStatus with connection state and motor positions.

        Example:
            >>> status = motor.get_device_status()
            >>> print(f"Connected: {status['connected']}")
            >>> print(f"Alt position: {status.get('altitude_position')}")
        """
        from telescope_mcp.drivers.motors.types import MotorType

        base_status: MotorDeviceStatus = {
            "connected": self._connected,
            "type": self._info.type if self._info else None,
            "name": self._info.name if self._info else None,
            "connect_time": self._connect_time.isoformat()
            if self._connect_time
            else None,
        }

        if self._connected and self._instance is not None:
            try:
                alt_status = self._instance.get_status(MotorType.ALTITUDE)
                az_status = self._instance.get_status(MotorType.AZIMUTH)

                base_status["altitude_position"] = alt_status.position_steps
                base_status["azimuth_position"] = az_status.position_steps
                base_status["altitude_moving"] = alt_status.is_moving
                base_status["azimuth_moving"] = az_status.is_moving
                base_status["is_open"] = self._instance.is_open
            except Exception as e:
                base_status["error"] = str(e)

        return base_status

    async def __aenter__(self) -> Motor:
        """Enter async context manager, connecting if not connected.

        Enables "async with motor:" syntax for automatic connection
        and cleanup.

        Returns:
            Self (Motor instance) for use in with block.

        Raises:
            RuntimeError: If connection fails.

        Example:
            >>> async with Motor(driver) as motor:
            ...     await motor.move_to(MotorType.ALTITUDE, 70000)
        """
        if not self._connected:
            await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Exit async context manager, disconnecting motor controller.

        Automatically disconnects when leaving "async with" block.

        Args:
            exc_type: Exception type if occurred, None otherwise.
            exc_val: Exception value if occurred, None otherwise.
            exc_tb: Exception traceback if occurred, None otherwise.

        Returns:
            None

        Example:
            >>> async with Motor(driver) as motor:
            ...     await motor.move_to(MotorType.ALTITUDE, 70000)
            # Auto-disconnects on exit
        """
        await self.disconnect()

    def __repr__(self) -> str:
        """Return string representation.

        Shows motor controller type and connection status.

        Returns:
            String like "Motor(type='digital_twin', connected=True)".

        Example:
            >>> print(repr(motor))
            Motor(type='digital_twin', connected=True)
        """
        if self._info:
            return f"Motor(type={self._info.type!r}, connected={self._connected})"
        return f"Motor(connected={self._connected})"
