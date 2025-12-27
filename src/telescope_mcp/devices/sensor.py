"""Sensor device abstraction for telescope orientation sensing.

Provides high-level interface to IMU sensors that report telescope pointing
direction. Supports driver injection for testing with DigitalTwin or real
hardware with Arduino driver.

Key Components:
- Sensor: High-level device abstraction with driver injection
- SensorConfig: Configuration for sensor behavior
- SensorInfo: Sensor capabilities and properties

Architecture:
    The Sensor device follows a driver injection pattern:

    ┌─────────────────────────────────────────────────┐
    │                   Sensor                         │
    │  (high-level abstraction, business logic)       │
    └─────────────────────────────────────────────────┘
                          │
                          │ uses
                          ▼
    ┌─────────────────────────────────────────────────┐
    │              SensorDriver Protocol              │
    │   (interface for sensor implementations)        │
    └─────────────────────────────────────────────────┘
                    │                │
        ┌───────────┘                └───────────┐
        ▼                                        ▼
    ┌─────────────────┐              ┌─────────────────┐
    │ DigitalTwin     │              │  Arduino        │
    │ SensorDriver    │              │  SensorDriver   │
    │ (simulation)    │              │  (hardware)     │
    └─────────────────┘              └─────────────────┘

Example:
    from telescope_mcp.devices.sensor import Sensor
    from telescope_mcp.drivers.sensors import DigitalTwinSensorDriver

    # Create sensor with digital twin (for testing)
    driver = DigitalTwinSensorDriver()
    sensor = Sensor(driver)

    # Connect and read
    sensor.connect()
    reading = sensor.read()
    print(f"Pointing: ALT {reading.altitude:.2f}°, AZ {reading.azimuth:.2f}°")

    # Calibrate with known position
    sensor.calibrate(true_altitude=45.0, true_azimuth=180.0)

    # Disconnect when done
    sensor.disconnect()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    from telescope_mcp.drivers.sensors import (
        SensorDriver,
        SensorInstance,
        SensorReading,
    )

logger = get_logger(__name__)


@dataclass
class SensorConfig:
    """Configuration for Sensor device.

    Attributes:
        auto_connect: Automatically connect when Sensor is created.
        sensor_id: Specific sensor ID or port to connect to.
        reconnect_on_error: Attempt to reconnect on read errors.
        max_reconnect_attempts: Maximum reconnection attempts.
    """

    auto_connect: bool = False
    sensor_id: int | str = 0  # 0 = first available, or specific port string
    reconnect_on_error: bool = True
    max_reconnect_attempts: int = 3


@dataclass
class SensorInfo:
    """Information about a sensor device.

    Attributes:
        type: Sensor type (e.g., "digital_twin", "arduino_ble33").
        name: Human-readable sensor name.
        has_accelerometer: Whether sensor has accelerometer.
        has_magnetometer: Whether sensor has magnetometer.
        has_temperature: Whether sensor has temperature sensing.
        has_humidity: Whether sensor has humidity sensing.
        sample_rate_hz: Sensor sample rate in Hz.
        port: Serial port (for hardware sensors).
        extra: Additional driver-specific info.
    """

    type: str
    name: str
    has_accelerometer: bool = True
    has_magnetometer: bool = True
    has_temperature: bool = True
    has_humidity: bool = True
    sample_rate_hz: float = 10.0
    port: str | None = None
    extra: dict = field(default_factory=dict)


class Sensor:
    """High-level abstraction for telescope orientation sensor.

    Provides a unified interface to read sensor data regardless of
    the underlying hardware. Supports driver injection for testing
    with DigitalTwin or production with Arduino hardware.

    Example:
        # Using digital twin for testing
        from telescope_mcp.drivers.sensors import DigitalTwinSensorDriver
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        # Using real hardware
        from telescope_mcp.drivers.sensors import ArduinoSensorDriver
        driver = ArduinoSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()  # Connects to first available Arduino
    """

    def __init__(
        self,
        driver: SensorDriver,
        config: SensorConfig | None = None,
    ) -> None:
        """Initialize Sensor with driver and optional configuration.

        Args:
            driver: SensorDriver implementation (DigitalTwin or Arduino).
            config: Optional sensor configuration.
        """
        self._driver = driver
        self._config = config or SensorConfig()
        self._instance: SensorInstance | None = None
        self._info: SensorInfo | None = None
        self._connected = False
        self._connect_time: datetime | None = None
        self._last_reading: SensorReading | None = None
        self._read_count = 0
        self._error_count = 0

        logger.info("Sensor device initialized")

        if self._config.auto_connect:
            self.connect()

    @property
    def connected(self) -> bool:
        """Whether sensor is currently connected.

        Returns:
            True if connected and ready for reading.
        """
        return self._connected and self._instance is not None

    @property
    def info(self) -> SensorInfo | None:
        """Get sensor information.

        Returns:
            SensorInfo if connected, None otherwise.
        """
        return self._info

    def get_available_sensors(self) -> list[dict]:
        """List available sensors from the driver.

        Returns:
            List of sensor info dicts with id, type, name, port.
        """
        return self._driver.get_available_sensors()

    def connect(self, sensor_id: int | str | None = None) -> None:
        """Connect to a sensor.

        Args:
            sensor_id: Specific sensor ID or port. Uses config default if None.

        Raises:
            RuntimeError: If already connected or connection fails.
        """
        if self._connected:
            raise RuntimeError("Sensor already connected. Call disconnect() first.")

        target_id = sensor_id if sensor_id is not None else self._config.sensor_id

        logger.info("Connecting to sensor", sensor_id=target_id)

        try:
            self._instance = self._driver.open(target_id)
            self._connected = True
            self._connect_time = datetime.now(UTC)

            # Get sensor info
            raw_info = self._instance.get_info()
            self._info = SensorInfo(
                type=raw_info.get("type", "unknown"),
                name=raw_info.get("name", "Unknown Sensor"),
                has_accelerometer=raw_info.get("has_accelerometer", True),
                has_magnetometer=raw_info.get("has_magnetometer", True),
                has_temperature=raw_info.get("has_temperature", True),
                has_humidity=raw_info.get("has_humidity", True),
                sample_rate_hz=raw_info.get("sample_rate_hz", 10.0),
                port=raw_info.get("port"),
                extra={
                    k: v
                    for k, v in raw_info.items()
                    if k
                    not in (
                        "type",
                        "name",
                        "has_accelerometer",
                        "has_magnetometer",
                        "has_temperature",
                        "has_humidity",
                        "sample_rate_hz",
                        "port",
                    )
                },
            )

            logger.info(
                "Sensor connected",
                type=self._info.type,
                name=self._info.name,
            )

        except Exception as e:
            self._connected = False
            self._instance = None
            raise RuntimeError(f"Failed to connect to sensor: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from the sensor.

        Safe to call even if not connected.
        """
        if self._instance is not None:
            try:
                self._instance.close()
            except Exception as e:
                logger.warning("Error closing sensor", error=str(e))

            self._instance = None

        self._connected = False
        self._info = None
        logger.info("Sensor disconnected")

    def read(self) -> SensorReading:
        """Read current sensor values.

        Returns:
            SensorReading with accelerometer, magnetometer, position,
            temperature, and humidity data.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")

        try:
            reading = self._instance.read()
            self._last_reading = reading
            self._read_count += 1
            return reading

        except Exception as e:
            self._error_count += 1
            logger.error("Sensor read error", error=str(e))

            if self._config.reconnect_on_error:
                self._attempt_reconnect()
                # Retry after reconnect
                return self._instance.read()

            raise

    def calibrate(
        self,
        true_altitude: float,
        true_azimuth: float,
    ) -> None:
        """Calibrate sensor to known true position.

        Sets calibration transform so that current reading maps to
        the provided true position. Should be called when telescope
        is pointed at a known position (e.g., from plate solving).

        Args:
            true_altitude: Known true altitude in degrees (0-90).
            true_azimuth: Known true azimuth in degrees (0-360).

        Raises:
            RuntimeError: If not connected.
            ValueError: If position values out of range.
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")

        # Validate ranges
        if not 0 <= true_altitude <= 90:
            raise ValueError(f"Altitude must be 0-90°, got {true_altitude}")
        if not 0 <= true_azimuth <= 360:
            raise ValueError(f"Azimuth must be 0-360°, got {true_azimuth}")

        logger.info(
            "Calibrating sensor",
            true_altitude=true_altitude,
            true_azimuth=true_azimuth,
        )

        self._instance.calibrate(true_altitude, true_azimuth)

    def reset(self) -> None:
        """Reset sensor to initial state.

        Clears calibration and reinitializes sensor.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")

        self._instance.reset()
        logger.info("Sensor reset")

    def get_status(self) -> dict:
        """Get comprehensive sensor status.

        Returns:
            Dict with connection state, calibration, statistics, etc.
        """
        base_status = {
            "connected": self._connected,
            "type": self._info.type if self._info else None,
            "name": self._info.name if self._info else None,
            "connect_time": self._connect_time.isoformat()
            if self._connect_time
            else None,
            "read_count": self._read_count,
            "error_count": self._error_count,
        }

        if self._connected and self._instance is not None:
            try:
                driver_status = self._instance.get_status()
                base_status.update(driver_status)
            except Exception as e:
                base_status["status_error"] = str(e)

        return base_status

    @property
    def last_reading(self) -> SensorReading | None:
        """Get the most recent sensor reading.

        Returns:
            Last SensorReading or None if no readings taken.
        """
        return self._last_reading

    @property
    def statistics(self) -> dict:
        """Get sensor usage statistics.

        Returns:
            Dict with read count, error count, uptime, etc.
        """
        uptime = None
        if self._connect_time:
            uptime = (datetime.now(UTC) - self._connect_time).total_seconds()

        return {
            "read_count": self._read_count,
            "error_count": self._error_count,
            "uptime_seconds": uptime,
            "error_rate": self._error_count / max(1, self._read_count),
        }

    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to the sensor.

        Called automatically on read errors if reconnect_on_error is True.
        """
        sensor_id = self._config.sensor_id

        for attempt in range(self._config.max_reconnect_attempts):
            logger.info(
                "Attempting reconnect",
                attempt=attempt + 1,
                max_attempts=self._config.max_reconnect_attempts,
            )

            try:
                self.disconnect()
                self.connect(sensor_id)
                logger.info("Reconnect successful")
                return

            except Exception as e:
                logger.warning(
                    "Reconnect attempt failed",
                    attempt=attempt + 1,
                    error=str(e),
                )

        raise RuntimeError(
            f"Failed to reconnect after {self._config.max_reconnect_attempts} attempts"
        )

    def __enter__(self) -> Sensor:
        """Context manager entry - connect to sensor.

        Returns:
            Self for use in with statement.
        """
        if not self._connected:
            self.connect()
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Context manager exit - disconnect from sensor."""
        self.disconnect()

    def __repr__(self) -> str:
        """String representation of Sensor.

        Returns:
            String showing connection state and sensor type.
        """
        if self._info:
            return f"Sensor(type={self._info.type!r}, connected={self._connected})"
        return f"Sensor(connected={self._connected})"
