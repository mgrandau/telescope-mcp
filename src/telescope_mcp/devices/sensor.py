"""Sensor device abstraction for telescope orientation sensing.

Provides high-level async interface to IMU sensors that report telescope
pointing direction. Supports driver injection for testing with DigitalTwin
or real hardware with Arduino driver.

Key Components:
- Sensor: High-level device abstraction with driver injection
- SensorConfig: Configuration for sensor behavior
- DeviceSensorInfo: Device-layer sensor metadata

Example:
    import asyncio
    from telescope_mcp.devices.sensor import Sensor
    from telescope_mcp.drivers.sensors import DigitalTwinSensorDriver

    async def main():
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        await sensor.connect()
        reading = await sensor.read()
        print(f"ALT {reading.altitude:.2f}, AZ {reading.azimuth:.2f}")

        # Read with averaging
        avg = await sensor.read(samples=10)

        # Or read for duration
        reading = await sensor.read_for(duration_ms=1000)

        sensor.calibrate(true_altitude=45.0, true_azimuth=180.0)
        await sensor.disconnect()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    from telescope_mcp.drivers.sensors import (
        AvailableSensor,
        SensorDriver,
        SensorInstance,
        SensorReading,
    )

logger = get_logger(__name__)

DEFAULT_SAMPLE_RATE_HZ = 10.0
STATUS_SETTLE_DELAY_SEC = 0.1  # Delay after STOP command before STATUS query
_SAMPLE_RATE_PATTERN = re.compile(r"Sample Rate:\s*(\d+)\s*Hz")

__all__ = [
    "Sensor",
    "SensorConfig",
    "DeviceSensorInfo",
    "SensorDeviceStatus",
    "DEFAULT_SAMPLE_RATE_HZ",
    "STATUS_SETTLE_DELAY_SEC",
]


@dataclass
class SensorConfig:
    """Configuration for Sensor device.

    Attributes:
        sensor_id: Default sensor to connect to (index or port path).
    """

    sensor_id: int | str = 0


class SensorDeviceStatus(TypedDict, total=False):
    """Status from Sensor device layer.

    Attributes:
        connected: Whether sensor is currently connected.
        type: Sensor type string (e.g., "digital_twin", "arduino_ble33").
        name: Human-readable sensor name.
        connect_time: ISO timestamp when connection was established.
        sample_rate_hz: Sensor sample rate in Hz.
        calibrated: Whether calibration offsets have been applied.
        is_open: Whether underlying driver connection is open.
        error: Error message if sensor is in error state.
        status_error: Error from get_status() call itself.
    """

    connected: bool
    type: str | None
    name: str | None
    connect_time: str | None
    sample_rate_hz: float
    calibrated: bool
    is_open: bool
    error: str | None
    status_error: str


@dataclass
class DeviceSensorInfo:
    """Device-layer sensor metadata.

    Attributes:
        type: Sensor type string (e.g., "digital_twin", "arduino_ble33").
        name: Human-readable sensor name.
        has_accelerometer: Whether sensor provides acceleration data.
        has_magnetometer: Whether sensor provides magnetic field data.
        has_temperature: Whether sensor provides temperature readings.
        has_humidity: Whether sensor provides humidity readings.
        sample_rate_hz: Sensor sample rate in Hz.
        port: Connection port/path (e.g., "/dev/ttyACM0").
        extra: Additional driver-specific metadata.
    """

    type: str
    name: str
    has_accelerometer: bool = True
    has_magnetometer: bool = True
    has_temperature: bool = True
    has_humidity: bool = True
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    port: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


class Sensor:
    """High-level async abstraction for telescope orientation sensor.

    Example:
        async with Sensor(driver) as sensor:
            reading = await sensor.read()
            avg = await sensor.read(samples=10)
            reading = await sensor.read_for(duration_ms=1000)
    """

    __slots__ = (
        "_driver",
        "_config",
        "_instance",
        "_info",
        "_connected",
        "_connect_time",
        "_sample_rate_hz",
    )

    def __init__(
        self,
        driver: SensorDriver,
        config: SensorConfig | None = None,
    ) -> None:
        """Initialize Sensor with driver and optional configuration.

        Args:
            driver: Sensor driver for hardware communication.
            config: Optional configuration. Uses defaults if None.
        """
        self._driver = driver
        self._config = config or SensorConfig()
        self._instance: SensorInstance | None = None
        self._info: DeviceSensorInfo | None = None
        self._connected = False
        self._connect_time: datetime | None = None
        self._sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
        logger.info("Sensor device initialized")

    @property
    def connected(self) -> bool:
        """Check whether sensor is currently connected."""
        return self._connected and self._instance is not None

    @property
    def info(self) -> DeviceSensorInfo | None:
        """Get detailed information about connected sensor."""
        return self._info

    @property
    def sample_rate_hz(self) -> float:
        """Get sensor sample rate in Hz (queried from device on connect)."""
        return self._sample_rate_hz

    def get_available_sensors(self) -> list[AvailableSensor]:
        """Enumerate all sensors available through the configured driver."""
        return self._driver.get_available_sensors()

    async def connect(self, sensor_id: int | str | None = None) -> None:
        """Connect to a sensor and initialize for reading."""
        if self._connected:
            raise RuntimeError("Sensor already connected. Call disconnect() first.")

        target_id = sensor_id if sensor_id is not None else self._config.sensor_id
        logger.info("Connecting to sensor", sensor_id=target_id)

        try:
            self._instance = self._driver.open(target_id)
            self._connected = True
            self._connect_time = datetime.now(UTC)

            await self._query_sample_rate()

            raw_info = self._instance.get_info()
            self._info = DeviceSensorInfo(
                type=str(raw_info.get("type", "unknown")),
                name=str(raw_info.get("name", "Unknown Sensor")),
                has_accelerometer=bool(raw_info.get("has_accelerometer", True)),
                has_magnetometer=bool(raw_info.get("has_magnetometer", True)),
                has_temperature=bool(raw_info.get("has_temperature", True)),
                has_humidity=bool(raw_info.get("has_humidity", True)),
                sample_rate_hz=self._sample_rate_hz,
                port=str(raw_info.get("port")) if raw_info.get("port") else None,
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
                sample_rate_hz=self._sample_rate_hz,
            )

        except Exception as e:
            self._connected = False
            self._instance = None
            available = self._driver.get_available_sensors()
            available_msg = (
                ", ".join(s.get("name", str(s.get("id"))) for s in available) or "none"
            )
            msg = f"Failed to connect to sensor {target_id}: {e}. "
            msg += f"Available: {available_msg}"
            raise RuntimeError(msg) from e

    async def _query_sample_rate(self) -> None:
        """Query sample rate from device STATUS response.

        Sends STOP command, waits for settle delay, then queries STATUS.
        Parses "Sample Rate: N Hz" from response.
        Falls back to DEFAULT_SAMPLE_RATE_HZ on any error.
        """
        if self._instance is None:
            return

        try:
            # Prefer protocol method (clean API) - DigitalTwin, Arduino
            if hasattr(self._instance, "get_sample_rate"):
                self._sample_rate_hz = self._instance.get_sample_rate()
            # Fall back to Arduino-specific STATUS command parsing
            elif hasattr(self._instance, "_send_command"):
                self._instance._send_command("STOP", wait_response=False)
                await asyncio.sleep(STATUS_SETTLE_DELAY_SEC)
                status_response = self._instance._send_command("STATUS", timeout=3.0)
                match = _SAMPLE_RATE_PATTERN.search(status_response)
                if match:
                    self._sample_rate_hz = float(match.group(1))
                    logger.debug("Parsed sample rate", rate=self._sample_rate_hz)
            else:
                self._sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ
        except Exception as e:
            logger.warning("Failed to query sample rate", error=str(e))
            self._sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ

    async def disconnect(self) -> None:
        """Disconnect from sensor and release hardware resources."""
        if self._instance is not None:
            try:
                self._instance.close()
            except Exception as e:
                logger.warning("Error closing sensor", error=str(e))
            self._instance = None

        self._connected = False
        self._info = None
        self._connect_time = None
        logger.info("Sensor disconnected")

    async def read(self, samples: int = 1) -> SensorReading:
        """Read sensor values with optional averaging.

        Reads one or more samples from the sensor and returns a single
        reading. When samples > 1, values are component-wise averaged.

        Args:
            samples: Number of readings to average. Default 1.

        Returns:
            SensorReading with:
                - accelerometer: {aX, aY, aZ} in g units
                - magnetometer: {mX, mY, mZ} in µT
                - altitude: Calculated altitude in degrees (0-90)
                - azimuth: Calculated azimuth in degrees (0-360)
                - temperature: Ambient temperature in Celsius
                - humidity: Relative humidity in %RH
                - timestamp: Reading timestamp (last sample for averaged)
                - raw_values: "averaged_N_samples" if samples > 1

        Raises:
            RuntimeError: If sensor not connected.
            ValueError: If samples < 1.
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")

        if samples < 1:
            raise ValueError("samples must be >= 1")

        if samples == 1:
            return await self._read_single()

        return await self._read_averaged(samples)

    async def _read_single(self) -> SensorReading:
        """Read a single sensor value asynchronously.

        Uses run_in_executor to wrap blocking driver read() call.
        """
        assert self._instance is not None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._instance.read)

    async def _read_averaged(self, samples: int) -> SensorReading:
        """Collect multiple samples and return averaged reading."""
        assert self._instance is not None
        interval_sec = 1.0 / self._sample_rate_hz
        readings: list[SensorReading] = []

        for _ in range(samples):
            reading = await self._read_single()
            readings.append(reading)
            if len(readings) < samples:
                await asyncio.sleep(interval_sec)

        return self._average_readings(readings)

    def _average_readings(self, readings: list[SensorReading]) -> SensorReading:
        """Compute component-wise average of multiple readings."""
        from telescope_mcp.drivers.sensors.types import (
            AccelerometerData,
            MagnetometerData,
        )
        from telescope_mcp.drivers.sensors.types import (
            SensorReading as SensorReadingType,
        )

        n = len(readings)
        if n == 0:
            raise ValueError("No readings to average")
        if n == 1:
            return readings[0]

        avg_accel: AccelerometerData = {
            "aX": sum(r.accelerometer["aX"] for r in readings) / n,
            "aY": sum(r.accelerometer["aY"] for r in readings) / n,
            "aZ": sum(r.accelerometer["aZ"] for r in readings) / n,
        }
        avg_mag: MagnetometerData = {
            "mX": sum(r.magnetometer["mX"] for r in readings) / n,
            "mY": sum(r.magnetometer["mY"] for r in readings) / n,
            "mZ": sum(r.magnetometer["mZ"] for r in readings) / n,
        }

        # Average azimuth using circular mean to handle 0°/360° wraparound
        az_rad = [math.radians(r.azimuth) for r in readings]
        avg_sin = sum(math.sin(a) for a in az_rad) / n
        avg_cos = sum(math.cos(a) for a in az_rad) / n
        avg_azimuth = math.degrees(math.atan2(avg_sin, avg_cos)) % 360

        return SensorReadingType(
            accelerometer=avg_accel,
            magnetometer=avg_mag,
            altitude=sum(r.altitude for r in readings) / n,
            azimuth=avg_azimuth,
            temperature=sum(r.temperature for r in readings) / n,
            humidity=sum(r.humidity for r in readings) / n,
            timestamp=readings[-1].timestamp,
            raw_values=f"averaged_{n}_samples",
        )

    async def read_for(self, duration_ms: int) -> SensorReading:
        """Read sensor for a time duration, averaging all samples.

        Args:
            duration_ms: Collection time in milliseconds.

        Returns:
            SensorReading with averaged values over the duration.

        Raises:
            ValueError: If duration_ms < 1.
            RuntimeError: If sensor not connected.
        """
        if duration_ms < 1:
            raise ValueError("duration_ms must be >= 1")

        samples = max(1, int(duration_ms / 1000.0 * self._sample_rate_hz))
        return await self.read(samples=samples)

    def calibrate(self, true_altitude: float, true_azimuth: float) -> None:
        """Calibrate sensor to a known true telescope position.

        Sets calibration offsets so subsequent readings match the true
        telescope position. Call when telescope is pointed at a known
        reference (star, landmark) with verified coordinates.

        Args:
            true_altitude: Known true altitude in degrees (0-90).
                Horizon=0, zenith=90.
            true_azimuth: Known true azimuth in degrees (0-360).
                North=0, East=90, South=180, West=270.

        Returns:
            None. Calibration applied to subsequent readings.

        Raises:
            RuntimeError: If sensor not connected.
            ValueError: If altitude not in [0, 90] or azimuth not in [0, 360).

        Example:
            >>> # After slewing to Polaris (alt=40°, az=0°)
            >>> sensor.calibrate(true_altitude=40.0, true_azimuth=0.0)
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")

        if not 0 <= true_altitude <= 90:
            raise ValueError(f"Altitude must be in range [0, 90], got {true_altitude}")
        if not 0 <= true_azimuth < 360:
            raise ValueError(f"Azimuth must be in range [0, 360), got {true_azimuth}")

        logger.info(
            "Calibrating sensor", true_altitude=true_altitude, true_azimuth=true_azimuth
        )
        self._instance.calibrate(true_altitude, true_azimuth)

    def get_status(self) -> SensorDeviceStatus:
        """Get comprehensive sensor status for debugging.

        Returns:
            SensorDeviceStatus with connection state, sensor info,
            sample rate, calibration status, and driver status fields.
        """
        base_status: SensorDeviceStatus = {
            "connected": self._connected,
            "type": self._info.type if self._info else None,
            "name": self._info.name if self._info else None,
            "connect_time": self._connect_time.isoformat()
            if self._connect_time
            else None,
            "sample_rate_hz": self._sample_rate_hz,
        }

        if self._connected and self._instance is not None:
            try:
                driver_status = self._instance.get_status()
                if (cal := driver_status.get("calibrated")) is not None:
                    base_status["calibrated"] = cal
                if (is_open := driver_status.get("is_open")) is not None:
                    base_status["is_open"] = is_open
                if (err := driver_status.get("error")) is not None:
                    base_status["error"] = err
            except Exception as e:
                base_status["status_error"] = str(e)

        return base_status

    async def __aenter__(self) -> Sensor:
        """Enter async context manager, connecting if not connected."""
        if not self._connected:
            await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Exit async context manager, disconnecting sensor."""
        await self.disconnect()

    def __repr__(self) -> str:
        """Return string representation."""
        if self._info:
            return (
                f"Sensor(type={self._info.type!r}, connected={self._connected}, "
                f"sample_rate={self._sample_rate_hz} Hz)"
            )
        return f"Sensor(connected={self._connected})"
