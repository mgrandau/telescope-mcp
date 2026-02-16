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

        Creates sensor device abstraction with specified driver for
        hardware communication. Does not connect to sensor - call
        connect() separately.

        Args:
            driver: Sensor driver for hardware communication.
            config: Optional configuration. Uses defaults if None.

        Returns:
            None

        Raises:
            None

        Example:
            >>> from telescope_mcp.drivers.sensors import DigitalTwinSensorDriver
            >>> driver = DigitalTwinSensorDriver()
            >>> sensor = Sensor(driver)
            >>> await sensor.connect()
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
        """Check whether sensor is currently connected.

        Returns True only if both the connection flag is set and a valid
        sensor instance exists. Used to verify sensor is ready for readings.

        Business context: Essential guard before attempting sensor reads.
        Prevents errors from trying to read disconnected hardware. Common
        in UI status displays and health checks.

        Args:
            None

        Returns:
            True if sensor connected and instance available, False otherwise.

        Raises:
            None

        Example:
            >>> if sensor.connected:
            ...     reading = await sensor.read()
        """
        return self._connected and self._instance is not None

    @property
    def info(self) -> DeviceSensorInfo | None:
        """Get detailed information about connected sensor.

        Returns sensor capabilities (accelerometer, magnetometer, etc),
        sample rate, and driver-specific metadata. None if not connected.

        Business context: Used to verify sensor capabilities before operation.
        For example, checking has_magnetometer before attempting compass
        calibration, or confirming sample rate for time-sensitive operations.

        Args:
            None

        Returns:
            DeviceSensorInfo with sensor type, name, capabilities, sample rate,
            and extra metadata. None if sensor not connected.

        Raises:
            None

        Example:
            >>> await sensor.connect()
            >>> print(f"Sensor: {sensor.info.name}")
            >>> print(f"Sample rate: {sensor.info.sample_rate_hz} Hz")
        """
        return self._info

    @property
    def sample_rate_hz(self) -> float:
        """Get sensor sample rate in Hz (queried from device on connect).

        Returns the actual sample rate reported by the sensor hardware.
        Used to calculate timing for averaged reads and duration-based
        sampling.

        Business context: Critical for accurate time-series data collection.
        When reading sensor for a specific duration, sample rate determines
        how many samples to collect. Also important for power management
        (higher rates drain batteries faster).

        Args:
            None

        Returns:
            Sample rate in Hz (samples per second). Defaults to
            DEFAULT_SAMPLE_RATE_HZ if not yet queried from device.

        Raises:
            None

        Example:
            >>> await sensor.connect()
            >>> print(f"Sampling at {sensor.sample_rate_hz} Hz")
            >>> # Read for 1 second
            >>> samples = int(sensor.sample_rate_hz)
            >>> reading = await sensor.read(samples=samples)
        """
        return self._sample_rate_hz

    def get_available_sensors(self) -> list[AvailableSensor]:
        """Enumerate all sensors available through the configured driver.

        Discovers all sensors that can be opened with this driver instance.
        Useful for presenting a list of sensors to users or auto-discovering
        hardware.

        Business context: Essential for multi-sensor setups. Observatory may
        have multiple IMU sensors (primary, backup, calibration reference).
        This lists all available options before calling connect().

        Args:
            None

        Returns:
            List of AvailableSensor dicts with keys like 'id', 'name', 'port'.
            Empty list if no sensors found.

        Raises:
            None. Driver exceptions propagate to caller.

        Example:
            >>> sensor = Sensor(driver)
            >>> available = sensor.get_available_sensors()
            >>> for s in available:
            ...     print(f"{s['id']}: {s['name']}")
            0: Digital Twin Sensor
            1: Arduino BLE33 Sense
        """
        return self._driver.get_available_sensors()

    async def connect(self, sensor_id: int | str | None = None) -> None:
        """Connect to a sensor and initialize for reading.

        Opens connection to specified sensor (or default from config),
        queries sample rate, and retrieves device information. After
        successful connection, sensor is ready for read operations.

        Business context: First step in telescope sensor workflow. Must
        complete successfully before any orientation readings. Connection
        establishes communication and validates sensor is responsive.

        Args:
            sensor_id: Sensor ID to connect to. Can be int (index) or
                str (port path like "/dev/ttyACM0"). If None, uses
                default from SensorConfig.

        Returns:
            None

        Raises:
            RuntimeError: If sensor already connected or connection fails.
                Error message includes list of available sensors for debugging.

        Example:
            >>> sensor = Sensor(driver)
            >>> await sensor.connect()  # Uses default from config
            >>> # Or connect to specific sensor:
            >>> await sensor.connect("/dev/ttyACM1")
        """
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

        Tries multiple methods to get sample rate: protocol method
        (get_sample_rate), Arduino STATUS command parsing, or defaults.
        Sends STOP command before STATUS for Arduino sensors to ensure
        clean response. Falls back to DEFAULT_SAMPLE_RATE_HZ on any error.

        Args:
            None

        Returns:
            None. Sets self._sample_rate_hz internally.

        Raises:
            None. All exceptions caught and logged as warnings.

        Example:
            >>> # Internal use during connect()
            >>> await sensor._query_sample_rate()
            >>> print(f"Rate: {sensor._sample_rate_hz} Hz")
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
                # Resume streaming after STATUS query (issue #8)
                self._instance._send_command("START", wait_response=False)
            else:
                self._sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ
        except Exception as e:
            logger.warning("Failed to query sample rate", error=str(e))
            self._sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ

    async def disconnect(self) -> None:
        """Disconnect from sensor and release hardware resources.

        Closes the sensor connection and resets all state. Safe to call
        even if sensor not connected. After disconnect, must call connect()
        again before reading.

        Business context: Essential for clean shutdown and resource management.
        Releases serial ports or USB connections so other processes can access
        the sensor. Always disconnect before program exit to avoid leaving
        hardware in undefined state.

        Args:
            None

        Returns:
            None

        Raises:
            None. Exceptions from driver.close() are logged as warnings
            but don't propagate.

        Example:
            >>> await sensor.connect()
            >>> reading = await sensor.read()
            >>> await sensor.disconnect()
            >>> # Or use context manager for automatic cleanup
            >>> async with sensor:
            ...     reading = await sensor.read()
        """
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

        Business context: Core operation for telescope pointing. Single
        samples provide real-time orientation, while averaged samples
        reduce noise for precise positioning. Astrophotography typically
        uses 5-10 samples for stable star tracking.

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

        Example:
            >>> # Single reading
            >>> reading = await sensor.read()
            >>> print(f"ALT: {reading.altitude:.1f}°, AZ: {reading.azimuth:.1f}°")
            >>> # Averaged reading for stability
            >>> stable = await sensor.read(samples=10)
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")

        if samples < 1:
            raise ValueError("samples must be >= 1")

        if samples == 1:
            return await self._read_single()

        return await self._read_averaged(samples)

    def read_sync(self) -> SensorReading | None:
        """Read a single sensor value synchronously (blocking).

        Performs blocking I/O to read from sensor. Use in sync contexts
        or when wrapped in run_in_executor for async contexts.

        Business context: Used by SensorCoordinateProvider for coordinate
        injection during camera capture, where sync access is required.

        Returns:
            SensorReading with current sensor values, or None if not connected.

        Raises:
            None. Returns None if sensor not connected or read fails.

        Example:
            >>> reading = sensor.read_sync()
            >>> if reading:
            ...     print(f"ALT: {reading.altitude}, AZ: {reading.azimuth}")
        """
        if not self._connected or self._instance is None:
            return None
        try:
            return self._instance.read()
        except Exception as e:
            logger.warning("Sync read failed", error=str(e))
            return None

    async def _read_single(self) -> SensorReading:
        """Read a single sensor value asynchronously.

        Uses run_in_executor to wrap blocking driver read() call,
        making it compatible with async code without blocking the
        event loop.

        Business context: Foundation of all sensor reads. Wraps synchronous
        driver I/O in async executor pattern, essential for non-blocking
        telescope control systems.

        Args:
            None

        Returns:
            Single SensorReading from driver.

        Raises:
            Exceptions from driver.read() propagate.

        Example:
            >>> # Internal use by read(samples=1)
            >>> reading = await sensor._read_single()
        """
        assert self._instance is not None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._instance.read)

    async def _read_averaged(self, samples: int) -> SensorReading:
        """Collect multiple samples and return averaged reading.

        Reads samples sequentially, waiting between each based on sample rate
        to avoid overwhelming the sensor. Collects all samples into a list,
        then calls _average_readings().

        Args:
            samples: Number of samples to collect and average. Must be >= 1.

        Returns:
            Single averaged SensorReading from _average_readings().

        Raises:
            None directly. Exceptions from _read_single() propagate.

        Example:
            >>> # Internal use by read(samples=10)
            >>> averaged = await sensor._read_averaged(10)
        """
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
        """Compute component-wise average of multiple readings.

        Averages accelerometer and magnetometer vectors component-wise.
        Uses circular mean for azimuth to handle 0°/360° wraparound correctly.
        Altitude, temperature, and humidity are arithmetically averaged.

        Business context: Reduces noise in sensor readings. Multiple samples
        averaged together provide more stable orientation measurements,
        critical for accurate telescope pointing.

        Args:
            readings: List of SensorReading objects to average. Must contain
                at least one reading.

        Returns:
            Single SensorReading with averaged values. Uses timestamp from
            last reading. Sets raw_values to "averaged_N_samples".

        Raises:
            ValueError: If readings list is empty.

        Example:
            >>> readings = [sensor.read_sync() for _ in range(10)]
            >>> averaged = sensor._average_readings(readings)
            >>> # Less noisy than single reading
        """
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

        Calculates number of samples based on duration and sample rate,
        then collects and averages them. Convenient for time-based
        averaging without manual sample calculation.

        Business context: Used for timed calibrations and environment
        monitoring. "Read for 5 seconds" is more intuitive than calculating
        samples manually. Common in telescope auto-calibration sequences.

        Args:
            duration_ms: Collection time in milliseconds.

        Returns:
            SensorReading with averaged values over the duration.

        Raises:
            ValueError: If duration_ms < 1.
            RuntimeError: If sensor not connected.

        Example:
            >>> # Read for 2 seconds for stable average
            >>> reading = await sensor.read_for(duration_ms=2000)
            >>> # At 10 Hz, this averages ~20 samples
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

        Returns a dictionary with connection state, device information,
        sample rate, calibration status, and underlying driver status.
        Useful for troubleshooting sensor issues and verifying configuration.

        Business context: Essential for debugging telescope pointing issues.
        When coordinates are incorrect, status reveals if sensor is connected,
        calibrated, and reporting valid data. Also useful for system health
        monitoring in automated observatories.

        Args:
            None

        Returns:
            SensorDeviceStatus with connection state, sensor info,
            sample rate, calibration status, and driver status fields.

        Raises:
            None. Exceptions from driver.get_status() are caught and
            reported in the "status_error" field.

        Example:
            >>> sensor = Sensor(driver)
            >>> await sensor.connect()
            >>> status = sensor.get_status()
            >>> print(f"Connected: {status['connected']}")
            >>> print(f"Calibrated: {status.get('calibrated', False)}")
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
        """Enter async context manager, connecting if not connected.

        Enables "async with sensor:" syntax for automatic connection
        and cleanup. Connects if not already connected.

        Args:
            None

        Returns:
            Self (Sensor instance) for use in with block.

        Raises:
            RuntimeError: If connection fails (from connect()).

        Example:
            >>> async with Sensor(driver) as sensor:
            ...     reading = await sensor.read()
            >>> # Auto-disconnects on exit
        """
        if not self._connected:
            await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Exit async context manager, disconnecting sensor.

        Automatically disconnects sensor when leaving "async with" block,
        ensuring clean resource cleanup even if exceptions occur.

        Args:
            exc_type: Exception type if exception occurred, None otherwise.
            exc_val: Exception value if exception occurred, None otherwise.
            exc_tb: Exception traceback if exception occurred, None otherwise.

        Returns:
            None

        Raises:
            None. Disconnect errors are logged but not propagated.

        Example:
            >>> async with Sensor(driver) as sensor:
            ...     reading = await sensor.read()
            ...     # Exception here still triggers disconnect
        """
        await self.disconnect()

    def __repr__(self) -> str:
        """Return string representation.

        Shows sensor type, connection status, and sample rate if connected.
        Useful for debugging and logging.

        Args:
            None

        Returns:
            String like "Sensor(type='digital_twin', connected=True,
            sample_rate=10.0 Hz)" or "Sensor(connected=False)".

        Raises:
            None

        Example:
            >>> sensor = Sensor(driver)
            >>> print(repr(sensor))
            Sensor(connected=False)
            >>> await sensor.connect()
            >>> print(repr(sensor))
            Sensor(type='digital_twin', connected=True, sample_rate=10.0 Hz)
        """
        if self._info:
            return (
                f"Sensor(type={self._info.type!r}, connected={self._connected}, "
                f"sample_rate={self._sample_rate_hz} Hz)"
            )
        return f"Sensor(connected={self._connected})"
