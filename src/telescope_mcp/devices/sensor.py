"""Sensor device abstraction for telescope orientation sensing.

Provides high-level interface to IMU sensors that report telescope pointing
direction. Supports driver injection for testing with DigitalTwin or real
hardware with Arduino driver.

Key Components:
- Sensor: High-level device abstraction with driver injection
- SensorConfig: Configuration for sensor behavior
- DeviceSensorInfo: Device-layer sensor metadata

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

# Constants
DEFAULT_SAMPLE_RATE_HZ = 10.0  # Typical IMU sensor polling rate

__all__ = [
    "Sensor",
    "SensorConfig",
    "DeviceSensorInfo",
    "SensorDeviceStatus",
    "SensorStatistics",
    "DEFAULT_SAMPLE_RATE_HZ",
]


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


class SensorDeviceStatus(TypedDict, total=False):
    """Status from Sensor device layer.

    Keys:
        connected: Current connection state.
        type: Sensor type if connected.
        name: Sensor name if connected.
        connect_time: ISO timestamp of connection.
        read_count: Total successful readings.
        error_count: Total read errors.
        calibrated: From driver if connected.
        is_open: Connection state from driver.
        error: Error string from driver if problem detected.
        last_reading_age_ms: Milliseconds since last read (from driver).
        reading_rate_hz: Current sample rate (from driver).
        status_error: Error string if driver status query fails.
    """

    connected: bool
    type: str | None
    name: str | None
    connect_time: str | None
    read_count: int
    error_count: int
    calibrated: bool
    is_open: bool
    error: str | None
    last_reading_age_ms: float
    reading_rate_hz: float
    status_error: str


class SensorStatistics(TypedDict):
    """Usage statistics for Sensor device.

    Keys:
        read_count: Total successful sensor reads.
        error_count: Total failed read attempts.
        uptime_seconds: Seconds since connect(), None if never connected.
        error_rate: Ratio of errors to total reads.
    """

    read_count: int
    error_count: int
    uptime_seconds: float | None
    error_rate: float


@dataclass
class DeviceSensorInfo:
    """Device-layer sensor metadata (enriched from driver SensorInfo).

    This is the Sensor device's view of sensor information, distinct from
    the driver-level SensorInfo TypedDict in drivers/sensors/types.py.

    Attributes:
        type: Sensor type (e.g., "digital_twin", "arduino_ble33").
        name: Human-readable sensor name.
        has_accelerometer: Whether sensor has accelerometer.
        has_magnetometer: Whether sensor has magnetometer.
        has_temperature: Whether sensor has temperature sensing.
        has_humidity: Whether sensor has humidity sensing.
        sample_rate_hz: Sensor sample rate in Hz.
        port: Serial port (for hardware sensors).
        extra: Driver-specific metadata. Keys vary by driver:
            - DigitalTwin: simulation_id, target_position
            - Arduino: firmware_version, serial_number
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

        Creates a new Sensor device wrapper around a driver implementation.
        Sets up internal state for connection tracking, statistics collection,
        and reading history. Optionally auto-connects if configured.

        Business context: The Sensor class is the primary interface for
        telescope orientation data. It wraps hardware-specific drivers
        (Arduino, DigitalTwin) with a consistent API for MCP tools and
        automation scripts. Dependency injection enables testing and
        hardware abstraction.

        Implementation: Initializes tracking variables (_read_count,
        _error_count, _last_reading, _connect_time) to defaults. If
        config.auto_connect is True, calls connect() immediately which
        may raise RuntimeError if no sensor available.

        Args:
            driver: SensorDriver implementation (DigitalTwin or Arduino).
                Must implement get_available_sensors() and open() methods.
            config: Optional sensor configuration. If None, uses defaults
                (auto_connect=False, reconnect_on_error=True).

        Returns:
            None. Instance initialized but not connected unless auto_connect.

        Raises:
            RuntimeError: If auto_connect enabled and connection fails.

        Example:
            >>> driver = ArduinoSensorDriver()
            >>> config = SensorConfig(auto_connect=False)
            >>> sensor = Sensor(driver, config)
            >>> sensor.connected
            False
            >>> sensor.connect()
        """
        self._driver = driver
        self._config = config or SensorConfig()
        self._instance: SensorInstance | None = None
        self._info: DeviceSensorInfo | None = None
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
        """Check whether the sensor is currently connected and operational.

        Indicates if the sensor hardware is connected and the driver instance
        is active. This property should be checked before attempting read
        operations to avoid RuntimeError exceptions.

        Business context: Telescope orientation tracking requires continuous
        sensor connectivity. UI components and automation scripts use this
        property to display connection status and gate sensor operations.

        Args:
            No arguments (property accessor).

        Returns:
            bool: True if both the internal connected flag is set AND a valid
                driver instance exists. False if disconnected, connection
                failed, or instance was cleaned up.

        Raises:
            No exceptions raised. Safe to call in any state.

        Example:
            >>> sensor = Sensor(driver, config)
            >>> sensor.connected
            False
            >>> sensor.connect()
            >>> sensor.connected
            True
            >>> if sensor.connected:
            ...     reading = sensor.read()
        """
        return self._connected and self._instance is not None

    @property
    def info(self) -> DeviceSensorInfo | None:
        """Get detailed information about the connected sensor hardware.

        Returns cached sensor metadata populated during connect(). Includes
        sensor type, name, available measurement capabilities (accelerometer,
        magnetometer, temperature, humidity), sample rate, and port info.

        Business context: Sensor capabilities vary between hardware models.
        The Arduino Nano BLE33 Sense provides IMU and environmental sensors,
        while other devices may have different feature sets. This info enables
        UI to display appropriate controls and data fields.

        Args:
            No arguments (property accessor).

        Returns:
            DeviceSensorInfo: Dataclass with sensor metadata if connected:
                - type: Hardware type identifier (e.g., 'arduino_nano_ble33')
                - name: Human-readable sensor name
                - has_accelerometer, has_magnetometer: IMU capabilities
                - has_temperature, has_humidity: Environmental sensors
                - sample_rate_hz: Maximum polling rate
                - port: Serial port or connection identifier
                - extra: Additional driver-specific metadata
            None: If sensor is not connected.

        Raises:
            No exceptions raised. Returns None if not connected.

        Example:
            >>> sensor.connect()
            >>> info = sensor.info
            >>> print(f"Connected to {info.name} on {info.port}")
            Connected to Arduino Nano 33 BLE Sense on /dev/ttyACM0
            >>> if info.has_magnetometer:
            ...     # Enable compass features
            ...     pass
        """
        return self._info

    def get_available_sensors(self) -> list[AvailableSensor]:
        """Enumerate all sensors available through the configured driver.

        Queries the driver for discoverable sensor hardware. For serial-based
        sensors like Arduino, this scans available COM/tty ports. For the
        digital twin driver, returns simulated sensor definitions.

        Business context: Users may have multiple sensors connected (e.g.,
        orientation sensor and environmental monitor). This method enables
        device selection UI and auto-discovery workflows. Called by MCP
        tools to let clients enumerate available hardware.

        Returns:
            list[AvailableSensor]: List of sensor descriptors (TypedDict), each
                containing:
                - id (int | str): Unique sensor identifier for connect()
                - type (str): Sensor type (e.g., 'arduino_nano_ble33')
                - name (str): Human-readable name
                - port (str): Connection port or address
                Additional driver-specific fields may be included.
                Empty list if no sensors found.

        Raises:
            RuntimeError: If driver fails to enumerate (e.g., permission
                denied on serial ports, driver not initialized).

        Example:
            >>> sensor = Sensor(driver, config)
            >>> available = sensor.get_available_sensors()
            >>> for s in available:
            ...     print(f"{s['name']} on {s['port']}")
            Arduino Nano 33 BLE Sense on /dev/ttyACM0
            >>> sensor.connect(available[0]['id'])
        """
        return self._driver.get_available_sensors()

    def connect(self, sensor_id: int | str | None = None) -> None:
        """Connect to a sensor and initialize for reading.

        Opens connection to specified sensor (or first available), retrieves
        sensor info, and prepares for read operations. Must be called before
        read(), calibrate(), or other sensor operations.

        Business context: Telescope orientation tracking requires explicit
        sensor connection. This method handles device discovery, connection
        establishment, and capability detection. Supports both ID-based
        selection (for multi-sensor setups) and auto-discovery.

        Args:
            sensor_id: Specific sensor ID or port. Uses config default if None.
                Can be int (index from get_available_sensors) or str (port path).

        Returns:
            None. Connection state updated, sensor info populated.

        Raises:
            RuntimeError: If already connected or connection fails.

        Example:
            >>> sensor = Sensor(ArduinoSensorDriver())
            >>> available = sensor.get_available_sensors()
            >>> sensor.connect(available[0]['id'])  # Connect to first
            >>> sensor.connected
            True
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
            self._info = DeviceSensorInfo(
                type=str(raw_info.get("type", "unknown")),
                name=str(raw_info.get("name", "Unknown Sensor")),
                has_accelerometer=bool(raw_info.get("has_accelerometer", True)),
                has_magnetometer=bool(raw_info.get("has_magnetometer", True)),
                has_temperature=bool(raw_info.get("has_temperature", True)),
                has_humidity=bool(raw_info.get("has_humidity", True)),
                sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ,
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
            )

        except Exception as e:
            self._connected = False
            self._instance = None
            available = self._driver.get_available_sensors()
            available_msg = (
                ", ".join(s.get("name", str(s.get("id"))) for s in available) or "none"
            )
            raise RuntimeError(
                f"Failed to connect to sensor {target_id}: {e}. "
                f"Available: {available_msg}"
            ) from e

    def disconnect(self) -> None:
        """Disconnect from the sensor and release hardware resources.

        Closes the driver instance connection and clears internal state.
        Safe to call even if already disconnected or never connected.
        After disconnecting, connect() must be called before reading again.

        Business context: Proper resource cleanup is essential for shared
        hardware. Serial ports must be released for other applications.
        Called during session shutdown, error recovery, and when switching
        sensors.

        Args:
            No arguments.

        Returns:
            None. Connection state reset regardless of outcome.

        Raises:
            No exceptions raised. Errors during close are logged but
            suppressed to ensure cleanup completes.

        Example:
            >>> sensor.connect()
            >>> reading = sensor.read()
            >>> sensor.disconnect()  # Release serial port
            >>> sensor.connected
            False
            >>> # Safe to call again
            >>> sensor.disconnect()  # No-op if already disconnected
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
        self._last_reading = None  # Clear stale reading from previous session
        # Note: _read_count and _error_count are intentionally preserved across
        # reconnects to track cumulative session reliability, not per-connection stats.
        logger.info("Sensor disconnected")

    def read(self) -> SensorReading:
        """Read current sensor values from the connected hardware.

        Polls the sensor driver for current IMU and environmental data.
        Updates internal statistics (read_count) and caches the reading
        in last_reading. May trigger automatic reconnection if configured.

        Business context: Core sensor operation for telescope orientation.
        Returns calibrated position data for pointing calculations.
        Called by MCP tools and UI polling loops for real-time display.

        Args:
            No arguments. Uses current calibration state.

        Returns:
            SensorReading: Dataclass containing:
                - accelerometer: dict with aX, aY, aZ in g's
                - magnetometer: dict with mX, mY, mZ in µT
                - altitude: Calibrated altitude in degrees (0-90)
                - azimuth: Calibrated azimuth in degrees (0-360)
                - temperature: Ambient temperature in °C (if available)
                - humidity: Relative humidity % (if available)
                - timestamp: datetime of reading (UTC)
                - raw_values: Original sensor data string

        Raises:
            RuntimeError: If sensor not connected. Call connect() first.
            Exception: Driver-specific errors if read fails and
                reconnect_on_error is False.

        Example:
            >>> sensor.connect()
            >>> reading = sensor.read()
            >>> print(f"Alt {reading.altitude:.1f}°, Az {reading.azimuth:.1f}°")
            Alt 45.2°, Az 180.5°
            >>> print(f"Temp: {reading.temperature:.1f}°C")
            Temp: 22.3°C
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
        """Calibrate sensor to a known true telescope position.

        Sets calibration offsets so subsequent readings map to the provided
        true position. Point the telescope at a known object (plate-solved
        star, landmark) and call this with the true coordinates.

        Business context: IMU-derived positions contain systematic errors from
        mounting orientation and magnetic interference. Calibration against
        a plate-solved image or known star corrects these errors. Essential
        for accurate Go-To pointing. Should be performed at session start.

        Args:
            true_altitude: Known true altitude in degrees.
                Valid range: 0.0 (horizon) to 90.0 (zenith).
                Obtained from plate solving or star catalog.
            true_azimuth: Known true azimuth in degrees.
                Valid range: [0.0, 360.0) (360 excluded, wraps to 0).
                0° = North, 90° = East, 180° = South, 270° = West.

        Returns:
            None. Calibration offsets stored in driver instance.

        Raises:
            RuntimeError: If sensor not connected.
            ValueError: If altitude not 0-90 or azimuth not 0-360.

        Example:
            >>> # Point telescope at Polaris, plate solve shows (89.26, 0.0)
            >>> sensor.calibrate(89.26, 0.0)
            >>> # Now readings are calibrated to true position
            >>> reading = sensor.read()
            >>> print(f"Calibrated: {reading.altitude:.2f}, {reading.azimuth:.2f}")
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")

        # Validate ranges (azimuth uses < 360 to match driver protocol)
        if not 0 <= true_altitude <= 90:
            raise ValueError(f"Altitude must be in range [0, 90], got {true_altitude}")
        if not 0 <= true_azimuth < 360:
            raise ValueError(f"Azimuth must be in range [0, 360), got {true_azimuth}")

        logger.info(
            "Calibrating sensor",
            true_altitude=true_altitude,
            true_azimuth=true_azimuth,
        )

        self._instance.calibrate(true_altitude, true_azimuth)

    def reset(self) -> None:
        """Reset sensor to initial state clearing calibration.

        Clears all calibration data (tilt, magnetometer offsets) and
        reinitializes the sensor to factory defaults. Use when calibration
        has drifted or before starting fresh alignment procedure.

        Business context: Telescope alignment workflows may need to restart
        calibration if conditions change (temperature drift, magnetic
        interference). Reset provides clean slate without full reconnection.

        Implementation: Delegates to driver instance's reset() method which
        clears internal calibration state. Does not affect connection or
        collected statistics.

        Args:
            No arguments.

        Returns:
            None. Calibration cleared, sensor reinitialized.

        Raises:
            RuntimeError: If not connected.

        Example:
            >>> sensor.calibrate(45.0, 180.0)  # Set calibration
            >>> # ... calibration drifted ...
            >>> sensor.reset()  # Clear and start over
            >>> sensor.calibrate(45.2, 180.1)  # Recalibrate
        """
        if not self._connected or self._instance is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")

        self._instance.reset()
        logger.info("Sensor reset")

    def get_status(self) -> SensorDeviceStatus:
        """Get comprehensive sensor status including connection and statistics.

        Combines device-level metadata with driver-reported status to provide
        a complete picture of sensor health. Includes connection state,
        timing information, usage statistics, and calibration status.

        Business context: Telescope control systems need real-time health
        monitoring. This method powers status dashboards, diagnostic tools,
        and automated health checks. MCP tools expose this for remote
        monitoring of observatory equipment.

        Returns:
            SensorDeviceStatus: TypedDict containing:
                - connected (bool): Current connection state
                - type (str | None): Sensor type if connected
                - name (str | None): Sensor name if connected
                - connect_time (str | None): ISO timestamp of connection
                - read_count (int): Total successful readings
                - error_count (int): Total read errors
                - calibrated (bool): From driver if connected
                - is_open (bool): Connection state from driver
                - status_error (str): If driver status query fails

        Raises:
            No exceptions raised; errors captured in status_error field.

        Example:
            >>> status = sensor.get_status()
            >>> print(f"Connected: {status['connected']}")
            Connected: True
            >>> print(f"Reads: {status['read_count']}, Errors: {status['error_count']}")
            Reads: 1523, Errors: 2
            >>> if status.get('calibrated'):
            ...     print("Sensor is calibrated")
        """
        base_status: SensorDeviceStatus = {
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
                # Merge driver status into base status
                if "calibrated" in driver_status:
                    base_status["calibrated"] = driver_status["calibrated"]
                if "is_open" in driver_status:
                    base_status["is_open"] = driver_status["is_open"]
                if "error" in driver_status:
                    base_status["error"] = driver_status["error"]
                if "last_reading_age_ms" in driver_status:
                    base_status["last_reading_age_ms"] = driver_status[
                        "last_reading_age_ms"
                    ]
                if "reading_rate_hz" in driver_status:
                    base_status["reading_rate_hz"] = driver_status["reading_rate_hz"]
            except Exception as e:
                base_status["status_error"] = str(e)

        return base_status

    @property
    def last_reading(self) -> SensorReading | None:
        """Get the most recent sensor reading without polling hardware.

        Returns the cached result from the last successful read() call.
        Useful for accessing recent data without triggering new hardware
        communication, which is important for high-frequency UI updates.

        Business context: Telescope pointing displays need frequent updates
        but shouldn't overwhelm the sensor with requests. Components can
        poll last_reading for cached data while a background task performs
        actual sensor reads at the appropriate sample rate.

        Args:
            No arguments (property accessor).

        Returns:
            SensorReading: The most recent reading containing:
                - timestamp: When reading was taken
                - accelerometer: 3-axis acceleration (g)
                - magnetometer: 3-axis magnetic field (µT)
                - altitude, azimuth: Computed position (degrees)
                - temperature: Ambient temp (°C) if available
                - humidity: Relative humidity (%) if available
            None: If no readings have been taken since connection,
                or if never connected.

        Raises:
            No exceptions raised. Returns None if no cached reading.

        Example:
            >>> sensor.connect()
            >>> sensor.read()  # Populate cache
            >>> reading = sensor.last_reading
            >>> if reading:
            ...     print(f"Alt: {reading.altitude:.1f}° Az: {reading.azimuth:.1f}°")
            Alt: 45.2° Az: 180.5°
            >>> # Can access repeatedly without new hardware calls
            >>> same_reading = sensor.last_reading
        """
        return self._last_reading

    @property
    def statistics(self) -> SensorStatistics:
        """Get sensor usage and reliability statistics.

        Computes operational metrics including read counts, error rates,
        and connection uptime. Statistics accumulate across reconnections
        within the same Sensor instance for session-wide reliability tracking.

        Business context: Long-running observatory sessions need reliability
        monitoring. High error rates may indicate hardware issues, loose
        connections, or environmental interference. These metrics help
        diagnose problems and validate sensor health over time.

        Args:
            No arguments (property accessor).

        Returns:
            SensorStatistics: TypedDict containing:
                - read_count (int): Total successful sensor reads
                - error_count (int): Total failed read attempts
                - uptime_seconds (float | None): Seconds since connect(),
                    None if never connected
                - error_rate (float): Ratio of errors to total reads,
                    computed as error_count / max(1, read_count) to
                    avoid division by zero

        Raises:
            No exceptions raised. Safe to call in any state.

        Example:
            >>> sensor.connect()
            >>> for _ in range(100):
            ...     sensor.read()
            >>> stats = sensor.statistics
            >>> print(f"Uptime: {stats['uptime_seconds']:.0f}s")
            Uptime: 45s
            >>> print(f"Error rate: {stats['error_rate']:.1%}")
            Error rate: 0.0%
            >>> if stats['error_rate'] > 0.05:
            ...     print("Warning: High error rate detected")
        """
        uptime: float | None = None
        if self._connect_time:
            uptime = (datetime.now(UTC) - self._connect_time).total_seconds()

        return SensorStatistics(
            read_count=self._read_count,
            error_count=self._error_count,
            uptime_seconds=uptime,
            error_rate=self._error_count / max(1, self._read_count),
        )

    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to the sensor after connection failure.

        Tries multiple reconnection attempts with the configured sensor_id.
        Disconnects first, then reconnects. Called automatically by read()
        when reconnect_on_error is enabled and a read fails.

        Business context: Long-running telescope sessions may experience
        transient USB disconnects. Automatic reconnection maintains sensor
        data flow without operator intervention during overnight observations.

        Implementation: Loops max_reconnect_attempts times, calling
        disconnect() then connect(sensor_id) each attempt. Logs each
        attempt and result. Raises RuntimeError if all attempts fail.

        Args:
            No arguments. Uses _config.sensor_id and max_reconnect_attempts.

        Returns:
            None. Sensor reconnected on success.

        Raises:
            RuntimeError: If all reconnection attempts fail.

        Example:
            >>> # Called internally by read() on error
            >>> sensor._attempt_reconnect()
            >>> sensor.connected
            True
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
        """Enter context manager, connecting sensor if not already connected.

        Enables 'with' statement usage for automatic resource management.
        Connects to sensor on entry if not already connected, ensuring
        sensor is ready for operations within the context block.

        Implementation: Checks _connected flag and calls connect() only
        if needed. Returns self to allow 'as sensor' binding in with
        statement. Connection uses config defaults (sensor_id from config).

        Args:
            No arguments (context manager protocol).

        Returns:
            Sensor: Self reference for use in with statement binding.

        Raises:
            RuntimeError: If connection fails (no sensor available,
                port in use, or hardware error).

        Example:
            >>> with Sensor(driver, config) as sensor:
            ...     reading = sensor.read()
            ...     print(f"Altitude: {reading.altitude}")
            >>> # Automatically disconnected here

            # Exceptions propagate but cleanup still runs:
            >>> try:
            ...     with Sensor(driver) as sensor:
            ...         sensor.calibrate(-10, 0)  # ValueError
            ... except ValueError:
            ...     pass  # sensor.disconnect() was called
        """
        if not self._connected:
            self.connect()
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Context manager exit - disconnect sensor and release resources.

        Called automatically when exiting a 'with' block. Ensures proper
        cleanup regardless of whether an exception occurred.

        Args:
            exc_type: Exception type if an exception was raised, else None.
            exc_val: Exception instance if raised, else None.
            exc_tb: Traceback object if exception raised, else None.

        Returns:
            None. Does not suppress exceptions (returns None/False).

        Raises:
            No exceptions raised. Cleanup errors logged but suppressed.

        Example:
            >>> with Sensor(driver, config) as sensor:
            ...     reading = sensor.read()
            ...     process(reading)
            >>> # sensor.disconnect() called automatically here
            >>> sensor.connected
            False
        """
        self.disconnect()

    def __repr__(self) -> str:
        """Return string representation showing sensor type and connection state.

        Generates a developer-friendly string for debugging and logging.
        Includes sensor type (from info) when available, always shows
        connection status.

        Business context: Useful for debugging sensor issues in logs and
        REPL sessions. Quickly shows what type of sensor and whether
        it's ready for operations.

        Implementation: Checks if _info is populated (set during connect).
        If available, includes type field. Always includes _connected flag.
        Uses !r for proper string quoting of type value.

        Args:
            No arguments.

        Returns:
            str: Representation like "Sensor(type='arduino_ble33', connected=True)"
                or "Sensor(connected=False)" if not yet connected.

        Raises:
            No exceptions raised.

        Example:
            >>> sensor = Sensor(driver)
            >>> repr(sensor)
            "Sensor(connected=False)"
            >>> sensor.connect()
            >>> repr(sensor)
            "Sensor(type='arduino_ble33', connected=True, uptime=45s)"
        """
        if self._info:
            uptime = ""
            if self._connect_time:
                secs = (datetime.now(UTC) - self._connect_time).total_seconds()
                uptime = f", uptime={secs:.0f}s"
            return (
                f"Sensor(type={self._info.type!r}, connected={self._connected}{uptime})"
            )
        return f"Sensor(connected={self._connected})"
