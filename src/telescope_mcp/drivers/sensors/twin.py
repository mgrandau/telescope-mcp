"""Digital twin sensor driver for testing without hardware.

Provides simulated IMU sensor responses for development and testing.
Supports configurable position, noise, drift, and calibration behavior.

Example:
    from telescope_mcp.drivers.sensors import DigitalTwinSensorDriver

    driver = DigitalTwinSensorDriver()
    instance = driver.open()

    data = instance.read()
    print(f"ALT: {data['altitude']:.2f}°, AZ: {data['azimuth']:.2f}°")
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from telescope_mcp.drivers.sensors.types import SensorReading
from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class DigitalTwinSensorConfig:
    """Configuration for digital twin sensor behavior.

    Attributes:
        initial_altitude: Starting altitude in degrees (0-90).
        initial_azimuth: Starting azimuth in degrees (0-360).
        noise_std_alt: Standard deviation of altitude noise (degrees).
        noise_std_az: Standard deviation of azimuth noise (degrees).
        drift_rate_alt: Altitude drift rate (degrees/hour).
        drift_rate_az: Azimuth drift rate (degrees/hour).
        temperature: Simulated temperature in Celsius.
        humidity: Simulated humidity in %RH.
        sample_rate_hz: Simulated sensor sample rate.
    """

    initial_altitude: float = 45.0
    initial_azimuth: float = 180.0
    noise_std_alt: float = 0.1  # degrees
    noise_std_az: float = 0.2  # degrees
    drift_rate_alt: float = 0.0  # degrees per hour
    drift_rate_az: float = 0.0  # degrees per hour
    temperature: float = 20.0  # Celsius
    humidity: float = 50.0  # %RH
    sample_rate_hz: float = 10.0


class DigitalTwinSensorInstance:
    """Simulated sensor instance for testing.

    Provides sensor readings with configurable noise, drift, and
    environmental data. Simulates BLE33 Sense IMU behavior.
    """

    def __init__(self, config: DigitalTwinSensorConfig) -> None:
        """Initialize simulated sensor with configuration.

        Creates a digital twin sensor that simulates IMU behavior including
        configurable noise, drift, and environmental readings.

        Business context: Digital twin enables development and testing
        without physical Arduino hardware. Simulates realistic sensor
        behavior including noise for algorithm testing.

        Implementation: Stores config, initializes position to config
        defaults, sets calibration transforms to identity (scale=1, offset=0),
        marks as open.

        Args:
            config: Sensor behavior configuration including noise levels,
                initial position, and sample rate.

        Returns:
            None. Instance ready for get_info(), read(), etc.

        Raises:
            No exceptions raised.

        Example:
            >>> config = DigitalTwinSensorConfig(noise_std_alt=0.1)
            >>> instance = DigitalTwinSensorInstance(config)
            >>> reading = instance.read()
        """
        self._config = config
        self._start_time = time.monotonic()

        # Current simulated position (can be set externally)
        self._true_altitude = config.initial_altitude
        self._true_azimuth = config.initial_azimuth

        # Calibration transform (applied to readings)
        self._cal_alt_scale = 1.0
        self._cal_alt_offset = 0.0
        self._cal_az_scale = 1.0
        self._cal_az_offset = 0.0

        # Magnetometer calibration offsets
        self._mag_offset_x = 0.0
        self._mag_offset_y = 0.0
        self._mag_offset_z = 0.0

        self._is_open = True

        logger.info(
            "Digital twin sensor initialized",
            altitude=self._true_altitude,
            azimuth=self._true_azimuth,
        )

    def get_info(self) -> dict:
        """Get sensor hardware information and capabilities.

        Returns metadata describing this digital twin sensor including
        supported measurement types, sample rates, and noise parameters.
        Used by the Sensor device layer to populate SensorInfo.

        Business context: Different sensor hardware has varying capabilities.
        The digital twin simulates a full-featured IMU sensor (accelerometer,
        magnetometer) plus environmental sensors. Noise parameters are exposed
        to help developers understand simulation accuracy.

        Returns:
            dict: Sensor metadata containing:
                - type (str): Always 'digital_twin'
                - name (str): Human-readable sensor name
                - has_accelerometer (bool): True - 3-axis supported
                - has_magnetometer (bool): True - 3-axis supported
                - has_temperature (bool): True - ambient temp available
                - has_humidity (bool): True - relative humidity available
                - sample_rate_hz (float): Configured sample rate
                - noise_std_alt (float): Altitude noise standard deviation
                - noise_std_az (float): Azimuth noise standard deviation

        Raises:
            No exceptions raised.

        Example:
            >>> instance = driver.open()
            >>> info = instance.get_info()
            >>> print(f"Sensor type: {info['type']}")
            Sensor type: digital_twin
            >>> print(f"Noise: ±{info['noise_std_alt']:.2f}° altitude")
            Noise: ±0.10° altitude
        """
        return {
            "type": "digital_twin",
            "name": "Digital Twin IMU Sensor",
            "has_accelerometer": True,
            "has_magnetometer": True,
            "has_temperature": True,
            "has_humidity": True,
            "sample_rate_hz": self._config.sample_rate_hz,
            "noise_std_alt": self._config.noise_std_alt,
            "noise_std_az": self._config.noise_std_az,
        }

    def read(self) -> SensorReading:
        """Read simulated sensor values with noise and drift.

        Generates sensor reading based on true position plus configured
        noise and drift effects. Simulates realistic IMU behavior including
        accelerometer/magnetometer physics and environmental sensors.

        Business context: Enables testing of pointing algorithms, calibration
        workflows, and UI without hardware. Configurable noise and drift
        parameters allow testing edge cases and error conditions.

        Args:
            No arguments.

        Returns:
            SensorReading: Dataclass containing simulated:
                - accelerometer: dict with aX, aY, aZ derived from altitude
                - magnetometer: dict with mX, mY, mZ derived from azimuth
                - altitude: Position with noise/drift/calibration applied
                - azimuth: Position with noise/drift/calibration applied
                - temperature: Configured temp with small noise
                - humidity: Configured humidity with small noise
                - timestamp: Current UTC datetime
                - raw_values: Tab-separated string matching Arduino format

        Raises:
            RuntimeError: If sensor has been closed.

        Example:
            >>> twin = driver.open()
            >>> twin.set_position(45.0, 180.0)
            >>> reading = twin.read()
            >>> assert abs(reading.altitude - 45.0) < 1.0  # Within noise
        """
        if not self._is_open:
            raise RuntimeError("Sensor is closed")

        # Calculate drift based on elapsed time
        elapsed_hours = (time.monotonic() - self._start_time) / 3600.0
        drift_alt = self._config.drift_rate_alt * elapsed_hours
        drift_az = self._config.drift_rate_az * elapsed_hours

        # Add noise to true position
        noise_alt = random.gauss(0, self._config.noise_std_alt)
        noise_az = random.gauss(0, self._config.noise_std_az)

        raw_altitude = self._true_altitude + drift_alt + noise_alt
        raw_azimuth = (self._true_azimuth + drift_az + noise_az) % 360

        # Apply calibration transform
        calibrated_alt = self._cal_alt_scale * raw_altitude + self._cal_alt_offset
        calibrated_az = (self._cal_az_scale * raw_azimuth + self._cal_az_offset) % 360

        # Generate accelerometer values from altitude
        # At altitude=0, gravity is in Y direction
        # At altitude=90, gravity is in X direction
        alt_rad = math.radians(raw_altitude)
        ax = math.sin(alt_rad) + random.gauss(0, 0.01)
        ay = math.cos(alt_rad) * random.gauss(0.95, 0.01)
        az = random.gauss(0, 0.05)

        # Generate magnetometer values from azimuth
        # Simplified: assume horizontal field, magnetic north = 0°
        az_rad = math.radians(raw_azimuth)
        field_strength = 45.0  # µT typical
        mx = field_strength * math.cos(az_rad) + random.gauss(0, 1.0)
        my = field_strength * math.sin(az_rad) + random.gauss(0, 1.0)
        mz = random.gauss(30, 2.0)  # Vertical component

        # Apply mag calibration
        mx -= self._mag_offset_x
        my -= self._mag_offset_y
        mz -= self._mag_offset_z

        # Environmental noise
        temp = self._config.temperature + random.gauss(0, 0.1)
        hum = self._config.humidity + random.gauss(0, 0.5)

        # Build raw string (matching Arduino output format)
        raw = (
            f"{ax:.2f}\t{ay:.2f}\t{az:.2f}\t"
            f"{mx:.2f}\t{my:.2f}\t{mz:.2f}\t{temp:.2f}\t{hum:.2f}"
        )

        return SensorReading(
            accelerometer={"aX": ax, "aY": ay, "aZ": az},
            magnetometer={"mX": mx, "mY": my, "mZ": mz},
            altitude=calibrated_alt,
            azimuth=calibrated_az,
            temperature=temp,
            humidity=hum,
            timestamp=datetime.now(UTC),
            raw_values=raw,
        )

    def _set_position(self, altitude: float, azimuth: float) -> None:
        """Set the true simulated telescope position for testing.

        Directly sets the underlying 'true' position that the digital twin
        simulates. Subsequent read() calls will return this position plus
        configured noise and drift. This is the primary method for driving
        the simulation in tests.

        Business context: Integration tests need deterministic sensor behavior.
        This method allows tests to simulate telescope movement by setting
        exact positions, then verify that calibration and reading logic
        works correctly with known ground truth.

        Args:
            altitude: True altitude in degrees. Valid range 0-90 for
                typical telescope orientations, but accepts any float.
                Represents angle above horizon.
            azimuth: True azimuth in degrees. Will be normalized to 0-360
                via modulo. 0° = North, 90° = East, 180° = South.

        Returns:
            None. Position is stored internally.

        Raises:
            No exceptions raised. Out-of-range values accepted for
            edge case testing.

        Example:
            >>> twin = driver.open()
            >>> twin.set_position(45.0, 180.0)  # Point south, 45° up
            >>> reading = twin.read()
            >>> assert abs(reading.altitude - 45.0) < 1.0  # Within noise
            >>> twin.set_position(0.0, 0.0)  # Horizon, north
        """
        self._true_altitude = altitude
        self._true_azimuth = azimuth % 360
        logger.debug(
            "Digital twin position set",
            altitude=altitude,
            azimuth=azimuth,
        )

    def calibrate(
        self,
        true_altitude: float,
        true_azimuth: float,
    ) -> None:
        """Calibrate sensor to match a known true position.

        Computes offset values so that the current sensor reading maps
        to the provided true position. Uses a simple offset model:
        corrected = raw + offset. Calibration persists until reset().

        Business context: Sensor readings contain systematic errors from
        mounting orientation and magnetic interference. Calibration against
        a known reference (plate-solved image, bright star) corrects these
        errors. Critical for accurate Go-To pointing.

        Args:
            true_altitude: Known true altitude in degrees (0-90).
                Obtained from plate solving, star catalog lookup,
                or other authoritative source.
            true_azimuth: Known true azimuth in degrees (0-360).
                0° = North, 90° = East, 180° = South, 270° = West.

        Returns:
            None. Calibration offsets stored internally.

        Raises:
            RuntimeError: If sensor is closed (in twin, always open).

        Example:
            >>> twin = driver.open()
            >>> # Point telescope at known star
            >>> twin.calibrate(45.0, 180.0)  # Calibrate to Polaris
            >>> reading = twin.read()
            >>> assert abs(reading.altitude - 45.0) < 0.5
        """
        # Get current raw reading
        reading = self.read()

        # Calculate offsets (simple offset model)
        self._cal_alt_offset = true_altitude - reading.altitude
        self._cal_az_offset = true_azimuth - reading.azimuth

        logger.info(
            "Sensor calibrated",
            alt_offset=self._cal_alt_offset,
            az_offset=self._cal_az_offset,
        )

    def _calibrate_magnetometer(self) -> dict:
        """Simulate magnetometer hard-iron calibration.

        Generates random calibration offsets to simulate the magnetometer
        calibration process. In real hardware, this would involve rotating
        the sensor to collect samples and computing center offsets.

        Business context: Magnetometer readings suffer from hard-iron
        distortion caused by nearby ferrous materials. Calibration computes
        offsets to center the magnetometer data sphere. The digital twin
        simulates this for testing calibration workflows.

        Args:
            No arguments required.

        Returns:
            dict: Calibration offsets containing:
                - offset_x (float): X-axis offset in µT (typically -5 to +5)
                - offset_y (float): Y-axis offset in µT
                - offset_z (float): Z-axis offset in µT
            These values are randomly generated for simulation.

        Raises:
            No exceptions raised.

        Example:
            >>> twin = driver.open()
            >>> offsets = twin.calibrate_magnetometer()
            >>> print(f"X offset: {offsets['offset_x']:.1f} µT")
            X offset: 2.3 µT
            >>> # Subsequent reads use calibrated values
        """
        # Simulate collecting samples and finding center
        self._mag_offset_x = random.gauss(0, 5)
        self._mag_offset_y = random.gauss(0, 5)
        self._mag_offset_z = random.gauss(0, 5)

        logger.info(
            "Magnetometer calibrated",
            offset_x=self._mag_offset_x,
            offset_y=self._mag_offset_y,
            offset_z=self._mag_offset_z,
        )

        return {
            "offset_x": self._mag_offset_x,
            "offset_y": self._mag_offset_y,
            "offset_z": self._mag_offset_z,
        }

    def reset(self) -> None:
        """Reset sensor to initial uncalibrated state.

        Clears all calibration offsets (position and magnetometer) and
        restarts the uptime timer. Simulates power-cycling the sensor.

        Business context: Testing calibration workflows requires ability
        to return to uncalibrated state. Also useful for simulating
        sensor reboot scenarios.

        Args:
            No arguments.

        Returns:
            None. All calibration cleared, timer restarted.

        Raises:
            No exceptions raised. Safe to call in any state.

        Example:
            >>> twin.calibrate(45.0, 180.0)
            >>> assert twin.get_status()['calibrated'] is True
            >>> twin.reset()
            >>> assert twin.get_status()['calibrated'] is False
        """
        self._cal_alt_offset = 0.0
        self._cal_az_offset = 0.0
        self._cal_alt_scale = 1.0
        self._cal_az_scale = 1.0
        self._mag_offset_x = 0.0
        self._mag_offset_y = 0.0
        self._mag_offset_z = 0.0
        self._start_time = time.monotonic()

        logger.info("Sensor reset")

    def get_status(self) -> dict:
        """Get current sensor status and calibration state.

        Returns operational status including connection state, calibration
        status for both position and magnetometer, and runtime information.
        Used by Sensor.get_status() to build comprehensive device status.

        Business context: Remote observatory monitoring requires real-time
        sensor health data. This status is exposed via MCP tools for
        dashboards and automated health checks. Calibration state is
        particularly important for assessing position accuracy.

        Returns:
            dict: Status dictionary containing:
                - connected (bool): True if sensor is open
                - type (str): Always 'digital_twin'
                - calibrated (bool): True if position calibration applied
                - mag_calibrated (bool): True if magnetometer calibrated
                - sample_rate_hz (float): Configured polling rate
                - uptime_seconds (float): Seconds since initialization

        Raises:
            No exceptions raised.

        Example:
            >>> instance = driver.open()
            >>> status = instance.get_status()
            >>> print(f"Connected: {status['connected']}")
            Connected: True
            >>> print(f"Calibrated: {status['calibrated']}")
            Calibrated: False
            >>> instance.calibrate(45.0, 180.0)
            >>> assert instance.get_status()['calibrated'] is True
        """
        return {
            "connected": self._is_open,
            "type": "digital_twin",
            "calibrated": self._cal_alt_offset != 0 or self._cal_az_offset != 0,
            "mag_calibrated": (
                self._mag_offset_x != 0
                or self._mag_offset_y != 0
                or self._mag_offset_z != 0
            ),
            "sample_rate_hz": self._config.sample_rate_hz,
            "uptime_seconds": time.monotonic() - self._start_time,
        }

    def close(self) -> None:
        """Close the simulated sensor connection.

        Marks sensor as closed. Subsequent read() calls will raise
        RuntimeError. Safe to call multiple times.

        Business context: Matches real sensor lifecycle for consistent
        testing. Closed state prevents reads, simulating disconnected
        hardware.

        Args:
            No arguments.

        Returns:
            None. Sensor marked as closed.

        Raises:
            No exceptions raised.

        Example:
            >>> twin = driver.open()
            >>> twin.close()
            >>> twin.read()  # Raises RuntimeError
        """
        self._is_open = False
        logger.info("Digital twin sensor closed")


class DigitalTwinSensorDriver:
    """Digital twin sensor driver for testing without hardware.

    Creates simulated sensor instances with configurable behavior
    for development and testing.

    Example:
        driver = DigitalTwinSensorDriver()
        instance = driver.open()
        reading = instance.read()
        print(f"ALT: {reading.altitude:.2f}°")
    """

    def __init__(self, config: DigitalTwinSensorConfig | None = None) -> None:
        """Initialize driver with optional configuration.

        Sets up driver with simulation configuration. Default config
        provides reasonable noise and behavior for testing.

        Business context: Driver manages digital twin sensor lifecycle.
        Configuration allows customizing simulation behavior for different
        test scenarios.

        Implementation: Stores config (or creates default), initializes
        _instance to None. No simulation runs until open() called.

        Args:
            config: Sensor behavior configuration. Uses defaults if None,
                including 0.1° noise, 45° initial altitude/azimuth.

        Returns:
            None. Driver ready for get_available_sensors() or open().

        Raises:
            No exceptions raised.

        Example:
            >>> driver = DigitalTwinSensorDriver()
            >>> instance = driver.open()
            >>> reading = instance.read()
        """
        self._config = config or DigitalTwinSensorConfig()
        self._instance: DigitalTwinSensorInstance | None = None

    def get_available_sensors(self) -> list[dict]:
        """List available sensors from this driver.

        For the digital twin driver, always returns a single simulated
        sensor. This matches the SensorDriver protocol interface and
        allows the digital twin to be used interchangeably with real
        hardware drivers.

        Business context: The digital twin enables development and testing
        without physical hardware. By implementing the same interface as
        real drivers, it allows full system testing including device
        enumeration workflows.

        Returns:
            list[dict]: Single-element list containing sensor descriptor:
                - id (int): Always 0
                - type (str): Always 'digital_twin'
                - name (str): Human-readable name
                - port (str): Always 'simulated'

        Raises:
            No exceptions raised.

        Example:
            >>> driver = DigitalTwinSensorDriver()
            >>> sensors = driver.get_available_sensors()
            >>> print(f"Found {len(sensors)} sensor(s)")
            Found 1 sensor(s)
            >>> print(sensors[0]['name'])
            Digital Twin IMU Sensor
        """
        return [
            {
                "id": 0,
                "type": "digital_twin",
                "name": "Digital Twin IMU Sensor",
                "port": "simulated",
            }
        ]

    def open(self, sensor_id: int | str = 0) -> DigitalTwinSensorInstance:
        """Open a simulated digital twin sensor instance.

        Creates a new DigitalTwinSensorInstance with the configured
        simulation parameters. Only one instance can be open at a time.

        Business context: Enables development and testing without physical
        hardware. The digital twin simulates realistic sensor behavior
        including noise, drift, and calibration. Essential for CI/CD
        pipelines and offline development.

        Args:
            sensor_id: Sensor identifier. Ignored for digital twin since
                only one simulated sensor exists. Accepts int or str for
                API compatibility with real drivers.

        Returns:
            DigitalTwinSensorInstance: Simulated sensor instance ready for
                reading. Provides read(), calibrate(), get_status() and
                other sensor operations.

        Raises:
            RuntimeError: If a sensor instance is already open.
                Call close() before opening again.

        Example:
            >>> driver = DigitalTwinSensorDriver()
            >>> instance = driver.open()
            >>> reading = instance.read()
            >>> print(f"Altitude: {reading.altitude:.1f}°")
            Altitude: 45.0°
            >>> driver.close()
        """
        if self._instance is not None and self._instance._is_open:
            raise RuntimeError("Sensor already open")

        self._instance = DigitalTwinSensorInstance(self._config)
        logger.info("Digital twin sensor opened")
        return self._instance

    def close(self) -> None:
        """Close the current simulated sensor instance.

        Closes underlying DigitalTwinSensorInstance if open. Safe to call
        when no instance open.

        Business context: Driver-level cleanup matching real driver API.
        Ensures consistent lifecycle management across driver types.

        Args:
            No arguments.

        Returns:
            None. Instance reference cleared.

        Raises:
            No exceptions raised.

        Example:
            >>> driver = DigitalTwinSensorDriver()
            >>> instance = driver.open()
            >>> reading = instance.read()
            >>> driver.close()
        """
        if self._instance is not None:
            self._instance.close()
            self._instance = None
