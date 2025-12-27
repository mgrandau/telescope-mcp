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

        Args:
            config: Sensor behavior configuration.
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
        """Get sensor information.

        Returns:
            Dict with sensor type, capabilities, and configuration.
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
        """Read current sensor values with simulated noise.

        Returns:
            SensorReading with accelerometer, magnetometer, position,
            temperature, and humidity data.

        Raises:
            RuntimeError: If sensor is closed.
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

    def set_position(self, altitude: float, azimuth: float) -> None:
        """Set the true simulated position (for testing).

        Args:
            altitude: True altitude in degrees (0-90).
            azimuth: True azimuth in degrees (0-360).
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
        """Calibrate sensor to known true position.

        Sets calibration transform so that current reading maps to
        the provided true position.

        Args:
            true_altitude: Known true altitude in degrees.
            true_azimuth: Known true azimuth in degrees.
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

    def calibrate_magnetometer(self) -> dict:
        """Simulate magnetometer calibration.

        Returns simulated calibration offsets.

        Returns:
            Dict with offset_x, offset_y, offset_z values.
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
        """Reset sensor (simulates hardware reset).

        Clears calibration and resets to initial state.
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
        """Get sensor status information.

        Returns:
            Dict with connection status, calibration state, etc.
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
        """Close the sensor connection."""
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

        Args:
            config: Sensor behavior configuration. Uses defaults if None.
        """
        self._config = config or DigitalTwinSensorConfig()
        self._instance: DigitalTwinSensorInstance | None = None

    def get_available_sensors(self) -> list[dict]:
        """List available sensors (always returns one simulated sensor).

        Returns:
            List with single sensor info dict.
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
        """Open a simulated sensor instance.

        Args:
            sensor_id: Sensor ID (ignored, always opens simulated sensor).

        Returns:
            DigitalTwinSensorInstance for reading sensor data.

        Raises:
            RuntimeError: If sensor already open.
        """
        if self._instance is not None and self._instance._is_open:
            raise RuntimeError("Sensor already open")

        self._instance = DigitalTwinSensorInstance(self._config)
        logger.info("Digital twin sensor opened")
        return self._instance

    def close(self) -> None:
        """Close the current sensor instance."""
        if self._instance is not None:
            self._instance.close()
            self._instance = None
