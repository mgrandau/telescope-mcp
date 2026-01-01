"""Coordinate provider implementations for automatic frame metadata injection.

Provides concrete implementations of the CoordinateProvider protocol that
read telescope pointing from various sources (sensors, encoders) and convert
to equatorial coordinates for frame metadata.

Example:
    from telescope_mcp.devices.coordinate_provider import SensorCoordinateProvider
    from telescope_mcp.devices.sensor import Sensor

    # Create sensor and connect
    sensor = Sensor(driver)
    await sensor.connect()

    # Create coordinate provider
    provider = SensorCoordinateProvider(
        sensor=sensor,
        lat=30.27,
        lon=-97.74,
        elevation=150.0,
    )

    # Inject into camera
    camera = Camera(driver, config, coordinate_provider=provider)

    # Now every capture automatically includes coordinates
    result = camera.capture()
    print(result.metadata["coordinates"]["ra_hms"])  # "12h 34m 56.7s"
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from telescope_mcp.devices.camera import CaptureCoordinates
from telescope_mcp.observability import get_logger
from telescope_mcp.utils.coordinates import altaz_to_radec

if TYPE_CHECKING:
    from telescope_mcp.devices.sensor import Sensor

logger = get_logger(__name__)

__all__ = [
    "SensorCoordinateProvider",
    "LocationConfig",
]


class LocationConfig:
    """Observer location configuration for coordinate conversion.

    Encapsulates geographic location needed for ALT/AZ to RA/Dec conversion.
    Can be constructed from a config dict or explicit parameters.

    Attributes:
        lat: Latitude in decimal degrees (-90 to +90, positive=North).
        lon: Longitude in decimal degrees (-180 to +180, positive=East).
        elevation: Elevation in meters above sea level.
    """

    __slots__ = ("lat", "lon", "elevation")

    def __init__(
        self,
        lat: float = 0.0,
        lon: float = 0.0,
        elevation: float = 0.0,
    ) -> None:
        """Create location configuration.

        Args:
            lat: Latitude in decimal degrees. Default 0.0 (equator).
            lon: Longitude in decimal degrees. Default 0.0 (prime meridian).
            elevation: Elevation in meters. Default 0.0 (sea level).
        """
        self.lat = lat
        self.lon = lon
        self.elevation = elevation

    @classmethod
    def from_dict(cls, config: dict[str, float]) -> LocationConfig:
        """Create LocationConfig from configuration dictionary.

        Factory method for creating location from config dicts as typically
        stored in application configuration files.

        Args:
            config: Dict with optional keys "lat", "lon", "alt" or "elevation".
                Missing keys use defaults (0.0).

        Returns:
            LocationConfig with values from dict.

        Example:
            >>> config = {"lat": 30.27, "lon": -97.74, "alt": 150.0}
            >>> loc = LocationConfig.from_dict(config)
            >>> print(f"{loc.lat}, {loc.lon}, {loc.elevation}")
            30.27, -97.74, 150.0
        """
        return cls(
            lat=config.get("lat", 0.0),
            lon=config.get("lon", 0.0),
            elevation=config.get("alt", config.get("elevation", 0.0)),
        )


class SensorCoordinateProvider:
    """Coordinate provider using IMU sensor for telescope pointing.

    Reads ALT/AZ from the connected Sensor device and converts to equatorial
    coordinates (RA/Dec) for the configured observer location. Designed for
    injection into Camera instances for automatic coordinate capture.

    Thread Safety:
        This provider performs synchronous sensor reads. In async contexts,
        consider wrapping in run_in_executor or using async-aware sensor reads.

    Example:
        sensor = Sensor(driver)
        await sensor.connect()

        provider = SensorCoordinateProvider(
            sensor=sensor,
            lat=30.27,
            lon=-97.74,
        )

        # Every capture now includes coordinates
        camera = Camera(driver, config, coordinate_provider=provider)
        result = camera.capture()
        coords = result.metadata["coordinates"]
        print(f"RA: {coords['ra_hms']}, Dec: {coords['dec_dms']}")
    """

    __slots__ = ("_sensor", "_location")

    def __init__(
        self,
        sensor: Sensor,
        lat: float | None = None,
        lon: float | None = None,
        elevation: float = 0.0,
        location: LocationConfig | None = None,
    ) -> None:
        """Create sensor-based coordinate provider.

        Args:
            sensor: Connected Sensor device for ALT/AZ readings.
            lat: Observer latitude in decimal degrees (if not using location).
            lon: Observer longitude in decimal degrees (if not using location).
            elevation: Observer elevation in meters (if not using location).
            location: LocationConfig (alternative to lat/lon/elevation params).
                If provided, lat/lon/elevation params are ignored.

        Example:
            # Using explicit params
            provider = SensorCoordinateProvider(sensor, lat=30.27, lon=-97.74)

            # Using LocationConfig
            loc = LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)
            provider = SensorCoordinateProvider(sensor, location=loc)
        """
        self._sensor = sensor

        if location is not None:
            self._location = location
        else:
            self._location = LocationConfig(
                lat=lat or 0.0,
                lon=lon or 0.0,
                elevation=elevation,
            )

    def get_coordinates(self) -> CaptureCoordinates | None:
        """Get current telescope pointing coordinates and environment from sensor.

        Reads ALT/AZ, temperature, and humidity from sensor synchronously,
        converts ALT/AZ to RA/Dec using current time and configured observer
        location. Returns None if sensor not connected or read fails.

        Business context: Called at capture time to inject coordinates and
        environmental data into frame metadata. Data captured at moment of
        exposure for accurate positional and atmospheric conditions.

        Returns:
            CaptureCoordinates with ALT/AZ, RA/Dec, temperature, humidity,
            or None if unavailable.

        Raises:
            None. All exceptions caught and logged, returns None on failure.

        Example:
            >>> coords = provider.get_coordinates()
            >>> if coords:
            ...     print(f"Pointing at RA {coords['ra_hms']}")
            ...     print(f"Temp: {coords['temperature']}°C")
        """
        if not self._sensor.connected:
            logger.debug("Sensor not connected, skipping coordinates")
            return None

        try:
            # Synchronous read - sensor.read() is async but we need sync here
            # Use the internal instance directly for sync access
            if self._sensor._instance is None:
                logger.debug("Sensor instance not available")
                return None

            reading = self._sensor._instance.read()
            capture_time = datetime.now(UTC)

            # Convert ALT/AZ to RA/Dec
            equatorial = altaz_to_radec(
                altitude=reading.altitude,
                azimuth=reading.azimuth,
                lat=self._location.lat,
                lon=self._location.lon,
                elevation=self._location.elevation,
                obstime=capture_time,
            )

            return CaptureCoordinates(
                altitude=reading.altitude,
                azimuth=reading.azimuth,
                ra=equatorial["ra"],
                dec=equatorial["dec"],
                ra_hours=equatorial["ra_hours"],
                ra_hms=equatorial["ra_hms"],
                dec_dms=equatorial["dec_dms"],
                temperature=reading.temperature,
                humidity=reading.humidity,
                coordinate_source="sensor",
                coordinate_timestamp=capture_time.isoformat(),
            )

        except Exception as e:
            logger.warning("Failed to get coordinates from sensor", error=str(e))
            return None
