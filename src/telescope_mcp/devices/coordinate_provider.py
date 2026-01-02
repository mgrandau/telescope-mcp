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

    Raises:
        ValueError: If lat not in [-90, 90], lon not in [-180, 180],
            or elevation < -500m.
    """

    __slots__ = ("lat", "lon", "elevation")

    def __init__(
        self,
        lat: float = 0.0,
        lon: float = 0.0,
        elevation: float = 0.0,
    ) -> None:
        """Create location configuration.

        Initializes observer location for ALT/AZ to RA/Dec coordinate
        transformations. Validates all inputs to ensure they fall within
        valid geographic ranges.

        Business context: Every coordinate transformation requires observer
        location. This encapsulates that data with validation to prevent
        invalid astronomy calculations.

        Args:
            lat: Latitude in decimal degrees. Default 0.0 (equator).
            lon: Longitude in decimal degrees. Default 0.0 (prime meridian).
            elevation: Elevation in meters. Default 0.0 (sea level).

        Returns:
            None

        Raises:
            ValueError: If lat not in [-90, 90], lon not in [-180, 180],
                or elevation < -500m (below Dead Sea).

        Example:
            >>> loc = LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)
            >>> print(f"Latitude: {loc.lat}")
            Latitude: 30.27
        """
        if not -90 <= lat <= 90:
            raise ValueError(f"Latitude must be in range [-90, 90], got {lat}")
        if not -180 <= lon <= 180:
            raise ValueError(f"Longitude must be in range [-180, 180], got {lon}")
        if elevation < -500:
            raise ValueError(f"Elevation cannot be less than -500m, got {elevation}")
        self.lat = lat
        self.lon = lon
        self.elevation = elevation

    def __repr__(self) -> str:
        """Return string representation for debugging.

        Formats the LocationConfig as a constructor call showing all
        attributes. Useful for logging, debugging, and interactive console.

        Business context: Clear string representation aids in debugging
        coordinate transformation issues. Shows exact configuration used
        for ALT/AZ to RA/Dec conversions.

        Args:
            None

        Returns:
            String in format "LocationConfig(lat=X, lon=Y, elevation=Z)".

        Raises:
            None

        Example:
            >>> loc = LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)
            >>> repr(loc)
            'LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)'
        """
        return (
            f"LocationConfig(lat={self.lat}, lon={self.lon}, "
            f"elevation={self.elevation})"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another LocationConfig.

        Compares latitude, longitude, and elevation values for equality.
        Implements proper equality protocol by returning NotImplemented
        for non-LocationConfig objects.

        Args:
            other: Object to compare with. Expected to be LocationConfig.

        Returns:
            True if all location attributes match, False otherwise.
            NotImplemented if other is not a LocationConfig instance.

        Example:
            >>> loc1 = LocationConfig(lat=30.0, lon=-97.0)
            >>> loc2 = LocationConfig(lat=30.0, lon=-97.0)
            >>> loc1 == loc2
            True
        """
        if not isinstance(other, LocationConfig):
            return NotImplemented
        return (
            self.lat == other.lat
            and self.lon == other.lon
            and self.elevation == other.elevation
        )

    def __hash__(self) -> int:
        """Return hash for use in sets and dict keys.

        Computes hash from tuple of (lat, lon, elevation). Required because
        we implement __eq__. Enables LocationConfig instances to be used in
        sets and as dictionary keys.

        Business context: Hashability enables using LocationConfig in lookup
        tables and caches. Common when managing multiple observatory sites
        or tracking location-specific calibrations.

        Args:
            None

        Returns:
            Integer hash value derived from location coordinates.

        Raises:
            None

        Example:
            >>> loc1 = LocationConfig(lat=30.0, lon=-97.0)
            >>> loc2 = LocationConfig(lat=30.0, lon=-97.0)
            >>> hash(loc1) == hash(loc2)
            True
            >>> locations = {loc1, loc2}  # Can use in sets
            >>> len(locations)
            1
        """
        return hash((self.lat, self.lon, self.elevation))

    @classmethod
    def from_dict(cls, config: dict[str, float]) -> LocationConfig:
        """Create LocationConfig from configuration dictionary.

        Factory method for creating location from config dicts as typically
        stored in application configuration files.

        Business context: Enables loading observer location from configuration
        files (YAML, JSON, TOML) without manually constructing LocationConfig.
        Common in telescope control systems where site location is configured
        once and loaded at startup.

        Args:
            config: Dict with optional keys "lat", "lon", "elevation" (or "alt").
                "elevation" takes priority over "alt" if both present.
                Missing keys use defaults (0.0).

        Returns:
            LocationConfig with values from dict.

        Raises:
            ValueError: If lat not in [-90, 90], lon not in [-180, 180],
                or elevation < -500m (validation happens in __init__).

        Example:
            >>> config = {"lat": 30.27, "lon": -97.74, "elevation": 150.0}
            >>> loc = LocationConfig.from_dict(config)
            >>> print(f"{loc.lat}, {loc.lon}, {loc.elevation}")
            30.27, -97.74, 150.0
        """
        return cls(
            lat=config.get("lat", 0.0),
            lon=config.get("lon", 0.0),
            elevation=config.get("elevation", config.get("alt", 0.0)),
        )


class SensorCoordinateProvider:
    """Coordinate provider using IMU sensor for telescope pointing.

    Reads ALT/AZ from the connected Sensor device and converts to equatorial
    coordinates (RA/Dec) for the configured observer location. Designed for
    injection into Camera instances for automatic coordinate capture.

    Thread Safety:
        get_coordinates() performs BLOCKING I/O via Sensor.read_sync().
        In async contexts, wrap the call:
            coords = await loop.run_in_executor(None, provider.get_coordinates)

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

        Accepts location either as explicit lat/lon/elevation parameters or
        as a pre-configured LocationConfig instance. If location is provided,
        it takes precedence over individual parameters.

        Business context: Enables automatic coordinate injection into every
        camera frame by reading sensor orientation at capture time. Essential
        for astrophotography where precise position metadata enables plate
        solving and image stacking.

        Args:
            sensor: Connected Sensor device for ALT/AZ readings.
            lat: Observer latitude in decimal degrees (if not using location).
            lon: Observer longitude in decimal degrees (if not using location).
            elevation: Observer elevation in meters (if not using location).
            location: LocationConfig (alternative to lat/lon/elevation params).
                If provided, lat/lon/elevation params are ignored.

        Returns:
            None

        Raises:
            ValueError: If creating LocationConfig from lat/lon/elevation and
                values are out of valid range (validation in LocationConfig).

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
                lat=lat if lat is not None else 0.0,
                lon=lon if lon is not None else 0.0,
                elevation=elevation,
            )

    @property
    def location(self) -> LocationConfig:
        """Get configured observer location for coordinate conversion.

        Returns the LocationConfig used for ALT/AZ to RA/Dec transformations.
        Useful for debugging coordinate issues or verifying configuration.

        Business context: Allows inspection of the provider's location settings
        without needing to track the original configuration. Essential for
        troubleshooting incorrect coordinate transformations.

        Args:
            None

        Returns:
            LocationConfig with lat, lon, elevation for coordinate conversion.

        Raises:
            None

        Example:
            >>> provider = SensorCoordinateProvider(sensor, lat=30.27, lon=-97.74)
            >>> print(f"Location: {provider.location.lat}, {provider.location.lon}")
            Location: 30.27, -97.74
        """
        return self._location

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
            # Use public sync read method (avoids accessing private _instance)
            reading = self._sensor.read_sync()
            if reading is None:
                logger.debug("Sensor read returned None")
                return None

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
