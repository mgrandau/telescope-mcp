"""Coordinate conversion utilities for astronomical observations.

Provides functions for converting between horizontal (ALT/AZ) and equatorial
(RA/Dec) coordinate systems using astropy. Requires observer location and
observation time for accurate transformations.

Business context: Essential for displaying where the telescope is pointing in
celestial coordinates. When a user moves the telescope (changes ALT/AZ), the
web UI shows the corresponding RA/Dec so they know which part of the sky
they're observing. Also used for goto functionality (target RA/Dec → ALT/AZ).

Example:
    >>> from telescope_mcp.utils.coordinates import altaz_to_radec, format_ra_hms
    >>> ra, dec = altaz_to_radec(45.0, 180.0, lat=30.27, lon=-97.74)
    >>> print(f"RA: {format_ra_hms(ra)}, Dec: {format_dec_dms(dec)}")
    RA: 12h 34m 56.7s, Dec: +23° 45' 12.3"
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

if TYPE_CHECKING:
    pass

__all__ = [
    "EquatorialCoords",
    "altaz_to_radec",
    "radec_to_altaz",
    "format_ra_hms",
    "format_dec_dms",
]


class EquatorialCoords(TypedDict):
    """Equatorial coordinates with formatted strings.

    Attributes:
        ra: Right Ascension in degrees (0-360).
        dec: Declination in degrees (-90 to +90).
        ra_hours: Right Ascension in hours (0-24).
        ra_hms: RA formatted as hours:minutes:seconds (e.g., "12h 34m 56.7s").
        dec_dms: Dec formatted as degrees:arcmin:arcsec (e.g., "+23° 45' 12.3\"").
    """

    ra: float
    dec: float
    ra_hours: float
    ra_hms: str
    dec_dms: str


def altaz_to_radec(
    altitude: float,
    azimuth: float,
    *,
    lat: float,
    lon: float,
    elevation: float = 0.0,
    obstime: datetime | None = None,
) -> EquatorialCoords:
    """Convert horizontal coordinates to equatorial coordinates.

    Transforms altitude/azimuth (local horizontal) to right ascension/declination
    (equatorial) coordinates for a given observer location and time. Uses astropy
    for accurate coordinate transformations including precession and nutation.

    Business context: When the telescope moves to a new ALT/AZ position, this
    function computes the corresponding RA/Dec to display in the web UI. Enables
    operators to know which celestial object they're pointing at without needing
    to manually consult star charts.

    Args:
        altitude: Altitude angle in degrees above horizon.
            Range: -90 (nadir) to +90 (zenith). Typically 0-90 for visible sky.
            Negative values indicate below horizon.
        azimuth: Azimuth angle in degrees clockwise from north.
            Range: 0-360 where 0=North, 90=East, 180=South, 270=West.
        lat: Observer latitude in decimal degrees.
            Range: -90 (South Pole) to +90 (North Pole). Positive=North.
        lon: Observer longitude in decimal degrees.
            Range: -180 to +180. Positive=East, Negative=West.
        elevation: Observer elevation in meters above sea level.
            Default 0.0. Used for minor atmospheric refraction correction.
        obstime: Observation time in UTC. If None, uses current UTC time.
            Must be timezone-aware or will be treated as UTC.

    Returns:
        EquatorialCoords dict containing:
            - ra: Right Ascension in degrees (0-360)
            - dec: Declination in degrees (-90 to +90)
            - ra_hours: Right Ascension in hours (0-24)
            - ra_hms: RA formatted string (e.g., "12h 34m 56.7s")
            - dec_dms: Dec formatted string (e.g., "+23° 45' 12.3\"")

    Raises:
        None. Invalid coordinates produce valid but possibly unusual results.

    Example:
        >>> coords = altaz_to_radec(45.0, 180.0, lat=30.27, lon=-97.74)
        >>> print(f"RA: {coords['ra_hms']}, Dec: {coords['dec_dms']}")
        RA: 06h 12m 34.5s, Dec: -15° 23' 45.6"

        >>> # With explicit time
        >>> from datetime import datetime, timezone
        >>> t = datetime(2025, 6, 21, 22, 0, 0, tzinfo=timezone.utc)
        >>> coords = altaz_to_radec(60.0, 90.0, lat=40.0, lon=-74.0, obstime=t)
    """
    if obstime is None:
        obstime = datetime.now(UTC)

    # Create astropy EarthLocation
    location = EarthLocation(
        lat=lat * u.deg,
        lon=lon * u.deg,
        height=elevation * u.m,
    )

    # Create astropy Time object
    obs_time = Time(obstime)

    # Create AltAz frame for this location and time
    altaz_frame = AltAz(obstime=obs_time, location=location)

    # Create coordinate in AltAz frame
    altaz_coord = SkyCoord(
        alt=altitude * u.deg,
        az=azimuth * u.deg,
        frame=altaz_frame,
    )

    # Transform to ICRS (equatorial)
    icrs_coord = altaz_coord.transform_to("icrs")

    # Extract values
    ra_deg = float(icrs_coord.ra.deg)
    dec_deg = float(icrs_coord.dec.deg)
    ra_hours = ra_deg / 15.0  # 360° = 24h, so 15°/h

    return EquatorialCoords(
        ra=ra_deg,
        dec=dec_deg,
        ra_hours=ra_hours,
        ra_hms=format_ra_hms(ra_deg),
        dec_dms=format_dec_dms(dec_deg),
    )


def radec_to_altaz(
    ra: float,
    dec: float,
    *,
    lat: float,
    lon: float,
    elevation: float = 0.0,
    obstime: datetime | None = None,
) -> tuple[float, float]:
    """Convert equatorial coordinates to horizontal coordinates.

    Transforms right ascension/declination (equatorial) to altitude/azimuth
    (local horizontal) coordinates for a given observer location and time.
    Inverse of altaz_to_radec().

    Business context: Used for goto functionality - when a user wants to point
    at a celestial object (specified by RA/Dec), this computes the ALT/AZ the
    telescope motors need to move to.

    Args:
        ra: Right Ascension in degrees (0-360) or hours (0-24).
            Values < 24 are assumed to be hours and converted to degrees.
        dec: Declination in degrees. Range: -90 to +90.
            Positive = North of celestial equator.
        lat: Observer latitude in decimal degrees [-90, +90].
        lon: Observer longitude in decimal degrees [-180, +180].
        elevation: Observer elevation in meters above sea level (default 0.0).
        obstime: Observation time in UTC. If None, uses current UTC time.

    Returns:
        Tuple of (altitude, azimuth) in degrees.
            - altitude: Degrees above horizon (0=horizon, 90=zenith)
            - azimuth: Degrees clockwise from north (0=N, 90=E, 180=S, 270=W)

    Raises:
        None. Invalid coordinates produce valid but possibly unusual results.

    Example:
        >>> alt, az = radec_to_altaz(83.82, -5.39, lat=30.27, lon=-97.74)
        >>> print(f"ALT: {alt:.2f}°, AZ: {az:.2f}°")
        ALT: 45.23°, AZ: 123.45°

        >>> # RA in hours (auto-converted)
        >>> alt, az = radec_to_altaz(5.588, -5.39, lat=30.27, lon=-97.74)
    """
    if obstime is None:
        obstime = datetime.now(UTC)

    # Auto-convert RA from hours to degrees if needed
    if ra < 24:
        ra = ra * 15.0  # hours to degrees

    # Create astropy EarthLocation
    location = EarthLocation(
        lat=lat * u.deg,
        lon=lon * u.deg,
        height=elevation * u.m,
    )

    # Create astropy Time object
    obs_time = Time(obstime)

    # Create AltAz frame for this location and time
    altaz_frame = AltAz(obstime=obs_time, location=location)

    # Create coordinate in ICRS (equatorial)
    sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # Transform to AltAz
    altaz_coord = sky_coord.transform_to(altaz_frame)

    return float(altaz_coord.alt.deg), float(altaz_coord.az.deg)


def format_ra_hms(ra_degrees: float) -> str:
    """Format Right Ascension in hours:minutes:seconds notation.

    Converts RA from degrees to traditional sexagesimal HMS format used in
    astronomy catalogs and star charts.

    Business context: Human-readable display format for the web UI. Astronomers
    expect RA in HMS format (e.g., "05h 35m 17.3s" for Orion Nebula).

    Args:
        ra_degrees: Right Ascension in degrees (0-360).

    Returns:
        Formatted string like "12h 34m 56.7s".

    Example:
        >>> format_ra_hms(83.82)
        '05h 35m 16.8s'
        >>> format_ra_hms(0.0)
        '00h 00m 00.0s'
    """
    # Normalize to 0-360 range
    ra_degrees = ra_degrees % 360.0

    # Convert to hours (24h = 360°)
    hours_total = ra_degrees / 15.0

    hours = int(hours_total)
    minutes_total = (hours_total - hours) * 60
    minutes = int(minutes_total)
    seconds = (minutes_total - minutes) * 60

    return f"{hours:02d}h {minutes:02d}m {seconds:04.1f}s"


def format_dec_dms(dec_degrees: float) -> str:
    """Format Declination in degrees:arcminutes:arcseconds notation.

    Converts Dec from decimal degrees to traditional sexagesimal DMS format
    used in astronomy catalogs and star charts.

    Business context: Human-readable display format for the web UI. Astronomers
    expect Dec in DMS format with sign (e.g., "-05° 23' 28.0\"" for Orion Nebula).

    Args:
        dec_degrees: Declination in degrees (-90 to +90).

    Returns:
        Formatted string like "+23° 45' 12.3\"" or "-05° 23' 28.0\"".
        Includes leading +/- sign.

    Example:
        >>> format_dec_dms(-5.391)
        '-05° 23\' 27.6"'
        >>> format_dec_dms(23.456)
        '+23° 27\' 21.6"'
    """
    sign = "+" if dec_degrees >= 0 else "-"
    dec_abs = abs(dec_degrees)

    degrees = int(dec_abs)
    arcmin_total = (dec_abs - degrees) * 60
    arcmin = int(arcmin_total)
    arcsec = (arcmin_total - arcmin) * 60

    return f"{sign}{degrees:02d}° {arcmin:02d}' {arcsec:04.1f}\""
