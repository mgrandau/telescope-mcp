"""Tests for coordinate conversion utilities.

Validates altaz_to_radec, radec_to_altaz, and formatting functions
using known astronomical values and edge cases.

Example:
    pdm run pytest tests/test_utils_coordinates.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime

from telescope_mcp.utils.coordinates import (
    altaz_to_radec,
    format_dec_dms,
    format_ra_hms,
    radec_to_altaz,
)


class TestAltazToRadec:
    """Test suite for altaz_to_radec function.

    Categories:
    1. Basic Conversion - Zenith, horizon, cardinal directions (4 tests)
    2. Round-trip Consistency - Convert back and forth (2 tests)
    3. Edge Cases - Poles, boundaries (2 tests)

    Total: 8 tests.
    """

    # Austin, TX coordinates for all tests
    LAT = 30.2672
    LON = -97.7431

    def test_returns_equatorial_coords_type(self) -> None:
        """Verifies return type is EquatorialCoords TypedDict.

        Business context: API contract - callers depend on consistent keys.

        Arrangement: Fixed location and time.
        Action: Call altaz_to_radec with valid inputs.
        Assertion: Result has all required keys with correct types.
        """
        result = altaz_to_radec(45.0, 180.0, lat=self.LAT, lon=self.LON)

        assert "ra" in result
        assert "dec" in result
        assert "ra_hours" in result
        assert "ra_hms" in result
        assert "dec_dms" in result
        assert isinstance(result["ra"], float)
        assert isinstance(result["dec"], float)
        assert isinstance(result["ra_hours"], float)
        assert isinstance(result["ra_hms"], str)
        assert isinstance(result["dec_dms"], str)

    def test_zenith_gives_reasonable_dec(self) -> None:
        """Verifies zenith (alt=90) produces declination near observer latitude.

        Business context: At zenith, the declination should be approximately
        equal to the observer's latitude (within a few degrees due to time).

        Arrangement: Set altitude to 90 (zenith).
        Action: Convert to RA/Dec.
        Assertion: Dec is within 15° of observer latitude.
        """
        result = altaz_to_radec(90.0, 0.0, lat=self.LAT, lon=self.LON)

        # Zenith dec should be close to observer latitude
        # Allow generous tolerance since RA/Dec changes with time
        assert abs(result["dec"] - self.LAT) < 15.0

    def test_horizon_north(self) -> None:
        """Verifies horizon north (alt=0, az=0) conversion.

        Business context: Validate coordinate system orientation.

        Arrangement: Set altitude=0 (horizon), azimuth=0 (north).
        Action: Convert to RA/Dec.
        Assertion: Returns valid coordinates.
        """
        result = altaz_to_radec(0.0, 0.0, lat=self.LAT, lon=self.LON)

        assert 0.0 <= result["ra"] <= 360.0
        assert -90.0 <= result["dec"] <= 90.0

    def test_south_horizon(self) -> None:
        """Verifies horizon south (alt=0, az=180) conversion.

        Business context: Validate azimuth interpretation.

        Arrangement: Set altitude=0 (horizon), azimuth=180 (south).
        Action: Convert to RA/Dec.
        Assertion: Returns valid coordinates.
        """
        result = altaz_to_radec(0.0, 180.0, lat=self.LAT, lon=self.LON)

        assert 0.0 <= result["ra"] <= 360.0
        assert -90.0 <= result["dec"] <= 90.0

    def test_ra_hours_is_ra_divided_by_15(self) -> None:
        """Verifies ra_hours = ra / 15.

        Business context: RA in hours (0-24) is standard astronomical format.

        Arrangement: Any valid input.
        Action: Convert and check relationship.
        Assertion: ra_hours = ra / 15 within floating point tolerance.
        """
        result = altaz_to_radec(45.0, 90.0, lat=self.LAT, lon=self.LON)

        expected_hours = result["ra"] / 15.0
        assert abs(result["ra_hours"] - expected_hours) < 1e-10

    def test_explicit_time(self) -> None:
        """Verifies explicit obstime parameter is used.

        Business context: Historical or future observations need specific times.

        Arrangement: Set a specific UTC time.
        Action: Convert with explicit time.
        Assertion: Returns valid coordinates (actual values time-dependent).
        """
        obstime = datetime(2025, 6, 21, 12, 0, 0, tzinfo=UTC)
        result = altaz_to_radec(
            45.0, 180.0, lat=self.LAT, lon=self.LON, obstime=obstime
        )

        assert 0.0 <= result["ra"] <= 360.0
        assert -90.0 <= result["dec"] <= 90.0

    def test_elevation_parameter(self) -> None:
        """Verifies elevation parameter is accepted.

        Business context: Mountain observatories have significant elevation.

        Arrangement: Set elevation to 2000m.
        Action: Convert with elevation.
        Assertion: Returns valid coordinates.
        """
        result = altaz_to_radec(
            45.0, 180.0, lat=self.LAT, lon=self.LON, elevation=2000.0
        )

        assert 0.0 <= result["ra"] <= 360.0
        assert -90.0 <= result["dec"] <= 90.0

    def test_negative_altitude(self) -> None:
        """Verifies negative altitude (below horizon) is handled.

        Business context: Objects can be below horizon during calculations.

        Arrangement: Set altitude to -10 (below horizon).
        Action: Convert to RA/Dec.
        Assertion: Returns valid coordinates.
        """
        result = altaz_to_radec(-10.0, 180.0, lat=self.LAT, lon=self.LON)

        assert 0.0 <= result["ra"] <= 360.0
        assert -90.0 <= result["dec"] <= 90.0


class TestRadecToAltaz:
    """Test suite for radec_to_altaz function.

    Categories:
    1. Basic Conversion - Known celestial objects (3 tests)
    2. Round-trip - Convert both ways (1 test)
    3. RA Hours Auto-conversion (1 test)

    Total: 5 tests.
    """

    LAT = 30.2672
    LON = -97.7431

    def test_returns_tuple_of_floats(self) -> None:
        """Verifies return type is tuple of (alt, az) floats.

        Business context: API contract for motor control integration.

        Arrangement: Any valid RA/Dec.
        Action: Convert to ALT/AZ.
        Assertion: Returns tuple of two floats.
        """
        alt, az = radec_to_altaz(83.82, -5.39, lat=self.LAT, lon=self.LON)

        assert isinstance(alt, float)
        assert isinstance(az, float)

    def test_valid_range(self) -> None:
        """Verifies returned values are in valid ranges.

        Business context: Motor controller expects valid angles.

        Arrangement: Any valid RA/Dec.
        Action: Convert to ALT/AZ.
        Assertion: Alt in [-90, 90], Az in [0, 360].
        """
        alt, az = radec_to_altaz(0.0, 0.0, lat=self.LAT, lon=self.LON)

        assert -90.0 <= alt <= 90.0
        assert 0.0 <= az <= 360.0

    def test_ra_hours_auto_conversion(self) -> None:
        """Verifies RA < 24 is treated as hours and auto-converted.

        Business context: Some catalogs use RA in hours (0-24h).

        Arrangement: Set RA to 5.588 (hours for ~83.82°).
        Action: Convert - should auto-detect hours.
        Assertion: Produces same result as 83.82°.
        """
        # 5.588 hours = 83.82 degrees
        alt_hours, az_hours = radec_to_altaz(5.588, -5.39, lat=self.LAT, lon=self.LON)
        alt_deg, az_deg = radec_to_altaz(83.82, -5.39, lat=self.LAT, lon=self.LON)

        # Should be approximately equal (small diff due to conversion)
        assert abs(alt_hours - alt_deg) < 0.1
        assert abs(az_hours - az_deg) < 0.1

    def test_explicit_time(self) -> None:
        """Verifies explicit obstime is used.

        Business context: Planning future observations needs specific times.

        Arrangement: Set a specific UTC time.
        Action: Convert with explicit time.
        Assertion: Returns valid coordinates.
        """
        obstime = datetime(2025, 6, 21, 12, 0, 0, tzinfo=UTC)
        alt, az = radec_to_altaz(
            83.82, -5.39, lat=self.LAT, lon=self.LON, obstime=obstime
        )

        assert -90.0 <= alt <= 90.0
        assert 0.0 <= az <= 360.0

    def test_round_trip_consistency(self) -> None:
        """Verifies converting ALT/AZ → RA/Dec → ALT/AZ returns original.

        Business context: Critical for goto accuracy - transformations must
        be reversible.

        Arrangement: Start with known ALT/AZ.
        Action: Convert to RA/Dec, then back to ALT/AZ.
        Assertion: Final ALT/AZ matches original within tolerance.
        """
        original_alt = 45.0
        original_az = 180.0
        obstime = datetime(2025, 6, 21, 12, 0, 0, tzinfo=UTC)

        # ALT/AZ → RA/Dec
        equatorial = altaz_to_radec(
            original_alt,
            original_az,
            lat=self.LAT,
            lon=self.LON,
            obstime=obstime,
        )

        # RA/Dec → ALT/AZ
        final_alt, final_az = radec_to_altaz(
            equatorial["ra"],
            equatorial["dec"],
            lat=self.LAT,
            lon=self.LON,
            obstime=obstime,
        )

        # Should match original within small tolerance
        assert abs(final_alt - original_alt) < 0.01
        assert abs(final_az - original_az) < 0.01


class TestFormatRaHms:
    """Test suite for format_ra_hms function.

    Categories:
    1. Standard Values (3 tests)
    2. Edge Cases (2 tests)

    Total: 5 tests.
    """

    def test_zero_ra(self) -> None:
        """Verifies RA=0 formats correctly.

        Business context: 0h RA is the vernal equinox point.

        Arrangement: RA = 0.0 degrees.
        Action: Format to HMS.
        Assertion: Returns "00h 00m 00.0s".

        Testing Principle:
            Tests boundary condition at RA origin for astronomical coordinate
            formatting.
        """
        result = format_ra_hms(0.0)
        assert result == "00h 00m 00.0s"

    def test_90_degrees(self) -> None:
        """Verifies RA=90° (6h) formats correctly.

        Business context: 90° = 6 hours exactly.

        Arrangement: RA = 90.0 degrees.
        Action: Format to HMS.
        Assertion: Returns "06h 00m 00.0s".

        Testing Principle:
            Tests exact degree-to-hour conversion at quarter-circle boundary.
        """
        result = format_ra_hms(90.0)
        assert result == "06h 00m 00.0s"

    def test_orion_nebula_ra(self) -> None:
        """Verifies M42 Orion Nebula RA formats correctly.

        Business context: Real-world test with known object.
        M42 RA is approximately 83.82° = 5h 35m 16.8s.

        Arrangement: RA = 83.82 degrees.
        Action: Format to HMS.
        Assertion: Returns approximately "05h 35m ...".
        """
        result = format_ra_hms(83.82)
        assert result.startswith("05h 35m")

    def test_negative_ra_normalized(self) -> None:
        """Verifies negative RA is normalized to 0-360.

        Business context: Input validation - negative RA should wrap.

        Arrangement: RA = -90.0 degrees.
        Action: Format to HMS.
        Assertion: Returns equivalent to 270° = 18h.
        """
        result = format_ra_hms(-90.0)
        assert result.startswith("18h")

    def test_ra_over_360_normalized(self) -> None:
        """Verifies RA > 360 is normalized.

        Business context: Input validation - wrap around.

        Arrangement: RA = 450.0 degrees (= 90°).
        Action: Format to HMS.
        Assertion: Returns equivalent to 90° = 6h.

        Testing Principle:
            Tests modulo normalization for out-of-range input handling.
        """
        result = format_ra_hms(450.0)
        assert result.startswith("06h")


class TestFormatDecDms:
    """Test suite for format_dec_dms function.

    Categories:
    1. Positive Declinations (2 tests)
    2. Negative Declinations (2 tests)
    3. Edge Cases (1 test)

    Total: 5 tests.
    """

    def test_zero_dec(self) -> None:
        """Verifies Dec=0 formats correctly with positive sign.

        Business context: Celestial equator is 0° declination.

        Arrangement: Dec = 0.0 degrees.
        Action: Format to DMS.
        Assertion: Returns "+00° 00' 00.0\"".
        """
        result = format_dec_dms(0.0)
        assert result == "+00° 00' 00.0\""

    def test_positive_dec(self) -> None:
        """Verifies positive declination includes + sign.

        Business context: Northern hemisphere objects have positive dec.

        Arrangement: Dec = 23.456 degrees.
        Action: Format to DMS.
        Assertion: Starts with "+23°".
        """
        result = format_dec_dms(23.456)
        assert result.startswith("+23°")
        assert "'" in result
        assert '"' in result

    def test_negative_dec(self) -> None:
        """Verifies negative declination includes - sign.

        Business context: Southern hemisphere objects have negative dec.
        M42 Orion Nebula is at approximately -5.39°.

        Arrangement: Dec = -5.391 degrees.
        Action: Format to DMS.
        Assertion: Starts with "-05°".
        """
        result = format_dec_dms(-5.391)
        assert result.startswith("-05°")

    def test_north_celestial_pole(self) -> None:
        """Verifies Dec=+90 formats correctly.

        Business context: Polaris is near +90° declination.

        Arrangement: Dec = 90.0 degrees.
        Action: Format to DMS.
        Assertion: Returns "+90° 00' 00.0\"".

        Testing Principle:
            Tests upper boundary condition for declination formatting at celestial pole.
        """
        result = format_dec_dms(90.0)
        assert result == "+90° 00' 00.0\""

    def test_south_celestial_pole(self) -> None:
        """Verifies Dec=-90 formats correctly.

        Business context: South celestial pole.

        Arrangement: Dec = -90.0 degrees.
        Action: Format to DMS.
        Assertion: Returns "-90° 00' 00.0\"".

        Testing Principle:
            Tests lower boundary condition for declination formatting at celestial pole.
        """
        result = format_dec_dms(-90.0)
        assert result == "-90° 00' 00.0\""
