"""Unit tests for coordinate_provider module.

Tests LocationConfig and SensorCoordinateProvider classes including
validation, error handling, and coordinate conversion.

Example:
    pdm run pytest tests/test_coordinate_provider.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from telescope_mcp.devices.coordinate_provider import (
    LocationConfig,
    SensorCoordinateProvider,
)

if TYPE_CHECKING:
    pass


class TestLocationConfig:
    """Tests for LocationConfig class."""

    def test_default_values(self) -> None:
        """Verifies LocationConfig defaults to (0, 0, 0) when no params provided.

        Tests zero-argument constructor uses sensible defaults.

        Business Context:
        Default origin (Null Island) enables testing without configuration.

        Arrangement:
        Construct LocationConfig with no parameters.

        Action:
        Access lat, lon, elevation attributes.

        Assertion Strategy:
        All three values default to 0.0.

        Testing Principle:
        Validates default constructor enables quick instantiation for testing.
        """
        loc = LocationConfig()
        assert loc.lat == 0.0
        assert loc.lon == 0.0
        assert loc.elevation == 0.0

    def test_explicit_values(self) -> None:
        """Verifies LocationConfig stores provided lat/lon/elevation.

        Tests constructor parameter assignment.

        Business Context:
        Observatory location configuration (Austin, TX coordinates).

        Arrangement:
        Construct with Austin coordinates (30.27°N, 97.74°W, 150m).

        Action:
        Access stored attributes.

        Assertion Strategy:
        Values match constructor arguments exactly.

        Testing Principle:
        Validates basic data storage in immutable config object.
        """
        loc = LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)
        assert loc.lat == 30.27
        assert loc.lon == -97.74
        assert loc.elevation == 150.0

    def test_from_dict_full(self) -> None:
        """Verifies from_dict creates LocationConfig from complete dict.

        Tests factory method for dict-based configuration.

        Business Context:
        Config files (JSON/YAML) provide location as dict.

        Arrangement:
        Prepare dict with lat, lon, elevation keys.

        Action:
        Call LocationConfig.from_dict(config).

        Assertion Strategy:
        Resulting object attributes match dict values.

        Testing Principle:
        Validates factory method enables dict-based config loading.
        """
        config = {"lat": 45.0, "lon": -122.0, "elevation": 100.0}
        loc = LocationConfig.from_dict(config)
        assert loc.lat == 45.0
        assert loc.lon == -122.0
        assert loc.elevation == 100.0

    def test_from_dict_alt_alias(self) -> None:
        """Verifies from_dict accepts 'alt' as alias for 'elevation'.

        Tests backward compatibility with alternative key names.

        Business Context:
        Legacy configs may use 'alt' instead of 'elevation'.

        Arrangement:
        Create dict with 'alt' key (200.0) instead of 'elevation'.

        Action:
        Call from_dict() with alt-keyed config.

        Assertion Strategy:
        elevation attribute populated from 'alt' value.

        Testing Principle:
        Validates alias support ensures backward compatibility.
        """
        config = {"lat": 45.0, "lon": -122.0, "alt": 200.0}
        loc = LocationConfig.from_dict(config)
        assert loc.elevation == 200.0

    def test_from_dict_elevation_priority(self) -> None:
        """Verifies 'elevation' key takes precedence over 'alt' when both present.

        Tests key priority resolution for conflicting aliases.

        Business Context:
        Prevents ambiguity when config has both keys.

        Arrangement:
        Create dict with both alt=300.0 and elevation=400.0.

        Action:
        Call from_dict() with conflicting keys.

        Assertion Strategy:
        elevation uses 'elevation' value (400.0), ignoring 'alt'.

        Testing Principle:
        Validates precedence rule eliminates configuration ambiguity.
        """
        config = {"lat": 45.0, "lon": -122.0, "alt": 300.0, "elevation": 400.0}
        loc = LocationConfig.from_dict(config)
        # elevation key takes priority over alt
        assert loc.elevation == 400.0

    def test_from_dict_missing_keys(self) -> None:
        """Verifies from_dict uses defaults for missing keys.

        Tests graceful handling of incomplete configuration.

        Business Context:
        Partial configs should work with sensible defaults.

        Arrangement:
        Pass empty dict to from_dict().

        Action:
        Create LocationConfig from empty config.

        Assertion Strategy:
        All attributes default to 0.0 (same as no-arg constructor).

        Testing Principle:
        Validates robustness handles incomplete config gracefully.
        """
        loc = LocationConfig.from_dict({})
        assert loc.lat == 0.0
        assert loc.lon == 0.0
        assert loc.elevation == 0.0

    def test_repr(self) -> None:
        """Verifies repr returns constructor-style string representation.

        Tests debugging/logging string format.

        Business Context:
        Logging needs human-readable location representation.

        Arrangement:
        Create LocationConfig with Austin coordinates.

        Action:
        Call repr() to get string representation.

        Assertion Strategy:
        Format matches "LocationConfig(lat=..., lon=..., elevation=...)".

        Testing Principle:
        Validates repr enables easy debugging and log analysis.
        """
        loc = LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)
        assert repr(loc) == "LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)"

    def test_equality(self) -> None:
        """Verifies equality based on coordinate values.

        Tests __eq__ implementation for value-based comparison.

        Business Context:
        Location comparison for config validation/deduplication.

        Arrangement:
        Create three LocationConfig instances (two identical, one different).

        Action:
        Compare using == and != operators.

        Assertion Strategy:
        loc1 == loc2 (same coordinates).
        loc1 != loc3 (different latitude).

        Testing Principle:
        Validates value equality enables proper config comparison.
        """
        loc1 = LocationConfig(lat=30.0, lon=-97.0, elevation=100.0)
        loc2 = LocationConfig(lat=30.0, lon=-97.0, elevation=100.0)
        loc3 = LocationConfig(lat=31.0, lon=-97.0, elevation=100.0)

        assert loc1 == loc2
        assert loc1 != loc3

    def test_equality_not_implemented(self) -> None:
        """Verifies equality with non-LocationConfig returns NotImplemented.

        Tests type safety in equality comparison.

        Business Context:
        Python equality protocol for incompatible types.

        Arrangement:
        Create LocationConfig and compare to string.

        Action:
        Call __eq__ directly with wrong type.

        Assertion Strategy:
        Returns NotImplemented (not False), allowing Python to try reverse comparison.

        Testing Principle:
        Validates proper equality protocol implementation.
        """
        loc = LocationConfig()
        assert loc.__eq__("not a location") == NotImplemented

    def test_hash(self) -> None:
        """Verifies LocationConfig is hashable for use in sets/dicts.

        Tests __hash__ implementation enables collection membership.

        Business Context:
        Location-based caching/deduplication in config management.

        Arrangement:
        Create three LocationConfig instances (two equal, one different).

        Action:
        Add to set, use as dict keys, check hash equality.

        Assertion Strategy:
        - Equal objects have equal hashes.
        - Set deduplicates equal locations (2 items from 3 inputs).
        - Dict key lookup works with equal objects.

        Testing Principle:
        Validates hash contract (equal objects → equal hashes)
        enables use in hash-based collections.
        """
        loc1 = LocationConfig(lat=30.0, lon=-97.0, elevation=100.0)
        loc2 = LocationConfig(lat=30.0, lon=-97.0, elevation=100.0)
        loc3 = LocationConfig(lat=31.0, lon=-97.0, elevation=100.0)

        # Equal objects have same hash
        assert hash(loc1) == hash(loc2)
        # Can use in sets
        location_set = {loc1, loc2, loc3}
        assert len(location_set) == 2  # loc1 and loc2 are equal
        # Can use as dict keys
        location_dict = {loc1: "Austin", loc3: "Dallas"}
        assert location_dict[loc2] == "Austin"  # loc2 == loc1

    # Validation tests
    def test_lat_too_high(self) -> None:
        """Verifies latitude > 90° raises ValueError.

        Tests upper bound validation.

        Arrangement: Attempt to create LocationConfig with lat=91.0.
        Action: Constructor validates latitude range.
        Assertion Strategy: ValueError with range message.
        Testing Principle: Validates input constraints prevent invalid coordinates.
        """
        with pytest.raises(ValueError, match="Latitude must be in range"):
            LocationConfig(lat=91.0)

    def test_lat_too_low(self) -> None:
        """Verifies latitude < -90° raises ValueError.

        Tests lower bound validation.

        Arrangement: Attempt to create LocationConfig with lat=-91.0.
        Action: Constructor validates latitude range.
        Assertion Strategy: ValueError with range message.
        Testing Principle: Validates bounds checking prevents invalid coordinates.
        """
        with pytest.raises(ValueError, match="Latitude must be in range"):
            LocationConfig(lat=-91.0)

    def test_lon_too_high(self) -> None:
        """Verifies longitude > 180° raises ValueError.

        Tests upper bound validation.

        Arrangement: Attempt to create LocationConfig with lon=181.0.
        Action: Constructor validates longitude range.
        Assertion Strategy: ValueError with range message.
        Testing Principle: Validates input constraints prevent invalid coordinates.
        """
        with pytest.raises(ValueError, match="Longitude must be in range"):
            LocationConfig(lon=181.0)

    def test_lon_too_low(self) -> None:
        """Verifies longitude < -180° raises ValueError.

        Tests lower bound validation.

        Arrangement: Attempt to create LocationConfig with lon=-181.0.
        Action: Constructor validates longitude range.
        Assertion Strategy: ValueError with range message.
        Testing Principle: Validates bounds checking prevents invalid coordinates.
        """
        with pytest.raises(ValueError, match="Longitude must be in range"):
            LocationConfig(lon=-181.0)

    def test_elevation_too_low(self) -> None:
        """Verifies elevation < -500m raises ValueError.

        Tests minimum elevation limit (below Mariana Trench).

        Arrangement: Attempt elevation=-501.0 (below limit).
        Action: Constructor validates minimum elevation.
        Assertion Strategy: ValueError with minimum message.
        Testing Principle: Validates reasonable Earth surface bounds.
        """
        with pytest.raises(ValueError, match="Elevation cannot be less than"):
            LocationConfig(elevation=-501.0)

    def test_boundary_lat_values(self) -> None:
        """Verifies boundary latitudes (±90°) are accepted.

        Tests poles are valid coordinates.

        Arrangement: Create configs at North Pole (90°) and South Pole (-90°).
        Action: Validate both boundary values accepted.
        Assertion Strategy: Values stored exactly (no rounding/rejection).
        Testing Principle: Validates inclusive bounds allow extreme values.
        """
        loc_north = LocationConfig(lat=90.0)
        loc_south = LocationConfig(lat=-90.0)
        assert loc_north.lat == 90.0
        assert loc_south.lat == -90.0

    def test_boundary_lon_values(self) -> None:
        """Verifies boundary longitudes (±180°) are accepted.

        Tests antimeridian is valid longitude.

        Arrangement: Create configs at ±180° (International Date Line).
        Action: Validate both boundary values accepted.
        Assertion Strategy: Values stored exactly.
        Testing Principle: Validates inclusive bounds allow extreme values.
        """
        loc_east = LocationConfig(lon=180.0)
        loc_west = LocationConfig(lon=-180.0)
        assert loc_east.lon == 180.0
        assert loc_west.lon == -180.0

    def test_dead_sea_elevation(self) -> None:
        """Verifies negative elevations like Dead Sea (-430m) are accepted.

        Tests real-world below-sea-level location.

        Arrangement: Create config with elevation=-430.0.
        Action: Constructor accepts negative elevation within -500m limit.
        Assertion Strategy: Value stored correctly.
        Testing Principle: Validates practical below-sea-level locations work.
        """
        loc = LocationConfig(elevation=-430.0)
        assert loc.elevation == -430.0


class TestSensorCoordinateProvider:
    """Tests for SensorCoordinateProvider class."""

    @pytest.fixture
    def mock_sensor(self) -> MagicMock:
        """Create a mock sensor for coordinate provider testing.

        Provides a connected sensor mock with read_sync capability.
        Used to test SensorCoordinateProvider without hardware dependencies.

        Returns:
            MagicMock configured as connected sensor (connected=True).
            read_sync() method available for mocking sensor readings.

        Raises:
            Not applicable - fixture construction always succeeds.

        Example:
            >>> provider = SensorCoordinateProvider(
            ...     sensor=mock_sensor, lat=30.0, lon=-97.0
            ... )
            >>> mock_sensor.read_sync.return_value = mock_reading
            >>> coords = provider.get_coordinates()

        Business Context:
            Coordinate providers require sensor access for telescope pointing data.
            Mocking isolates provider logic from hardware sensor dependencies.
        """
        sensor = MagicMock()
        sensor.connected = True
        return sensor

    @pytest.fixture
    def mock_reading(self) -> MagicMock:
        """Create a mock sensor reading with telescope pointing data.

        Provides sample ALT/AZ coordinates and environmental data.
        Matches SensorReading structure from sensor.py module.

        Returns:
            MagicMock with altitude=45.0°, azimuth=180.0°, temperature=20.0°C,
            humidity=50.0%. Simulates typical sensor output.

        Raises:
            Not applicable - fixture construction always succeeds.

        Example:
            >>> mock_sensor.read_sync.return_value = mock_reading
            >>> coords = provider.get_coordinates()
            >>> assert coords["altitude"] == 45.0

        Business Context:
            ALT/AZ coordinates represent telescope physical pointing direction.
            Temperature/humidity tracked for environmental correlation analysis.
        """
        reading = MagicMock()
        reading.altitude = 45.0
        reading.azimuth = 180.0
        reading.temperature = 20.0
        reading.humidity = 50.0
        return reading

    def test_init_with_location(self, mock_sensor: MagicMock) -> None:
        """Verifies initialization with LocationConfig object.

        Tests that provider accepts pre-configured LocationConfig.

        Business Context:
        Shared location configs enable consistent observer location across
        multiple coordinate providers in multi-camera telescope systems.

        Arrangement:
        1. Create LocationConfig with Austin, TX coordinates (30.27°N, 97.74°W, 150m).
        2. Mock sensor in connected state.
        3. Initialize provider with location object.

        Action:
        Construct SensorCoordinateProvider passing location=loc parameter.

        Assertion Strategy:
        - Internal _location attribute references same LocationConfig instance.
        - Validates location object stored correctly.

        Testing Principle:
        Validates LocationConfig dependency injection pattern enables
        shared configuration across coordinate provider instances.
        """
        loc = LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)
        provider = SensorCoordinateProvider(sensor=mock_sensor, location=loc)
        assert provider._location == loc

    def test_init_with_explicit_params(self, mock_sensor: MagicMock) -> None:
        """Verifies initialization with explicit lat/lon/elevation parameters.

        Tests that provider creates LocationConfig from individual parameters.

        Business Context:
        Direct parameter passing simplifies single-provider setup without
        requiring separate LocationConfig instantiation.

        Arrangement:
        1. Mock sensor in connected state.
        2. Prepare Austin, TX coordinates as separate parameters.
        3. No LocationConfig object created beforehand.

        Action:
        Construct SensorCoordinateProvider with lat=30.27, lon=-97.74, elevation=150.0.
        Provider internally creates LocationConfig from parameters.

        Assertion Strategy:
        - Internal _location.lat matches provided value (30.27).
        - Internal _location.lon matches provided value (-97.74).
        - Internal _location.elevation matches provided value (150.0).

        Testing Principle:
        Validates convenience constructor pattern auto-creates LocationConfig,
        reducing boilerplate for simple single-provider configurations.
        """
        provider = SensorCoordinateProvider(
            sensor=mock_sensor,
            lat=30.27,
            lon=-97.74,
            elevation=150.0,
        )
        assert provider._location.lat == 30.27
        assert provider._location.lon == -97.74
        assert provider._location.elevation == 150.0

    def test_init_with_zero_lat_lon(self, mock_sensor: MagicMock) -> None:
        """Verifies zero lat/lon treated as valid values (not falsy None).

        Tests edge case where Null Island (0°, 0°) is intentional location.

        Business Context:
        Zero is valid for equator/prime meridian. Must not be treated as
        missing value requiring default substitution.

        Arrangement:
        1. Mock sensor in connected state.
        2. Prepare coordinates for Null Island (0.0°N, 0.0°E).
        3. Provider should accept zeros as explicit values.

        Action:
        Construct SensorCoordinateProvider with lat=0.0, lon=0.0.
        Tests falsy value handling (0.0 should not trigger default).

        Assertion Strategy:
        - _location.lat exactly 0.0 (not substituted with default).
        - _location.lon exactly 0.0 (not substituted with default).
        - Validates explicit None check prevents falsy value bug.

        Testing Principle:
        Validates edge case handling distinguishes explicit zero from
        missing value, preventing subtle coordinate calculation bugs.
        """
        provider = SensorCoordinateProvider(
            sensor=mock_sensor,
            lat=0.0,
            lon=0.0,
        )
        assert provider._location.lat == 0.0
        assert provider._location.lon == 0.0

    def test_init_with_none_params(self, mock_sensor: MagicMock) -> None:
        """Verifies None parameters trigger default value substitution.

        Tests that missing coordinates use sensible defaults (0, 0).

        Business Context:
        Default coordinates enable basic functionality when location
        configuration unavailable (testing, development environments).

        Arrangement:
        1. Mock sensor in connected state.
        2. Pass lat=None, lon=None explicitly.
        3. Provider should substitute default values.

        Action:
        Construct SensorCoordinateProvider with explicit None parameters.
        Triggers default value logic for missing coordinates.

        Assertion Strategy:
        - _location.lat becomes 0.0 (default substituted).
        - _location.lon becomes 0.0 (default substituted).
        - Validates None detection triggers default path.

        Testing Principle:
        Validates default value logic handles missing configuration,
        enabling graceful degradation when location unknown.
        """
        provider = SensorCoordinateProvider(
            sensor=mock_sensor,
            lat=None,
            lon=None,
        )
        assert provider._location.lat == 0.0
        assert provider._location.lon == 0.0

    def test_location_property(self, mock_sensor: MagicMock) -> None:
        """Verifies location property exposes configured LocationConfig.

        Tests public accessor for observer location metadata.

        Business Context:
        Applications need observer location for plate solving, weather
        correlation, and multi-site observation coordination.

        Arrangement:
        1. Create LocationConfig with Austin, TX coordinates.
        2. Initialize provider with this location object.
        3. Location stored internally as _location attribute.

        Action:
        Access provider.location property to retrieve configuration.
        Verifies public accessor exposes internal location data.

        Assertion Strategy:
        - property returns same LocationConfig instance.
        - Individual attributes accessible (lat, lon, elevation).
        - Validates encapsulation with public property access.

        Testing Principle:
        Validates public API exposes necessary configuration data
        while maintaining encapsulation of internal state.
        """
        loc = LocationConfig(lat=30.27, lon=-97.74, elevation=150.0)
        provider = SensorCoordinateProvider(sensor=mock_sensor, location=loc)

        assert provider.location == loc
        assert provider.location.lat == 30.27
        assert provider.location.lon == -97.74
        assert provider.location.elevation == 150.0

    def test_get_coordinates_not_connected(self, mock_sensor: MagicMock) -> None:
        """Verifies get_coordinates returns None when sensor disconnected.

        Tests graceful degradation when sensor hardware unavailable.

        Business Context:
        Captures should succeed even without sensor coordinates.
        Coordinate absence logged as warning, not fatal error.

        Arrangement:
        1. Mock sensor with connected=False (simulates disconnected state).
        2. Initialize provider with valid location configuration.
        3. Provider will check sensor.connected before reading.

        Action:
        Call provider.get_coordinates() on disconnected sensor.
        Early return path triggered by connection check.

        Assertion Strategy:
        - Returns None (no coordinates available).
        - Validates precondition check prevents read attempt.
        - Ensures graceful handling of unavailable sensor.

        Testing Principle:
        Validates error resilience ensures coordinate failures don't
        crash capture workflow, maintaining operational robustness.
        """
        mock_sensor.connected = False
        provider = SensorCoordinateProvider(sensor=mock_sensor, lat=30.0, lon=-97.0)

        result = provider.get_coordinates()
        assert result is None

    def test_get_coordinates_read_returns_none(self, mock_sensor: MagicMock) -> None:
        """Verifies get_coordinates returns None when sensor read fails.

        Tests handling of sensor read timeout or transient failure.

        Business Context:
        Sensor reads may timeout during recalibration or interference.
        Coordinate provider must handle gracefully without crashing.

        Arrangement:
        1. Mock sensor configured as connected.
        2. Mock read_sync() to return None (simulates read failure).
        3. Provider will attempt read but receive no data.

        Action:
        Call provider.get_coordinates() with failing sensor read.
        Read succeeds but returns None (timeout/transient error).

        Assertion Strategy:
        - Returns None (no coordinates from failed read).
        - Validates None check after read attempt.
        - Ensures read failures propagate as None result.

        Testing Principle:
        Validates error handling for transient sensor failures,
        ensuring captures continue despite temporary read issues.
        """
        mock_sensor.read_sync.return_value = None
        provider = SensorCoordinateProvider(sensor=mock_sensor, lat=30.0, lon=-97.0)

        result = provider.get_coordinates()
        assert result is None

    def test_get_coordinates_success(
        self, mock_sensor: MagicMock, mock_reading: MagicMock
    ) -> None:
        """Verifies successful coordinate retrieval with RA/Dec conversion.

        Tests complete happy-path workflow from sensor read to coordinate dict.

        Business Context:
        Frame metadata embedding requires ALT/AZ from sensor plus RA/Dec
        conversion using observer location for astrometry workflows.

        Arrangement:
        1. Mock sensor configured to return valid reading.
        2. Provider initialized with Austin, TX location.
        3. mock_reading contains ALT=45°, AZ=180°, temp=20°C, humidity=50%.

        Action:
        Call provider.get_coordinates() which:
        - Checks sensor.connected (True)
        - Calls sensor.read_sync() (returns mock_reading)
        - Extracts ALT/AZ from reading
        - Converts to RA/Dec using observer location
        - Packages all data into result dict

        Assertion Strategy:
        - Result is not None (successful retrieval).
        - altitude and azimuth match sensor reading.
        - temperature and humidity preserved from reading.
        - coordinate_source="sensor" identifies data origin.
        - Validates complete pipeline from read to formatted output.

        Testing Principle:
        Validates end-to-end coordinate pipeline ensures sensor data
        correctly flows through conversion to frame metadata format.
        """
        mock_sensor.read_sync.return_value = mock_reading
        provider = SensorCoordinateProvider(
            sensor=mock_sensor,
            lat=30.27,
            lon=-97.74,
            elevation=150.0,
        )

        result = provider.get_coordinates()

        assert result is not None
        assert result["altitude"] == 45.0
        assert result["azimuth"] == 180.0
        assert result["temperature"] == 20.0
        assert result["humidity"] == 50.0
        assert result["coordinate_source"] == "sensor"
        assert "ra" in result
        assert "dec" in result
        assert "ra_hms" in result
        assert "dec_dms" in result
        assert "coordinate_timestamp" in result

    def test_get_coordinates_exception_returns_none(
        self, mock_sensor: MagicMock
    ) -> None:
        """Verifies get_coordinates returns None when sensor read raises exception.

        Tests exception handling during sensor communication errors.

        Business Context:
        Serial communication errors (unplugged cable, hardware fault)
        should not crash capture pipeline. Graceful degradation required.

        Arrangement:
        1. Mock sensor configured as connected.
        2. Mock read_sync() to raise RuntimeError (simulates comm failure).
        3. Provider will catch exception during coordinate retrieval.

        Action:
        Call provider.get_coordinates() with error-throwing sensor.
        Exception caught and logged, None returned to caller.

        Assertion Strategy:
        - Returns None (exception suppressed).
        - Validates exception handler prevents crash.
        - Ensures hardware errors don't propagate to capture.

        Testing Principle:
        Validates exception resilience ensures hardware failures
        degrade gracefully rather than crashing observation session.
        """
        mock_sensor.read_sync.side_effect = RuntimeError("Sensor error")
        provider = SensorCoordinateProvider(sensor=mock_sensor, lat=30.0, lon=-97.0)

        result = provider.get_coordinates()
        assert result is None

    def test_location_config_location_param_takes_priority(
        self, mock_sensor: MagicMock
    ) -> None:
        """Verifies location parameter overrides individual lat/lon/elevation.

        Tests parameter precedence when both location object and individual
        parameters provided simultaneously.

        Business Context:
        When migrating config from individual params to LocationConfig,
        location object should take precedence to avoid confusion.

        Arrangement:
        1. Create LocationConfig with one set of coordinates (50°N, 100°W, 500m).
        2. Prepare conflicting individual parameters (30°N, 97°W, 100m).
        3. Pass both location object AND individual params to constructor.

        Action:
        Initialize provider with both location=loc and lat/lon/elevation params.
        Constructor must choose which parameters to use.

        Assertion Strategy:
        - _location contains values from location object (50, -100, 500).
        - Individual parameters ignored when location provided.
        - Validates explicit precedence rule prevents ambiguity.

        Testing Principle:
        Validates parameter precedence rule eliminates configuration
        ambiguity when multiple sources provide same data.
        """
        loc = LocationConfig(lat=50.0, lon=-100.0, elevation=500.0)
        provider = SensorCoordinateProvider(
            sensor=mock_sensor,
            lat=30.0,  # Should be ignored
            lon=-97.0,  # Should be ignored
            elevation=100.0,  # Should be ignored
            location=loc,
        )
        assert provider._location.lat == 50.0
        assert provider._location.lon == -100.0
        assert provider._location.elevation == 500.0


class TestSensorReadSync:
    """Tests for Sensor.read_sync() method."""

    def test_read_sync_not_connected(self) -> None:
        """Test read_sync returns None when not connected.

        Arrangement:
            Create Sensor without connecting.

        Action:
            Call read_sync() on disconnected sensor.

        Assertion Strategy:
            Verify None returned (no blocking on disconnected sensor).

        Testing Principle:
            Tests synchronous read safety check for disconnected state.
        """
        from telescope_mcp.devices.sensor import Sensor

        mock_driver = MagicMock()
        sensor = Sensor(mock_driver)

        result = sensor.read_sync()
        assert result is None

    def test_read_sync_instance_none(self) -> None:
        """Test read_sync returns None when _instance is None.

        Arrangement:
            Create Sensor, set connected=True but _instance=None (edge case).

        Action:
            Call read_sync() with null instance.

        Assertion Strategy:
            Verify None returned (defensive null check).

        Testing Principle:
            Tests defensive programming for inconsistent state.
        """
        from telescope_mcp.devices.sensor import Sensor

        mock_driver = MagicMock()
        sensor = Sensor(mock_driver)
        sensor._connected = True
        sensor._instance = None

        result = sensor.read_sync()
        assert result is None

    def test_read_sync_success(self) -> None:
        """Test read_sync returns reading on success.

        Arrangement:
            Create Sensor with connected instance that returns mock reading.

        Action:
            Call read_sync() to get synchronous reading.

        Assertion Strategy:
            Verify mock_reading returned and instance.read() called once.

        Testing Principle:
            Tests successful synchronous read path for blocking operations.
        """
        from telescope_mcp.devices.sensor import Sensor

        mock_driver = MagicMock()
        mock_instance = MagicMock()
        mock_reading = MagicMock()
        mock_instance.read.return_value = mock_reading

        sensor = Sensor(mock_driver)
        sensor._connected = True
        sensor._instance = mock_instance

        result = sensor.read_sync()
        assert result == mock_reading
        mock_instance.read.assert_called_once()

    def test_read_sync_exception_returns_none(self) -> None:
        """Test read_sync returns None on exception.

        Arrangement:
            Create Sensor with instance that raises on read().

        Action:
            Call read_sync() which triggers exception.

        Assertion Strategy:
            Verify None returned (exception swallowed for sync compatibility).

        Testing Principle:
            Tests exception handling in synchronous read for robustness.
        """
        from telescope_mcp.devices.sensor import Sensor

        mock_driver = MagicMock()
        mock_instance = MagicMock()
        mock_instance.read.side_effect = RuntimeError("Read error")

        sensor = Sensor(mock_driver)
        sensor._connected = True
        sensor._instance = mock_instance

        result = sensor.read_sync()
        assert result is None
