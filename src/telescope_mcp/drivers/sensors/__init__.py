"""Position sensor driver.

Reads altitude and azimuth position from sensors (TBD - accelerometer, encoder, etc.)
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TelescopePosition:
    """Current telescope position in degrees."""
    altitude: float  # 0-90 degrees
    azimuth: float   # 0-360 degrees


class PositionSensor(Protocol):
    """Protocol for position sensor implementations.
    
    Defines interface for reading telescope position from sensors
    like accelerometers, encoders, or IMUs.
    """

    def read(self) -> TelescopePosition:
        """Read current telescope pointing position from hardware sensors.
        
        Queries position sensors (accelerometer, IMU, or encoders) to determine
        current telescope orientation. Returns absolute angles in standard
        coordinate system. Call frequently for real-time tracking or validation.
        
        Business context: Essential for closed-loop tracking, goto verification,
        and manual positioning feedback. Enables comparison of commanded vs actual
        position. Required for any position-dependent automation.
        
        Args:
            None. Reads from configured hardware sensors.
        
        Returns:
            TelescopePosition dataclass with:
            - altitude: Elevation angle in degrees, 0 (horizon) to 90 (zenith)
            - azimuth: Rotation angle in degrees, 0 (north) clockwise to 360
            Values are post-calibration absolute coordinates.
        
        Raises:
            RuntimeError: If sensor hardware not responding or disconnected.
            CalibrationError: If sensor not calibrated (requires calibrate() call).
            ValueError: If sensor returns out-of-range values.
        
        Example:
            >>> sensor = AccelerometerSensor()
            >>> pos = sensor.read()
            >>> print(f"Pointing at altitude={pos.altitude:.1f}°, azimuth={pos.azimuth:.1f}°")
            Pointing at altitude=45.0°, azimuth=180.0°
        """
        ...

    def calibrate(self, altitude: float, azimuth: float) -> None:
        """Calibrate sensor to match known telescope position.
        
        Sets sensor reference frame by asserting current position. Point telescope
        at known target (star, landmark) and provide true coordinates. Subsequent
        read() calls will be relative to this calibration. Required after power-on
        or if significant drift detected.
        
        Business context: Establishes accurate absolute positioning for goto and
        tracking. Without calibration, sensor readings may have arbitrary offset.
        Periodic recalibration compensates for sensor drift over time.
        
        Args:
            altitude: True altitude in degrees (0-90). 0=horizon, 90=zenith.
                Must match actual telescope pointing at calibration time.
            azimuth: True azimuth in degrees (0-360). 0=north, increases clockwise.
                Must match actual telescope pointing at calibration time.
        
        Returns:
            None. Calibration takes effect immediately for next read().
        
        Raises:
            ValueError: If altitude not in [0, 90] or azimuth not in [0, 360].
            RuntimeError: If sensor hardware error during calibration.
            CalibrationError: If sensor unable to accept calibration (hardware issue).
        
        Example:
            >>> sensor = AccelerometerSensor()
            >>> # Point telescope at Polaris (known position)
            >>> sensor.calibrate(altitude=40.0, azimuth=0.0)
            >>> # Now sensor returns calibrated absolute coordinates
        """
        ...


class StubPositionSensor:
    """Stub implementation for development without hardware.
    
    Returns a fixed position that can be updated via calibrate().
    """

    def __init__(self) -> None:
        """Initialize stub position sensor with default telescope pointing.
        
        Creates simulated encoder sensor returning fixed position (45° altitude, 180° azimuth)
        until calibrated. No hardware communication. Position fixed until calibrate() called.
        
        Business context: Enables development of position-aware telescope features without physical
        encoders or IMU sensors. Developers test goto accuracy calculations, plate solving integration,
        position displays in UI. CI/CD validates positioning logic with predictable values. Critical
        for development environments lacking sensor hardware.
        
        Implementation details: Initializes _position as TelescopePosition(45.0, 180.0). read() returns
        this value. calibrate() updates stored position. No drift, noise, or hardware errors. Used
        by DriverFactory in DIGITAL_TWIN mode.
        
        Args:
            None.
        
        Returns:
            None. Sensor ready for read() calls.
        
        Raises:
            None. Stub never fails.
        
        Example:
            >>> sensor = StubPositionSensor()
            >>> pos = sensor.read()  # TelescopePosition(altitude=45.0, azimuth=180.0)
            >>> sensor.calibrate(TelescopePosition(67.5, 123.4))
            >>> pos = sensor.read()  # TelescopePosition(altitude=67.5, azimuth=123.4)
        """
        self._position = TelescopePosition(altitude=45.0, azimuth=180.0)

    def read(self) -> TelescopePosition:
        """Return simulated position from internal state.
        
        Returns last position set by calibrate() or default (45°, 180°). Does
        not reflect actual motor movements - purely for testing position-dependent
        logic without hardware. Position is fixed until calibrate() called.
        
        Business context: Enables development of position-aware features without
        physical sensors. Validates UI position displays, tracking algorithms,
        and goto logic against predictable values.
        
        Args:
            None.
        
        Returns:
            TelescopePosition with:
            - altitude: Simulated elevation (default 45.0°)
            - azimuth: Simulated rotation (default 180.0°)
            Values match last calibrate() call or initialization defaults.
        
        Raises:
            None. Stub implementation never fails.
        
        Example:
            >>> stub = StubPositionSensor()
            >>> pos = stub.read()
            >>> print(f"{pos.altitude:.1f}°, {pos.azimuth:.1f}°")
            45.0°, 180.0°
            >>> stub.calibrate(30.0, 90.0)
            >>> pos = stub.read()
            >>> print(f"{pos.altitude:.1f}°, {pos.azimuth:.1f}°")
            30.0°, 90.0°
        """
        return self._position

    def calibrate(self, altitude: float, azimuth: float) -> None:
        """Set simulated position to specified test values.
        
        Updates internal position state immediately. Prints action for debugging.
        No validation of range limits - accepts any float values for testing
        edge cases and error handling.
        
        Business context: Enables testing of calibration workflows, position
        changes, and coordinate validation logic. Simulates successful calibration
        for UI and automation testing without hardware.
        
        Args:
            altitude: Altitude in degrees. Typically 0-90 but not enforced.
            azimuth: Azimuth in degrees. Typically 0-360 but not enforced.
        
        Returns:
            None. Position change is immediate.
        
        Raises:
            None. Stub implementation accepts any values including out-of-range.
        
        Example:
            >>> stub = StubPositionSensor()
            >>> stub.calibrate(75.0, 270.0)
            [STUB] Calibrated to alt=75.0, az=270.0
            >>> pos = stub.read()
            >>> print(f"Now at {pos.altitude:.1f}°, {pos.azimuth:.1f}°")
            Now at 75.0°, 270.0°
        """
        self._position = TelescopePosition(altitude=altitude, azimuth=azimuth)
        print(f"[STUB] Calibrated to alt={altitude}, az={azimuth}")


# TODO: Implement real sensor drivers
# - AccelerometerSensor (for IMU-based sensing)
# - EncoderSensor (for rotary encoder-based sensing)
