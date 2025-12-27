# Sensor Calibration Architecture

## Overview

The telescope uses an Arduino Nano BLE33 Sense as an orientation sensor to provide rapid ALT/AZ (Altitude/Azimuth) position estimates. This document describes the sensor calibration workflow that combines IMU-based orientation with camera-based plate solving for accurate celestial positioning.

## System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Telescope Orientation System                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────┐                      ┌─────────────────────────┐  │
│   │  BLE33 Sense    │                      │  Imaging Camera         │  │
│   │  IMU Sensor     │                      │  (Plate Solving)        │  │
│   ├─────────────────┤                      ├─────────────────────────┤  │
│   │ • Accelerometer │                      │ • Star field image      │  │
│   │ • Magnetometer  │                      │ • Astrometry.net or     │  │
│   │ • Temperature   │                      │   local solver          │  │
│   │ • Humidity      │                      │ • True RA/DEC output    │  │
│   └────────┬────────┘                      └────────────┬────────────┘  │
│            │                                            │               │
│            ▼                                            ▼               │
│   ┌─────────────────┐                      ┌─────────────────────────┐  │
│   │ Raw ALT/AZ      │                      │ True RA/DEC             │  │
│   │ (approximate)   │                      │ (precise)               │  │
│   └────────┬────────┘                      └────────────┬────────────┘  │
│            │                                            │               │
│            │         ┌─────────────────────┐            │               │
│            └────────▶│ Transform Function  │◀───────────┘               │
│                      │ (Calibration)       │                            │
│                      ├─────────────────────┤                            │
│                      │ • Offset correction │                            │
│                      │ • Rotation matrix   │                            │
│                      │ • Scale factors     │                            │
│                      └─────────┬───────────┘                            │
│                                │                                        │
│                                ▼                                        │
│                      ┌─────────────────────┐                            │
│                      │ Corrected ALT/AZ    │                            │
│                      │ (real-time, ~10Hz)  │                            │
│                      └─────────────────────┘                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Coordinate Systems

### ALT/AZ (Horizontal Coordinates)
- **Altitude (ALT)**: Angle above horizon (0° = horizon, 90° = zenith)
- **Azimuth (AZ)**: Compass bearing (0° = North, 90° = East, 180° = South, 270° = West)
- Local to observer's position and time

### RA/DEC (Equatorial Coordinates)
- **Right Ascension (RA)**: Celestial longitude (0-24 hours)
- **Declination (DEC)**: Celestial latitude (-90° to +90°)
- Fixed on the celestial sphere, independent of observer location

### Conversion Requirements
- Observer latitude/longitude
- Current UTC time (for sidereal time calculation)
- Atmospheric refraction correction (optional)

## Sensor Hardware

### Arduino Nano BLE33 Sense
- **LSM9DS1 IMU**
  - 3-axis accelerometer → Tilt (Altitude)
  - 3-axis magnetometer → Compass heading (Azimuth)
- **HTS221**
  - Temperature sensor
  - Humidity sensor (for atmospheric correction)

### Serial Protocol
```
Output: aX\taY\taZ\tmX\tmY\tmZ\ttemperature\thumidity\r\n
Rate: 10 Hz (100ms interval)
Baud: 115200
```

### Commands
| Command | Description |
|---------|-------------|
| `RESET` | Reinitialize all sensors |
| `STATUS` | Report sensor status |
| `CALIBRATE` | Run magnetometer hard-iron calibration |
| `STOP` | Pause sensor output |
| `START` | Resume sensor output |

## Calibration Workflow

### Phase 1: Sensor Self-Calibration

1. **Accelerometer calibration** (built into tilt calculation)
   - Uses gravity vector to determine tilt
   - Linear offset correction from known reference angles (0° and 90°)

2. **Magnetometer hard-iron calibration**
   - Run `CALIBRATE` command
   - Rotate sensor in all directions for 10 seconds
   - Calculates center offset of min/max envelope
   - Corrects for local magnetic field distortion

### Phase 2: Transform Calibration

This is the key calibration that maps sensor readings to true celestial coordinates.

```python
class SensorCalibration:
    """Calibrate IMU sensor against plate-solved camera images."""

    def add_calibration_point(
        self,
        sensor_alt: float,
        sensor_az: float,
        true_ra: float,
        true_dec: float,
        timestamp: datetime
    ):
        """Add a calibration point from plate-solved image."""
        # Convert RA/DEC to ALT/AZ at given time
        true_alt, true_az = radec_to_altaz(true_ra, true_dec, timestamp)
        # Store (sensor, true) pair
        self.points.append((sensor_alt, sensor_az, true_alt, true_az))

    def compute_transform(self):
        """Compute best-fit transformation from calibration points."""
        # Calculate:
        # - Altitude offset and scale
        # - Azimuth offset (includes magnetic declination)
        # - Cross-coupling terms (optional)

    def apply(self, sensor_alt: float, sensor_az: float) -> tuple[float, float]:
        """Apply transform to get corrected ALT/AZ."""
        return corrected_alt, corrected_az
```

### Phase 3: Runtime Operation

1. **Initial slew**: Use corrected sensor ALT/AZ for approximate pointing
2. **Plate solve**: Take image, solve for precise RA/DEC
3. **Refinement**: Adjust pointing based on plate solution
4. **Tracking**: Continue using sensor for motion feedback

## Transform Model

### Simple Linear Model
For initial implementation, use offset + scale:

```
corrected_alt = m_alt * sensor_alt + b_alt
corrected_az  = m_az  * sensor_az  + b_az
```

Where:
- `m` = scale factor (ideally 1.0)
- `b` = offset (mounting error + magnetic declination for AZ)

### Advanced Model (Future)
For better accuracy, consider:
- 3D rotation matrix for cross-axis coupling
- Non-linear corrections for accelerometer
- Temperature compensation
- Magnetic field strength normalization

## Error Sources

| Source | Typical Error | Mitigation |
|--------|---------------|------------|
| Sensor mounting angle | 1-5° | Transform calibration |
| Magnetic declination | 0-20° | AZ offset in transform |
| Hard-iron distortion | 5-50µT | `CALIBRATE` command |
| Soft-iron distortion | Variable | Future: ellipsoid fit |
| Accelerometer non-linearity | 0.5-1° | Linear fit at 0° and 90° |
| Temperature drift | 0.1°/10°C | Temperature compensation |

## Integration with MCP Tools

The sensor system integrates with telescope-mcp as:

1. **Position tool**: `get_telescope_position()` returns current ALT/AZ
2. **Calibration tool**: `calibrate_sensor(ra, dec, timestamp)` adds calibration point
3. **Session logging**: Calibration events recorded in ASDF session files

### Sensor Device Architecture

The sensor system follows a driver injection pattern for testability:

```
┌─────────────────────────────────────────────┐
│              Sensor Device                   │
│    (high-level: connect/read/calibrate)     │
└─────────────────────────────────────────────┘
                    │
         SensorDriver Protocol
           ┌───────┴───────┐
           ▼               ▼
┌─────────────────┐  ┌─────────────────┐
│  DigitalTwin    │  │    Arduino      │
│  SensorDriver   │  │  SensorDriver   │
│  (simulation)   │  │  (serial USB)   │
└─────────────────┘  └─────────────────┘
```

**Key Classes:**
- `Sensor` - High-level device with connect/disconnect, read, calibrate
- `SensorConfig` - Configuration (auto_connect, reconnect_on_error)
- `SensorInfo` - Sensor capabilities and metadata
- `SensorReading` - Complete reading (accel, mag, alt, az, temp, humidity)

**Driver Implementations:**
- `DigitalTwinSensorDriver` - Simulated sensor for testing
- `ArduinoSensorDriver` - Real hardware via serial USB

### Server Configuration

The MCP server accepts location arguments for proper ALT/AZ ↔ RA/DEC conversion:

```json
{
  "servers": {
    "telescope-mcp": {
      "command": "/path/to/.venv/bin/python",
      "args": [
        "-m", "telescope_mcp.server",
        "--dashboard-host", "127.0.0.1",
        "--dashboard-port", "8080",
        "--mode", "digital_twin",
        "--latitude", "40.7128",
        "--longitude", "-74.0060",
        "--height", "10.0"
      ]
    }
  }
}
```

| Argument | Type | Description |
|----------|------|-------------|
| `--latitude` | float | Observer latitude in degrees (N positive) |
| `--longitude` | float | Observer longitude in degrees (E positive) |
| `--height` | float | Height above sea level in meters (default: 0) |

These values are required for accurate coordinate conversions using astropy's `EarthLocation`.

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Serial USB  │────▶│ Python      │────▶│ MCP Server  │
│ /dev/ttyACM0│     │ pyserial    │     │ Tools       │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Threaded    │
                    │ Reader      │
                    │ (10 Hz)     │
                    └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Accel    │ │ Mag      │ │ Environ  │
       │ → ALT    │ │ → AZ     │ │ Temp/Hum │
       └──────────┘ └──────────┘ └──────────┘
              │            │
              ▼            ▼
       ┌─────────────────────────┐
       │ Transform Function      │
       │ (from calibration)      │
       └─────────────────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Corrected ALT/AZ        │
       │ → Used for pointing     │
       └─────────────────────────┘
```

## Future Enhancements

1. **Gyroscope integration**: LSM9DS1 has gyro - could add for smoother tracking
2. **Kalman filter**: Fuse accelerometer, magnetometer, and gyro
3. **EEPROM storage**: Save calibration on Arduino, persist across power cycles
4. **Automatic re-calibration**: Periodic plate-solve to update transform
5. **Weather integration**: Use humidity/temperature for atmospheric refraction

## References

- [Using an Accelerometer for Inclination Sensing](https://www.digikey.com/en/articles/using-an-accelerometer-for-inclination-sensing)
- [W3C Magnetometer API](https://www.w3.org/TR/magnetometer/)
- [Arduino Nano 33 BLE Sense Datasheet](./ABX00031-datasheet.pdf)
- [Astropy Coordinates](https://docs.astropy.org/en/stable/coordinates/)
