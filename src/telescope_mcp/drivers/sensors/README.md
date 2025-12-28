# Sensor Drivers Architecture

## 1. Component Overview

| Attribute | Value |
|-----------|-------|
| **Name** | `drivers.sensors` |
| **Type** | Package |
| **Responsibility** | Telescope orientation sensing via IMU (accelerometer/magnetometer) |
| **Context** | Hardware abstraction layer for position feedback |
| **Public Surface** | `SensorReading`, `SensorInstance`, `SensorDriver`, `ArduinoSensorDriver`, `DigitalTwinSensorDriver` |
| **Patterns** | Protocol-based DI, Factory (Driver→Instance), Digital Twin |
| **Language** | Python 3.13+ |
| **Stack** | pyserial, threading, dataclasses |
| **Entry Points** | `ArduinoSensorDriver.open()`, `DigitalTwinSensorDriver.open()` |
| **State** | Stateful (calibration, position cache, serial connection) |

### Key Decisions
- **Protocol-based abstraction**: `SensorInstance`/`SensorDriver` use Python `Protocol` with `...` (ellipsis) bodies = structural typing, implementations must provide all methods
- **Background reader thread**: Arduino instance uses daemon thread for continuous ~10Hz data
- **Offset calibration model**: `calibrated = scale * raw + offset` for position correction

### Protocol Pattern
```python
class SensorInstance(Protocol):  # pragma: no cover
    def read(self) -> SensorReading:
        ...  # Abstract - implementations MUST override
```
The `...` (ellipsis) marks abstract methods. `# pragma: no cover` excludes from coverage since protocol methods are never executed directly.

### Risks
- Serial port contention (only one process can own port)
- Thread safety: reader thread writes state, main thread reads
- Calibration drift over time without periodic recalibration

## 2. Code Layout

```
drivers/sensors/
├── __init__.py          # Public exports, re-exports SerialPort/PortEnumerator
├── types.py             # 🔒 SensorReading dataclass, SensorInstance/SensorDriver protocols
├── arduino.py           # ArduinoSensorInstance/Driver - real hardware via serial
└── twin.py              # DigitalTwinSensorInstance/Driver - simulation for testing
```

## 3. Public Surface

### 🔒 Frozen (ABI stable)

| Symbol | Signature | Stability | Change Impact |
|--------|-----------|-----------|---------------|
| `SensorReading` | `@dataclass(accelerometer, magnetometer, altitude, azimuth, temperature, humidity, timestamp, raw_values)` | 🔒 | Breaks all sensor consumers |
| `SensorInstance` | `Protocol: get_info(), read(), calibrate(), get_status(), reset(), close()` | 🔒 | Breaks device layer |
| `SensorDriver` | `Protocol: get_available_sensors(), open(), close()` | 🔒 | Breaks config/factory |

### ⚠️ Internal (may change)

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `ArduinoSensorInstance._create_with_serial()` | `classmethod(serial_port, port_name, start_reader) -> Self` | Testing factory |
| `ArduinoSensorDriver._create_with_enumerator()` | `classmethod(port_enumerator, serial_factory, baudrate) -> Self` | Testing factory |
| `ArduinoSensorInstance._parse_line()` | `(line: str) -> bool` | Parse serial data |
| `ArduinoSensorInstance._send_command()` | `(cmd, wait, timeout) -> str` | Serial command |
| `ArduinoSensorInstance._calibrate_magnetometer()` | `() -> str` | Mag calibration |
| `ArduinoSensorInstance._set_tilt_calibration()` | `(slope, intercept) -> None` | Tilt correction |
| `ArduinoSensorInstance._stop_output()` | `() -> None` | Pause streaming |
| `ArduinoSensorInstance._start_output()` | `() -> None` | Resume streaming |
| `DigitalTwinSensorInstance._set_position()` | `(altitude, azimuth) -> None` | Test control |
| `DigitalTwinSensorInstance._calibrate_magnetometer()` | `() -> dict` | Simulated mag cal |

### Data Contracts

**Input (calibrate)**:
```python
true_altitude: float  # 0-90 degrees
true_azimuth: float   # 0-360 degrees
```

**Output (read)**:
```python
SensorReading(
    accelerometer={'aX': float, 'aY': float, 'aZ': float},  # g units
    magnetometer={'mX': float, 'mY': float, 'mZ': float},   # µT
    altitude=float,      # 0-90° calibrated
    azimuth=float,       # 0-360° calibrated
    temperature=float,   # Celsius
    humidity=float,      # %RH
    timestamp=datetime,  # UTC
    raw_values=str,      # Tab-separated Arduino format
)
```

## 4. Dependencies

### depends_on
| Module | Purpose |
|--------|---------|
| `drivers.serial` | `SerialPort`, `PortEnumerator` protocols |
| `observability` | `get_logger()` structured logging |
| `pyserial` | Serial communication (runtime, optional for twin) |

### required_by
| Module | Purpose |
|--------|---------|
| `devices.sensor` | `Sensor` wrapper uses driver/instance |
| `drivers.config` | `create_sensor_driver()` factory |
| `tools.sensors` | MCP tool implementations |

### IO
- **Serial**: `/dev/ttyACM*` @ 115200 baud (Arduino)
- **Protocol**: Tab-separated values: `aX\taY\taZ\tmX\tmY\tmZ\ttemp\thumidity\r\n`
- **Commands**: `RESET`, `STATUS`, `CALIBRATE`, `STOP`, `START`

## 5. Invariants & Errors

### ⚠️ Must Preserve

| Invariant | Threshold | Verification |
|-----------|-----------|--------------|
| Altitude range | 0-90° | `assert 0 <= reading.altitude <= 90` |
| Azimuth range | 0-360° | `reading.azimuth % 360` normalization |
| Single instance per driver | 1 | `RuntimeError` if `open()` called twice |
| Background reader is daemon | Always | Thread exits with main program |

### Errors

| Error | When Raised |
|-------|-------------|
| `RuntimeError("Sensor is closed")` | `read()` after `close()` |
| `RuntimeError("No sensor data available yet")` | `read()` before first data received |
| `RuntimeError("Sensor already open")` | `open()` without `close()` |
| `RuntimeError("pyserial not installed")` | Arduino used without pyserial |

### Side Effects
- **Serial port**: Exclusive lock during connection
- **Thread**: Daemon thread spawned on Arduino open
- **State**: Calibration offsets persist until `reset()`

### Concurrency
- Reader thread writes `_accelerometer`, `_magnetometer`, `_last_update`
- Main thread reads cached values (eventual consistency, ~100ms staleness max)
- No explicit locking (dict assignment is atomic in CPython)

## 6. Usage

### Setup
```python
from telescope_mcp.drivers.sensors import (
    ArduinoSensorDriver,
    DigitalTwinSensorDriver,
)

# Real hardware
driver = ArduinoSensorDriver()
sensors = driver.get_available_sensors()
if sensors:
    instance = driver.open(sensors[0]["port"])
    reading = instance.read()
    instance.close()

# Testing (no hardware)
driver = DigitalTwinSensorDriver()
instance = driver.open()
instance._set_position(45.0, 180.0)  # Private: test-only
reading = instance.read()
```

### Config
| Variable | Default | Description |
|----------|---------|-------------|
| Baudrate | 115200 | Arduino serial speed |
| Sample rate | 10 Hz | Arduino streaming rate |

### Testing
```bash
pdm run pytest tests/drivers/sensors/ -v
```

### Pitfalls

| Issue | Fix |
|-------|-----|
| `No sensor data available` | Wait 0.5s after `open()` for first data |
| Port busy | Close other serial monitors (Arduino IDE) |
| Calibration lost | Calibration clears on `reset()` or reconnect |

## 7. AI-Accessibility Map

| Task | Target | Guards | Change Impact |
|------|--------|--------|---------------|
| Add sensor type | `types.py` + new file | Must impl `SensorInstance` protocol | Low (additive) |
| Modify reading format | `SensorReading` in `types.py` | 🔒 Breaking change | High (all consumers) |
| Add calibration method | `SensorInstance` protocol | 🔒 Protocol change | High (all impls) |
| Change serial protocol | `arduino.py:_parse_line()` | Must match Arduino firmware | Medium |
| Adjust noise model | `twin.py:read()` | Test-only impact | Low |
| Add driver discovery | Driver class | Follow `get_available_sensors()` pattern | Low |

## 8. Architecture Diagram

```mermaid
%%{init: {'theme': 'neutral'}}%%
classDiagram
    class SensorDriver {
        <<protocol>>
        +get_available_sensors()* list~dict~
        +open(sensor_id)* SensorInstance
        +close()*
    }

    class SensorInstance {
        <<protocol>>
        +get_info()* dict
        +read()* SensorReading
        +calibrate(alt, az)*
        +get_status()* dict
        +reset()*
        +close()*
    }

    class SensorReading {
        <<dataclass>>
        +accelerometer: dict
        +magnetometer: dict
        +altitude: float
        +azimuth: float
        +temperature: float
        +humidity: float
        +timestamp: datetime
    }

    class ArduinoSensorDriver {
        -_baudrate: int
        -_instance: ArduinoSensorInstance
        +get_available_sensors() list~dict~
        +open(sensor_id) ArduinoSensorInstance
        +close()
    }

    class ArduinoSensorInstance {
        -_serial: SerialPort
        -_reader_thread: Thread
        -_cal_alt_offset: float
        +get_info() dict
        +read() SensorReading
        +calibrate(alt, az)
        +get_status() dict
        +reset()
        +close()
        -_parse_line(line) bool
        -_send_command(cmd) str
        -_calibrate_magnetometer() str
        -_set_tilt_calibration(slope, intercept)
        -_stop_output()
        -_start_output()
    }

    class DigitalTwinSensorDriver {
        -_config: DigitalTwinSensorConfig
        -_instance: DigitalTwinSensorInstance
        +get_available_sensors() list~dict~
        +open(sensor_id) DigitalTwinSensorInstance
        +close()
    }

    class DigitalTwinSensorInstance {
        -_true_altitude: float
        -_true_azimuth: float
        +get_info() dict
        +read() SensorReading
        +calibrate(alt, az)
        +get_status() dict
        +reset()
        +close()
        -_set_position(alt, az)
        -_calibrate_magnetometer() dict
    }

    SensorDriver <|.. ArduinoSensorDriver : implements
    SensorDriver <|.. DigitalTwinSensorDriver : implements
    SensorInstance <|.. ArduinoSensorInstance : implements
    SensorInstance <|.. DigitalTwinSensorInstance : implements
    ArduinoSensorDriver --> ArduinoSensorInstance : creates
    DigitalTwinSensorDriver --> DigitalTwinSensorInstance : creates
    SensorInstance --> SensorReading : returns
```

```mermaid
%%{init: {'theme': 'neutral'}}%%
flowchart TB
    subgraph External
        Arduino[Arduino Nano BLE33]
        Serial[/dev/ttyACM*/]
    end

    subgraph drivers.sensors
        AD[ArduinoSensorDriver]
        AI[ArduinoSensorInstance]
        TD[DigitalTwinSensorDriver]
        TI[DigitalTwinSensorInstance]
        Types[types.py<br/>Protocols + SensorReading]
    end

    subgraph Consumers
        Device[devices.sensor.Sensor]
        Config[drivers.config]
        Tools[tools.sensors]
    end

    Arduino -->|USB| Serial
    Serial -->|pyserial| AI
    AD --> AI
    TD --> TI

    Types -.->|protocol| AI
    Types -.->|protocol| TI

    Device --> AD
    Device --> TD
    Config --> AD
    Config --> TD
    Tools --> Device
```
