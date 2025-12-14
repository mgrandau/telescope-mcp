# Telescope Control Drivers Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Linux Ubuntu Box (at telescope site)                                       │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  telescope-mcp Server                                                 │  │
│  │                                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Camera    │  │   Motor     │  │  Position   │  │    Web      │  │  │
│  │  │   Driver    │  │   Driver    │  │   Sensor    │  │  Dashboard  │  │  │
│  │  │  (zwoasi)   │  │  (serial)   │  │   (IMU)     │  │  (FastAPI)  │  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │  │
│  │         │                │                │                │         │  │
│  └─────────┼────────────────┼────────────────┼────────────────┼─────────┘  │
│            │                │                │                │            │
│            ▼                ▼                ▼                ▼            │
│      ┌──────────┐    ┌──────────┐    ┌──────────┐      ┌──────────┐       │
│      │   USB    │    │  Serial  │    │   I2C    │      │  HTTP    │       │
│      │  Port    │    │  Port    │    │   Bus    │      │  :8080   │       │
│      └────┬─────┘    └────┬─────┘    └────┬─────┘      └────┬─────┘       │
└───────────┼───────────────┼───────────────┼─────────────────┼─────────────┘
            │               │               │                 │
            ▼               ▼               ▼                 │
     ┌────────────┐  ┌────────────┐  ┌────────────┐           │
     │ ZWO ASI    │  │  Stepper   │  │    IMU     │           │
     │ Cameras    │  │  Motors    │  │  Sensor    │           │
     │ (2x)       │  │ (Alt/Az)   │  │            │           │
     └────────────┘  └────────────┘  └────────────┘           │
                                                              │
                                                         Network/WiFi
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │  Remote Client   │
                                                    │  (Browser)       │
                                                    │  - Live camera   │
                                                    │  - Motor control │
                                                    │  - Position view │
                                                    └──────────────────┘
```

## Hardware Components

### Cameras (USB)
- **Finder Camera**: ZWO ASI camera mounted on finder scope
- **Main Camera**: ZWO ASI camera at telescope eyepiece/OTA

Both cameras connect via USB and are controlled through the ZWO ASI SDK.

### Motors (Serial)
- **Altitude Motor**: NEMA 23 stepper - controls up/down (0° to 90°)
- **Azimuth Motor**: NEMA 17 stepper - controls rotation (belt-driven, ~270° range)

Motors connect via serial port to a custom stepper controller board.

### Position Sensor (I2C)
- **IMU/Accelerometer**: Provides real-time altitude and azimuth readings
- Used for closed-loop position feedback and goto functionality

---

## Driver Modules

### `asi_sdk/` - ZWO ASI Camera 2 SDK
The native shared library for controlling ZWO ASI cameras.

```
asi_sdk/
├── __init__.py              # SDK path helper functions
├── asi.rules                # udev rules for USB permissions
└── x64/
    └── libASICamera2.so.1.40
```

**Usage:**
```python
from telescope_mcp.drivers.asi_sdk import get_sdk_library_path
import zwoasi as asi

asi.init(get_sdk_library_path())
cameras = asi.list_cameras()
```

**USB Setup (one-time):**
```bash
sudo cp asi_sdk/asi.rules /etc/udev/rules.d/99-asi.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### `motors/` - Stepper Motor Control
Controls the altitude and azimuth stepper motors via serial connection.

```
motors/
├── __init__.py              # MotorController protocol & StubMotorController
└── serial_controller.py     # (TODO) Real serial implementation
```

**Motor Specifications:**
| Axis | Motor | Range | Steps | Notes |
|------|-------|-------|-------|-------|
| Altitude | NEMA 23 | 0° - 90° | 0 - 140,000 | Home = 90° (zenith) |
| Azimuth | NEMA 17 | ~270° | Center = home | Belt-driven |

**Serial Protocol:**
- `A0` / `A1` - Select axis (0=altitude, 1=azimuth)
- `o{steps}` - Move to absolute position
- `?` - Help/status

### `sensors/` - Position Sensing
Reads altitude and azimuth from IMU/accelerometer sensor.

```
sensors/
├── __init__.py              # PositionSensor protocol & StubPositionSensor
└── imu_sensor.py            # (TODO) Real IMU implementation
```

**Provides:**
- Real-time alt/az position in degrees
- Calibration to known reference point
- Closed-loop feedback for goto operations

### `config.py` - Hardware Configuration
Centralized configuration for all hardware parameters.

---

## Data Flow

### Camera Capture
```
Browser Request → Web Dashboard → Camera Tool → zwoasi → ASI SDK → USB → Camera
                                                                          │
Browser ← JPEG/Stream ← Web Dashboard ← Camera Tool ← numpy array ←───────┘
```

### Motor Movement
```
Browser Request → Web Dashboard → Motor Tool → Serial Controller → Motor
                                                                     │
Browser ← Status ← Web Dashboard ← Motor Tool ← Position Sensor ←────┘
```

### Goto Operation (Closed Loop)
```
1. User requests goto(alt=45°, az=180°)
2. Motor tool calculates required steps
3. Motors move toward target
4. Position sensor provides feedback
5. Loop until position within tolerance
6. Report completion to user
```

---

## Network Configuration

The telescope-mcp server exposes:

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| Web Dashboard | 8080 | HTTP | Browser UI for control |
| MCP Server | stdio | JSON-RPC | AI assistant integration |

**Remote Access:**
- Ensure Ubuntu box has static IP or hostname
- Configure firewall to allow port 8080
- Access via `http://<telescope-ip>:8080`

---

## File Structure

```
drivers/
├── README.md           # This file
├── __init__.py
├── config.py           # Hardware configuration
├── asi_sdk/            # ZWO ASI Camera 2 SDK
│   ├── __init__.py
│   ├── asi.rules
│   └── x64/
│       └── libASICamera2.so.1.40
├── motors/             # Stepper motor control
│   └── __init__.py
├── sensors/            # Position sensing (IMU)
│   └── __init__.py
└── pyasi/              # (Legacy - replaced by zwoasi + asi_sdk)
    └── __init__.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `zwoasi` | Python bindings for ASI cameras |
| `pyserial` | Serial communication for motors |
| `smbus2` | I2C communication for IMU (optional) |
| `numpy` | Image array handling |
| `opencv-python` | Image processing |
| `fastapi` | Web dashboard backend |

---

## Future Enhancements

- [ ] Multiple motor controller backends (GPIO, Arduino)
- [ ] Encoder-based position sensing
- [ ] Celestial coordinate conversion (RA/Dec ↔ Alt/Az)
- [ ] Automated star alignment
