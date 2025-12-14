# telescope-control → telescope-mcp Port TODO

This document outlines the steps needed to port functionality from `telescope-control` to `telescope-mcp`.

---

## Phase 1: PyASI Camera Driver (Core)

### 1.1 SDK Setup ✅ (Completed 2024-12-14)
- [x] Using `python-zwoasi` PyPI package instead of custom Cython bindings
- [x] Added ZWO ASI SDK V1.40 (`libASICamera2.so.1.40`) for x64 Linux
- [x] Created `telescope-mcp/src/telescope_mcp/drivers/asi_sdk/` with path helpers
- [x] SDK auto-initializes from `get_sdk_library_path()`

### 1.2 Camera Driver ✅ (Completed 2024-12-14)
- [x] Using `zwoasi` package (ctypes-based, no compilation needed)
- [x] Camera access via `asi.Camera(id)` - handles open/close
- [x] Control via `camera.set_control_value()` / `get_control_value()`
- [x] Capture via `camera.start_exposure()` + `camera.get_data_after_exposure()`

### 1.3 Implement Camera MCP Tools ✅ (Completed 2024-12-14)
- [x] Implement `_list_cameras()` in `tools/cameras.py`
- [x] Implement `_get_camera_info()` in `tools/cameras.py`
- [x] Implement `_capture_frame()` - return base64 JPEG
- [x] Implement `_set_camera_control()` 
- [x] Implement `_get_camera_control()`
- [ ] Add new tool: `start_video_capture` (streaming support)
- [ ] Add new tool: `stop_video_capture`
- [ ] Add new tool: `apply_overlay` (for finder camera alignment)

---

## Phase 2: Motor Control

### 2.1 Serial Motor Driver
- [ ] Create `telescope-mcp/src/telescope_mcp/drivers/motors/serial_controller.py`
- [ ] Port serial communication from `telescope-control/notebooks/motor-control/motor-control.ipynb`
- [ ] Implement `SerialMotorController` class following existing `MotorController` protocol
- [ ] Add device discovery (list serial ports)
- [ ] Add connection management (open/close/reconnect)
- [ ] Implement command protocol (`A0`, `o{steps}`, etc.)

### 2.2 Motor Configuration
- [ ] Add motor calibration data to `drivers/config.py`
  - Axis 0 (altitude): 0-140000 steps = 0-90 degrees
  - Axis 1 (azimuth): center = home, ~270 degrees range
- [ ] Add steps-per-degree conversion functions
- [ ] Add physical limit enforcement (soft limits)
- [ ] Add backlash compensation

### 2.3 Wire Up Motor MCP Tools
- [ ] Implement `move_altitude()` in `tools/motors.py`
- [ ] Implement `move_azimuth()` in `tools/motors.py`
- [ ] Implement `stop_motors()` - emergency stop
- [ ] Implement `get_motor_status()` 
- [ ] Implement `home_motors()`
- [ ] Add new tool: `set_motor_speed` (default speed config)
- [ ] Add new tool: `get_motor_limits` (report physical limits)

---

## Phase 3: Position Sensing

### 3.1 Evaluate Sensor Options
- [ ] Review `telescope-control/notebooks/arduino-nano-ble33-sense.ipynb`
- [ ] Review `telescope-control/notebooks/thunderboard-sense-2.ipynb`
- [ ] Decide on sensor hardware (accelerometer/IMU vs encoder)
- [ ] Document sensor selection in `docs/`

### 3.2 Implement Position Driver
- [ ] Create concrete sensor implementation in `drivers/sensors/`
- [ ] Implement `PositionSensor` protocol methods
- [ ] Add calibration storage/persistence
- [ ] Add sensor fusion if using multiple sensors

### 3.3 Wire Up Position MCP Tools
- [ ] Implement `get_position()` in `tools/position.py`
- [ ] Implement `calibrate_position()`
- [ ] Implement `goto_position()` - closed-loop control with motor + sensor

---

## Phase 4: Alignment & Goto

### 4.1 Star Alignment
- [ ] Port concepts from `telescope-control/notebooks/align-telescope.ipynb`
- [ ] Port concepts from `telescope-control/notebooks/goto-scope.ipynb`
- [ ] Add tool: `align_to_star` (set known position from star)
- [ ] Add tool: `get_celestial_position` (convert alt/az to RA/Dec)
- [ ] Add time/location handling for coordinate conversion

### 4.2 Camera Alignment
- [ ] Port from `telescope-control/notebooks/cameras/camera_alignment.ipynb`
- [ ] Add tool: `capture_alignment_frame`
- [ ] Add tool: `set_finder_overlay_offset`
- [ ] Store overlay configuration

---

## Phase 5: Web Dashboard Integration

### 5.1 Camera Streaming
- [ ] Add live camera preview to web dashboard
- [ ] Add camera control panel (exposure, gain, etc.)
- [ ] Add frame capture button with download

### 5.2 Motor Control UI
- [ ] Add directional buttons (up/down/left/right)
- [ ] Add speed slider
- [ ] Add position display
- [ ] Add emergency stop button (prominent!)

### 5.3 Status Display
- [ ] Show connected hardware status
- [ ] Show current position (alt/az)
- [ ] Show camera settings

---

## Phase 6: Configuration & Persistence

### 6.1 Hardware Config
- [ ] Extend `drivers/config.py` for all hardware settings
- [ ] Add YAML/JSON config file support
- [ ] Add per-session vs persistent settings

### 6.2 Calibration Storage
- [ ] Store motor calibration data
- [ ] Store position sensor calibration
- [ ] Store camera overlay offsets
- [ ] Store last known position (for session resume)

---

## Phase 7: Documentation & Testing

### 7.1 Documentation
- [ ] Document hardware setup requirements
- [ ] Document MCP tool API
- [ ] Port relevant notebooks as usage examples
- [ ] Add troubleshooting guide

### 7.2 Testing
- [ ] Add unit tests for drivers (with mocks)
- [ ] Add integration tests for MCP tools
- [ ] Add hardware-in-loop tests (optional, CI skip)
- [ ] Test on Raspberry Pi target

---

## Dependencies to Add

```toml
# pyproject.toml additions
[build-system]
requires = ["setuptools", "cython", "numpy"]

[project.optional-dependencies]
camera = ["opencv-python", "numpy"]
motors = ["pyserial"]
sensors = ["smbus2"]  # For I2C sensors
```

---

## Files to Port (Reference)

| Source (telescope-control)                | Destination (telescope-mcp)                          |
|------------------------------------------|-----------------------------------------------------|
| `pyasi/asi.pyx`                          | `src/telescope_mcp/drivers/pyasi/asi.pyx`           |
| `pyasi/casi.pxd`                         | `src/telescope_mcp/drivers/pyasi/casi.pxd`          |
| `ASI_linux_mac_SDK_V1.27/`               | `src/telescope_mcp/drivers/pyasi/sdk/`              |
| `src/cameras/asi_camera.py`              | `src/telescope_mcp/drivers/pyasi/camera.py`         |
| `src/cameras/finder_camera.py`           | `src/telescope_mcp/drivers/pyasi/finder_camera.py`  |
| `notebooks/motor-control/motor-control.ipynb` | `src/telescope_mcp/drivers/motors/serial_controller.py` |

---

## Priority Order

1. **PyASI Camera Driver** - Core functionality, most complex
2. **Motor Control** - Essential for telescope operation  
3. **Position Sensing** - Required for goto functionality
4. **Alignment/Goto** - Advanced features
5. **Web Dashboard** - Nice to have UI improvements
6. **Config/Persistence** - Polish
7. **Docs/Testing** - Ongoing throughout

---

*Last updated: 2025-12-13*
