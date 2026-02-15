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

### 1.4 Implement ASICameraDriver ✅ (Completed 2025-12-17)
- [x] Created `drivers/cameras/asi.py` wrapping zwoasi
- [x] Implemented `ASICameraDriver` following CameraDriver protocol
- [x] Implemented `ASICameraInstance` following CameraInstance protocol
- [x] SDK initialization with error handling
- [x] Control mapping (Gain, Exposure, WB_R, WB_B, etc.)
- [x] Image capture with RAW8 → JPEG encoding
- [x] Updated `DriverFactory` to use real hardware
- [x] Created test notebook `notebooks/test_asi_driver.ipynb`

### 1.5 CameraController for Synchronized Captures ✅ (Completed 2025-12-17)
- [x] `CameraController` already implemented in `devices/controller.py`
- [x] Supports multi-camera coordination
- [x] Calculates precise timing for centered exposures
- [x] ThreadPoolExecutor for parallel capture
- [x] Timing accuracy tracking (ideal vs actual)
- [x] Injectable Clock protocol for testing
- [x] Created test notebook `notebooks/test_sync_capture.ipynb`
- [x] Already exported from `devices/__init__.py`

### 1.6 Refactor MCP Tools to Use CameraRegistry ✅ (Completed 2025-12-17)
- [x] Removed direct zwoasi calls from `tools/cameras.py`
- [x] Updated to use `CameraRegistry` and device layer
- [x] `list_cameras()` uses `registry.discover()`
- [x] `get_camera_info()` uses `registry.get()` with `auto_connect=True`
- [x] `capture_frame()` uses `camera.capture(CaptureOptions())`
- [x] `set_camera_control()` uses `camera._instance.set_control()`
- [x] `get_camera_control()` uses `camera._instance.get_control()`
- [x] Added `init_registry()` call in `server.py` create_server()
- [x] Added `shutdown_registry()` cleanup in server shutdown
- [x] Backwards compatible with ASI_ prefix stripping

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
- [x] Decide on sensor hardware (Arduino Nano BLE33 Sense IMU)
- [x] Document sensor selection in `docs/architecture/sensor-calibration.md`

### 3.2 Implement Position Driver
- [x] Create concrete sensor implementation in `drivers/sensors/`
- [x] Implement `SensorDriver` protocol (DigitalTwin + Arduino drivers)
- [x] Create high-level `Sensor` device abstraction
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

## Future Ideas

### Virtual Star Parties — Remote Observing via Web Stream

**Concept:** The MJPEG camera streams (`/stream/main`, `/stream/finder`) already work on a phone over the local network. By proxying or relaying these streams through an AWS-hosted webpage, remote participants could observe the telescope feed in real-time without any local network access.

**How it would work:**
- Telescope controller streams camera feeds locally (already working)
- A relay service forwards the MJPEG stream to a public-facing AWS endpoint (EC2, Lambda + API Gateway, or CloudFront)
- Viewers open a URL in their browser and see the live telescope feed
- Combine with voice/video chat (Discord, Zoom, etc.) for interactive sessions
- The operator (Mark) controls the telescope; viewers observe and suggest targets

**Benefits:**
- True virtual star parties — anyone with a link can watch
- No hardware or software required for viewers (just a browser)
- Operator retains full control of the telescope
- Could integrate chat/reactions directly into the web page

**Considerations:**
- Latency: MJPEG over the internet adds delay, but fine for observing (not tracking)
- Bandwidth: Outbound stream from home → AWS, then fan-out to viewers
- Security: Need auth or link-based access to prevent open access to the telescope
- Could start simple (ngrok or Tailscale Funnel for one-off sessions) before building a proper AWS relay

#### Streaming Options — Low Cost, Audience Building

The goal: stream the telescope feed cheaply, build a following over time. Here are the approaches ranked by cost and complexity:

**Option 1: YouTube Live Stream (Free, Best for Audience Building)**
- Convert MJPEG → RTMP on the telescope controller or AWS relay, push to YouTube Live
- YouTube handles all the CDN, scaling, chat, and discovery — zero cost to you
- Viewers find you through YouTube search, recommendations, notifications
- Built-in chat for interactivity during star parties
- Can schedule events ("Live Meteor Shower Watch — Geminids 2026") to build anticipation
- Recordings auto-save as YouTube videos — free content library
- **How:** `ffmpeg` on the Pi/controller converts MJPEG → RTMP: `ffmpeg -i http://localhost:8080/stream/main -c:v libx264 -f flv rtmp://a.rtmp.youtube.com/live2/<stream-key>`
- **Cost: $0** (YouTube is free for streaming)

**Option 2: Discord Stage/Stream (Free, Good for Community)**
- Stream directly into the Discord server using a bot or OBS → Discord
- Viewers are already in the community — low friction
- Stage channels allow voice interaction (people can ask to speak)
- Good for small interactive sessions (< 25 viewers ideal)
- Less discoverable than YouTube — better for existing community
- **Cost: $0**

**Option 3: Twitch (Free, Gaming/Niche Audience)**
- Same RTMP approach as YouTube
- Twitch has a Science & Technology category — some astronomy streamers do well
- Better real-time chat interaction than YouTube
- Smaller discovery potential than YouTube for astronomy content
- Can multi-stream to both YouTube + Twitch simultaneously with `ffmpeg` or restream.io
- **Cost: $0**

**Option 4: AWS Static Site + CloudFront (Low Cost, Full Control)**
- Simple webpage that embeds the stream, hosted on S3 + CloudFront
- Telescope → AWS relay (small EC2 or Lightsail, ~$3.50-5/month) → CloudFront CDN → viewers
- Full control over the experience (custom UI, overlays, annotations)
- Can embed YouTube chat or Discord widget for interaction
- Good as a "home base" that links to YouTube/Twitch streams
- **Cost: ~$5-10/month** (mostly the relay instance; CloudFront bandwidth is cheap)

**Option 5: HLS via AWS MediaLive (Overkill for Now)**
- Professional-grade: MJPEG → AWS MediaLive → MediaPackage → CloudFront
- Auto-scaling, adaptive bitrate, DVR/rewind
- Way too expensive for starting out (~$50-100+/month)
- **Skip this until audience justifies it**

#### Recommended Path (Low Cost → Growth)

1. **Start with YouTube Live** — free, discoverable, zero infrastructure. Just need `ffmpeg` running on the controller to push RTMP. Schedule streams around interesting events (meteor showers, conjunctions, ISS passes).
2. **Simultaneously stream to Discord** for your community — they get the interactive experience with voice chat.
3. **Add a simple AWS landing page** later (S3 static site, ~$1/month) as a hub: links to YouTube, Discord, GitHub repo, schedule of upcoming sessions.
4. **Multi-stream to Twitch** when you want to reach that audience — restream.io free tier handles 2 destinations.

The key insight: **YouTube/Twitch handle the hard part (CDN, scaling, chat, discovery) for free.** Don't pay for infrastructure until you need custom features they don't provide.

**Building a following:**
- Consistent schedule matters more than frequency (e.g., "First clear Friday of each month")
- Meteor showers are natural events to build around — post the schedule on the YouTube channel
- Time-lapse compilations from the all-sky camera make great short-form content (YouTube Shorts, TikTok)
- Plate-solved images with annotations ("Here's what we're looking at") make great thumbnails/posts
- Cross-post to r/telescopes, r/astrophotography, astronomy Discord servers

**This is a stretch goal — no immediate action needed, but a compelling use case for the web dashboard.**

### Meteor Shower / Satellite Streak Monitoring with Wide-Field Camera

**Concept:** During meteor showers, set the telescope outside with the wide-field finder camera (ASI120MC-S, 150° all-sky) running continuously. The camera's massive field of view should capture meteor streaks and satellite trails depending on the exposure time.

**How it would work:**
- Deploy the telescope during known meteor showers (Perseids, Geminids, etc.)
- Run the finder camera on continuous capture with appropriate exposure (a few seconds per frame should show streaks as lines)
- Stream via `/stream/finder` — could combine with the virtual star party relay for remote meteor watching
- Optionally save frames to disk for later review or stacking

**What you'd see:**
- **Meteor streaks** — bright lines across the frame during active showers
- **Satellite trails** — Starlink trains, ISS passes, etc. as slower-moving lines
- **Star trails** — at longer exposures, stars will trail too (which helps distinguish meteors by their brightness and speed)

**Future possibilities:**
- Automated meteor detection (frame differencing — compare consecutive frames, flag new streaks)
- Count rate tracking over the night (plot meteors/hour vs. time)
- Plate-solve the frames (#7) to determine radiant direction
- Time-lapse compilation of an entire shower night

**The finder camera's all-sky FOV makes this nearly zero-effort — just point up and record.**

---

*Last updated: 2026-02-15*
