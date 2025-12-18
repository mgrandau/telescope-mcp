# Camera Architecture

## Overview

This document describes the camera subsystem architecture, explaining how MCP tools interact with camera hardware through abstraction layers.

## Hardware Configuration

The telescope system uses two ZWO ASI cameras with different purposes:

### Camera 0: ASI120MC-S (Finder/Spotter Scope)

| Specification | Value | Notes |
|---------------|-------|-------|
| **Resolution** | 1280 × 960 (1.2 MP) | |
| **Pixel Size** | 3.75 µm | |
| **Sensor Size** | 4.8 × 3.6 mm | |
| **Bit Depth** | 8-bit | |
| **Lens** | 150° All-Sky | Fish-eye for wide field |
| **FOV per Pixel** | 421.875 arcseconds | 150° × 3600 / 1280 |
| **Purpose** | Plate solving, alignment | Wide field acquisition |

### Camera 1: ASI482MC (Main Imaging Camera)

| Specification | Value | Notes |
|---------------|-------|-------|
| **Resolution** | 1920 × 1080 (2.07 MP) | |
| **Pixel Size** | 5.8 µm | |
| **Sensor Size** | 11.13 × 6.26 mm | |
| **Bit Depth** | 12-bit | Higher dynamic range |
| **Optics** | 1600mm focal length | Through telescope |
| **FOV per Pixel** | 0.748 arcseconds | High resolution imaging |
| **FOV (total)** | 23.9' × 13.4' | Arcminutes |
| **Purpose** | Deep sky imaging | Primary science camera |

```mermaid
%%{init: {'theme': 'dark'}}%%
graph TB
    subgraph "Physical Setup"
        SKY[🌌 Sky]
        
        subgraph "Finder Scope"
            LENS[150° All-Sky Lens]
            ASI120[ASI120MC-S<br/>1280×960<br/>421.875 arcsec/px]
        end
        
        subgraph "Main Telescope"
            OTA[1600mm OTA]
            ASI482[ASI482MC<br/>1920×1080<br/>0.748 arcsec/px]
        end
    end
    
    SKY --> LENS
    SKY --> OTA
    LENS --> ASI120
    OTA --> ASI482
    
    style ASI120 fill:#9cf,stroke:#333,stroke-width:2px,color:#000
    style ASI482 fill:#f9c,stroke:#333,stroke-width:2px,color:#000
```

## Architecture Overview

### Previous State (Flat) — Replaced

```mermaid
%%{init: {'theme': 'dark'}}%%
graph LR
    T[tools/cameras.py] -->|direct calls| Z[zwoasi library]
    Z --> H[ASI Camera Hardware]
```

**Problem:** Tools called `zwoasi` directly. No abstraction, no way to inject a digital twin for testing.

### Current Architecture (Layered) ✅ Implemented

```mermaid
%%{init: {'theme': 'dark'}}%%
graph TB
    subgraph "MCP Tools Layer"
        TC[tools/cameras.py<br/>list_cameras, capture_frame, etc.]
    end

    subgraph "Logical Device Layer"
        REG[CameraRegistry<br/>devices/registry.py]
        CAM[Camera<br/>devices/camera.py]
    end

    subgraph "Driver Layer"
        ASI[ASICameraDriver<br/>drivers/cameras/asi.py]
        TWIN[DigitalTwinCameraDriver<br/>drivers/cameras/twin.py]
    end

    subgraph "Hardware / Simulation"
        HW[ASI Camera USB]
        IMG[Image Files<br/>directory/file]
    end
    
    subgraph "Observability"
        LOG[StructuredLogger]
        STATS[CameraStats]
    end

    TC --> REG
    REG --> CAM
    CAM -->|driver injection| ASI
    CAM -->|driver injection| TWIN
    ASI --> HW
    TWIN --> IMG
    
    TC -.-> LOG
    CAM -.-> LOG
    TWIN -.-> LOG
    CAM -.-> STATS

    style REG fill:#9cf,stroke:#333,stroke-width:2px,color:#000
    style CAM fill:#ff9,stroke:#333,stroke-width:2px,color:#000
    style LOG fill:#9f9,stroke:#333,stroke-width:2px,color:#000
```

## Class Diagrams

The camera subsystem is organized into three layers. Each diagram below focuses on one layer for clarity.

### Device Layer (Camera Class)

The main `Camera` class with its injectable dependencies and data types:

```mermaid
%%{init: {'theme': 'dark'}}%%
classDiagram
    direction LR
    
    class Camera {
        +config: CameraConfig
        +is_connected: bool
        +is_streaming: bool
        +info: CameraInfo
        +connect() CameraInfo
        +disconnect() None
        +capture(options?) CaptureResult
        +capture_raw() CaptureResult
        +set_overlay(config) None
        +stream(options?, max_fps?) Iterator
        +stop_stream() None
    }
    
    class CameraConfig {
        <<dataclass>>
        +camera_id: int
        +name: str?
        +default_gain: int
        +default_exposure_us: int
    }
    
    class CaptureOptions {
        <<dataclass>>
        +exposure_us: int?
        +gain: int?
        +apply_overlay: bool
        +format: str
    }
    
    class CaptureResult {
        <<dataclass>>
        +image_data: bytes
        +timestamp: datetime
        +exposure_us: int
        +gain: int
        +has_overlay: bool
    }
    
    class StreamFrame {
        <<dataclass>>
        +image_data: bytes
        +timestamp: datetime
        +sequence_number: int
    }
    
    Camera --> CameraConfig : configured by
    Camera --> CaptureOptions : accepts
    Camera --> CaptureResult : returns
    Camera --> StreamFrame : yields
```

### Injectable Dependencies

Protocols that can be injected for testing and customization:

```mermaid
%%{init: {'theme': 'dark'}}%%
classDiagram
    direction TB
    
    class Camera {
        -_driver: CameraDriver
        -_renderer: OverlayRenderer
        -_clock: Clock
        -_hooks: CameraHooks
        -_recovery: RecoveryStrategy
    }
    
    class CameraDriver {
        <<protocol>>
        +get_connected_cameras() dict
        +open(camera_id) CameraInstance
    }
    
    class OverlayRenderer {
        <<protocol>>
        +render(image, config, info) bytes
    }
    
    class Clock {
        <<protocol>>
        +monotonic() float
        +sleep(seconds) None
    }
    
    class RecoveryStrategy {
        <<protocol>>
        +attempt_recovery(camera_id) bool
    }
    
    class CameraHooks {
        <<dataclass>>
        +on_connect: Callable?
        +on_capture: Callable?
        +on_error: Callable?
    }
    
    class NullRenderer {
        +render() bytes
    }
    
    class SystemClock {
        +monotonic() float
        +sleep() None
    }
    
    class NullRecoveryStrategy {
        +attempt_recovery() bool
    }
    
    Camera --> CameraDriver : injects
    Camera --> OverlayRenderer : injects
    Camera --> Clock : injects
    Camera --> RecoveryStrategy : injects
    Camera --> CameraHooks : has
    
    NullRenderer ..|> OverlayRenderer : implements
    SystemClock ..|> Clock : implements
    NullRecoveryStrategy ..|> RecoveryStrategy : implements
```

### Driver Implementations

Real hardware driver and digital twin for development:

```mermaid
%%{init: {'theme': 'dark'}}%%
classDiagram
    direction TB
    
    class CameraDriver {
        <<protocol>>
        +get_connected_cameras() dict
        +open(camera_id) CameraInstance
    }
    
    class CameraInstance {
        <<protocol>>
        +get_info() dict
        +get_controls() dict
        +set_control(name, value) dict
        +capture(exposure_us) bytes
        +close() None
    }
    
    class DigitalTwinCameraDriver {
        -_config: DigitalTwinConfig
        +get_connected_cameras() dict
        +open(camera_id) CameraInstance
    }
    
    class DigitalTwinConfig {
        <<dataclass>>
        +image_source: ImageSource
        +image_path: Path?
        +cycle_images: bool
    }
    
    class ASICameraDriver {
        <<future>>
        +get_connected_cameras() dict
        +open(camera_id) CameraInstance
    }
    
    CameraDriver --> CameraInstance : creates
    DigitalTwinCameraDriver ..|> CameraDriver : implements
    DigitalTwinCameraDriver --> DigitalTwinConfig : configured by
    ASICameraDriver ..|> CameraDriver : implements
```

> **Note:** `DigitalTwinCameraDriver.open()` returns a `CameraInstance` protocol implementation.
> The concrete `DigitalTwinCameraInstance` class is an internal implementation detail.

### CameraRegistry

Centralized camera discovery and singleton management:

```mermaid
%%{init: {'theme': 'dark'}}%%
classDiagram
    direction LR
    
    class CameraRegistry {
        -_driver: CameraDriver
        -_cameras: dict~int, Camera~
        -_discovery_cache: dict~int, CameraInfo~
        +discover(refresh?) dict~int, CameraInfo~
        +get(camera_id, name?, auto_connect?) Camera
        +has(camera_id) bool
        +remove(camera_id) Camera?
        +clear() None
        +camera_ids: list~int~
        +discovered_ids: list~int~
    }
    
    class RecoveryStrategy {
        -_registry: CameraRegistry
        +attempt_recovery(camera_id) bool
    }
    
    note for CameraRegistry "Context manager support\\nwith CameraRegistry(driver) as reg:"
    
    CameraRegistry --> Camera : creates/manages
    CameraRegistry --> CameraDriver : uses
    CameraRegistry --> CameraInfo : caches
    RecoveryStrategy --> CameraRegistry : uses for discovery
```

## Visibility Summary

| Class | Location | Visibility | Used By |
|-------|----------|------------|---------|
| `Camera` | `devices/camera.py` | **PUBLIC** | MCP tools, user code |
| `CameraConfig` | `devices/camera.py` | **PUBLIC** | User code (create cameras) |
| `CaptureOptions` | `devices/camera.py` | **PUBLIC** | Passed to `capture()` |
| `CaptureResult` | `devices/camera.py` | **PUBLIC** | Returned by `capture()` |
| `StreamFrame` | `devices/camera.py` | **PUBLIC** | Yielded by `stream()` |
| `OverlayRenderer` | `devices/camera.py` | **PUBLIC** | Protocol for custom renderers |
| `RecoveryStrategy` | `devices/camera.py` | **PUBLIC** | Protocol for disconnect recovery |
| `Clock` | `devices/camera.py` | **PUBLIC** | Protocol for time injection |
| `CameraHooks` | `devices/camera.py` | **PUBLIC** | Event callbacks |
| `CameraRegistry` | `devices/registry.py` | **PUBLIC** | MCP tools, application startup |
| `CameraDriver` | `drivers/cameras/` | **PUBLIC** | Protocol for drivers |
| `CameraInstance` | `drivers/cameras/` | **PUBLIC** | Protocol returned by `open()` |
| `DigitalTwinCameraDriver` | `drivers/cameras/twin.py` | **PUBLIC** | Development, tests |
| `DigitalTwinConfig` | `drivers/cameras/twin.py` | **PUBLIC** | Configure twin behavior |
| `ImageSource` | `drivers/cameras/twin.py` | **PUBLIC** | Enum for twin image source |

## Layer Responsibilities

### 1. MCP Tools Layer (`tools/cameras.py`)

**Purpose:** Expose camera functionality as MCP tools that AI agents can call.

**Responsibilities:**
- Define tool schemas (input/output)
- Convert between MCP types and domain types
- Handle errors and format responses
- Delegate to Camera device

**Does NOT:**
- Know about ASI SDK specifics
- Manage camera state directly
- Handle hardware communication

### 2. Logical Device Layer (`devices/camera.py`) ⭐ NEW

**Purpose:** Represent "what a camera is" independent of hardware.

**Responsibilities:**
- Maintain camera state (settings, connection status)
- Provide clean interface (capture, set_gain, get_info)
- Validate inputs (gain range, exposure limits)
- Log operations to active session
- Accept driver via dependency injection

**Does NOT:**
- Know which driver implementation is injected
- Handle raw SDK calls
- Format MCP responses

```python
class Camera:
    """Logical camera device with injected driver."""
    
    def __init__(self, driver: CameraDriver, camera_id: int):
        self._driver = driver
        self._camera_id = camera_id
        self._instance: CameraInstance | None = None
        self._settings = CameraSettings()
    
    def connect(self) -> None:
        """Connect to camera via driver."""
        self._instance = self._driver.open(self._camera_id)
    
    def capture(self, exposure_us: int | None = None) -> CaptureResult:
        """Capture a frame, return structured result."""
        # Use provided exposure or current setting
        exp = exposure_us or self._settings.exposure_us
        
        # Log to session
        get_session_manager().log("DEBUG", f"Capturing frame", 
            camera_id=self._camera_id, exposure_us=exp)
        
        # Delegate to driver
        jpeg_bytes = self._instance.capture(exp)
        
        return CaptureResult(
            image_data=jpeg_bytes,
            settings=self._settings.copy(),
            timestamp=datetime.now(timezone.utc),
        )
```

### 3. Driver Layer (`drivers/cameras/`)

**Purpose:** Implement hardware-specific communication.

**Two implementations:**

#### ASICameraDriver (`drivers/cameras/asi.py`)
- Wraps `zwoasi` library
- Handles ASI SDK initialization
- Maps control names to SDK constants
- Manages USB camera lifecycle

#### TwinCameraDriver (`drivers/cameras/twin.py`)
- Loads images from file/directory
- Simulates camera properties
- Mimics ASI control behavior
- No hardware dependency

**Both implement the same protocol:**

```python
class CameraDriver(Protocol):
    def get_connected_cameras(self) -> dict: ...
    def open(self, camera_id: int) -> CameraInstance: ...

class CameraInstance(Protocol):
    def get_info(self) -> dict: ...
    def get_controls(self) -> dict: ...
    def set_control(self, control: str, value: int) -> dict: ...
    def get_control(self, control: str) -> dict: ...
    def capture(self, exposure_us: int) -> bytes: ...
    def close(self) -> None: ...
```

## Directory Structure

```
src/telescope_mcp/
├── tools/
│   └── cameras.py          # MCP tool definitions
├── devices/                 # Logical device layer
│   ├── __init__.py
│   ├── camera.py           # Camera class + protocols
│   ├── controller.py       # CameraController (multi-camera sync)
│   └── registry.py         # CameraRegistry (singleton management)
└── drivers/
    └── cameras/
        ├── __init__.py     # Protocol definitions
        ├── asi.py          # Real ASI SDK driver (future)
        └── twin.py         # Digital twin driver
```

## Dependency Injection Flow

```mermaid
%%{init: {'theme': 'dark'}}%%
sequenceDiagram
    participant Tools as tools/cameras.py
    participant Registry as CameraRegistry
    participant Camera as Camera
    participant Driver as ASI/Twin Driver
    participant HW as Hardware/Files

    Note over Tools: Application startup
    Tools->>Registry: init_registry(driver)
    
    Note over Tools: list_cameras tool called
    Tools->>Registry: discover()
    Registry->>Driver: get_connected_cameras()
    Driver-->>Registry: {0: info, 1: info}
    Registry-->>Tools: dict[int, CameraInfo]
    
    Note over Tools: capture_frame tool called
    Tools->>Registry: get(camera_id=0)
    Registry->>Camera: Camera(driver, config, recovery=strategy)
    Registry-->>Tools: Camera (singleton)
    
    Tools->>Camera: connect()
    Camera->>Driver: open(0)
    Driver->>HW: Initialize camera
    Driver-->>Camera: CameraInstance
    
    Tools->>Camera: capture()
    Camera->>Driver: capture(exposure_us)
    Driver->>HW: Capture frame
    Driver-->>Camera: JPEG bytes
    Camera-->>Tools: CaptureResult
```

## Configuration

```python
from telescope_mcp.drivers import DriverConfig, DriverMode, configure

# Use real hardware
configure(DriverConfig(mode=DriverMode.HARDWARE))

# Use digital twin with image directory
configure(DriverConfig(
    mode=DriverMode.DIGITAL_TWIN,
    twin_image_path=Path("/data/test-captures/"),
))
```

## Benefits of This Architecture

| Aspect | Benefit |
|--------|---------|
| **Testability** | Inject twin driver for unit tests |
| **Development** | Work without hardware connected |
| **Separation** | Tools don't know about ASI SDK |
| **Session logging** | Camera layer logs all operations |
| **Validation** | Camera layer validates inputs |
| **State management** | Camera tracks settings, connection |
| **Swappability** | Add new driver without changing tools |

## Observability Integration

Structured logging is integrated throughout the camera subsystem via the `telescope_mcp.observability` module.

### Logging Architecture

```mermaid
%%{init: {'theme': 'dark'}}%%
graph TB
    subgraph "Camera Subsystem"
        TOOLS[tools/cameras.py]
        DEVICES[devices/camera.py<br/>devices/registry.py]
        DRIVERS[drivers/cameras/twin.py<br/>drivers/cameras/asi.py]
    end
    
    subgraph "Observability Module"
        LOGGER[StructuredLogger]
        CONTEXT[LogContext]
        STATS[CameraStats]
    end
    
    subgraph "Output"
        CONSOLE[Console<br/>Human-readable]
        JSON[JSON Logs<br/>Machine-parseable]
        SESSION[Session ASDF<br/>Persistent storage]
    end
    
    TOOLS --> LOGGER
    DEVICES --> LOGGER
    DRIVERS --> LOGGER
    
    DEVICES --> STATS
    
    LOGGER --> CONSOLE
    LOGGER --> JSON
    LOGGER --> SESSION
    
    style LOGGER fill:#9cf,stroke:#333,stroke-width:2px,color:#000
    style STATS fill:#f9c,stroke:#333,stroke-width:2px,color:#000
```

### Structured Logging Usage

```python
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)

# Structured logging with keyword arguments
logger.info("Frame captured", 
    camera_id=0,
    exposure_us=100000,
    gain=50,
    size_bytes=len(data)
)

# Output (console): Frame captured camera_id=0 exposure_us=100000 gain=50 size_bytes=245760
# Output (JSON): {"timestamp": "...", "level": "INFO", "message": "Frame captured", "camera_id": 0, ...}
```

### Camera Statistics

```python
from telescope_mcp.observability import CameraStats

stats = CameraStats(camera_id=0)
stats.record_capture(duration_ms=150.5, size_bytes=245760)
stats.record_capture(duration_ms=148.2, size_bytes=245120)

summary = stats.get_summary()
# StatsSummary(
#     total_captures=2,
#     avg_duration_ms=149.35,
#     p95_duration_ms=150.5,
#     total_bytes=490880,
#     ...
# )
```

## Implementation Status

| Phase | Task | Status |
|-------|------|--------|
| **Phase 1** | Device Layer | ✅ Complete |
| | `devices/camera.py` with Camera class | ✅ |
| | `devices/registry.py` with CameraRegistry | ✅ |
| | `devices/controller.py` with CameraController | ✅ |
| | `drivers/cameras/twin.py` with accurate specs | ✅ |
| | `drivers/cameras/asi.py` ASI SDK wrapper | ✅ |
| **Phase 2** | Enhanced Digital Twin | ✅ Complete |
| | File/directory image sources | ✅ |
| | Simulated control responses | ✅ |
| | Real camera specs (ASI120MC-S, ASI482MC) | ✅ |
| **Phase 3** | Observability | ✅ Complete |
| | Structured logging module | ✅ |
| | Camera statistics collection | ✅ |
| | Integration across all layers | ✅ |
| **Phase 4** | Motors & Sensors | 🔲 Planned |
| | `devices/motor.py` | 🔲 |
| | `devices/sensor.py` | 🔲 |
| | Same pattern: device layer + driver injection | 🔲 |

## CameraController: Multi-Camera Coordination

### Purpose

Orchestrates synchronized captures across **two or more** cameras for alignment operations.

The controller manages a named collection of cameras and coordinates their capture timing. While the current telescope setup uses two cameras (finder + main), the architecture supports any number of cameras for future expansion (e.g., guide camera, spectrograph).

### Synchronized Capture Problem

For telescope alignment via plate solving, we need:
- **Spotterscope (finder):** Long exposure (e.g., 176 seconds) with wide field of view
- **Main camera:** Short exposure (e.g., 312 ms) through telescope optics

The short exposure must be **centered** within the long exposure so star positions match:

```
Finder (176s):  |=====================================|
                              ↑ midpoint (t=88s)
Main (312ms):               |===|
                            ↑ starts at t=87.844s
```

### Timing Calculation

```python
delay = (primary_exposure / 2) - (secondary_exposure / 2)

# Example: 176s primary, 312ms secondary
delay = (176_000_000 / 2) - (312_000 / 2)
      = 88_000_000 - 156_000
      = 87_844_000 µs (87.844 seconds)
```

### Class Diagram

```mermaid
%%{init: {'theme': 'dark'}}%%
classDiagram
    direction LR
    
    class CameraController {
        -_cameras: dict~str, Camera~
        -_clock: Clock
        +add_camera(name, camera) None
        +remove_camera(name) Camera
        +get_camera(name) Camera
        +camera_names: list~str~
        +camera_count: int
        +calculate_sync_timing(primary_us, secondary_us) int
        +sync_capture(config) SyncCaptureResult
    }
    
    note for CameraController "Manages 2+ cameras\nCurrent setup: finder + main"
    
    class SyncCaptureConfig {
        <<dataclass>>
        +primary: str
        +secondary: str
        +primary_exposure_us: int
        +secondary_exposure_us: int
        +primary_gain: int?
        +secondary_gain: int?
    }
    
    class SyncCaptureResult {
        <<dataclass>>
        +primary_frame: CaptureResult
        +secondary_frame: CaptureResult
        +primary_start: datetime
        +secondary_start: datetime
        +ideal_secondary_start_us: int
        +actual_secondary_start_us: int
        +timing_error_us: int
        +timing_error_ms: float
    }
    
    class Clock {
        <<protocol>>
        +monotonic() float
        +sleep(seconds) None
    }
    
    CameraController --> Camera : manages
    CameraController --> Clock : uses
    CameraController --> SyncCaptureConfig : accepts
    CameraController --> SyncCaptureResult : returns
```

### Usage Example

```python
from telescope_mcp.devices import (
    Camera, CameraConfig, CameraController, SyncCaptureConfig
)

# Create cameras
finder = Camera(driver, CameraConfig(camera_id=0, name="Finder"))
main = Camera(driver, CameraConfig(camera_id=1, name="Main"))

# Connect both
finder.connect()
main.connect()

# Create controller
controller = CameraController({
    "finder": finder,
    "main": main,
})

# Synchronized capture for alignment
result = controller.sync_capture(SyncCaptureConfig(
    primary="finder",
    secondary="main",
    primary_exposure_us=176_000_000,  # 176 seconds
    secondary_exposure_us=312_000,     # 312 ms
))

print(f"Timing error: {result.timing_error_ms:.1f}ms")

# Both frames now ready for plate solving
# result.primary_frame   -> finder image
# result.secondary_frame -> main image (centered in finder exposure)
```

### Testability

The `Clock` protocol allows injecting fake time for testing:

```python
class FakeClock:
    def __init__(self):
        self._time = 0.0
        self._sleeps = []
    
    def monotonic(self) -> float:
        return self._time
    
    def sleep(self, seconds: float) -> None:
        self._sleeps.append(seconds)
        self._time += seconds

# Test timing calculation without waiting
clock = FakeClock()
controller = CameraController(cameras, clock=clock)
result = controller.sync_capture(config)
assert clock._sleeps[0] == pytest.approx(87.844, rel=0.001)
```

## Design Decisions

### Camera Singleton Pattern ✅ DECIDED

**Decision:** Use **Singleton per camera_id** via `CameraRegistry` class.

**Rationale:**

| Factor | Why Singleton Wins |
|--------|-------------------|
| **Physical reality** | One camera = one USB connection = one object |
| **ASI SDK expectation** | `ASIOpenCamera` → use → `ASICloseCamera` lifecycle |
| **Connection cost** | ~300ms per connect; reuse is efficient |
| **CameraController** | Assumes persistent cameras for sync capture |
| **State consistency** | Settings (gain, exposure) stay synchronized |

**Implementation:**

```python
# Encapsulated in CameraRegistry class (devices/registry.py)
class CameraRegistry:
    def __init__(self, driver: CameraDriver):
        self._driver = driver
        self._cameras: dict[int, Camera] = {}
        self._discovery_cache: dict[int, CameraInfo] | None = None
    
    def get(self, camera_id: int) -> Camera:
        """Get or create a Camera singleton for this camera_id."""
        if camera_id not in self._cameras:
            config = CameraConfig(camera_id=camera_id)
            self._cameras[camera_id] = Camera(
                driver=self._driver,
                config=config,
                recovery=RecoveryStrategy(self),  # Inject recovery
            )
        return self._cameras[camera_id]
    
    def clear(self) -> None:
        """Disconnect all cameras and clear registry."""
        for camera in self._cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self._cameras.clear()

# Usage with context manager
with CameraRegistry(driver) as registry:
    camera = registry.get(0)
    camera.connect()
    result = camera.capture()
# All cameras disconnected on exit
```

**Testing story:**

```python
# Tests bypass registry - create fresh instances directly
def test_capture():
    camera = Camera(mock_driver, config)  # Direct instantiation
    camera.connect()
    result = camera.capture()
    # No global state touched

# Integration tests use context manager
def test_with_registry():
    with CameraRegistry(mock_driver) as registry:
        camera = registry.get(0)
        # ... test ...
    # Automatic cleanup

# Or explicit cleanup in fixture
@pytest.fixture
def registry(mock_driver):
    reg = CameraRegistry(mock_driver)
    yield reg
    reg.clear()
```

### Camera Discovery in Device Layer ✅ DECIDED

**Decision:** Discovery happens in `CameraRegistry` class alongside singleton management.

**Rationale:**

| Factor | Why Device Layer Wins |
|--------|----------------------|
| **Centralized** | One class for all "where do I get cameras" questions |
| **Clean tools layer** | Tools just call `registry.discover()`, no driver knowledge |
| **Caching opportunity** | Can cache discovery results without changing tools |
| **Validation** | Can transform/validate driver output before returning |
| **Encapsulated state** | No module-level globals, all state in registry instance |

**Implementation:**

```python
# CameraRegistry.discover() in devices/registry.py
class CameraRegistry:
    def __init__(self, driver: CameraDriver):
        self._driver = driver
        self._discovery_cache: dict[int, CameraInfo] | None = None
    
    def discover(self, refresh: bool = False) -> dict[int, CameraInfo]:
        """Discover connected cameras with caching.
        
        Args:
            refresh: Force re-discovery (ignore cache)
        
        Returns:
            Dict mapping camera_id to CameraInfo
        """
        if self._discovery_cache is None or refresh:
            raw = self._driver.get_connected_cameras()
            self._discovery_cache = {
                cam_id: CameraInfo(**info) if isinstance(info, dict) else info
                for cam_id, info in raw.items()
            }
        return self._discovery_cache
```

**Usage from tools layer:**

```python
# tools/cameras.py
from telescope_mcp.devices import get_registry

@tool
def list_cameras() -> list[dict]:
    """List connected cameras."""
    registry = get_registry()
    cameras = registry.discover()
    return [asdict(info) for info in cameras.values()]

@tool  
def capture_frame(camera_id: int, exposure_us: int) -> dict:
    """Capture a frame from specified camera."""
    registry = get_registry()
    camera = registry.get(camera_id, auto_connect=True)
    return asdict(camera.capture(CaptureOptions(exposure_us=exposure_us)))
```

### Hot-Plug Handling: Lazy Re-validation ✅ DECIDED

**Decision:** Use **lazy re-validation** via injectable `RecoveryStrategy` — no polling, auto-recover on failure.

**Context:** Cameras don't typically come/go during operation, but if a USB cable gets bumped, we want recovery without restarting the MCP server.

**Rationale:**

| Factor | Why Lazy Re-validation Wins |
|--------|----------------------------|
| **No background threads** | Simple, no concurrency issues |
| **No polling overhead** | Don't waste cycles checking constantly |
| **Auto-recovery** | If camera comes back, next capture succeeds |
| **AI-agent friendly** | Retry semantics work naturally |
| **Singleton stays valid** | Camera object recovers in place |
| **Testable** | Inject mock RecoveryStrategy for testing |

**Implementation:**

```python
# RecoveryStrategy protocol (devices/camera.py)
class RecoveryStrategy(Protocol):
    def attempt_recovery(self, camera_id: int) -> bool:
        """Returns True if camera is available after recovery attempt."""
        ...

# Registry-based implementation (devices/registry.py)
class RegistryRecoveryStrategy:
    def __init__(self, registry: CameraRegistry):
        self._registry = registry
    
    def attempt_recovery(self, camera_id: int) -> bool:
        cameras = self._registry.discover(refresh=True)
        return camera_id in cameras

# Camera uses injected strategy (devices/camera.py)
class Camera:
    def __init__(self, ..., recovery: RecoveryStrategy | None = None):
        self._recovery = recovery or NullRecoveryStrategy()
    
    def _recover_and_capture(self, exposure_us: int, ...) -> CaptureResult:
        """Attempt to recover from camera disconnect."""
        self._instance = None
        
        # Use injected strategy (no circular import!)
        if not self._recovery.attempt_recovery(self._config.camera_id):
            raise CameraDisconnectedError(...)
        
        # Reconnect and retry
        self.connect()
        return self._instance.capture(exposure_us)
```

**Behavior:**

| Scenario | Result |
|----------|--------|
| Normal capture | Works as usual |
| Cable bumped, replugged | First capture fails, auto-recovers, retries |
| Camera unplugged permanently | `CameraDisconnectedError` after recovery attempt |
| AI agent retries failed capture | Second attempt succeeds if camera back |

**What this does NOT handle:**
- Proactive "camera available" notifications
- Hot-plug of *new* cameras (call `registry.discover(refresh=True)` manually)
- Registry cleanup of permanently removed cameras

### SOLID Improvements ✅ APPLIED

The following SOLID-principled improvements have been applied:

| # | Improvement | SOLID Principle | Benefit |
|---|-------------|-----------------|---------|
| 1 | `RecoveryStrategy` protocol | Dependency Inversion | No circular imports; testable recovery |
| 2 | `CameraRegistry` class | Single Responsibility | Encapsulated state; context manager support |
| 3 | Registry replaces `get_configured_driver()` | Dependency Inversion | Single point of configuration |
| 4 | Drivers can return `CameraInfo` | Type Safety | End-to-end type checking |
| 5 | Registry as context manager | Python Best Practice | Automatic cleanup |

## Open Questions

*All architectural questions have been resolved.*
