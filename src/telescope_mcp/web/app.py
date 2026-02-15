"""FastAPI web application for telescope dashboard."""

import asyncio
import datetime
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
import zwoasi as asi
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from telescope_mcp.devices.sensor import Sensor
from telescope_mcp.drivers.asi_sdk import get_sdk_library_path
from telescope_mcp.drivers.config import get_factory
from telescope_mcp.observability import get_logger
from telescope_mcp.utils.coordinates import altaz_to_radec
from telescope_mcp.utils.image import CV2ImageEncoder, ImageEncoder

logger = get_logger(__name__)

# Paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Camera state management
_sdk_initialized = False
_cameras: dict[int, asi.Camera] = {}  # Open camera instances
_camera_streaming: dict[int, bool] = {}  # Track which cameras are streaming
_camera_settings: dict[
    int, dict[str, int]
] = {}  # Store camera settings (exposure_us, gain)
# Latest RAW16 frame buffers - both cameras stream RAW16, capture grabs from here
# No mode switch needed for capture on either camera
_latest_frames: dict[int, np.ndarray] = {}  # camera_id -> RAW16 frame
_latest_frame_info: dict[int, dict[str, object]] = {}  # camera_id -> metadata

# Motor state management
# Tracks continuous motion state for start/stop control pattern
_motor_moving: dict[str, bool] = {"altitude": False, "azimuth": False}
_motor_direction: dict[str, int] = {"altitude": 0, "azimuth": 0}  # -1, 0, +1
_motor_speed: dict[str, int] = {"altitude": 0, "azimuth": 0}  # 0-100%

# USB bandwidth management for dual-camera operation
# ASI_BANDWIDTHOVERLOAD range: 0-100 (percentage of USB bus)
# Two cameras sharing one USB controller need <=50% each
USB_BANDWIDTH_SINGLE = 80  # Single camera can use most of the bus
USB_BANDWIDTH_DUAL = 40  # Each camera gets 40% when both streaming

# Stream error recovery
MAX_CONSECUTIVE_ERRORS = 10  # Stop stream after this many errors in a row
ERROR_BACKOFF_BASE_S = 0.5  # Base sleep between error retries
ERROR_BACKOFF_MAX_S = 5.0  # Maximum backoff sleep
STREAM_TIMEOUT_BUFFER_US = 5_000_000  # 5s buffer added to exposure for timeout

# Motor configuration
MOTOR_NUDGE_DEGREES = 0.1  # Default nudge amount in degrees
MOTOR_STEPS_PER_DEGREE_ALT = 140000 / 90.0  # ~1556 steps per degree
MOTOR_STEPS_PER_DEGREE_AZ = 110000 / 135.0  # ~815 steps per degree

# IMU sensor for position feedback
_sensor: Sensor | None = None

# Image encoder (injectable for testing)
_encoder: ImageEncoder | None = None

# Default settings (per-camera)
# Finder camera (0): Long exposures for wide-field, up to ~180 seconds
DEFAULT_FINDER_EXPOSURE_US = 10_000_000  # 10 second (quick startup, user can increase)
# Main camera (1): Short exposures for high-res, ~300ms typical
DEFAULT_MAIN_EXPOSURE_US = 60_000  # 60ms
DEFAULT_FINDER_GAIN = 80
DEFAULT_MAIN_GAIN = 80
DEFAULT_FPS = 15


def configure_camera_defaults(
    finder_exposure_us: int | None = None,
    finder_gain: int | None = None,
    main_exposure_us: int | None = None,
    main_gain: int | None = None,
) -> None:
    """Configure per-camera default settings from MCP config.

    Called at startup to override hardcoded defaults with values from
    the MCP configuration (mcp.json CLI args). If a value is None,
    the hardcoded default is retained.

    Business context: Different observing conditions (moonlit vs dark sky)
    require different camera defaults. Config-driven approach avoids code
    changes for routine adjustments. Values are set via --finder-exposure-us,
    --finder-gain, --main-exposure-us, --main-gain CLI args in mcp.json.

    Args:
        finder_exposure_us: Finder camera (0) exposure in microseconds.
            None keeps DEFAULT_FINDER_EXPOSURE_US (10,000,000 = 10s).
        finder_gain: Finder camera (0) gain value (0-510).
            None keeps DEFAULT_FINDER_GAIN (80).
        main_exposure_us: Main camera (1) exposure in microseconds.
            None keeps DEFAULT_MAIN_EXPOSURE_US (60,000 = 60ms).
        main_gain: Main camera (1) gain value (0-570).
            None keeps DEFAULT_MAIN_GAIN (80).

    Returns:
        None. Modifies module-level defaults.

    Raises:
        None.

    Example:
        >>> configure_camera_defaults(
        ...     finder_exposure_us=5_000_000,  # 5s finder
        ...     finder_gain=100,
        ...     main_exposure_us=300_000,  # 300ms main
        ...     main_gain=50,
        ... )
    """
    global DEFAULT_FINDER_EXPOSURE_US, DEFAULT_MAIN_EXPOSURE_US
    global DEFAULT_FINDER_GAIN, DEFAULT_MAIN_GAIN

    if finder_exposure_us is not None:
        DEFAULT_FINDER_EXPOSURE_US = finder_exposure_us
        logger.info(f"Finder exposure configured: {finder_exposure_us}us")
    if finder_gain is not None:
        DEFAULT_FINDER_GAIN = finder_gain
        logger.info(f"Finder gain configured: {finder_gain}")
    if main_exposure_us is not None:
        DEFAULT_MAIN_EXPOSURE_US = main_exposure_us
        logger.info(f"Main exposure configured: {main_exposure_us}us")
    if main_gain is not None:
        DEFAULT_MAIN_GAIN = main_gain
        logger.info(f"Main gain configured: {main_gain}")


def _get_default_exposure(camera_id: int) -> int:
    """Get default exposure for a camera in microseconds.

    Returns sensible default exposure times for finder vs main camera.
    Finder camera uses shorter exposure for real-time tracking, main camera
    uses longer exposure for deep sky imaging.

    Args:
        camera_id: Camera index (0=finder, 1=main).

    Returns:
        Exposure time in microseconds. Finder: 100ms (100000μs),
        Main: 5s (5000000μs).

    Raises:
        None. Unknown camera IDs default to main camera exposure.

    Example:
        >>> _get_default_exposure(0)
        100000  # Finder: 100ms
        >>> _get_default_exposure(1)
        5000000  # Main: 5s

    Business Context:
        Finder cameras need fast frame rates for tracking/guiding (10-30 FPS),
        so shorter exposures prevent motion blur. Main cameras capture faint
        deep sky objects requiring long exposures (seconds to minutes) to
        accumulate photons from distant galaxies/nebulae.
    """
    if camera_id == 0:
        return DEFAULT_FINDER_EXPOSURE_US
    return DEFAULT_MAIN_EXPOSURE_US


def _get_default_gain(camera_id: int) -> int:
    """Get default gain for a camera.

    Returns per-camera gain defaults. Configurable via
    configure_camera_defaults() at startup.

    Args:
        camera_id: Camera index (0=finder, 1=main).

    Returns:
        Gain value for the specified camera.

    Example:
        >>> _get_default_gain(0)
        80
        >>> _get_default_gain(1)
        80
    """
    if camera_id == 0:
        return DEFAULT_FINDER_GAIN
    return DEFAULT_MAIN_GAIN


def _init_sdk() -> None:
    """Initialize the ZWO ASI SDK if not already initialized.

    Performs one-time SDK initialization required before any camera
    operations. Uses the SDK library path from the driver configuration.
    Thread-safe via global flag; subsequent calls are no-ops.

    Business context: ZWO ASI cameras require SDK initialization before
    use. The web app initializes lazily on first camera access, allowing
    startup without cameras connected.

    This function is idempotent and safe to call multiple times.
    Initialization failures are logged but not raised, allowing the
    application to start even without cameras connected.

    Args:
        None.

    Returns:
        None. Sets global _sdk_initialized flag on success or failure.

    Raises:
        None. Exceptions are caught and logged as warnings.

    Example:
        >>> _init_sdk()  # First call initializes
        >>> _init_sdk()  # Subsequent calls are no-ops
    """
    global _sdk_initialized
    if not _sdk_initialized:  # pragma: no cover - SDK init with real hardware
        try:
            sdk_path = get_sdk_library_path()
            asi.init(sdk_path)
            _sdk_initialized = True
            logger.info(f"ASI SDK initialized from {sdk_path}")
        except Exception as e:
            # P2-4: Set flag to prevent retry storm on repeated failures
            _sdk_initialized = True
            logger.warning(f"ASI SDK init failed (no cameras?): {e}")


def _get_camera(camera_id: int, force_reopen: bool = False) -> asi.Camera | None:
    """Get or lazily open a camera instance by ID.

    Manages the camera connection lifecycle, opening cameras on first
    access and caching instances for reuse. Initializes the SDK if
    needed. Camera settings are initialized to defaults on first open.

    This is the primary camera access point for the web application,
    ensuring consistent state management across stream and API handlers.

    Args:
        camera_id: Zero-based camera index (0=finder, 1=main imaging).
            Must be less than the number of connected cameras.
        force_reopen: If True, close and reopen the camera to reset state.

    Returns:
        asi.Camera instance if available, None if camera not found or
        failed to open. Check return value before use.

    Raises:
        None. Errors are logged and None is returned.

    Example:
        >>> camera = _get_camera(0)
        >>> if camera:
        ...     info = camera.get_camera_property()
    """
    # Force reopen if requested (resets camera state)
    if force_reopen and camera_id in _cameras:
        try:
            old_camera = _cameras.pop(camera_id)
            try:
                old_camera.stop_video_capture()
            except Exception:
                pass
            old_camera.close()
            logger.info(f"Closed camera {camera_id} for reopen")
        except Exception as e:
            logger.warning(f"Error closing camera {camera_id}: {e}")

    if camera_id not in _cameras:
        try:
            _init_sdk()
            num_cameras = asi.get_num_cameras()
            if camera_id >= num_cameras:
                logger.error(f"Camera {camera_id} not found (have {num_cameras})")
                return None
            camera = asi.Camera(camera_id)
            _cameras[camera_id] = camera
            _camera_streaming[camera_id] = False
            _camera_settings[camera_id] = {
                "exposure_us": _get_default_exposure(camera_id),
                "gain": _get_default_gain(camera_id),
            }
            logger.info(f"Opened camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to open camera {camera_id}: {e}")
            return None
    return _cameras.get(camera_id)


def _close_all_cameras() -> None:
    """Close all open camera instances and stop any active streams.

    Performs graceful shutdown of all camera resources. Stops video
    capture on streaming cameras before closing. Called during
    application shutdown via the lifespan context manager.

    This function clears all camera state including instances,
    streaming flags, and settings. Safe to call even if no cameras
    are open.

    Args:
        None.

    Returns:
        None. Clears _cameras and _camera_streaming dicts.

    Raises:
        None. Errors are logged per-camera but don't stop cleanup.
    """
    for camera_id, camera in list(_cameras.items()):
        try:
            if _camera_streaming.get(camera_id):
                camera.stop_video_capture()
            camera.close()
            logger.info(f"Closed camera {camera_id}")
        except Exception as e:
            logger.error(f"Error closing camera {camera_id}: {e}")
    _cameras.clear()
    _camera_streaming.clear()


async def _init_sensor() -> None:
    """Initialize IMU sensor for position feedback.

    Creates sensor device with driver from factory, connects to hardware
    or digital twin based on configuration. Sensor provides ALT/AZ position
    readings for /api/position endpoint.

    Business context: IMU-based position feedback enables real-time pointing
    display on dashboard. Critical for operator awareness during telescope
    movement. Digital twin mode allows UI development without hardware.

    Args:
        None. Uses global factory configuration.

    Returns:
        None. Sets global _sensor on success.

    Raises:
        None. Connection failures are logged but don't prevent startup.

    Example:
        >>> await _init_sensor()
        >>> if _sensor:
        ...     reading = await _sensor.read()
    """
    global _sensor
    try:
        factory = get_factory()
        driver = factory.create_sensor_driver()
        _sensor = Sensor(driver)
        await _sensor.connect()
        logger.info("IMU sensor initialized", sensor_type=type(driver).__name__)
    except Exception as e:
        logger.warning(
            "Failed to initialize sensor, position will be 0,0", error=str(e)
        )
        _sensor = None


async def _cleanup_sensor() -> None:
    """Disconnect and cleanup IMU sensor.

    Gracefully disconnects sensor connection on application shutdown.
    Safe to call even if sensor was never initialized.

    Args:
        None. Uses global _sensor.

    Returns:
        None. Clears global _sensor.

    Raises:
        None. Errors are logged but don't prevent shutdown.

    Example:
        >>> await _cleanup_sensor()
    """
    global _sensor
    if _sensor is not None:
        try:
            await _sensor.disconnect()
            logger.info("IMU sensor disconnected")
        except Exception as e:
            logger.error("Error disconnecting sensor", error=str(e))
        _sensor = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle for startup and shutdown.

    FastAPI lifespan context manager that handles resource initialization
    on startup and cleanup on shutdown. This ensures cameras and other
    hardware resources are properly managed across the application lifecycle.

    Startup actions:
    - Initialize ASI SDK for camera access
    - Log service startup

    Shutdown actions:
    - Close all open camera connections
    - Stop any active video streams
    - Log service shutdown

    Args:
        app: The FastAPI application instance (provided by FastAPI).

    Yields:
        None. Control returns to FastAPI to run the application.

    Raises:
        None. SDK init failures are logged but don't prevent startup.

    Example:
        >>> app = FastAPI(lifespan=lifespan)
    """
    # Startup: Initialize cameras, motors, sensors
    logger.info("Starting telescope control services...")
    _init_sdk()
    await _init_sensor()
    yield
    # Shutdown: Clean up
    logger.info("Shutting down telescope control services...")
    await _cleanup_sensor()
    _close_all_cameras()


def create_app(encoder: ImageEncoder | None = None) -> FastAPI:
    """Create and configure the FastAPI telescope control application.

    Factory function that builds the complete web application with all
    routes, middleware, and configuration. Mounts static files, sets up
    Jinja2 templates, and registers all API endpoints.

    The application provides:
    - Dashboard UI at / for browser-based telescope control
    - MJPEG camera streams at /stream/{camera_id}
    - REST API for camera and motor control at /api/*

    This factory pattern allows testing with fresh app instances and
    supports different configurations per environment. The encoder
    parameter enables dependency injection for testing without cv2.

    Args:
        encoder: Optional ImageEncoder for JPEG encoding. If None,
            uses CV2ImageEncoder (real OpenCV). Pass mock encoder
            for testing without cv2 dependency.

    Returns:
        Configured FastAPI application instance ready for uvicorn.run().
        Includes lifespan handler for proper startup/shutdown.

    Raises:
        None. Missing static/template dirs are handled gracefully.

    Example:
        >>> app = create_app()  # Production with real cv2
        >>> uvicorn.run(app, host="0.0.0.0", port=8080)
        >>> # Testing with mock encoder:
        >>> mock_enc = MockImageEncoder()
        >>> app = create_app(encoder=mock_enc)
    """
    global _encoder
    _encoder = encoder if encoder is not None else CV2ImageEncoder()

    app = FastAPI(
        title="Telescope Control",
        description="Web dashboard for telescope control",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Templates
    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    # Routes
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        """Render main telescope control dashboard (web UI entry point).

        Serves HTML dashboard with embedded camera streams (MJPEG), motor
        controls (altitude/azimuth), position display (alt/az readout), and
        camera settings (exposure/gain). Page uses JavaScript to interact with
        REST API endpoints (/api/cameras, /api/motor/*, /api/position) for
        real-time control without page refreshes.

        Business context: Primary operator interface for telescope control
        during observation sessions. Enables remote telescope operation from
        observatory control room or over internet (VPN). Critical for
        unattended observatories where physical access difficult. Combines
        live camera preview (alignment verification, target centering) with
        motion control (goto, tracking corrections) in single interface. Used
        during setup (camera alignment, focus), acquisition (target centering,
        platesolving), and guiding (drift monitoring). Alternative to desktop
        control software - platform-independent (works on tablets, laptops,
        phones).

        Implementation details: Renders dashboard.html Jinja2 template with
        title context. Template contains MJPEG <img> tags pointing to
        /stream/finder and /stream/main, AJAX controls calling /api/*
        endpoints, WebSocket connections for position updates (if enabled).
        FastAPI handles async rendering. Static assets (CSS, JS) served from
        /static mount. Template not found raises TemplateNotFound (500 error).
        Request object required by Jinja2 f/stream/finderor URL generation.

        Args:
            request: FastAPI Request object providing URL context, headers,
                session. Required by Jinja2Templates.TemplateResponse for
                generating absolute URLs in template.

        Returns:
            HTMLResponse with rendered dashboard.html containing complete
            telescope control interface. Content-Type: text/html. Status 200
            on success.

        Raises:0
            None explicitly. TemplateNotFound (500) if dashboard.html missing
            from templates directory.

        Example:
            >>> # Access dashboard at http://localhost:8000/
            >>> # Browser displays:
            >>> # - Finder camera stream (top left)
            >>> # - Main camera stream (top right)
            >>> # - Motor controls (center)
            >>> # - Position readout (bottom)
            >>> # Click altitude control -> AJAX POST /api/motor/altitude
            >>> # Stream auto-refreshes at configured FPS
        """
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {"title": "Telescope Control"},
        )

    @app.get("/stream/finder")
    async def finder_stream(
        exposure_us: int | None = Query(
            None, ge=1, le=60_000_000, description="Exposure in microseconds (1-60s)"
        ),
        gain: int | None = Query(None, ge=0, le=600, description="Gain value (0-600)"),
        fps: int = Query(
            DEFAULT_FPS, ge=1, le=60, description="Target frames per second (1-60)"
        ),
    ) -> StreamingResponse:
        """Stream MJPEG video from the finder camera (camera 0).

                Convenience endpoint for the finder/guide camera. Returns a
                continuous MJPEG stream suitable for <img> tags or video players.
                The finder camera is typically used for alignment and tracking.
        /stream/finder
                Business context: Enables real-time finder camera display in web
                dashboards for telescope alignment and target acquisition. The shorter
                exposures and higher frame rates typical of finder cameras make them
                ideal for live view while the main camera takes long exposures.
                Critical for "goto" operation verification and autoguiding feedback.

                Args:
                    exposure_us: Exposure time in microseconds. None uses default.
                    gain: Gain value (camera-specific range). None uses default.
                    fps: Target frame rate, default 15. Actual rate may be lower
                        if exposure time exceeds frame interval.

                Returns:
                    StreamingResponse with multipart MJPEG content.

                Raises:
                    None. Camera errors displayed as text on error frames.

                Example:
                    # In HTML dashboard:
                    <img src="/stream/finder?exposure_us=50000&gain=50&fps=15">

                    # Or via requests:
                    import requests
                    resp = requests.get(
                        'http://localhost:8080/stream/finder?fps=10', stream=True
                    )
                    for chunk in resp.iter_content(chunk_size=1024):
                        # Process MJPEG frames
                        pass
        """
        return StreamingResponse(  # pragma: no cover - infinite stream
            _generate_camera_stream(
                camera_id=0, exposure_us=exposure_us, gain=gain, fps=fps
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/main")
    async def main_stream(
        exposure_us: int | None = Query(
            None, ge=1, le=60_000_000, description="Exposure in microseconds (1-60s)"
        ),
        gain: int | None = Query(None, ge=0, le=600, description="Gain value (0-600)"),
        fps: int = Query(
            DEFAULT_FPS, ge=1, le=60, description="Target frames per second (1-60)"
        ),
    ) -> StreamingResponse:
        """Stream MJPEG video from the main imaging camera (camera 1).

        Convenience endpoint for the primary imaging camera. Returns a
        continuous MJPEG stream for live preview during astrophotography.
        The main camera is typically higher resolution with better sensitivity.

        Business context: Provides live preview of the main imaging camera for
        focusing, framing, and quick target verification before committing to
        long exposures. Short preview exposures (50-200 ms) allow responsive
        framing while the main camera actual imaging uses much longer exposures
        (1-10 minutes). Essential for unguided setups to verify
        field before starting imaging sequences.

        Args:
            exposure_us: Exposure time in microseconds. None uses default.
            gain: Gain value (camera-specific range). None uses default.
            fps: Target frame rate, default 15. Preview streams often use
                shorter exposures than actual imaging.

        Returns:
            StreamingResponse with multipart MJPEG content.

        Raises:
            None. Camera errors displayed as text on error frames.

        Example:
            # HTML dashboard for main camera preview
            <img src="/stream/main?exposure_us=100000&gain=80&fps=10">

            # Adjust exposure dynamically via URL params
            <img id="main-stream" src="/stream/main">
            <script>
            function updateExposure(exp_us) {
                document.getElementById('main-stream').src =
                    `/stream/main?exposure_us=${exp_us}&fps=15`;
            }
            </script>
        """
        return StreamingResponse(  # pragma: no cover - infinite stream
            _generate_camera_stream(
                camera_id=1, exposure_us=exposure_us, gain=gain, fps=fps
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/{camera_id}")
    async def camera_stream(
        camera_id: int,
        exposure_us: int | None = Query(
            None, ge=1, le=60_000_000, description="Exposure in microseconds (1-60s)"
        ),
        gain: int | None = Query(None, ge=0, le=600, description="Gain value (0-600)"),
        fps: int = Query(
            DEFAULT_FPS, ge=1, le=60, description="Target frames per second (1-60)"
        ),
    ) -> StreamingResponse:
        """Stream MJPEG video from any camera by numeric ID.

        Generic streaming endpoint for any connected camera. Use camera_id
        0 for finder, 1 for main, or higher indices for additional cameras.
        Returns error frame if camera_id does not exist.

        Business context: Enables flexible multi-camera setups where users can
        add cameras without hardcoding endpoints. Particularly useful for
        observatory setups with multiple guide cameras, all-sky cameras, or
        specialized imaging cameras. Dynamic UI generation can query
        /api/cameras then create stream views for each camera_id returned.

        Args:
            camera_id: Zero-based camera index from /api/cameras list.
            exposure_us: Exposure time in microseconds. None uses stored
                settings or defaults.
            gain: Gain value (camera-specific range). None uses stored
                settings or defaults.
            fps: Target frame rate, default 15.

        Returns:
            StreamingResponse with multipart MJPEG content.

        Raises:
            None. Invalid camera_id returns stream with error frame.

        Example:
            # Query available cameras first
            response = requests.get('http://localhost:8080/api/cameras')
            cameras = response.json()['cameras']

            # Generate stream URLs dynamically
            for cam in cameras:
                stream_url = f"/stream/{cam['id']}?fps=15"
                print(f"{cam['name']}: {stream_url}")

            # Use in HTML
            <div id="camera-grid"></div>
            <script>
            cameras.forEach(cam => {
                const img = document.createElement('img');
                img.src = `/stream/${cam.id}?fps=10`;
                document.getElementById('camera-grid').appendChild(img);
            });
            </script>
        """
        return StreamingResponse(  # pragma: no cover - infinite stream
            _generate_camera_stream(
                camera_id=camera_id, exposure_us=exposure_us, gain=gain, fps=fps
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/cameras")
    async def api_list_cameras() -> JSONResponse:
        """List all connected ASI cameras with basic info (discovery endpoint).

        Returns camera count and names for all detected ZWO ASI cameras via SDK
        enumeration. Initializes SDK lazily if not already done (_init_sdk). Use
        camera IDs from response for subsequent /stream/{camera_id} and
        /api/camera/{camera_id}/* endpoints. Discovery endpoint for dashboard
        initialization and camera availability checks.

        Business context: Essential for dynamic camera configuration in
        multi-camera systems. Dashboard JavaScript calls this on page load to
        populate camera selectors, enable/disable stream buttons based on
        availability. Enables hot-plug detection (refresh to see newly
        connected cameras). Critical for diagnostic workflows ("which cameras
        are connected?", "why is stream endpoint 404?"). Used by setup scripts
        to verify expected cameras present before starting observations (e.g.,
        both finder and main camera must be available).

        Implementation details: Calls _init_sdk() (loads libASICamera2.so, calls
        ASIGetNumOfConnectedCameras), then asi.get_num_cameras() and
        asi.list_cameras() for enumeration. Returns JSON with count and array
        of {id, name} objects. IDs are 0-based sequential matching SDK
        enumeration order (typically USB port order). Empty array if no cameras
        (not an error). SDK errors (library load failure, USB access denied)
        return {"error": str} with 500 status. No caching - always queries
        hardware (enables hot-plug detection but adds 50-200 ms latency).

        Args:
            None. Query parameters not needed for simple enumeration.

        Returns:
            JSONResponse with camera list:
            {"count": int, "cameras": [{"id": int, "name": str}, ...]}.
            Example: {"count": 2, "cameras": [{"id": 0, "name": "ZWO ASI120MC"},
            {"id": 1, "name": "ZWO ASI290MM"}]}.
            Error response: {"error": "SDK initialization failed: ..."}  with
            status 500 if SDK errors occur.

        Raises:
            None explicitly. All exceptions caught and returned as JSON
            {"error": str} with 500 status.

        Example:
            >>> # JavaScript dashboard code
            >>> fetch('/api/cameras')
            ...     .then(r => r.json())
            ...     .then(data => {
            ...         console.log(`Found ${data.count} cameras`);
            ...         data.cameras.forEach(cam => {
            ...             addStreamButton(cam.id, cam.name);
            ...         });
            ...     });
            >>> # Response: {"count": 2, "cameras": [{"id": 0, ...}, ...]}
        """
        _init_sdk()
        try:
            num_cameras = asi.get_num_cameras()
            if num_cameras == 0:
                return JSONResponse({"count": 0, "cameras": []})
            names = asi.list_cameras()
            return JSONResponse(
                {
                    "count": num_cameras,
                    "cameras": [{"id": i, "name": n} for i, n in enumerate(names)],
                }
            )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # API routes for dashboard JavaScript
    @app.post("/api/motor/altitude")
    async def api_move_altitude(steps: int, speed: int = 100) -> dict[str, object]:
        """Move the altitude (elevation) motor by specified steps.

        Controls vertical telescope movement for pointing adjustment.
        Positive steps move up (toward zenith), negative move down (toward
        horizon). This is a programmatic interface - for UI buttons, prefer
        /api/motor/altitude/nudge or /api/motor/altitude/start.

        Business context: Enables programmatic telescope pointing control for
        goto functionality, drift correction, and scripted observations.
        Altitude control is essential for object tracking as celestial objects
        change elevation throughout the night.

        Note: Currently a stub - returns placeholder response.

        Args:
            steps: Number of motor steps to move. Range: ±140000.
                0 = zenith (90°), 140000 = horizon (0°).
            speed: Motor speed as percentage (1-100). Default 100.

        Returns:
            Dict with status and parameters:
            {"status": "ok", "axis": "altitude", "steps": int, "speed": int}

        Example:
            response = requests.post(
                'http://localhost:8080/api/motor/altitude',
                params={'steps': 1000, 'speed': 50}
            )
        """
        logger.info("Motor altitude move", steps=steps, speed=speed)
        # TODO: Integrate with motor controller
        return {"status": "ok", "axis": "altitude", "steps": steps, "speed": speed}

    @app.post("/api/motor/azimuth")
    async def api_move_azimuth(steps: int, speed: int = 100) -> dict[str, object]:
        """Move the azimuth (rotation) motor by specified steps.

        Controls horizontal telescope rotation for pointing adjustment.
        Positive steps rotate counter-clockwise (looking down from above),
        negative rotate clockwise.

        Business context: Enables horizontal telescope positioning for target
        acquisition and tracking. Azimuth adjustments compensate for Earth
        rotation and allow slewing between targets.

        Note: Currently a stub - returns placeholder response.

        Args:
            steps: Number of motor steps to move. Range: ±110000.
            speed: Motor speed as percentage (1-100). Default 100.

        Returns:
            Dict with status and parameters:
            {"status": "ok", "axis": "azimuth", "steps": int, "speed": int}

        Example:
            response = requests.post(
                'http://localhost:8080/api/motor/azimuth',
                params={'steps': 2000, 'speed': 100}
            )
        """
        logger.info("Motor azimuth move", steps=steps, speed=speed)
        # TODO: Integrate with motor controller
        return {"status": "ok", "axis": "azimuth", "steps": steps, "speed": speed}

    @app.post("/api/motor/altitude/nudge")
    async def api_nudge_altitude(
        direction: str = Query(..., pattern="^(up|down)$"),
        degrees: float = Query(default=MOTOR_NUDGE_DEGREES, ge=0.01, le=10.0),
    ) -> dict[str, object]:
        """Nudge altitude motor by small fixed amount (tap gesture).

        Moves altitude axis by specified degrees in given direction. Designed
        for single-tap UI interactions where user wants discrete, predictable
        movement. For continuous motion (hold gesture), use /start instead.

        Business context: Enables fine pointing adjustments via dashboard arrow
        buttons. Tap-to-nudge provides predictable movement for centering
        objects in eyepiece/camera field of view.

        UI Pattern: Bind to button click event. Each tap moves fixed amount.

        Args:
            direction: Movement direction - "up" (toward zenith) or "down"
                (toward horizon).
            degrees: Amount to move in degrees. Default 0.1°.
                Range: 0.01 to 10.0 degrees.

        Returns:
            Dict with movement details:
            {"status": "ok", "axis": "altitude", "direction": str,
             "degrees": float, "steps": int}

        Example:
            >>> # JavaScript tap handler
            >>> upBtn.onclick = () => fetch(
            ...     '/api/motor/altitude/nudge?direction=up&degrees=0.1',
            ...     {method: 'POST'}
            ... );
        """
        steps = int(degrees * MOTOR_STEPS_PER_DEGREE_ALT)
        if direction == "down":
            steps = -steps

        logger.info(
            "Motor altitude nudge", direction=direction, degrees=degrees, steps=steps
        )
        # TODO: Integrate with motor controller - call move_relative()
        return {
            "status": "ok",
            "axis": "altitude",
            "direction": direction,
            "degrees": degrees,
            "steps": steps,
        }

    @app.post("/api/motor/azimuth/nudge")
    async def api_nudge_azimuth(
        direction: str = Query(..., pattern="^(cw|ccw|left|right)$"),
        degrees: float = Query(default=MOTOR_NUDGE_DEGREES, ge=0.01, le=10.0),
    ) -> dict[str, object]:
        """Nudge azimuth motor by small fixed amount (tap gesture).

        Moves azimuth axis by specified degrees in given direction. Designed
        for single-tap UI interactions where user wants discrete, predictable
        movement. For continuous motion (hold gesture), use /start instead.

        Business context: Enables fine rotation adjustments via dashboard arrow
        buttons. Tap-to-nudge provides predictable movement for centering
        objects in eyepiece/camera field of view.

        UI Pattern: Bind to button click event. Each tap moves fixed amount.

        Args:
            direction: Movement direction:
                - "cw" or "right": Clockwise (looking down from above)
                - "ccw" or "left": Counter-clockwise
            degrees: Amount to move in degrees. Default 0.1°.
                Range: 0.01 to 10.0 degrees.

        Returns:
            Dict with movement details:
            {"status": "ok", "axis": "azimuth", "direction": str,
             "degrees": float, "steps": int}

        Example:
            >>> # JavaScript tap handler
            >>> leftBtn.onclick = () => fetch(
            ...     '/api/motor/azimuth/nudge?direction=left',
            ...     {method: 'POST'}
            ... );
        """
        steps = int(degrees * MOTOR_STEPS_PER_DEGREE_AZ)
        if direction in ("cw", "right"):
            steps = -steps  # CW = negative direction

        logger.info(
            "Motor azimuth nudge", direction=direction, degrees=degrees, steps=steps
        )
        # TODO: Integrate with motor controller - call move_relative()
        return {
            "status": "ok",
            "axis": "azimuth",
            "direction": direction,
            "degrees": degrees,
            "steps": steps,
        }

    @app.post("/api/motor/altitude/start")
    async def api_start_altitude(
        direction: str = Query(..., pattern="^(up|down)$"),
        speed: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, object]:
        """Start continuous altitude motion (hold gesture).

        Begins continuous motor movement that continues until /stop is called.
        Designed for press-and-hold UI interactions where user wants smooth,
        ongoing motion. Call /api/motor/stop when button is released.

        Business context: Enables smooth slewing via dashboard controls. Hold
        to start continuous motion allows rapid telescope repositioning without
        multiple clicks. Essential for comfortable manual pointing.

        UI Pattern:
        - Bind to mousedown/touchstart: call /start
        - Bind to mouseup/touchend/mouseleave: call /stop

        Args:
            direction: Movement direction - "up" (toward zenith) or "down"
                (toward horizon).
            speed: Motor speed as percentage (1-100). Default 50%.
                Higher = faster slewing, lower = finer control.

        Returns:
            Dict confirming motion started:
            {"status": "moving", "axis": "altitude", "direction": str,
             "speed": int}

        Example:
            >>> # JavaScript hold handler
            >>> upBtn.onmousedown = () => fetch(
            ...     '/api/motor/altitude/start?direction=up&speed=50',
            ...     {method: 'POST'}
            ... );
            >>> upBtn.onmouseup = () => fetch('/api/motor/stop', {method: 'POST'});
        """
        _motor_moving["altitude"] = True
        _motor_direction["altitude"] = 1 if direction == "up" else -1
        _motor_speed["altitude"] = speed

        logger.info("Motor altitude start", direction=direction, speed=speed)
        # TODO: Integrate with motor controller - start continuous move
        return {
            "status": "moving",
            "axis": "altitude",
            "direction": direction,
            "speed": speed,
        }

    @app.post("/api/motor/azimuth/start")
    async def api_start_azimuth(
        direction: str = Query(..., pattern="^(cw|ccw|left|right)$"),
        speed: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, object]:
        """Start continuous azimuth motion (hold gesture).

        Begins continuous motor movement that continues until /stop is called.
        Designed for press-and-hold UI interactions where user wants smooth,
        ongoing motion. Call /api/motor/stop when button is released.

        Business context: Enables smooth rotation via dashboard controls. Hold
        to start continuous motion allows rapid telescope repositioning without
        multiple clicks. Essential for comfortable manual pointing.

        UI Pattern:
        - Bind to mousedown/touchstart: call /start
        - Bind to mouseup/touchend/mouseleave: call /stop

        Args:
            direction: Movement direction:
                - "cw" or "right": Clockwise (looking down from above)
                - "ccw" or "left": Counter-clockwise
            speed: Motor speed as percentage (1-100). Default 50%.
                Higher = faster slewing, lower = finer control.

        Returns:
            Dict confirming motion started:
            {"status": "moving", "axis": "azimuth", "direction": str,
             "speed": int}

        Example:
            >>> # JavaScript hold handler
            >>> leftBtn.onmousedown = () => fetch(
            ...     '/api/motor/azimuth/start?direction=left&speed=50',
            ...     {method: 'POST'}
            ... );
            >>> leftBtn.onmouseup = () => fetch('/api/motor/stop', {method: 'POST'});
        """
        _motor_moving["azimuth"] = True
        _motor_direction["azimuth"] = -1 if direction in ("cw", "right") else 1
        _motor_speed["azimuth"] = speed

        logger.info("Motor azimuth start", direction=direction, speed=speed)
        # TODO: Integrate with motor controller - start continuous move
        return {
            "status": "moving",
            "axis": "azimuth",
            "direction": direction,
            "speed": speed,
        }

    @app.post("/api/motor/stop")
    async def api_stop_motors(
        axis: str | None = Query(default=None, pattern="^(altitude|azimuth)$"),
    ) -> dict[str, object]:
        """Stop motor movement (emergency stop or release gesture).

        Immediately halts motor movement. Can stop specific axis or all motors.
        Called when user releases hold button or for emergency stop.

        Business context: Critical safety function and normal operation endpoint.
        Used both for:
        1. Normal release of hold buttons (stop one axis)
        2. Emergency stop (stop all axes immediately)

        This endpoint is idempotent - safe to call even when motors not moving.

        Args:
            axis: Optional - "altitude" or "azimuth" to stop specific motor.
                If None (default), stops ALL motors (emergency stop behavior).

        Returns:
            Dict confirming motors stopped:
            {"status": "stopped", "axes": ["altitude", "azimuth"]}

        Example:
            >>> # Stop all motors (emergency)
            >>> fetch('/api/motor/stop', {method: 'POST'});
            >>> # Stop just altitude (button release)
            >>> fetch('/api/motor/stop?axis=altitude', {method: 'POST'});
        """
        stopped_axes: list[str] = []

        if axis is None:
            # Emergency stop - all motors
            for motor in ["altitude", "azimuth"]:
                _motor_moving[motor] = False
                _motor_direction[motor] = 0
                _motor_speed[motor] = 0
                stopped_axes.append(motor)
            logger.warning("Emergency stop - all motors")
        else:
            # Single axis stop
            _motor_moving[axis] = False
            _motor_direction[axis] = 0
            _motor_speed[axis] = 0
            stopped_axes.append(axis)
            logger.info("Motor stop", axis=axis)

        # TODO: Integrate with motor controller - send stop commands
        return {"status": "stopped", "axes": stopped_axes}

    @app.get("/api/position")
    async def api_get_position() -> dict[str, object]:
        """Get current telescope pointing position (encoder readout).

        Returns altitude (elevation) and azimuth angles from motor encoders
        representing current telescope pointing direction. Values in degrees
        using alt-az coordinate system. Dashboard polls this endpoint (e.g.,
        1 Hz) to update position display for operator awareness.

        Business context: Essential for operator situational awareness during
        telescope operation. Shows where we are pointing in real-time, critical
        for verifying goto accuracy (did telescope reach target coordinates?),
        monitoring tracking (is telescope drifting?), and safety (are we
        pointed too low, risk hitting pier/horizon?). Used by platesolving
        workflows to compare actual pointing (encoder readout) vs solved
        pointing (image analysis) for alignment model refinement. Enables
        creation of pointing logs for post-observation analysis (where did we
        observe throughout night?). Diagnostic tool for encoder issues (why is
        readout stuck/jumping?).

        Implementation details: When implemented, will query motor controller
        get_position() reading encoder values from altitude/azimuth motors,
        converting encoder counts to degrees. Altitude: 0 deg (horizon) to
        90 deg (zenith), may support negative for below-horizon if mount
        allows. Azimuth: 0 deg (north) to 360 deg clockwise (east=90,
        south=180, west=270). Currently returns placeholder
        {"altitude": 45.0, "azimuth": 180.0} - TODO: integrate encoder
        drivers. Resolution typically 0.01 degrees (36 arcseconds) or better.
        Update rate ~10 Hz max (encoder query latency).

        Args:
            None. Position is current instantaneous state, no parameters
            needed.

        Returns:
            Dict with telescope pointing in alt-az degrees:
            {"altitude": float, "azimuth": float}.
            Example: {"altitude": 67.5, "azimuth": 123.4} = pointing 67.5 deg
            above horizon, azimuth 123.4 deg.
            When implemented, may include: {"altitude": 67.5, "azimuth": 123.4,
            "tracking": true, "timestamp": "2025-12-18T12:34:56.789Z",
            "encoder_counts": {"alt": 123456, "az": 789012}}.

        Raises:
            None currently (placeholder). TODO: Will raise HTTPException(503)
            if motor controller unavailable or encoders return invalid data
            (disconnected, out of range).

        Example:
            >>> # JavaScript dashboard polling for position display
            >>> setInterval(async () => {
            ...     const response = await fetch('/api/position');
            ...     const pos = await response.json();
            ...     document.getElementById('alt-display').innerText =
            ...         `Alt: ${pos.altitude.toFixed(2)} deg`;
            ...     document.getElementById('az-display').innerText =
            ...         `Az: ${pos.azimuth.toFixed(2)} deg`;
            ... }, 1000);  // Update every second
        """
        # Read position from IMU sensor if available
        altitude = 0.0
        azimuth = 0.0
        sensor_status = "no_sensor"

        if _sensor is not None:
            try:
                reading = await _sensor.read()
                altitude = reading.altitude
                azimuth = reading.azimuth
                sensor_status = "ok"
            except Exception as e:
                logger.warning("Sensor read failed", error=str(e))
                sensor_status = "error"

        # Get observer location from config for coordinate conversion
        config = get_factory().config
        location = config.location

        # Default location if not configured (Austin, TX)
        lat = location.get("lat", 30.2672)
        lon = location.get("lon", -97.7431)
        elevation = location.get("alt", 0.0)

        # Convert ALT/AZ to RA/Dec for display
        equatorial = altaz_to_radec(
            altitude=altitude,
            azimuth=azimuth,
            lat=lat,
            lon=lon,
            elevation=elevation,
        )

        return {
            "altitude": altitude,
            "azimuth": azimuth,
            "ra": equatorial["ra"],
            "dec": equatorial["dec"],
            "ra_hours": equatorial["ra_hours"],
            "ra_hms": equatorial["ra_hms"],
            "dec_dms": equatorial["dec_dms"],
            "location": {"lat": lat, "lon": lon, "elevation": elevation},
            "sensor_status": sensor_status,
            "status": "ok",
        }

    @app.post("/api/camera/{camera_id}/control")
    async def api_set_camera_control(
        camera_id: int,
        control: str = Query(
            ..., description="Control name (ASI_GAIN, ASI_EXPOSURE, etc.)"
        ),
        value: int = Query(..., description="Value to set"),
    ) -> JSONResponse:
        """Set a camera control parameter value.

        Adjusts camera settings like gain, exposure, white balance, etc.
        Changes take effect immediately and persist for the camera session.
        Settings are also stored for use by streaming endpoints.

        Business context: Primary interface for dynamic camera adjustment
        during observation sessions. Enables adaptive exposure control as sky
        conditions change, gain adjustment for different target brightness, and
        real-time optimization of image quality. Critical for automated
        routines like auto-focus (adjusting exposure for star detection) and
        auto-exposure (adjusting gain/exposure for target brightness). Used by
        both manual UI controls and automated imaging scripts.

        Valid control names (with ASI_ prefix):
        - ASI_GAIN: Amplification (0-500 typical)
        - ASI_EXPOSURE: Exposure time in microseconds
        - ASI_GAMMA: Gamma correction
        - ASI_WB_R, ASI_WB_B: White balance red/blue
        - ASI_BRIGHTNESS: Brightness offset
        - ASI_OFFSET: Black level offset
        - ASI_BANDWIDTHOVERLOAD: USB bandwidth percentage
        - ASI_FLIP: Image flip mode
        - ASI_HIGH_SPEED_MODE: Enable high-speed readout

        Args:
            camera_id: Camera index from /api/cameras.
            control: Control name string, must be one of valid options.
            value: Integer value to set. Valid range depends on control
                and camera model.

        Returns:
            JSONResponse with result:
            {"camera_id": int, "control": str, "value_set": int,
             "value_current": int, "auto": bool}
            Returns {"error": str} with status 400/404/500 on failure.

        Raises:
            None. Errors returned as JSON with appropriate status code.

        Example:
            # Set gain for bright target
            response = requests.post(
                'http://localhost:8080/api/camera/0/control',
                params={'control': 'ASI_GAIN', 'value': 80}
            )
            result = response.json()
            print(f"Gain set to {result['value_current']}")

            # Set exposure for 5 second frame
            response = requests.post(
                'http://localhost:8080/api/camera/1/control',
                params={'control': 'ASI_EXPOSURE', 'value': 5_000_000}
            )

            # JavaScript UI slider
            gainSlider.oninput = async (e) => {
                const url = `/api/camera/0/control?control=ASI_GAIN`
                    + `&value=${e.target.value}`;
                const response = await fetch(url, {method: 'POST'});
                const result = await response.json();
                const display = document.getElementById('gain-display');
                display.textContent = result.value_current;
            };
        """
        control_map = {
            "ASI_GAIN": asi.ASI_GAIN,
            "ASI_EXPOSURE": asi.ASI_EXPOSURE,
            "ASI_GAMMA": asi.ASI_GAMMA,
            "ASI_WB_R": asi.ASI_WB_R,
            "ASI_WB_B": asi.ASI_WB_B,
            "ASI_BRIGHTNESS": asi.ASI_BRIGHTNESS,
            "ASI_OFFSET": asi.ASI_OFFSET,
            "ASI_BANDWIDTHOVERLOAD": asi.ASI_BANDWIDTHOVERLOAD,
            "ASI_FLIP": asi.ASI_FLIP,
            "ASI_HIGH_SPEED_MODE": asi.ASI_HIGH_SPEED_MODE,
        }

        if control not in control_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown control: {control}. Valid: {list(control_map.keys())}",
            )

        camera = _get_camera(camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

        try:
            camera.set_control_value(control_map[control], value)

            # Update stored settings for exposure/gain (affects stream)
            if control == "ASI_EXPOSURE":
                _camera_settings.setdefault(camera_id, {})["exposure_us"] = value
            elif control == "ASI_GAIN":
                _camera_settings.setdefault(camera_id, {})["gain"] = value

            current = camera.get_control_value(control_map[control])
            return JSONResponse(
                {
                    "camera_id": camera_id,
                    "control": control,
                    "value_set": value,
                    "value_current": current[0],
                    "auto": current[1],
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/camera/{camera_id}/capture")
    async def api_capture_raw(
        camera_id: int,
        frame_type: str = Query(default="light", pattern="^(light|dark|flat|bias)$"),
    ) -> JSONResponse:
        """Capture a single RAW image and save to session ASDF archive.

        Captures a full-resolution RAW frame (RAW16 for maximum dynamic range)
        using current camera settings (exposure, gain). Saves to a single
        session ASDF archive containing frames from both cameras.

        Business context: Primary capture endpoint for actual astrophotography
        work. RAW16 format preserves full sensor data for calibration and
        stacking. Session ASDF archives contain all frames (finder + main,
        lights/darks/flats/bias) with full metadata. Use notebook to extract
        FITS files for stacking software compatibility.

        Args:
            camera_id: Camera index (0=finder, 1=main).
            frame_type: Type of frame: "light", "dark", "flat", "bias".

        Returns:
            JSONResponse with result:
            {"status": "success", "filename": str, "camera": str,
             "frame_type": str, "frame_index": int, ...}
            Returns {"status": "error", "error": str} on failure.

        Raises:
            None. Returns JSONResponse with status_code 400 if stream not running
            or no frame available. Returns status_code 500 if capture/save fails.

        Example:
            >>> # POST /api/camera/1/capture?frame_type=light
            >>> # Response (success):
            >>> {
            ...   "status": "success",
            ...   "filename": "session_20260101.asdf",
            ...   "camera": "main",
            ...   "frame_type": "light",
            ...   "frame_index": 0,
            ...   "exposure_us": 5000000,
            ...   "gain": 100,
            ...   "width": 4144,
            ...   "height": 2822,
            ...   "capture_mode": "raw16_stream"
            ... }
            >>> # Response (error - stream not running):
            >>> {"status": "error",
            ...  "error": "Main stream not running - start stream first"}
        """
        import datetime
        from pathlib import Path

        # Camera name for organizing in ASDF
        camera_key = "finder" if camera_id == 0 else "main"

        # ===== ALL CAMERAS: Grab RAW16 from video stream =====
        # Both cameras stream in RAW16 mode, no mode switch needed for capture

        # Check if stream is running and we have a frame
        if not _camera_streaming.get(camera_id, False):
            return JSONResponse(
                {
                    "status": "error",
                    "error": (
                        f"{camera_key.title()} stream not running - "
                        "start stream first"
                    ),
                },
                status_code=400,
            )

        if camera_id not in _latest_frames or _latest_frames[camera_id] is None:
            return JSONResponse(
                {
                    "status": "error",
                    "error": f"No frame available from {camera_key} stream yet",
                },
                status_code=400,
            )

        try:
            # Grab the RAW16 frame directly from stream (no mode switch, no conversion!)
            img = _latest_frames[camera_id].copy()
            frame_info = _latest_frame_info.get(camera_id, {})

            width = frame_info.get("width", img.shape[1])
            height = frame_info.get("height", img.shape[0])
            is_color = frame_info.get("is_color", False)
            exp = frame_info.get("exposure_us", _get_default_exposure(camera_id))
            gain = frame_info.get("gain", _get_default_gain(camera_id))

            logger.info(
                f"Capturing RAW16 {frame_type} from {camera_key} stream: "
                f"exp={exp}us, gain={gain}"
            )

            # Get camera info for metadata
            camera = _get_camera(camera_id)
            info = camera.get_camera_property() if camera else {}

            # Create output directory
            capture_dir = Path("data/captures")
            capture_dir.mkdir(parents=True, exist_ok=True)

            # Single session file per day (contains both cameras)
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"session_{date_str}.asdf"
            filepath = capture_dir / filename

            # Frame metadata
            capture_time = datetime.datetime.now(datetime.UTC)
            frame_meta: dict[str, object] = {
                "timestamp": capture_time.isoformat(),
                "exposure_us": exp,
                "gain": gain,
                "width": width,
                "height": height,
                "camera_id": camera_id,
                "camera_name": info.get("Name", f"Camera {camera_id}"),
                "camera_temp": info.get("Temperature", 0) / 10.0 if info else 0.0,
                "is_color": is_color,
                "bayer_pattern": "RGGB"
                if is_color
                else None,  # RAW16 preserves bayer pattern
                # RAW16 from stream, same quality as still capture
                "capture_mode": "raw16_stream",
            }

            # Add coordinates
            await _add_coordinates_to_metadata(frame_meta, capture_time)

            # Save to ASDF
            frame_index = await _save_frame_to_asdf(
                filepath, camera_key, frame_type, img, frame_meta, info
            )

            logger.info(
                f"Saved {camera_key}/{frame_type} #{frame_index} to: {filepath}"
            )

            return JSONResponse(
                {
                    "status": "success",
                    "filename": str(filename),
                    "filepath": str(filepath),
                    "camera": camera_key,
                    "frame_type": frame_type,
                    "frame_index": frame_index,
                    "exposure_us": exp,
                    "gain": gain,
                    "width": width,
                    "height": height,
                    "capture_mode": "raw16_stream",
                }
            )

        except Exception as e:
            logger.error(f"{camera_key.title()} capture failed: {e}")
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    return app


async def _add_coordinates_to_metadata(
    frame_meta: dict[str, object],
    capture_time: datetime.datetime,
) -> None:
    """Add telescope coordinates and environmental data to frame metadata.

    Reads sensor for ALT/AZ position, converts to RA/Dec, and adds to metadata.
    Logs warnings if sensor unavailable but does not raise exceptions.

    Args:
        frame_meta: Frame metadata dict to update (modified in place).
        capture_time: Timestamp for coordinate calculation.

    Returns:
        None. Modifies frame_meta dict in-place by adding 'coordinates' and
        'environmental' keys.

    Raises:
        None. Logs warnings if sensor unavailable or disconnected, but does
        not raise exceptions. Captures succeed even without coordinates.

    Example:
        >>> import datetime
        >>> meta = {"exposure_ms": 5000}
        >>> capture_time = datetime.datetime.now(datetime.timezone.utc)
        >>> await _add_coordinates_to_metadata(meta, capture_time)
        >>> # If sensor connected:
        >>> # meta["coordinates"] = {"alt": 45.0, "az": 180.0, "ra": 12.5, "dec": 30.0}
        >>> # meta["environmental"] = {"temperature": 20.5, "humidity": 45.0, ...}

    Business Context:
        Embedding telescope pointing and environmental data in each frame enables:
        - Plate solving verification (compare commanded vs actual pointing)
        - Mosaics/panoramas (stitch frames using coordinate overlap)
        - Astrometry calibration (match star positions to catalog)
        - Environmental correlation (track seeing conditions vs image quality)
        Non-blocking design ensures captures succeed even if sensors fail,
        critical for unattended observing sessions.
    """

    logger.info(
        "Checking sensor for coordinates",
        sensor_exists=_sensor is not None,
        sensor_connected=_sensor.connected if _sensor else False,
    )
    if _sensor is None:
        logger.warning(
            "No sensor available for frame coordinates - sensor not initialized"
        )
    elif not _sensor.connected:
        logger.warning(
            "Sensor not connected for frame coordinates",
            sensor_type=type(_sensor).__name__,
        )
    else:
        try:
            reading = await _sensor.read()
            logger.debug(
                "Sensor reading obtained",
                altitude=reading.altitude,
                azimuth=reading.azimuth,
                temperature=reading.temperature,
            )

            # Get observer location from config
            config = get_factory().config
            location = config.location
            lat = location.get("lat", 30.2672)
            lon = location.get("lon", -97.7431)
            elevation = location.get("alt", 0.0)

            # Convert ALT/AZ to RA/Dec
            equatorial = altaz_to_radec(
                altitude=reading.altitude,
                azimuth=reading.azimuth,
                lat=lat,
                lon=lon,
                elevation=elevation,
                obstime=capture_time,
            )

            # Add coordinates to frame metadata
            frame_meta["coordinates"] = {
                "altitude": reading.altitude,
                "azimuth": reading.azimuth,
                "ra": equatorial["ra"],
                "dec": equatorial["dec"],
                "ra_hours": equatorial["ra_hours"],
                "ra_hms": equatorial["ra_hms"],
                "dec_dms": equatorial["dec_dms"],
                "temperature": reading.temperature,
                "humidity": reading.humidity,
                "coordinate_source": "sensor",
                "coordinate_timestamp": capture_time.isoformat(),
            }
            logger.info(
                "Added coordinates to frame metadata",
                ra=equatorial["ra_hms"],
                dec=equatorial["dec_dms"],
            )
        except Exception as e:
            logger.error(
                "Failed to read sensor for frame coordinates",
                error=str(e),
                error_type=type(e).__name__,
            )


async def _save_frame_to_asdf(
    filepath: "Path",
    camera_key: str,
    frame_type: str,
    img: np.ndarray,
    frame_meta: dict[str, object],
    info: dict[str, object],
) -> int:
    """Save frame to ASDF session archive.

    Creates or updates session ASDF file with the captured frame.
    Handles both new file creation and appending to existing files.

    Args:
        filepath: Path to ASDF file.
        camera_key: Camera identifier ("finder" or "main").
        frame_type: Frame type ("light", "dark", "flat", "bias").
        img: Image data as numpy array.
        frame_meta: Frame metadata dict.
        info: Camera info dict from get_camera_property().

    Returns:
        Frame index within the frame_type list.

    Raises:
        OSError: If file I/O fails (disk full, permissions).
        asdf.exceptions.AsdfError: If ASDF serialization fails.

    Example:
        >>> img = np.random.randint(0, 65535, (1080, 1920), dtype=np.uint16)
        >>> meta = {"width": 1920, "height": 1080, "exposure_ms": 5000}
        >>> info = {"Name": "ASI Camera"}
        >>> idx = await _save_frame_to_asdf(
        ...     Path("session.asdf"), "main", "light", img, meta, info
        ... )
        >>> # Returns 0 for first light frame, 1 for second, etc.

    Business Context:
        ASDF (Advanced Scientific Data Format) provides efficient storage
        for large astronomy datasets. Organizes frames by camera and type,
        enabling batch processing workflows (stacking, calibration).
        Each frame includes full metadata for traceability.
    """
    import datetime

    import asdf

    width = frame_meta.get("width", img.shape[1] if len(img.shape) > 1 else 0)
    height = frame_meta.get("height", img.shape[0])
    is_color = frame_meta.get("is_color", False)

    if filepath.exists():
        with asdf.open(str(filepath), mode="rw") as af:
            # Ensure camera section exists
            if "cameras" not in af.tree:
                af.tree["cameras"] = {}
            if camera_key not in af.tree["cameras"]:
                af.tree["cameras"][camera_key] = {
                    "info": {
                        "name": info.get("Name", f"Camera {camera_key}"),
                        "sensor_width": width,
                        "sensor_height": height,
                        "is_color": is_color,
                        "bayer_pattern": frame_meta.get("bayer_pattern"),
                    },
                    "light": [],
                    "dark": [],
                    "flat": [],
                    "bias": [],
                }

            # Add frame
            af.tree["cameras"][camera_key][frame_type].append(
                {
                    "data": img.copy(),
                    "meta": frame_meta,
                }
            )
            frame_index = len(af.tree["cameras"][camera_key][frame_type]) - 1
            af.update()
    else:
        # Create new session ASDF archive
        capture_time = datetime.datetime.now(datetime.UTC)
        date_str = datetime.datetime.now().strftime("%Y%m%d")

        tree = {
            "metadata": {
                "created": capture_time.isoformat(),
                "session_date": date_str,
                "format_version": "1.0",
            },
            "cameras": {
                camera_key: {
                    "info": {
                        "name": info.get("Name", f"Camera {camera_key}"),
                        "sensor_width": width,
                        "sensor_height": height,
                        "is_color": is_color,
                        "bayer_pattern": frame_meta.get("bayer_pattern"),
                    },
                    "light": [],
                    "dark": [],
                    "flat": [],
                    "bias": [],
                }
            },
        }
        # Add the first frame
        cameras_dict = tree["cameras"]
        assert isinstance(cameras_dict, dict)
        camera_dict = cameras_dict[camera_key]
        assert isinstance(camera_dict, dict)
        frame_list = camera_dict[frame_type]
        assert isinstance(frame_list, list)
        frame_list.append(
            {
                "data": img.copy(),
                "meta": frame_meta,
            }
        )
        frame_index = 0

        af = asdf.AsdfFile(tree)
        af.write_to(str(filepath))

    return frame_index


async def _generate_camera_stream(
    camera_id: int,
    exposure_us: int | None = None,
    gain: int | None = None,
    fps: int = DEFAULT_FPS,
) -> AsyncGenerator[bytes, None]:
    """Generate continuous MJPEG video stream from camera (async generator).

    Async generator continuously capturing frames from specified camera,
    yielding MJPEG-formatted chunks. Uses ASI SDK video capture mode
    (start_video_capture) for efficient streaming without re-initialization
    per frame. Auto-stretches each frame (histogram normalization) for
    visibility of dim astronomical objects. Error frames displayed in-stream
    (red text on black) rather than breaking connection. Handles full capture
    lifecycle: configuration, video start, frame loop, cleanup on generator
    close.

    Business context: Core streaming engine for web dashboard live preview
    enabling real-time telescope alignment, focus adjustment, target
    acquisition. Critical for remote operation - operators see what telescope
    sees without physical presence. Finder camera streams used for goto
    verification ("did telescope point correctly?"), guiding feedback (drift
    monitoring), field identification. Main camera streams used for focus
    tuning (star size), framing verification, exposure preview ("will 5min
    exposure work?"). Auto-stretch essential - raw astronomical frames mostly
    black (faint stars invisible without processing). MJPEG format enables
    simple HTML <img src="/stream"> integration without WebRTC complexity.

    Implementation details: AsyncGenerator yields MJPEG multipart chunks
    (--frame boundary, JPEG data). Frame loop: capture_video_frame() -> numpy
    reshape -> auto-stretch (normalize to 0-255) -> imencode JPEG -> yield.
    Frame rate controlled by sleep(1/fps) minus capture time. Camera
    configured once at start (gain, exposure, bandwidth, RAW8 format). Video
    mode faster than repeated start_exposure/stop_exposure. Cleanup on
    generator close (finally block) stops video, updates _camera_streaming
    flag. Error handling: camera not found -> single error frame -> return;
    capture errors -> error frame with exception text -> continue (do not
    break stream).

    Args:
        camera_id: Camera index (0=finder, 1=main). Must match ASI SDK camera
            enumeration order. If camera not found, yields single error frame
            then returns (HTTP 200 with error image).
        exposure_us: Exposure time in microseconds (controls brightness). None
            uses stored settings from _camera_settings dict or per-camera
            defaults (finder: 10s, main: 300ms). Finder supports up to ~180s,
            main typically 10-5000ms.
        gain: Gain value (amplification). None uses stored settings or
            per-camera default (80). Range 0-600 (camera-dependent). Higher
            gain = brighter but more noise.
        fps: Target frames per second (yield rate). Default 15. Actual FPS
            limited by exposure time (cannot exceed 1000000/exposure_us).
            Higher FPS reduces latency but increases bandwidth.

    Yields:
        Bytes in MJPEG multipart format suitable for StreamingResponse:
        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n<jpeg_data>\r\n"
        Each yield is one complete frame. Browser buffers and displays as
        video.

    Returns:
        AsyncGenerator yielding frame bytes until client disconnects or
        _camera_streaming[camera_id] set to False. Generator cleanup (finally)
        stops video capture.

    Raises:
        None explicitly. Errors rendered as frames (cv2.putText exception
        message on black image). Generator continues after capture errors to
        maintain stream stability.

    Example:
        >>> # FastAPI endpoint
        >>> @app.get("/stream/0")
        >>> async def finder_stream():
        ...     return StreamingResponse(
        ...         _generate_camera_stream(0, exposure_us=50000, gain=80,
        ...                                 fps=15),
        ...         media_type="multipart/x-mixed-replace; boundary=frame"
        ...     )
        >>> # Browser: <img src="http://localhost:8000/stream/0">
        >>> # Displays live video until page closed or stream stopped
    """
    # Force reopen camera to ensure clean state for streaming
    camera = _get_camera(camera_id, force_reopen=True)
    if camera is None:
        # Yield error frame
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        assert _encoder is not None
        _encoder.put_text(
            error_img,
            f"Camera {camera_id} not found",
            (50, 240),
            1.0,
            (0, 0, 255),
            2,
        )
        jpeg = _encoder.encode_jpeg(error_img)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        return

    try:
        # Get camera info for dimensions
        info = camera.get_camera_property()
        width = info["MaxWidth"]
        height = info["MaxHeight"]

        # Use settings from arguments or stored settings
        settings = _camera_settings.get(camera_id, {})
        exp = (
            exposure_us
            if exposure_us is not None
            else settings.get("exposure_us", _get_default_exposure(camera_id))
        )
        g = (
            gain
            if gain is not None
            else settings.get("gain", _get_default_gain(camera_id))
        )

        # Configure camera for video
        camera.set_control_value(asi.ASI_GAIN, g)
        camera.set_control_value(asi.ASI_EXPOSURE, exp)
        # USB bandwidth: reduce per-camera when both are streaming to
        # prevent USB contention that crashes the secondary camera.
        # Count how many cameras are currently streaming (including this one)
        active_streams = sum(
            1
            for cid, streaming in _camera_streaming.items()
            if streaming and cid != camera_id
        )
        bandwidth = USB_BANDWIDTH_DUAL if active_streams > 0 else USB_BANDWIDTH_SINGLE
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, bandwidth)
        logger.info(
            "USB bandwidth configured",
            camera_id=camera_id,
            bandwidth_pct=bandwidth,
            other_active_streams=active_streams,
        )

        # Finder camera (0): Use RAW16 for maximum quality, no mode switch for capture
        # Main camera (1): Use RGB24 for color preview (debayered by SDK)
        is_color = info.get("IsColorCam", False)
        if camera_id == 0:
            # Finder: RAW16 mode - grayscale preview, but capture-ready
            camera.set_roi(
                width=width, height=height, bins=1, image_type=asi.ASI_IMG_RAW16
            )
            buffer_size = width * height * 2  # 16-bit = 2 bytes per pixel
            logger.info("Finder camera using RAW16 mode for capture-ready streaming")
        else:  # pragma: no cover - ASI SDK hardware setup for main camera
            # Main camera: RAW16 for maximum quality, grayscale preview
            camera.set_roi(
                width=width, height=height, bins=1, image_type=asi.ASI_IMG_RAW16
            )
            buffer_size = width * height * 2  # 16-bit = 2 bytes per pixel
            logger.info("Main camera using RAW16 mode for capture-ready streaming")

        # Pre-allocate frame buffer (required for full-frame capture)
        frame_buffer = bytearray(buffer_size)

        # Stop any existing video capture before starting new one
        try:
            camera.stop_video_capture()
        except Exception:  # pragma: no cover - ASI SDK hardware exception
            pass  # May not have been capturing

        # Start video capture
        camera.start_video_capture()
        _camera_streaming[camera_id] = True
        logger.info(
            f"Started video capture on camera {camera_id} "
            f"({width}x{height}, exp={exp}us, gain={g}, color={is_color})"
        )

        frame_interval = 1.0 / fps
        frame_count = 0
        consecutive_errors = 0

        # Timeout: exposure time + generous buffer for USB transfer,
        # SDK overhead, and contention with other cameras.
        # Previous: 2x exposure + 2s — too tight for long exposures
        # or dual-camera USB contention scenarios.
        timeout_ms = max(
            (exp + STREAM_TIMEOUT_BUFFER_US) // 1000,
            3000,
        )
        logger.info(
            "Stream timeout configured",
            camera_id=camera_id,
            exposure_us=exp,
            timeout_ms=timeout_ms,
            fps=fps,
            frame_interval_s=round(frame_interval, 3),
        )

        while _camera_streaming.get(
            camera_id, False
        ):  # pragma: no cover - infinite loop
            frame_start = asyncio.get_event_loop().time()
            try:
                # Run blocking capture in executor to not block event loop
                # This allows other streams to process while waiting
                # for long exposures
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: camera.capture_video_frame(
                        buffer_=frame_buffer, timeout=timeout_ms
                    ),
                )

                frame_count += 1
                consecutive_errors = 0  # Reset on success

                # All cameras use RAW16 - reshape and convert for display
                img_raw = np.frombuffer(frame_buffer, dtype=np.uint16).reshape(
                    (height, width)
                )

                # Store RAW16 frame for capture (before any processing)
                # Both cameras can grab from stream without mode switch
                _latest_frames[camera_id] = img_raw.copy()
                _latest_frame_info[camera_id] = {
                    "width": width,
                    "height": height,
                    "dtype": "uint16",
                    "is_color": is_color,
                    "exposure_us": exp,
                    "gain": g,
                }

                # Convert to 8-bit for display (scale 16-bit to 8-bit)
                img = (img_raw >> 8).astype(np.uint8)

                # Apply auto-stretch for visibility
                if img.max() > img.min():
                    # Use float arithmetic then convert back to uint8
                    img = (
                        ((img.astype(np.float32) - img.min()) * 255.0)
                        / (img.max() - img.min())
                    ).astype(np.uint8)

                # Encode as JPEG
                assert _encoder is not None
                jpeg = _encoder.encode_jpeg(img, quality=85)

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

                # Log frame timing periodically (every 100 frames)
                if frame_count % 100 == 0:
                    elapsed = loop.time() - frame_start
                    logger.info(
                        "Stream health",
                        camera_id=camera_id,
                        frames=frame_count,
                        last_frame_s=round(elapsed, 3),
                        timeout_ms=timeout_ms,
                    )

                await asyncio.sleep(frame_interval)

            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)

                # Classify error severity
                is_timeout = "timeout" in error_msg.lower()
                logger.warning(
                    "Frame capture error",
                    camera_id=camera_id,
                    error=error_msg[:80],
                    is_timeout=is_timeout,
                    consecutive_errors=consecutive_errors,
                    max_errors=MAX_CONSECUTIVE_ERRORS,
                    frame_count=frame_count,
                )

                # Stop stream after too many consecutive errors
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        "Stream stopped: too many consecutive errors",
                        camera_id=camera_id,
                        consecutive_errors=consecutive_errors,
                        last_error=error_msg[:80],
                    )
                    # Yield final error frame
                    err_img = np.zeros((height, width), dtype=np.uint8)
                    assert _encoder is not None
                    _encoder.put_text(
                        err_img,
                        f"Stream stopped: {consecutive_errors} errors",
                        (10, height // 2),
                        0.5,
                        255,
                        1,
                    )
                    jpeg = _encoder.encode_jpeg(err_img)
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                    break

                # Yield error frame but keep trying
                frame_error_img = np.zeros((height, width), dtype=np.uint8)
                assert _encoder is not None
                _encoder.put_text(
                    frame_error_img,
                    f"Frame error: {error_msg[:30]}",
                    (10, height // 2),
                    0.5,
                    255,
                    1,
                )
                jpeg = _encoder.encode_jpeg(frame_error_img)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

                # Exponential backoff on errors
                backoff = min(
                    ERROR_BACKOFF_BASE_S * (2 ** (consecutive_errors - 1)),
                    ERROR_BACKOFF_MAX_S,
                )
                await asyncio.sleep(backoff)

    except Exception as e:  # pragma: no cover - stream fatal error
        logger.error(f"Video stream error for camera {camera_id}: {e}")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        assert _encoder is not None
        _encoder.put_text(
            error_img,
            f"Stream error: {str(e)[:40]}",
            (20, 240),
            0.7,
            (0, 0, 255),
            2,
        )
        jpeg = _encoder.encode_jpeg(error_img)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
    finally:  # pragma: no cover - cleanup after stream ends
        # Stop video capture
        try:
            if _camera_streaming.get(camera_id):
                camera.stop_video_capture()
                _camera_streaming[camera_id] = False
                logger.info(f"Stopped video capture on camera {camera_id}")
        except Exception as e:
            logger.error(f"Error stopping video capture: {e}")


def main() -> None:
    """Run the telescope control web server.

    Entry point for running the dashboard as a standalone service.
    Creates the FastAPI application and starts uvicorn on all interfaces.

    Default configuration:
    - Host: 0.0.0.0 (accessible from network)
    - Port: 8080

    For production, consider running behind a reverse proxy (nginx)
    or using uvicorn worker processes.

    Args:
        None.

    Returns:
        None. Blocks until server is stopped (Ctrl+C).

    Raises:
        None. Uvicorn handles signals for graceful shutdown.

    Example:
        >>> # Run from command line:
        >>> # python -m telescope_mcp.web.app
        >>> main()
    """
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":  # pragma: no cover
    main()
