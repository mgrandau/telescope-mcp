"""FastAPI web application for telescope dashboard."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import uvicorn
import zwoasi as asi
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from telescope_mcp.drivers.asi_sdk import get_sdk_library_path
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)

# Paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Camera state management
_sdk_initialized = False
_cameras: dict[int, asi.Camera] = {}  # Open camera instances
_camera_streaming: dict[int, bool] = {}  # Track which cameras are streaming
_camera_settings: dict[int, dict] = {}  # Store camera settings (exposure_us, gain)

# Default settings
DEFAULT_EXPOSURE_US = 100_000  # 100ms
DEFAULT_GAIN = 50
DEFAULT_FPS = 15


def _init_sdk() -> None:
    """Initialize the ZWO ASI SDK if not already initialized.

    Performs one-time SDK initialization required before any camera
    operations. Uses the SDK library path from the driver configuration.
    Thread-safe via global flag; subsequent calls are no-ops.

    This function is idempotent and safe to call multiple times.
    Initialization failures are logged but not raised, allowing the
    application to start even without cameras connected.

    Args:
        None.

    Returns:
        None. Sets global _sdk_initialized flag on success.

    Raises:
        None. Exceptions are caught and logged as warnings.
    """
    global _sdk_initialized
    if not _sdk_initialized:
        try:
            sdk_path = get_sdk_library_path()
            asi.init(sdk_path)
            _sdk_initialized = True
            logger.info(f"ASI SDK initialized from {sdk_path}")
        except Exception as e:
            logger.warning(f"ASI SDK init failed (no cameras?): {e}")


def _get_camera(camera_id: int) -> Optional[asi.Camera]:
    """Get or lazily open a camera instance by ID.

    Manages the camera connection lifecycle, opening cameras on first
    access and caching instances for reuse. Initializes the SDK if
    needed. Camera settings are initialized to defaults on first open.

    This is the primary camera access point for the web application,
    ensuring consistent state management across stream and API handlers.

    Args:
        camera_id: Zero-based camera index (0=finder, 1=main imaging).
            Must be less than the number of connected cameras.

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
                "exposure_us": DEFAULT_EXPOSURE_US,
                "gain": DEFAULT_GAIN,
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
    yield
    # Shutdown: Clean up
    logger.info("Shutting down telescope control services...")
    _close_all_cameras()


def create_app() -> FastAPI:
    """Create and configure the FastAPI telescope control application.

    Factory function that builds the complete web application with all
    routes, middleware, and configuration. Mounts static files, sets up
    Jinja2 templates, and registers all API endpoints.

    The application provides:
    - Dashboard UI at / for browser-based telescope control
    - MJPEG camera streams at /stream/{camera_id}
    - REST API for camera and motor control at /api/*

    This factory pattern allows testing with fresh app instances and
    supports different configurations per environment.

    Args:
        None.

    Returns:
        Configured FastAPI application instance ready for uvicorn.run().
        Includes lifespan handler for proper startup/shutdown.

    Raises:
        None. Missing static/template dirs are handled gracefully.

    Example:
        >>> app = create_app()
        >>> uvicorn.run(app, host="0.0.0.0", port=8080)
    """
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
        """Render the main telescope control dashboard page.

        Serves the HTML dashboard with camera streams, motor controls,
        and position display. The page uses JavaScript to interact
        with the REST API endpoints for real-time control.

        Args:
            request: FastAPI Request object (required for templates).

        Returns:
            HTMLResponse with rendered dashboard.html template.

        Raises:
            None. Template errors would raise TemplateNotFound.
        """
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "title": "Telescope Control"},
        )

    @app.get("/stream/finder")
    async def finder_stream(
        exposure_us: Optional[int] = Query(None, description="Exposure in microseconds"),
        gain: Optional[int] = Query(None, description="Gain value"),
        fps: int = Query(DEFAULT_FPS, description="Target frames per second"),
    ) -> StreamingResponse:
        """Stream MJPEG video from the finder camera (camera 0).

        Convenience endpoint for the finder/guide camera. Returns a
        continuous MJPEG stream suitable for <img> tags or video players.
        The finder camera is typically used for alignment and tracking.
        
        Business context: Enables real-time finder camera display in web dashboards
        for telescope alignment and target acquisition. The shorter exposures and
        higher frame rates typical of finder cameras make them ideal for live view
        while the main camera takes long exposures. Critical for "goto" operation
        verification and autoguiding feedback.

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
            response = requests.get('http://localhost:8080/stream/finder?fps=10', stream=True)
            for chunk in response.iter_content(chunk_size=1024):
                # Process MJPEG frames
                pass
        """
        return StreamingResponse(
            _generate_camera_stream(camera_id=0, exposure_us=exposure_us, gain=gain, fps=fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/main")
    async def main_stream(
        exposure_us: Optional[int] = Query(None, description="Exposure in microseconds"),
        gain: Optional[int] = Query(None, description="Gain value"),
        fps: int = Query(DEFAULT_FPS, description="Target frames per second"),
    ) -> StreamingResponse:
        """Stream MJPEG video from the main imaging camera (camera 1).

        Convenience endpoint for the primary imaging camera. Returns a
        continuous MJPEG stream for live preview during astrophotography.
        The main camera is typically higher resolution with better sensitivity.
        
        Business context: Provides live preview of the main imaging camera for
        focusing, framing, and quick target verification before committing to
        long exposures. Short preview exposures (50-200ms) allow responsive
        framing while the main camera's actual imaging uses much longer exposures
        (1-10 minutes). Essential for unguided setups to verify field before
        starting imaging sequences.

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
        return StreamingResponse(
            _generate_camera_stream(camera_id=1, exposure_us=exposure_us, gain=gain, fps=fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/{camera_id}")
    async def camera_stream(
        camera_id: int,
        exposure_us: Optional[int] = Query(None, description="Exposure in microseconds"),
        gain: Optional[int] = Query(None, description="Gain value"),
        fps: int = Query(DEFAULT_FPS, description="Target frames per second"),
    ) -> StreamingResponse:
        """Stream MJPEG video from any camera by numeric ID.

        Generic streaming endpoint for any connected camera. Use camera_id
        0 for finder, 1 for main, or higher indices for additional cameras.
        Returns error frame if camera_id doesn't exist.
        
        Business context: Enables flexible multi-camera setups where users can
        add cameras without hardcoding endpoints. Particularly useful for
        observatory setups with multiple guide cameras, all-sky cameras, or
        specialized imaging cameras. Dynamic UI generation can query /api/cameras
        then create stream views for each camera_id returned.

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
        return StreamingResponse(
            _generate_camera_stream(camera_id=camera_id, exposure_us=exposure_us, gain=gain, fps=fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/cameras")
    async def api_list_cameras() -> JSONResponse:
        """List all connected ASI cameras with basic information.

        Returns camera count and names for all detected ZWO ASI cameras.
        Initializes the SDK if not already done. Use camera IDs from this
        list for stream and control endpoints.

        Args:
            None.

        Returns:
            JSONResponse with structure:
            {"count": int, "cameras": [{"id": int, "name": str}, ...]}
            Returns {"error": str} with status 500 on SDK errors.

        Raises:
            None. Errors returned as JSON with appropriate status code.
        """
        _init_sdk()
        try:
            num_cameras = asi.get_num_cameras()
            if num_cameras == 0:
                return JSONResponse({"count": 0, "cameras": []})
            names = asi.list_cameras()
            return JSONResponse({
                "count": num_cameras,
                "cameras": [{"id": i, "name": n} for i, n in enumerate(names)],
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # API routes for dashboard JavaScript
    @app.post("/api/motor/altitude")
    async def api_move_altitude(steps: int, speed: int = 100) -> dict:
        """Move the altitude (elevation) motor by specified steps.

        Controls vertical telescope movement for pointing adjustment.
        Positive steps move up, negative move down (convention may vary
        by mount configuration).
        
        Business context: Enables programmatic telescope pointing control for
        goto functionality, drift correction, and manual adjustment via web UI.
        Altitude control is essential for object tracking as celestial objects
        change elevation throughout the night. Used by autoguiding systems to
        correct for atmospheric refraction and mount tracking errors.

        Note: Not yet implemented - returns placeholder response.

        Args:
            steps: Number of motor steps to move. Range depends on motor.
            speed: Motor speed as percentage (1-100). Default 100.

        Returns:
            Dict with status and echo of parameters:
            {"status": "ok", "steps": int, "speed": int}

        Raises:
            None. TODO: Will raise on motor errors when implemented.
        
        Example:
            # Move up 1000 steps at 50% speed
            response = requests.post(
                'http://localhost:8080/api/motor/altitude',
                params={'steps': 1000, 'speed': 50}
            )
            print(response.json())  # {"status": "ok", "steps": 1000, "speed": 50}
            
            # In JavaScript dashboard
            async function moveUp() {
                const response = await fetch('/api/motor/altitude?steps=500&speed=75', {
                    method: 'POST'
                });
                const result = await response.json();
                console.log('Move complete:', result);
            }
        """
        # TODO: Implement
        return {"status": "ok", "steps": steps, "speed": speed}

    @app.post("/api/motor/azimuth")
    async def api_move_azimuth(steps: int, speed: int = 100) -> dict:
        """Move the azimuth (rotation) motor by specified steps.

        Controls horizontal telescope rotation for pointing adjustment.
        Positive steps rotate clockwise (looking down), negative counter-
        clockwise (convention may vary by mount configuration).
        
        Business context: Enables horizontal telescope positioning for target
        acquisition and tracking. Azimuth adjustments compensate for Earth's
        rotation and allow slewing between targets. Critical for automated goto
        systems and periodic error correction in equatorial mounts. Slower speed
        settings enable fine adjustments for centering objects in eyepiece/camera.

        Note: Not yet implemented - returns placeholder response.

        Args:
            steps: Number of motor steps to move. Range depends on motor.
            speed: Motor speed as percentage (1-100). Default 100.

        Returns:
            Dict with status and echo of parameters:
            {"status": "ok", "steps": int, "speed": int}

        Raises:
            None. TODO: Will raise on motor errors when implemented.
        
        Example:
            # Rotate clockwise 2000 steps at full speed
            response = requests.post(
                'http://localhost:8080/api/motor/azimuth',
                params={'steps': 2000, 'speed': 100}
            )
            
            # Fine adjustment at slow speed
            response = requests.post(
                'http://localhost:8080/api/motor/azimuth',
                params={'steps': 50, 'speed': 20}
            )
            
            # Dashboard button handler
            document.getElementById('rotate-left').onclick = async () => {
                await fetch('/api/motor/azimuth?steps=-500&speed=50', {method: 'POST'});
            };
        """
        # TODO: Implement
        return {"status": "ok", "steps": steps, "speed": speed}

    @app.post("/api/motor/stop")
    async def api_stop_motors() -> dict:
        """Emergency stop all telescope motors immediately.

        Halts all motor movement for safety. Use when unexpected movement
        occurs or before manual intervention. Should be called before
        any maintenance operations.

        Note: Not yet implemented - returns placeholder response.

        Args:
            None.

        Returns:
            Dict confirming stop: {"status": "stopped"}

        Raises:
            None. TODO: May raise if motors fail to stop when implemented.
        """
        # TODO: Implement
        return {"status": "stopped"}

    @app.get("/api/position")
    async def api_get_position() -> dict:
        """Get current telescope pointing position.

        Returns altitude (elevation) and azimuth angles representing
        where the telescope is currently pointed. Values are in degrees.

        Note: Not yet implemented - returns placeholder values.

        Args:
            None.

        Returns:
            Dict with position angles in degrees:
            {"altitude": float, "azimuth": float}
            Altitude: 0 (horizon) to 90 (zenith)
            Azimuth: 0 (north) to 360, clockwise

        Raises:
            None. TODO: Will raise on encoder errors when implemented.
        """
        # TODO: Implement
        return {"altitude": 45.0, "azimuth": 180.0}

    @app.post("/api/camera/{camera_id}/control")
    async def api_set_camera_control(
        camera_id: int,
        control: str = Query(..., description="Control name (ASI_GAIN, ASI_EXPOSURE, etc.)"),
        value: int = Query(..., description="Value to set"),
    ) -> JSONResponse:
        """Set a camera control parameter value.

        Adjusts camera settings like gain, exposure, white balance, etc.
        Changes take effect immediately and persist for the camera session.
        Settings are also stored for use by streaming endpoints.
        
        Business context: Primary interface for dynamic camera adjustment during
        observation sessions. Enables adaptive exposure control as sky conditions
        change, gain adjustment for different target brightness, and real-time
        optimization of image quality. Critical for automated routines like
        auto-focus (adjusting exposure for star detection) and auto-exposure
        (adjusting gain/exposure for target brightness). Used by both manual UI
        controls and automated imaging scripts.

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
                const response = await fetch(
                    `/api/camera/0/control?control=ASI_GAIN&value=${e.target.value}`,
                    {method: 'POST'}
                );
                const result = await response.json();
                document.getElementById('gain-display').textContent = result.value_current;
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
            return JSONResponse(
                {"error": f"Unknown control: {control}", "valid": list(control_map.keys())},
                status_code=400,
            )
        
        camera = _get_camera(camera_id)
        if camera is None:
            return JSONResponse({"error": f"Camera {camera_id} not found"}, status_code=404)
        
        try:
            camera.set_control_value(control_map[control], value)
            
            # Update stored settings for exposure/gain (affects stream)
            if control == "ASI_EXPOSURE":
                _camera_settings.setdefault(camera_id, {})["exposure_us"] = value
            elif control == "ASI_GAIN":
                _camera_settings.setdefault(camera_id, {})["gain"] = value
            
            current = camera.get_control_value(control_map[control])
            return JSONResponse({
                "camera_id": camera_id,
                "control": control,
                "value_set": value,
                "value_current": current[0],
                "auto": current[1],
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return app


async def _generate_camera_stream(
    camera_id: int,
    exposure_us: Optional[int] = None,
    gain: Optional[int] = None,
    fps: int = DEFAULT_FPS,
) -> AsyncGenerator[bytes, None]:
    """Generate MJPEG video stream from a camera.

    Async generator that continuously captures frames from the specified
    camera and yields them as MJPEG-formatted chunks. Uses the ASI SDK's
    video capture mode for efficient streaming.

    The stream automatically applies auto-stretch to each frame for
    better visibility of dim astronomical objects. Error frames are
    displayed in-stream rather than breaking the connection.

    This generator handles the full capture lifecycle including camera
    configuration, video start/stop, and cleanup on generator close.

    Args:
        camera_id: Camera index (0=finder, 1=main). If camera not found,
            yields a single error frame and returns.
        exposure_us: Exposure time in microseconds. If None, uses stored
            camera settings or DEFAULT_EXPOSURE_US (100000).
        gain: Gain value (0-100 typical range). If None, uses stored
            settings or DEFAULT_GAIN (50).
        fps: Target frames per second. Controls yield rate, not exposure.
            Default 15. Higher values may exceed camera capabilities.

    Yields:
        Bytes in MJPEG multipart format:
        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n<jpeg_data>\r\n"

    Raises:
        None. Errors are rendered as text on error frames.

    Example:
        >>> @app.get("/stream/0")
        >>> async def stream():
        ...     return StreamingResponse(
        ...         _generate_camera_stream(0, exposure_us=50000),
        ...         media_type="multipart/x-mixed-replace; boundary=frame"
        ...     )
    """
    camera = _get_camera(camera_id)
    if camera is None:
        # Yield error frame
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            error_img, f"Camera {camera_id} not found",
            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        _, jpeg = cv2.imencode(".jpg", error_img)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        return

    try:
        # Get camera info for dimensions
        info = camera.get_camera_property()
        width = info["MaxWidth"]
        height = info["MaxHeight"]
        
        # Use settings from arguments or stored settings
        settings = _camera_settings.get(camera_id, {})
        exp = exposure_us if exposure_us is not None else settings.get("exposure_us", DEFAULT_EXPOSURE_US)
        g = gain if gain is not None else settings.get("gain", DEFAULT_GAIN)
        
        # Configure camera for video
        camera.set_control_value(asi.ASI_GAIN, g)
        camera.set_control_value(asi.ASI_EXPOSURE, exp)
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 80)  # USB bandwidth
        camera.set_image_type(asi.ASI_IMG_RAW8)
        
        # Start video capture
        camera.start_video_capture()
        _camera_streaming[camera_id] = True
        logger.info(f"Started video capture on camera {camera_id} ({width}x{height}, exp={exp}us, gain={g})")
        
        frame_interval = 1.0 / fps
        
        while _camera_streaming.get(camera_id, False):
            try:
                # Capture video frame (timeout in ms)
                timeout_ms = max(int(exp / 1000) + 500, 1000)
                data = camera.capture_video_frame(timeout=timeout_ms)
                
                # Reshape to image
                img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
                
                # Apply auto-stretch for visibility
                if img.max() > img.min():
                    img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
                
                # Encode as JPEG
                _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                
                await asyncio.sleep(frame_interval)
                
            except Exception as e:
                logger.warning(f"Frame capture error: {e}")
                # Yield error frame but keep trying
                error_img = np.zeros((height, width), dtype=np.uint8)
                cv2.putText(
                    error_img, f"Frame error: {str(e)[:30]}",
                    (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1
                )
                _, jpeg = cv2.imencode(".jpg", error_img)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                await asyncio.sleep(0.5)
                
    except Exception as e:
        logger.error(f"Video stream error for camera {camera_id}: {e}")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            error_img, f"Stream error: {str(e)[:40]}",
            (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        _, jpeg = cv2.imencode(".jpg", error_img)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
    finally:
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
    or using uvicorn's worker processes.

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


if __name__ == "__main__":
    main()
