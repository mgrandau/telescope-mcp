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
    """Initialize ASI SDK if not already done."""
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
    """Get or open a camera instance."""
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
    """Close all open camera instances."""
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
    """Application lifespan - startup and shutdown."""
    # Startup: Initialize cameras, motors, sensors
    logger.info("Starting telescope control services...")
    _init_sdk()
    yield
    # Shutdown: Clean up
    logger.info("Shutting down telescope control services...")
    _close_all_cameras()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
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
        """Main dashboard page."""
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
        """MJPEG stream from finder camera."""
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
        """MJPEG stream from main imaging camera."""
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
        """MJPEG stream from any camera by ID."""
        return StreamingResponse(
            _generate_camera_stream(camera_id=camera_id, exposure_us=exposure_us, gain=gain, fps=fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/cameras")
    async def api_list_cameras() -> JSONResponse:
        """List available cameras."""
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
        """Move altitude motor."""
        # TODO: Implement
        return {"status": "ok", "steps": steps, "speed": speed}

    @app.post("/api/motor/azimuth")
    async def api_move_azimuth(steps: int, speed: int = 100) -> dict:
        """Move azimuth motor."""
        # TODO: Implement
        return {"status": "ok", "steps": steps, "speed": speed}

    @app.post("/api/motor/stop")
    async def api_stop_motors() -> dict:
        """Emergency stop all motors."""
        # TODO: Implement
        return {"status": "stopped"}

    @app.get("/api/position")
    async def api_get_position() -> dict:
        """Get current telescope position."""
        # TODO: Implement
        return {"altitude": 45.0, "azimuth": 180.0}

    @app.post("/api/camera/{camera_id}/control")
    async def api_set_camera_control(
        camera_id: int,
        control: str = Query(..., description="Control name (ASI_GAIN, ASI_EXPOSURE, etc.)"),
        value: int = Query(..., description="Value to set"),
    ) -> JSONResponse:
        """Set camera control value."""
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
    """Generate MJPEG stream from camera using video capture mode."""
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
    """Run the web server."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
