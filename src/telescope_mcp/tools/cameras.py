"""MCP Tools for camera control.

Uses the zwoasi Python package with the ASI Camera 2 SDK.
"""

import base64
import json
import logging
from typing import Any

import zwoasi as asi
from mcp.server import Server
from mcp.types import TextContent, Tool

from telescope_mcp.drivers.asi_sdk import get_sdk_library_path

logger = logging.getLogger(__name__)

# Initialize ASI SDK on module load
_sdk_initialized = False


def _ensure_sdk_initialized() -> None:
    """Initialize the ASI SDK if not already done."""
    global _sdk_initialized
    if not _sdk_initialized:
        try:
            sdk_path = get_sdk_library_path()
            asi.init(sdk_path)
            _sdk_initialized = True
            logger.info(f"ASI SDK initialized from {sdk_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ASI SDK: {e}")
            raise


# Tool definitions
TOOLS = [
    Tool(
        name="list_cameras",
        description="List all connected ASI cameras with their properties",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="get_camera_info",
        description="Get detailed information about a specific camera",
        inputSchema={
            "type": "object",
            "properties": {
                "camera_id": {
                    "type": "integer",
                    "description": "Camera ID (0 for finder, 1 for main)",
                },
            },
            "required": ["camera_id"],
        },
    ),
    Tool(
        name="capture_frame",
        description="Capture a single frame from a camera",
        inputSchema={
            "type": "object",
            "properties": {
                "camera_id": {
                    "type": "integer",
                    "description": "Camera ID (0 for finder, 1 for main)",
                },
                "exposure_us": {
                    "type": "integer",
                    "description": "Exposure time in microseconds",
                    "default": 100000,
                },
                "gain": {
                    "type": "integer",
                    "description": "Gain value",
                    "default": 50,
                },
            },
            "required": ["camera_id"],
        },
    ),
    Tool(
        name="set_camera_control",
        description="Set a camera control value (gain, exposure, white balance, etc.)",
        inputSchema={
            "type": "object",
            "properties": {
                "camera_id": {
                    "type": "integer",
                    "description": "Camera ID",
                },
                "control": {
                    "type": "string",
                    "description": "Control name (ASI_GAIN, ASI_EXPOSURE, ASI_WB_R, ASI_WB_B, etc.)",
                },
                "value": {
                    "type": "integer",
                    "description": "Control value to set",
                },
            },
            "required": ["camera_id", "control", "value"],
        },
    ),
    Tool(
        name="get_camera_control",
        description="Get current value of a camera control",
        inputSchema={
            "type": "object",
            "properties": {
                "camera_id": {
                    "type": "integer",
                    "description": "Camera ID",
                },
                "control": {
                    "type": "string",
                    "description": "Control name",
                },
            },
            "required": ["camera_id", "control"],
        },
    ),
]


def register(server: Server) -> None:
    """Register camera tools with the MCP server."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "list_cameras":
            return await _list_cameras()
        elif name == "get_camera_info":
            return await _get_camera_info(arguments["camera_id"])
        elif name == "capture_frame":
            return await _capture_frame(
                arguments["camera_id"],
                arguments.get("exposure_us", 100000),
                arguments.get("gain", 50),
            )
        elif name == "set_camera_control":
            return await _set_camera_control(
                arguments["camera_id"],
                arguments["control"],
                arguments["value"],
            )
        elif name == "get_camera_control":
            return await _get_camera_control(
                arguments["camera_id"],
                arguments["control"],
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


# Tool implementations using zwoasi + ASI SDK


async def _list_cameras() -> list[TextContent]:
    """List connected ASI cameras."""
    try:
        _ensure_sdk_initialized()
        num_cameras = asi.get_num_cameras()
        
        if num_cameras == 0:
            return [TextContent(type="text", text="No ASI cameras connected")]
        
        camera_names = asi.list_cameras()
        result = {
            "count": num_cameras,
            "cameras": [
                {"id": i, "name": name}
                for i, name in enumerate(camera_names)
            ]
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return [TextContent(type="text", text=f"Error listing cameras: {e}")]


async def _get_camera_info(camera_id: int) -> list[TextContent]:
    """Get detailed camera information."""
    try:
        _ensure_sdk_initialized()
        
        camera = asi.Camera(camera_id)
        info = camera.get_camera_property()
        controls = camera.get_controls()
        camera.close()
        
        result = {
            "camera_id": camera_id,
            "properties": info,
            "controls": {
                name: {
                    "min": ctrl["MinValue"],
                    "max": ctrl["MaxValue"],
                    "default": ctrl["DefaultValue"],
                    "is_auto_supported": ctrl["IsAutoSupported"],
                    "is_writable": ctrl["IsWritable"],
                }
                for name, ctrl in controls.items()
            }
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error getting camera {camera_id} info: {e}")
        return [TextContent(type="text", text=f"Error getting camera info: {e}")]


async def _capture_frame(camera_id: int, exposure_us: int, gain: int) -> list[TextContent]:
    """Capture a single frame and return as base64 JPEG."""
    try:
        _ensure_sdk_initialized()
        
        camera = asi.Camera(camera_id)
        camera_info = camera.get_camera_property()
        
        # Set up capture parameters
        camera.set_control_value(asi.ASI_GAIN, gain)
        camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
        
        # Use camera's native resolution
        camera.set_image_type(asi.ASI_IMG_RAW8)
        
        # Capture image
        camera.start_exposure()
        
        # Wait for exposure and get data
        import time
        time.sleep(exposure_us / 1_000_000 + 0.1)  # Wait for exposure + buffer
        
        status = camera.get_exposure_status()
        while status == asi.ASI_EXP_WORKING:
            time.sleep(0.01)
            status = camera.get_exposure_status()
        
        if status == asi.ASI_EXP_SUCCESS:
            # Get the image data
            img = camera.get_data_after_exposure()
            camera.close()
            
            # Convert to JPEG using opencv
            import cv2
            import numpy as np
            
            # Reshape based on camera info
            width = camera_info["MaxWidth"]
            height = camera_info["MaxHeight"]
            img_array = np.frombuffer(img, dtype=np.uint8).reshape((height, width))
            
            # Encode as JPEG
            _, jpeg_data = cv2.imencode('.jpg', img_array)
            b64_image = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
            
            result = {
                "camera_id": camera_id,
                "width": width,
                "height": height,
                "exposure_us": exposure_us,
                "gain": gain,
                "image_base64": b64_image,
            }
            return [TextContent(type="text", text=json.dumps(result))]
        else:
            camera.close()
            return [TextContent(type="text", text=f"Exposure failed with status: {status}")]
            
    except Exception as e:
        logger.error(f"Error capturing frame from camera {camera_id}: {e}")
        return [TextContent(type="text", text=f"Error capturing frame: {e}")]


async def _set_camera_control(camera_id: int, control: str, value: int) -> list[TextContent]:
    """Set a camera control value."""
    try:
        _ensure_sdk_initialized()
        
        # Map control name to ASI constant
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
            return [TextContent(type="text", text=f"Unknown control: {control}. Valid controls: {list(control_map.keys())}")]
        
        camera = asi.Camera(camera_id)
        camera.set_control_value(control_map[control], value)
        
        # Read back the value to confirm
        current = camera.get_control_value(control_map[control])
        camera.close()
        
        result = {
            "camera_id": camera_id,
            "control": control,
            "value_set": value,
            "value_current": current[0],
            "auto": current[1],
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error setting {control} on camera {camera_id}: {e}")
        return [TextContent(type="text", text=f"Error setting control: {e}")]


async def _get_camera_control(camera_id: int, control: str) -> list[TextContent]:
    """Get current value of a camera control."""
    try:
        _ensure_sdk_initialized()
        
        control_map = {
            "ASI_GAIN": asi.ASI_GAIN,
            "ASI_EXPOSURE": asi.ASI_EXPOSURE,
            "ASI_GAMMA": asi.ASI_GAMMA,
            "ASI_WB_R": asi.ASI_WB_R,
            "ASI_WB_B": asi.ASI_WB_B,
            "ASI_BRIGHTNESS": asi.ASI_BRIGHTNESS,
            "ASI_OFFSET": asi.ASI_OFFSET,
            "ASI_BANDWIDTHOVERLOAD": asi.ASI_BANDWIDTHOVERLOAD,
            "ASI_TEMPERATURE": asi.ASI_TEMPERATURE,
            "ASI_FLIP": asi.ASI_FLIP,
            "ASI_HIGH_SPEED_MODE": asi.ASI_HIGH_SPEED_MODE,
        }
        
        if control not in control_map:
            return [TextContent(type="text", text=f"Unknown control: {control}. Valid controls: {list(control_map.keys())}")]
        
        camera = asi.Camera(camera_id)
        current = camera.get_control_value(control_map[control])
        camera.close()
        
        result = {
            "camera_id": camera_id,
            "control": control,
            "value": current[0],
            "auto": current[1],
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error getting {control} from camera {camera_id}: {e}")
        return [TextContent(type="text", text=f"Error getting control: {e}")]
