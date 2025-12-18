"""MCP Tools for camera control.

Uses the device layer with CameraRegistry for hardware abstraction.
Supports both real ASI cameras and digital twin simulation.
"""

import base64
import json
from dataclasses import asdict
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from telescope_mcp.devices import CameraRegistry, CaptureOptions, get_registry
from telescope_mcp.drivers.config import get_factory
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)


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


# Tool implementations using device layer


async def _list_cameras() -> list[TextContent]:
    """List connected cameras via CameraRegistry."""
    try:
        registry = get_registry()
        cameras = registry.discover()
        
        if not cameras:
            return [TextContent(type="text", text="No cameras connected")]
        
        result = {
            "count": len(cameras),
            "cameras": [
                {
                    "id": cam_id,
                    "name": info.name,
                    "max_width": info.max_width,
                    "max_height": info.max_height,
                }
                for cam_id, info in cameras.items()
            ]
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return [TextContent(type="text", text=f"Error listing cameras: {e}")]


async def _get_camera_info(camera_id: int) -> list[TextContent]:
    """Get detailed camera information."""
    try:
        registry = get_registry()
        camera = registry.get(camera_id, auto_connect=True)
        
        # Get info from camera
        info_dict = asdict(camera.info)
        
        result = {
            "camera_id": camera_id,
            "info": info_dict,
            "is_connected": camera.is_connected,
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error getting camera {camera_id} info: {e}")
        return [TextContent(type="text", text=f"Error getting camera info: {e}")]


async def _capture_frame(camera_id: int, exposure_us: int, gain: int) -> list[TextContent]:
    """Capture a single frame and return as base64 JPEG."""
    try:
        registry = get_registry()
        camera = registry.get(camera_id, auto_connect=True)
        
        # Capture with specified settings
        capture_result = camera.capture(CaptureOptions(
            exposure_us=exposure_us,
            gain=gain,
        ))
        
        # Encode image as base64
        b64_image = base64.b64encode(capture_result.image_data).decode('utf-8')
        
        result = {
            "camera_id": camera_id,
            "exposure_us": capture_result.exposure_us,
            "gain": capture_result.gain,
            "timestamp": capture_result.timestamp.isoformat(),
            "image_base64": b64_image,
        }
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        logger.error(f"Error capturing frame from camera {camera_id}: {e}")
        return [TextContent(type="text", text=f"Error capturing frame: {e}")]


async def _set_camera_control(camera_id: int, control: str, value: int) -> list[TextContent]:
    """Set a camera control value.
    
    Note: Control names should not include 'ASI_' prefix.
    Valid names: Gain, Exposure, WB_R, WB_B, Gamma, etc.
    """
    try:
        registry = get_registry()
        camera = registry.get(camera_id, auto_connect=True)
        
        # Remove ASI_ prefix if provided for backwards compatibility
        control_name = control.replace("ASI_", "")
        
        # Set control via camera instance
        if not camera._instance:
            return [TextContent(type="text", text="Camera not connected")]
        
        result_dict = camera._instance.set_control(control_name, value)
        result_dict["camera_id"] = camera_id
        
        return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]
    except Exception as e:
        logger.error(f"Error setting {control} on camera {camera_id}: {e}")
        return [TextContent(type="text", text=f"Error setting control: {e}")]


async def _get_camera_control(camera_id: int, control: str) -> list[TextContent]:
    """Get current value of a camera control.
    
    Note: Control names should not include 'ASI_' prefix.
    Valid names: Gain, Exposure, WB_R, WB_B, Temperature, etc.
    """
    try:
        registry = get_registry()
        camera = registry.get(camera_id, auto_connect=True)
        
        # Remove ASI_ prefix if provided for backwards compatibility
        control_name = control.replace("ASI_", "")
        
        # Get control via camera instance
        if not camera._instance:
            return [TextContent(type="text", text="Camera not connected")]
        
        result_dict = camera._instance.get_control(control_name)
        result_dict["camera_id"] = camera_id
        
        return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]
    except Exception as e:
        logger.error(f"Error getting {control} from camera {camera_id}: {e}")
        return [TextContent(type="text", text=f"Error getting control: {e}")]
