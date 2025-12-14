"""MCP Tools for camera control."""

from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

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


# Tool implementations - stubs for now, will use drivers/pyasi

async def _list_cameras() -> list[TextContent]:
    """List connected cameras."""
    # TODO: Import and use drivers.pyasi
    return [TextContent(type="text", text="Camera listing not yet implemented - pyasi driver needed")]


async def _get_camera_info(camera_id: int) -> list[TextContent]:
    """Get camera info."""
    # TODO: Implement with pyasi
    return [TextContent(type="text", text=f"Camera {camera_id} info not yet implemented")]


async def _capture_frame(camera_id: int, exposure_us: int, gain: int) -> list[TextContent]:
    """Capture a frame."""
    # TODO: Implement with pyasi, return base64 JPEG
    return [TextContent(type="text", text=f"Capture from camera {camera_id} not yet implemented")]


async def _set_camera_control(camera_id: int, control: str, value: int) -> list[TextContent]:
    """Set camera control."""
    # TODO: Implement with pyasi
    return [TextContent(type="text", text=f"Set {control}={value} on camera {camera_id} not yet implemented")]


async def _get_camera_control(camera_id: int, control: str) -> list[TextContent]:
    """Get camera control value."""
    # TODO: Implement with pyasi
    return [TextContent(type="text", text=f"Get {control} from camera {camera_id} not yet implemented")]
