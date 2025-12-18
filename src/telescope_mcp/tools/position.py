"""MCP Tools for position sensing (altitude and azimuth)."""

from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from telescope_mcp.observability import get_logger

logger = get_logger(__name__)

# Tool definitions
TOOLS = [
    Tool(
        name="get_position",
        description="Get current telescope position (altitude and azimuth)",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="calibrate_position",
        description="Calibrate position sensors to a known reference",
        inputSchema={
            "type": "object",
            "properties": {
                "altitude": {
                    "type": "number",
                    "description": "Known altitude in degrees (0-90)",
                },
                "azimuth": {
                    "type": "number",
                    "description": "Known azimuth in degrees (0-360)",
                },
            },
            "required": ["altitude", "azimuth"],
        },
    ),
    Tool(
        name="goto_position",
        description="Move telescope to a specific alt/az position",
        inputSchema={
            "type": "object",
            "properties": {
                "altitude": {
                    "type": "number",
                    "description": "Target altitude in degrees (0-90)",
                },
                "azimuth": {
                    "type": "number",
                    "description": "Target azimuth in degrees (0-360)",
                },
            },
            "required": ["altitude", "azimuth"],
        },
    ),
]


def register(server: Server) -> None:
    """Register position tools with the MCP server."""
    # Note: Tools are registered via cameras.py list_tools handler
    pass


# Tool implementations - stubs for now

async def get_position() -> list[TextContent]:
    """Get current position from sensors."""
    # TODO: Implement with sensor driver
    return [TextContent(type="text", text="Position sensing not yet implemented")]


async def calibrate_position(altitude: float, azimuth: float) -> list[TextContent]:
    """Calibrate position sensors."""
    # TODO: Implement
    return [TextContent(type="text", text=f"Calibrate to alt={altitude}, az={azimuth} - not yet implemented")]


async def goto_position(altitude: float, azimuth: float) -> list[TextContent]:
    """Go to a specific position."""
    # TODO: Implement with motor + position feedback loop
    return [TextContent(type="text", text=f"Goto alt={altitude}, az={azimuth} - not yet implemented")]
