"""MCP Tools for position sensing (altitude and azimuth)."""

from mcp.server import Server
from mcp.types import TextContent, Tool

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
    """Register position sensing tools with the MCP server.

    Prepares position tools for registration. Currently a placeholder
    as tools are registered via the combined cameras.py handler.
    Position implementations are provided as separate async functions.

    Tools provided by this module:
    - get_position: Read current telescope pointing
    - calibrate_position: Set known reference point
    - goto_position: Slew to target coordinates

    Args:
        server: MCP Server instance (unused, kept for interface consistency).

    Returns:
        None.

    Raises:
        None.
    """
    # Note: Tools are registered via cameras.py list_tools handler
    pass


# Tool implementations - stubs for now


async def get_position() -> list[TextContent]:
    """Get current telescope pointing position from sensors.

    Reads position sensors to determine current altitude (elevation)
    and azimuth angles. Values are in degrees using standard
    alt-az coordinate system.

    Position may be unavailable if sensors not calibrated or if
    motors haven't been homed since power-on.

    Note: Not yet implemented - returns placeholder message.

    Args:
        None.

    Returns:
        List with TextContent containing position. When implemented,
        will contain JSON: {"altitude": float, "azimuth": float,
        "calibrated": bool, "timestamp": str}
        Altitude: 0 (horizon) to 90 (zenith) degrees
        Azimuth: 0 (north) to 360, clockwise

    Raises:
        None. TODO: May raise on sensor errors when implemented.

    Example:
        >>> result = await get_position()
        >>> pos = json.loads(result[0].text)
        >>> print(f"Pointing at alt={pos['altitude']}°, az={pos['azimuth']}°")
    """
    # TODO: Implement with sensor driver
    return [TextContent(type="text", text="Position sensing not yet implemented")]


async def calibrate_position(altitude: float, azimuth: float) -> list[TextContent]:
    """Calibrate position sensors to a known reference point.

    Sets the current position reading to match known coordinates.
    Use when the telescope is pointed at a known target (e.g., after
    visual alignment to a star or landmark with known position).

    Calibration corrects for sensor drift and mounting offsets.
    Should be performed at the start of each observing session.

    Note: Not yet implemented - returns placeholder message.

    Args:
        altitude: Known altitude in degrees. Range: 0 (horizon) to 90
            (zenith). Must match actual pointing direction.
        azimuth: Known azimuth in degrees. Range: 0 to 360, where 0=north,
            90=east, 180=south, 270=west (clockwise from north).

    Returns:
        List with TextContent containing calibration result. When
        implemented, will contain JSON confirming new calibration.

    Raises:
        None. TODO: Will raise on invalid values when implemented.

    Example:
        >>> # After aligning to Polaris (altitude ~90° at pole, az=north)
        >>> result = await calibrate_position(altitude=89.5, azimuth=0)
    """
    # TODO: Implement
    return [
        TextContent(
            type="text",
            text=f"Calibrate to alt={altitude}, az={azimuth} - not yet implemented",
        )
    ]


async def goto_position(altitude: float, azimuth: float) -> list[TextContent]:
    """Slew telescope to a specific alt-az position.

    Moves both altitude and azimuth motors to point at the specified
    coordinates. Uses position feedback loop for accurate targeting.
    Blocks until movement completes or timeout.

    The slew speed is automatically managed for safety. Large movements
    may take 30-60 seconds. Position accuracy depends on calibration.

    Note: Not yet implemented - returns placeholder message.

    Args:
        altitude: Target altitude in degrees. Range: 0 (horizon) to 90
            (zenith). Values below 0 or above 90 are clamped.
        azimuth: Target azimuth in degrees. Range: 0 to 360, where 0=north.
            Values outside range are normalized (e.g., 370 -> 10).

    Returns:
        List with TextContent containing slew result. When implemented,
        will contain JSON: {"status": "arrived", "altitude": float,
        "azimuth": float, "duration_seconds": float}

    Raises:
        None. TODO: Will raise on motor errors when implemented.

    Example:
        >>> # Slew to southern horizon
        >>> result = await goto_position(altitude=10, azimuth=180)
    """
    # TODO: Implement with motor + position feedback loop
    return [
        TextContent(
            type="text", text=f"Goto alt={altitude}, az={azimuth} - not yet implemented"
        )
    ]
