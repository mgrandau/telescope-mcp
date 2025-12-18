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

from telescope_mcp.devices import CaptureOptions, get_registry
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
                    "description": (
                        "Control name (ASI_GAIN, ASI_EXPOSURE, "
                        "ASI_WB_R, ASI_WB_B, etc.)"
                    ),
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
    """Register camera control tools with the MCP server.

    Attaches camera-related tools to the MCP server instance, enabling
    AI assistants to discover and use camera functionality. Registers
    both the tool listing and call handlers.

    Tools registered:
    - list_cameras: Enumerate connected cameras
    - get_camera_info: Query camera properties
    - capture_frame: Take a single exposure
    - set_camera_control: Adjust camera settings
    - get_camera_control: Read camera settings

    Args:
        server: MCP Server instance to register tools with. Must be
            initialized but not yet running.

    Returns:
        None. Modifies server in-place by adding handlers.

    Raises:
        None. Registration itself doesn't access hardware.

    Example:
        >>> server = Server("telescope-mcp")
        >>> register(server)
        >>> # Tools now available via server.list_tools()
    """

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return available camera tools (MCP tool discovery).

        MCP handler providing tool definitions to clients. Called during MCP
        handshake when clients request available tools. Returns TOOLS list
        defining camera operations.

        Business context: Enables AI agents to discover camera control
        capabilities at runtime. Critical for AI-powered telescope automation
        where LLMs orchestrate imaging workflows. Provides schema-driven
        interface enabling type-safe AI tool calls.

        Args:
            None.

        Returns:
            List[Tool] defining camera capabilities (list_cameras, capture_frame, etc.).

        Raises:
            None. Always succeeds returning pre-built TOOLS list.

        Example:
            >>> tools = await list_tools()
            >>> print([t.name for t in tools])  # [list_cameras, capture_frame, ...]
        """
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Route tool calls to appropriate camera implementations.

        MCP handler that dispatches incoming tool calls based on name.
        Extracts arguments and calls the corresponding _* function.
        This is the primary entry point for all camera-related MCP tool
        invocations from AI agents or external clients.

        Business context: Enables AI agents (Claude, GPT) to control telescope
        cameras through the Model Context Protocol. Provides a uniform interface
        for camera discovery, configuration, and capture across different camera
        types (ASI hardware or digital twin). Essential for building AI-powered
        telescope automation where LLMs orchestrate imaging workflows.

        Args:
            name: Tool name from TOOLS definitions (list_cameras, get_camera_info,
                capture_frame, set_camera_control, get_camera_control).
            arguments: Dict of arguments matching tool's inputSchema. Keys and
                types validated by MCP framework before this handler.

        Returns:
            List containing single TextContent with JSON result string or error
            message. Success responses have structured JSON, errors have
            descriptive text.

        Raises:
            None. Errors are caught and returned as TextContent with error details.

        Example:
            # MCP client invocation from AI agent
            result = await call_tool(
                "capture_frame",
                {"camera_id": 0, "exposure_us": 100000, "gain": 50}
            )
            # Returns: [TextContent(type="text", text='{"success": true, ...}')]
        """
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
    """List all connected cameras via the CameraRegistry.

    Queries the camera registry to discover connected ASI cameras.
    Returns basic identification info for each camera including ID,
    name, and sensor dimensions.

    Uses the device layer abstraction, which handles both real
    hardware and digital twin simulation based on configuration.

    Args:
        None.

    Returns:
        List with single TextContent containing JSON:
        {"count": int, "cameras": [{"id": int, "name": str,
         "max_width": int, "max_height": int}, ...]}
        Returns "No cameras connected" if registry is empty.
        Returns error message on exceptions.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _list_cameras()
        >>> print(result[0].text)
        {"count": 2, "cameras": [{"id": 0, "name": "ASI120MC"}, ...]}
    """
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
            ],
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return [TextContent(type="text", text=f"Error listing cameras: {e}")]


async def _get_camera_info(camera_id: int) -> list[TextContent]:
    """Get detailed information about a specific camera.

    Retrieves comprehensive camera properties including sensor specs,
    supported controls, and connection status. Auto-connects to the
    camera if not already connected.

    The info dict includes: name, max_width, max_height, pixel_size,
    color_type, bayer_pattern, and supported control ranges.

    Args:
        camera_id: Zero-based camera index (0=finder, 1=main).

    Returns:
        List with single TextContent containing JSON:
        {"camera_id": int, "info": {...}, "is_connected": bool}
        Returns error message if camera not found or on exceptions.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _get_camera_info(0)
        >>> info = json.loads(result[0].text)
        >>> info["info"]["name"]
        "ASI120MC-S"
    """
    try:
        registry = get_registry()
        camera = registry.get(camera_id, auto_connect=True)

        # Get info from camera (guaranteed non-None after successful get)
        camera_info = camera.info
        assert camera_info is not None
        info_dict = asdict(camera_info)

        result = {
            "camera_id": camera_id,
            "info": info_dict,
            "is_connected": camera.is_connected,
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error getting camera {camera_id} info: {e}")
        return [TextContent(type="text", text=f"Error getting camera info: {e}")]


async def _capture_frame(
    camera_id: int, exposure_us: int, gain: int
) -> list[TextContent]:
    """Capture a single frame from a camera and return as base64 JPEG.

    Takes a single exposure with the specified settings and returns
    the image data encoded for transmission. Auto-connects to the
    camera if not already connected.

    The image is returned as base64-encoded JPEG suitable for display
    in web interfaces or saving to files. Metadata includes capture
    parameters and timestamp.

    Args:
        camera_id: Zero-based camera index (0=finder, 1=main).
        exposure_us: Exposure duration in microseconds. Typical range
            1000 (1ms) to 60000000 (60s). Common values: 100000 (100ms).
        gain: Amplification value. Range varies by camera, typically
            0-500. Higher values increase brightness but add noise.

    Returns:
        List with single TextContent containing JSON:
        {"camera_id": int, "exposure_us": int, "gain": int,
         "timestamp": str (ISO), "image_base64": str}
        Returns error message on capture failure.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _capture_frame(0, 100000, 50)
        >>> data = json.loads(result[0].text)
        >>> image_bytes = base64.b64decode(data["image_base64"])
    """
    try:
        registry = get_registry()
        camera = registry.get(camera_id, auto_connect=True)

        # Capture with specified settings
        capture_result = camera.capture(
            CaptureOptions(
                exposure_us=exposure_us,
                gain=gain,
            )
        )

        # Encode image as base64
        b64_image = base64.b64encode(capture_result.image_data).decode("utf-8")

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


async def _set_camera_control(
    camera_id: int, control: str, value: int
) -> list[TextContent]:
    """Set a camera control parameter to a specified value.

    Adjusts camera hardware settings like gain, exposure, white balance.
    Control names should not include 'ASI_' prefix (added for backwards
    compatibility if present). Changes take effect immediately.

    Common controls: Gain, Exposure, WB_R, WB_B, Gamma, Brightness,
    Offset, Flip, HighSpeedMode, BandwidthOverload.

    Args:
        camera_id: Zero-based camera index (0=finder, 1=main).
        control: Control name without ASI_ prefix (e.g., "Gain").
            Legacy "ASI_GAIN" format also accepted.
        value: Integer value to set. Valid range depends on control
            and camera model. Use get_camera_info for ranges.

    Returns:
        List with single TextContent containing JSON with result:
        {"camera_id": int, "control": str, "value": int,
         "min": int, "max": int, "default": int}
        Returns error message if camera not connected or control invalid.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _set_camera_control(0, "Gain", 100)
        >>> print(result[0].text)
        {"camera_id": 0, "control": "Gain", "value": 100, ...}
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
    """Get the current value of a camera control parameter.

    Reads camera hardware settings including current value, valid range,
    and whether auto mode is enabled. Control names should not include
    'ASI_' prefix (added for backwards compatibility if present).

    Common controls: Gain, Exposure, WB_R, WB_B, Temperature, Gamma,
    Brightness, Offset, Flip, HighSpeedMode.

    Args:
        camera_id: Zero-based camera index (0=finder, 1=main).
        control: Control name without ASI_ prefix (e.g., "Temperature").
            Legacy "ASI_TEMPERATURE" format also accepted.

    Returns:
        List with single TextContent containing JSON:
        {"camera_id": int, "control": str, "value": int,
         "min": int, "max": int, "default": int, "is_auto": bool}
        Returns error message if camera not connected or control invalid.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _get_camera_control(0, "Temperature")
        >>> data = json.loads(result[0].text)
        >>> print(f"Sensor temp: {data['value'] / 10}°C")
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
