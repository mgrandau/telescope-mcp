"""MCP Server entry point for telescope control."""

import argparse
import asyncio
import logging
import threading

import uvicorn
from mcp.server import Server
from mcp.server.stdio import stdio_server

from telescope_mcp.observability import configure_logging, get_logger
from telescope_mcp.tools import cameras, motors, position, sessions
from telescope_mcp.web.app import create_app

logger = get_logger(__name__)

# Dashboard server thread and uvicorn server instance
_dashboard_thread: threading.Thread | None = None
_dashboard_server: uvicorn.Server | None = None


def create_server(mode: str = "digital_twin") -> Server:
    """Create and configure the MCP server for AI agent telescope control.

    Initializes the camera registry with the configured driver
    and registers all MCP tools (cameras, motors, position, sessions).
    Business context: Central server configuration that exposes telescope
    hardware to AI agents via Model Context Protocol. Enables Claude and
    other LLMs to control telescope operations through structured tool calls.
    Essential for AI-assisted astronomy, automated observations, and intelligent
    telescope scheduling.

    Args:
        mode: Driver mode - "hardware" for real cameras, "digital_twin" for simulation.
            Defaults to "digital_twin" for safety.

    Returns:
        Configured MCP Server instance with all tools registered.

    Raises:
        RuntimeError: If driver initialization or tool registration fails.

    Example:
        >>> server = create_server(mode="hardware")
        >>> # Server now exposes tools: list_cameras, capture_frame,
        >>> # move_motors, get_position, start_session, etc.
    """
    server = Server("telescope-mcp")

    # Initialize camera registry with configured driver
    from telescope_mcp.devices import init_registry
    from telescope_mcp.drivers.config import get_factory

    # Configure driver mode
    if mode.lower() == "hardware":
        from telescope_mcp.drivers.config import use_hardware

        use_hardware()
        logger.info("Using HARDWARE mode (real cameras)")
    else:
        logger.info("Using DIGITAL_TWIN mode (simulated cameras)")

    driver = get_factory().create_camera_driver()
    init_registry(driver)
    logger.info(f"Initialized camera registry with {type(driver).__name__}")

    # Register tool handlers
    cameras.register(server)
    motors.register(server)
    position.register(server)
    sessions.register(server)

    return server


def _run_dashboard(host: str, port: int) -> None:
    """Run the dashboard server in a background thread.

    Creates and runs a FastAPI/Uvicorn server for the web dashboard.

    Business context: Enables simultaneous human (web UI) and AI (MCP) control
    of telescope. Dashboard provides camera streams, manual controls, session
    monitoring while MCP handles AI agent requests. Critical for hybrid
    workflows and troubleshooting.

    Args:
        host: Host address to bind to (127.0.0.1 or 0.0.0.0).
        port: Port to listen on (typically 8080).

    Returns:
        None. Blocks running server until shutdown.

    Raises:
        OSError: If port already in use or binding fails.

    Example:
        >>> thread = Thread(
        ...     target=_run_dashboard, args=("127.0.0.1", 8080), daemon=True
        ... )
        >>> thread.start()
    """
    global _dashboard_server
    app = create_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",  # Reduce noise in MCP mode
    )
    _dashboard_server = uvicorn.Server(config)
    _dashboard_server.run()


def start_dashboard(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the web dashboard server in a background thread.

    Starts a daemon thread running the web dashboard. Safe to call
    multiple times (no-op if already running).
    Business context: Provides human-accessible web UI for telescope control,
    camera preview, and session monitoring while MCP server handles AI agent
    requests. Enables hybrid workflows where humans monitor/override AI decisions.
    Essential for troubleshooting, training, and manual intervention during
    automated observations.

    Args:
        host: Host address to bind to (default 127.0.0.1 for local only).
            Use 0.0.0.0 to allow remote access.
        port: Port to listen on (default 8080).

    Returns:
        None. Dashboard runs in background daemon thread.

    Raises:
        None. Errors during dashboard startup are logged but not raised.

    Example:
        >>> start_dashboard("0.0.0.0", 8080)  # Allow remote access
        >>> # Dashboard available at http://hostname:8080
        >>> start_dashboard()  # Already running - no-op
    """
    global _dashboard_thread
    if _dashboard_thread is not None and _dashboard_thread.is_alive():
        logger.warning("Dashboard already running")
        return

    _dashboard_thread = threading.Thread(
        target=_run_dashboard,
        args=(host, port),
        daemon=True,
        name="telescope-dashboard",
    )
    _dashboard_thread.start()
    logger.info(f"Dashboard started at http://{host}:{port}")


def stop_dashboard() -> None:
    """Stop the web dashboard server gracefully.

    Triggers shutdown of the uvicorn server running in the background thread.
    Safe to call even if dashboard is not running (no-op).

    Business context: Ensures clean shutdown when MCP server terminates,
    preventing orphaned web server processes. Critical for proper resource
    cleanup in production deployments and preventing port conflicts on restart.

    Args:
        None.

    Returns:
        None. Dashboard shutdown is asynchronous.

    Raises:
        None. Errors during shutdown are logged but not raised.

    Example:
        >>> start_dashboard("127.0.0.1", 8080)
        >>> # ... use dashboard ...
        >>> stop_dashboard()  # Triggers graceful shutdown
    """
    global _dashboard_server
    if _dashboard_server is not None:
        logger.info("Stopping dashboard server")
        _dashboard_server.should_exit = True
        _dashboard_server = None


async def run_server(
    dashboard_host: str | None, dashboard_port: int | None, mode: str = "digital_twin"
) -> None:
    """Run the MCP server over stdio for AI agent communication.

    Creates the MCP server, optionally starts the dashboard, then
    runs the MCP protocol over stdin/stdout. Cleans up registry on exit.

    Args:
        dashboard_host: Host for dashboard, or None to disable dashboard.
        dashboard_port: Port for dashboard, or None to disable dashboard.
        mode: Driver mode - "hardware" or "digital_twin" (default).

    Returns:
        None. Runs until stdin closes or process terminated.

    Raises:
        None. Errors are logged, registry cleanup always attempted.

    Example:
        >>> # Run with dashboard and hardware mode
        >>> await run_server("127.0.0.1", 8080, "hardware")
        >>> # Run without dashboard, simulation mode
        >>> await run_server(None, None, "digital_twin")
    """
    server = create_server(mode=mode)

    # Start dashboard if configured
    if dashboard_host and dashboard_port:
        start_dashboard(dashboard_host, dashboard_port)

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        # Stop dashboard server
        stop_dashboard()

        # Clean up camera registry on shutdown
        from telescope_mcp.devices import shutdown_registry

        shutdown_registry()
        logger.info("Camera registry shut down")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for telescope MCP server configuration.

    Processes command-line arguments to configure the MCP server and optional
    web dashboard. Arguments override default settings and enable customization
    for different deployment scenarios (development, production, multi-user).

    Business context: Enables flexible deployment of telescope control server
    across different environments. Development may use localhost:8080, production
    might use 0.0.0.0:80 behind nginx. Custom data directories allow multiple
    observers or separate test/production data. Essential for containerized
    deployments, remote observatories, and automated testing.

    Returns:
        argparse.Namespace with attributes:
        - dashboard_host: str | None - Host for web dashboard (None=no dashboard)
        - dashboard_port: int | None - Port for web dashboard (None=use default)
        - data_dir: str | None - Directory for ASDF session storage (None=default)

    Raises:
        SystemExit: On invalid arguments (e.g., --help, bad port number).

    Example:
        >>> args = parse_args()  # From sys.argv
        >>> if args.dashboard_host:
        ...     print(f"Dashboard: {args.dashboard_host}:{args.dashboard_port}")

    Raises:
        SystemExit: If invalid arguments provided (handled by argparse).

    Example:
        >>> # Run with dashboard on all interfaces, custom data directory
        >>> # Command: python -m telescope_mcp.server --dashboard-host 0.0.0.0
        >>> #          --dashboard-port 8080 --data-dir /data/telescope
        >>> args = parse_args()
        >>> print(f"Dashboard: {args.dashboard_host}:{args.dashboard_port}")
        >>> print(f"Data: {args.data_dir}")
        >>>
        >>> # Development mode (localhost only)
        >>> # Command: python -m telescope_mcp.server --dashboard-host 127.0.0.1
    """
    parser = argparse.ArgumentParser(
        description="Telescope MCP Server - Control cameras, motors, and sensors"
    )
    parser.add_argument(
        "--dashboard-host",
        type=str,
        default=None,
        help="Host to run the web dashboard on (e.g., 127.0.0.1)",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        help="Port to run the web dashboard on (e.g., 8080)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to store session data (ASDF files)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hardware", "digital_twin"],
        default="digital_twin",
        help=(
            "Driver mode: 'hardware' for real cameras, "
            "'digital_twin' for simulation (default)"
        ),
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=None,
        help=(
            "Observer latitude in degrees (e.g., 40.7128 for NYC). "
            "Required for ALT/AZ calculation."
        ),
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=None,
        help=(
            "Observer longitude in degrees (e.g., -74.0060 for NYC). "
            "Required for ALT/AZ calculation."
        ),
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.0,
        help="Observer height above sea level in meters (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for telescope-mcp server.

    Parses command-line arguments, configures structured logging and data
    directory, initializes session manager, and starts the MCP server over
    stdio. This is the entry point used by MCP clients (Claude Desktop,
    custom integrations) when launching the telescope control server.

    Business context: Primary server entry point for AI-controlled telescope
    operations. Called by MCP client configuration to spawn server process.
    Handles full lifecycle: argument parsing, resource initialization,
    server execution, and graceful shutdown. Essential for production
    deployments, development testing, and CI/CD validation.

    Implementation: Synchronous wrapper around async run_server(). Uses
    asyncio.run() to manage event loop lifecycle. Blocks until server
    terminates (stdin close or signal).

    Args:
        None. Reads sys.argv for command-line arguments.

    Returns:
        None. Exits when server terminates.

    Raises:
        SystemExit: On argument parsing errors or unhandled exceptions.

    Example:
        >>> # Typically called from MCP client config:
        >>> # "command": "python", "args": ["-m", "telescope_mcp.server"]
        >>> if __name__ == "__main__":
        ...     main()  # Runs until terminated
    """
    args = parse_args()

    # Configure structured logging
    configure_logging(level=logging.INFO)

    # Configure data directory if specified
    if args.data_dir:
        from pathlib import Path

        from telescope_mcp.drivers.config import set_data_dir

        set_data_dir(Path(args.data_dir))
        logger.info("Data directory configured", path=args.data_dir)

    # Configure observer location if specified
    if args.latitude is not None and args.longitude is not None:
        from telescope_mcp.drivers.config import set_location

        set_location(
            lat=args.latitude,
            lon=args.longitude,
            alt=args.height,
        )
        logger.info(
            "Observer location configured",
            latitude=args.latitude,
            longitude=args.longitude,
            height=args.height,
        )

    # Initialize session manager and log startup
    from telescope_mcp.drivers.config import get_session_manager

    manager = get_session_manager()
    manager.log("INFO", "Telescope MCP server starting", source="server")

    logger.info("Starting MCP server")
    asyncio.run(run_server(args.dashboard_host, args.dashboard_port, args.mode))


if __name__ == "__main__":
    main()
