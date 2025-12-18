"""MCP Server entry point for telescope control."""

import argparse
import asyncio
import logging
import threading
from typing import Optional

import uvicorn
from mcp.server import Server
from mcp.server.stdio import stdio_server

from telescope_mcp.observability import configure_logging, get_logger
from telescope_mcp.tools import cameras, motors, position, sessions
from telescope_mcp.web.app import create_app

logger = get_logger(__name__)

# Dashboard server thread
_dashboard_thread: Optional[threading.Thread] = None


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("telescope-mcp")
    
    # Initialize camera registry with configured driver
    from telescope_mcp.devices import init_registry
    from telescope_mcp.drivers.config import get_factory
    
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
    """Run the dashboard server in a background thread."""
    app = create_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",  # Reduce noise in MCP mode
    )
    server = uvicorn.Server(config)
    server.run()


def start_dashboard(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the dashboard server in a background thread."""
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


async def run_server(dashboard_host: Optional[str], dashboard_port: Optional[int]) -> None:
    """Run the MCP server over stdio."""
    server = create_server()
    
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
        # Clean up camera registry on shutdown
        from telescope_mcp.devices import shutdown_registry
        shutdown_registry()
        logger.info("Camera registry shut down")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Configure structured logging
    configure_logging(level=logging.INFO)
    
    # Configure data directory if specified
    if args.data_dir:
        from pathlib import Path
        from telescope_mcp.drivers.config import set_data_dir
        set_data_dir(Path(args.data_dir))
        logger.info("Data directory configured", path=args.data_dir)
    
    # Initialize session manager and log startup
    from telescope_mcp.drivers.config import get_session_manager
    manager = get_session_manager()
    manager.log("INFO", "Telescope MCP server starting", source="server")
    
    logger.info("Starting MCP server")
    asyncio.run(run_server(args.dashboard_host, args.dashboard_port))


if __name__ == "__main__":
    main()
