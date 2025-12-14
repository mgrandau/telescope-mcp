"""MCP Server entry point for telescope control."""

import argparse
import asyncio
import logging
import threading
from typing import Optional

import uvicorn
from mcp.server import Server
from mcp.server.stdio import stdio_server

from telescope_mcp.tools import cameras, motors, position
from telescope_mcp.web.app import create_app

logger = logging.getLogger(__name__)

# Dashboard server thread
_dashboard_thread: Optional[threading.Thread] = None


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("telescope-mcp")

    # Register tool handlers
    cameras.register(server)
    motors.register(server)
    position.register(server)

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
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


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
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    asyncio.run(run_server(args.dashboard_host, args.dashboard_port))


if __name__ == "__main__":
    main()
