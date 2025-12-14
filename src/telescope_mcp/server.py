"""MCP Server entry point for telescope control."""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

from telescope_mcp.tools import cameras, motors, position


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("telescope-mcp")

    # Register tool handlers
    cameras.register(server)
    motors.register(server)
    position.register(server)

    return server


async def run_server() -> None:
    """Run the MCP server over stdio."""
    server = create_server()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
