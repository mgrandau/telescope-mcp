"""CLI entry point for telescope-mcp.

Provides the ``telescope-mcp`` console script with subcommands:

- ``install`` — Generate ``.vscode/mcp.json`` for a project
- ``server`` — Run the MCP server (default if no subcommand)

Usage::

    # Install MCP config in current project
    telescope-mcp install

    # Install to global VS Code settings
    telescope-mcp install --global

    # Run MCP server (default, same as python -m telescope_mcp.server)
    telescope-mcp

    # Run MCP server with explicit subcommand
    telescope-mcp server --dashboard-host 127.0.0.1 --dashboard-port 8080

Module Structure:
    - ``main()`` — CLI entry point, dispatches subcommands
    - ``run_install()`` — Generate/update .vscode/mcp.json
    - ``_generate_mcp_template()`` — JSONC template with all options
    - ``_strip_jsonc_comments()`` — Parse existing JSONC configs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

# Constants
SERVER_NAME = "telescope-mcp"
MODULE_NAME = "telescope_mcp.server"
VSCODE_DIR = ".vscode"
CONFIG_FILE = "mcp.json"


@lru_cache(maxsize=1)
def _get_logger() -> logging.Logger:
    """Get module logger with cached initialization.

    Creates a logger with simple message-only format for CLI output.
    Cached to avoid handler duplication on repeated calls.

    Returns:
        logging.Logger: Configured logger for CLI output.

    Example:
        >>> logger = _get_logger()
        >>> logger.info("Server started")
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger(__name__)


def _log(message: str, *, emoji: str = "") -> None:
    """Log message with optional emoji prefix for CLI feedback.

    Args:
        message: Message text to log.
        emoji: Optional emoji prefix for visual status indicators.

    Example:
        >>> _log("Config created", emoji="✅")
        >>> _log("Backup saved", emoji="💾")
    """
    prefix = f"{emoji} " if emoji else ""
    _get_logger().info(f"{prefix}{message}")


def _strip_jsonc_comments(text: str) -> str:
    """Strip single-line // comments from JSONC text.

    Removes ``//`` comments (both standalone and trailing) and
    cleans up trailing commas that would result in invalid JSON.
    Does not handle ``/* */`` block comments.

    Args:
        text: JSONC text with ``//`` comments.

    Returns:
        Standard JSON text safe for ``json.loads()``.

    Example:
        >>> _strip_jsonc_comments('{"key": "val"} // comment')
        '{"key": "val"} '
        >>> _strip_jsonc_comments('["a", // comment\\n]')
        '["a" \\n]'
    """
    # Remove // comments (not inside strings — works for mcp.json)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    # Remove trailing commas before ] or } (invalid in JSON)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _detect_python_path() -> str:
    """Detect the Python executable path in the current environment.

    Returns the path to the Python interpreter running this script,
    which is the correct path for the venv where telescope-mcp is
    installed.

    Returns:
        Absolute path to the Python executable.

    Example:
        >>> _detect_python_path()
        '/home/user/project/.venv/bin/python'
    """
    return sys.executable


def _get_global_vscode_dir() -> Path:
    """Detect the global VS Code user settings directory.

    Checks for VS Code Insiders first (preferred), then regular
    VS Code. Supports Linux, macOS, and Windows.

    Returns:
        Path to the VS Code User settings directory.

    Example:
        >>> _get_global_vscode_dir()
        PosixPath('/home/user/.config/Code - Insiders/User')
    """
    home = Path.home()

    if sys.platform == "darwin":  # pragma: no cover
        base = home / "Library" / "Application Support"
    elif os.name == "nt":  # pragma: no cover
        base = home / "AppData" / "Roaming"
    else:  # Linux
        base = home / ".config"

    # Prefer Insiders if it exists
    insiders = base / "Code - Insiders" / "User"
    if insiders.exists():
        return insiders

    regular = base / "Code" / "User"
    return regular


def _generate_mcp_template(python_path: str) -> str:
    """Generate full JSONC mcp.json with all telescope-mcp options.

    Creates a complete configuration template with all available CLI
    arguments shown. Disabled options are commented out with their
    defaults, following the project convention of showing all options
    for discoverability.

    Args:
        python_path: Absolute path to the Python executable.

    Returns:
        JSONC string ready to write to mcp.json.

    Example:
        >>> t = _generate_mcp_template("/home/user/.venv/bin/python")
        >>> Path(".vscode/mcp.json").write_text(t)
    """
    # Template uses {{PYTHON_PATH}} placeholder to avoid f-string
    # issues with JSON curly braces
    template = """\
{
  "servers": {
    "telescope-mcp": {
      "command": "{{PYTHON_PATH}}",
      "args": [
        "-m",
        "telescope_mcp.server",
        // Dashboard settings
        "--dashboard-host", "127.0.0.1",
        "--dashboard-port", "8080",
        // Driver mode: "hardware" for real cameras,
        // "digital_twin" for simulation
        "--mode", "digital_twin",
        // Dashboard log level
        // (critical/error/warning/info/debug)
        // "--dashboard-log-level", "warning",
        // Data directory for ASDF session files
        // "--data-dir", "./data",
        // Observer location
        // (required for RA/Dec coordinate conversion)
        // "--latitude", "0.0",
        // "--longitude", "0.0",
        // "--height", "0.0",
        // Per-camera defaults
        // (uncomment to override code defaults)
        // Finder camera (Camera 0: ASI120MC-S)
        // Wide-field, long exposure
        // "--finder-exposure-us", "10000000",
        //   ^^ 10s (range: 1-180000000)
        // "--finder-gain", "80",
        //   ^^ (range: 0-510)
        // Main camera (Camera 1: ASI482MC)
        // High-res, short exposure
        // "--main-exposure-us", "60000",
        //   ^^ 60ms (range: 1-600000000)
        // "--main-gain", "80"
        //   ^^ (range: 0-570)
      ]
    }
  }
}
"""
    return template.replace("{{PYTHON_PATH}}", python_path)


def run_install(
    cwd: str | None = None,
    *,
    global_install: bool = False,
) -> None:
    """Install telescope-mcp MCP configuration.

    Creates or updates ``.vscode/mcp.json`` to include the telescope-mcp
    server configuration. Generates a JSONC template with all available
    options documented as comments for discoverability.

    Behavior:
    - **No existing config**: Writes full JSONC template with all options.
    - **Existing config without telescope-mcp**: Backs up original, parses
      JSONC, adds telescope-mcp server entry, writes merged config.
    - **Already installed**: Reports up-to-date, no changes.

    Args:
        cwd: Working directory (project root). Defaults to current
            directory. Pass explicitly for testing.
        global_install: If True, install to user's global VS Code
            settings instead of project ``.vscode/`` directory.

    Returns:
        None. Progress messages printed to stdout.

    Raises:
        None. Errors are logged, not raised.

    Example:
        >>> # Install in current project
        >>> run_install()
        ✅ Created .vscode/mcp.json
        >>> # Install globally
        >>> run_install(global_install=True)
        🌐 Installing globally to: ~/.config/Code - Insiders/User
    """
    working_dir = Path(cwd) if cwd else Path.cwd()
    python_path = _detect_python_path()

    # Determine target directory
    if global_install:
        vscode_dir = _get_global_vscode_dir()
        _log(f"Installing globally to: {vscode_dir}", emoji="🌐")
    else:
        vscode_dir = working_dir / VSCODE_DIR

    config_path = vscode_dir / CONFIG_FILE

    if not config_path.exists():
        # Fresh install: write full JSONC template
        vscode_dir.mkdir(parents=True, exist_ok=True)
        template = _generate_mcp_template(python_path)
        config_path.write_text(template)
        _log(f"Created {config_path}", emoji="✅")
    else:
        # Parse existing config
        existing_text = config_path.read_text()

        # Back up existing config
        backup_path = config_path.with_suffix(".json.bak")
        backup_path.write_text(existing_text)
        _log(f"Backed up to {backup_path.name}", emoji="💾")

        # Parse JSONC (strip comments, fix trailing commas)
        clean_json = _strip_jsonc_comments(existing_text)
        try:
            config: dict[str, object] = json.loads(clean_json)
        except json.JSONDecodeError:
            _log(
                "Could not parse existing config, writing fresh",
                emoji="⚠️",
            )
            template = _generate_mcp_template(python_path)
            config_path.write_text(template)
            _log(f"Created {config_path}", emoji="✅")
            _log(f"Python: {python_path}")
            return

        servers: dict[str, object] = config.setdefault(  # type: ignore[assignment]
            "servers", {}
        )

        if SERVER_NAME in servers:
            _log(
                f"{SERVER_NAME} already configured in {CONFIG_FILE}",
                emoji="✅",
            )
            return

        # Add telescope-mcp server config
        servers[SERVER_NAME] = {
            "command": python_path,
            "args": [
                "-m",
                MODULE_NAME,
                "--dashboard-host",
                "127.0.0.1",
                "--dashboard-port",
                "8080",
                "--mode",
                "digital_twin",
            ],
        }

        config_path.write_text(json.dumps(config, indent=2) + "\n")
        _log(f"Added {SERVER_NAME} to {CONFIG_FILE}", emoji="➕")
        _log(
            "Note: JSONC comments from original were not preserved",
            emoji="⚠️",
        )

    _log(f"Config: {config_path}")
    _log(f"Python: {python_path}")


def main() -> int:
    """Main CLI entry point for telescope-mcp.

    Dispatches to subcommands:
    - ``install``: Generate .vscode/mcp.json configuration
    - ``server`` or no subcommand: Run MCP server (delegates to
      ``server.main()`` which has its own arg parser)

    Returns:
        Exit code 0 for success, non-zero for errors.

    Raises:
        SystemExit: On --help or argument parsing errors.

    Example:
        >>> # Installed as console script:
        >>> # telescope-mcp install
        >>> # telescope-mcp install --global
        >>> # telescope-mcp server --dashboard-host 0.0.0.0
        >>> # telescope-mcp  (no args = run server)
    """
    parser = argparse.ArgumentParser(
        prog="telescope-mcp",
        description=(
            "Telescope MCP — AI-controlled telescope operations "
            "(cameras, motors, sensors)"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    # Install subcommand
    install_parser = subparsers.add_parser(
        "install",
        help="Create .vscode/mcp.json configuration",
    )
    install_parser.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install to global VS Code settings",
    )

    # Server subcommand (pass-through to server.main())
    subparsers.add_parser(
        "server",
        help="Run MCP server (default if no subcommand)",
        add_help=False,
    )

    # Only parse known args so server flags pass through
    args, _ = parser.parse_known_args()

    if args.command == "install":
        run_install(global_install=args.global_install)
        return 0

    # Default or "server": delegate to server.main()
    # Strip "server" subcommand so server.parse_args() works
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    from telescope_mcp.server import main as server_main

    server_main()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
