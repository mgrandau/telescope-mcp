"""Tests for telescope_mcp.cli — install command and CLI dispatch.

Tests the ``telescope-mcp install`` CLI command that generates
``.vscode/mcp.json`` configuration files, mirroring the pattern
from ai-session-tracker's install command.

Test Categories:
    - ``_strip_jsonc_comments``: JSONC comment removal and cleanup
    - ``_generate_mcp_template``: Template generation and validity
    - ``run_install``: Fresh install, merge, already-installed, corrupt
    - ``main``: CLI dispatch to install vs server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from telescope_mcp.cli import (
    _detect_python_path,
    _generate_mcp_template,
    _get_global_vscode_dir,
    _strip_jsonc_comments,
    main,
    run_install,
)

# =========================================================================
# _strip_jsonc_comments
# =========================================================================


class TestStripJsoncComments:
    """Tests for JSONC comment stripping.

    Verifies that // comments are removed and trailing commas
    are cleaned up to produce valid JSON.
    """

    def test_no_comments(self) -> None:
        """Plain JSON passes through unchanged."""
        text = '{"key": "value"}'
        result = _strip_jsonc_comments(text)
        assert json.loads(result) == {"key": "value"}

    def test_trailing_comment(self) -> None:
        """Trailing // comment on a line is removed."""
        text = '{"key": "value"} // this is a comment'
        result = _strip_jsonc_comments(text)
        assert json.loads(result) == {"key": "value"}

    def test_full_line_comment(self) -> None:
        """Full-line // comment becomes empty."""
        text = '{\n  // comment\n  "key": "value"\n}'
        result = _strip_jsonc_comments(text)
        assert json.loads(result) == {"key": "value"}

    def test_trailing_comma_before_bracket(self) -> None:
        """Trailing comma before ] is removed.

        This happens when commented-out args follow the last
        real arg in an array.
        """
        text = '["a", "b",\n  // commented\n]'
        result = _strip_jsonc_comments(text)
        assert json.loads(result) == ["a", "b"]

    def test_trailing_comma_before_brace(self) -> None:
        """Trailing comma before } is removed."""
        text = '{"a": 1, "b": 2,\n  // commented\n}'
        result = _strip_jsonc_comments(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_multiple_comments(self) -> None:
        """Multiple comment lines are all removed."""
        text = (
            "{\n"
            '  "servers": {\n'
            "    // server 1\n"
            '    "a": {},\n'
            "    // server 2\n"
            '    "b": {}\n'
            "  }\n"
            "}"
        )
        result = _strip_jsonc_comments(text)
        parsed = json.loads(result)
        assert "a" in parsed["servers"]
        assert "b" in parsed["servers"]

    def test_realistic_mcp_json(self) -> None:
        """Realistic mcp.json with mixed comments parses correctly.

        Simulates the actual telescope-mcp mcp.json structure with
        commented-out args following active args.
        """
        text = """\
{
  "servers": {
    "telescope-mcp": {
      "command": "/usr/bin/python3",
      "args": [
        "-m",
        "telescope_mcp.server",
        "--dashboard-host", "127.0.0.1",
        "--dashboard-port", "8080",
        "--mode", "digital_twin",
        // "--latitude", "42.0",
        // "--longitude", "-89.0",
        // "--height", "291.0",
        // Per-camera defaults
        // "--finder-exposure-us", "10000000",  // 10s
        // "--finder-gain", "80"  // range: 0-510
      ]
    }
  }
}"""
        result = _strip_jsonc_comments(text)
        parsed = json.loads(result)
        server = parsed["servers"]["telescope-mcp"]
        assert server["command"] == "/usr/bin/python3"
        assert "--mode" in server["args"]
        assert "digital_twin" in server["args"]


# =========================================================================
# _generate_mcp_template
# =========================================================================


class TestGenerateMcpTemplate:
    """Tests for JSONC template generation.

    Verifies the template contains expected content and is
    parseable as JSONC (after comment stripping).
    """

    def test_contains_python_path(self) -> None:
        """Template includes the specified Python path."""
        result = _generate_mcp_template("/usr/bin/python3")
        assert '"/usr/bin/python3"' in result

    def test_contains_server_module(self) -> None:
        """Template includes the server module name."""
        result = _generate_mcp_template("/usr/bin/python3")
        assert "telescope_mcp.server" in result

    def test_contains_all_options(self) -> None:
        """Template documents all configurable options.

        Every CLI arg from server.py should appear (commented or not)
        so users can discover and enable them.
        """
        result = _generate_mcp_template("/usr/bin/python3")
        assert "--dashboard-host" in result
        assert "--dashboard-port" in result
        assert "--mode" in result
        assert "--dashboard-log-level" in result
        assert "--data-dir" in result
        assert "--latitude" in result
        assert "--longitude" in result
        assert "--height" in result
        assert "--finder-exposure-us" in result
        assert "--finder-gain" in result
        assert "--main-exposure-us" in result
        assert "--main-gain" in result

    def test_valid_jsonc(self) -> None:
        """Template is valid JSONC (parseable after comment removal)."""
        result = _generate_mcp_template("/usr/bin/python3")
        clean = _strip_jsonc_comments(result)
        parsed = json.loads(clean)
        assert "servers" in parsed
        assert "telescope-mcp" in parsed["servers"]

    def test_parsed_structure(self) -> None:
        """Parsed template has correct server structure."""
        result = _generate_mcp_template("/test/python")
        clean = _strip_jsonc_comments(result)
        parsed = json.loads(clean)
        server = parsed["servers"]["telescope-mcp"]
        assert server["command"] == "/test/python"
        assert server["args"][0] == "-m"
        assert server["args"][1] == "telescope_mcp.server"


# =========================================================================
# _detect_python_path
# =========================================================================


class TestDetectPythonPath:
    """Tests for Python executable detection."""

    def test_returns_sys_executable(self) -> None:
        """Returns the current Python interpreter path."""
        result = _detect_python_path()
        assert result == sys.executable

    def test_returns_absolute_path(self) -> None:
        """Path is absolute, not relative."""
        result = _detect_python_path()
        assert Path(result).is_absolute()


# =========================================================================
# _get_global_vscode_dir
# =========================================================================


class TestGetGlobalVscodeDir:
    """Tests for VS Code settings directory detection."""

    def test_returns_path(self) -> None:
        """Returns a Path object."""
        result = _get_global_vscode_dir()
        assert isinstance(result, Path)

    def test_ends_with_user(self) -> None:
        """Path ends with 'User' directory."""
        result = _get_global_vscode_dir()
        assert result.name == "User"

    def test_contains_code_in_path(self) -> None:
        """Path includes 'Code' or 'Code - Insiders'."""
        result = _get_global_vscode_dir()
        path_str = str(result)
        assert "Code" in path_str


# =========================================================================
# run_install — Fresh install
# =========================================================================


class TestRunInstallFresh:
    """Tests for fresh install (no existing mcp.json)."""

    def test_creates_vscode_dir(self, tmp_path: Path) -> None:
        """Creates .vscode/ directory if it doesn't exist."""
        run_install(cwd=str(tmp_path))
        assert (tmp_path / ".vscode").is_dir()

    def test_creates_mcp_json(self, tmp_path: Path) -> None:
        """Creates mcp.json in .vscode/ directory."""
        run_install(cwd=str(tmp_path))
        config_path = tmp_path / ".vscode" / "mcp.json"
        assert config_path.exists()

    def test_mcp_json_is_valid_jsonc(self, tmp_path: Path) -> None:
        """Generated mcp.json is valid JSONC."""
        run_install(cwd=str(tmp_path))
        config_path = tmp_path / ".vscode" / "mcp.json"
        content = config_path.read_text()
        clean = _strip_jsonc_comments(content)
        parsed = json.loads(clean)
        assert "servers" in parsed
        assert "telescope-mcp" in parsed["servers"]

    def test_mcp_json_has_python_path(self, tmp_path: Path) -> None:
        """Generated config includes current Python path."""
        run_install(cwd=str(tmp_path))
        config_path = tmp_path / ".vscode" / "mcp.json"
        content = config_path.read_text()
        assert sys.executable in content

    def test_mcp_json_has_comments(self, tmp_path: Path) -> None:
        """Generated config includes JSONC comments for options."""
        run_install(cwd=str(tmp_path))
        config_path = tmp_path / ".vscode" / "mcp.json"
        content = config_path.read_text()
        assert "//" in content


# =========================================================================
# run_install — Existing config merge
# =========================================================================


class TestRunInstallMerge:
    """Tests for merging into existing mcp.json."""

    def _write_existing_config(self, tmp_path: Path, config: dict[str, Any]) -> Path:
        """Helper to write a pre-existing mcp.json.

        Args:
            tmp_path: Temporary directory for test.
            config: Config dict to write as JSON.

        Returns:
            Path to the created mcp.json.
        """
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        config_path = vscode_dir / "mcp.json"
        config_path.write_text(json.dumps(config, indent=2))
        return config_path

    def test_backs_up_existing(self, tmp_path: Path) -> None:
        """Creates .json.bak backup of existing config."""
        self._write_existing_config(
            tmp_path,
            {"servers": {"other-server": {"command": "echo"}}},
        )
        run_install(cwd=str(tmp_path))
        backup = tmp_path / ".vscode" / "mcp.json.bak"
        assert backup.exists()

    def test_preserves_other_servers(self, tmp_path: Path) -> None:
        """Other servers in existing config are preserved."""
        self._write_existing_config(
            tmp_path,
            {
                "servers": {
                    "other-server": {
                        "command": "echo",
                        "args": ["hello"],
                    }
                }
            },
        )
        run_install(cwd=str(tmp_path))
        config_path = tmp_path / ".vscode" / "mcp.json"
        parsed = json.loads(config_path.read_text())
        assert "other-server" in parsed["servers"]
        assert "telescope-mcp" in parsed["servers"]

    def test_adds_telescope_mcp(self, tmp_path: Path) -> None:
        """Adds telescope-mcp server to existing config."""
        self._write_existing_config(
            tmp_path,
            {"servers": {"other": {"command": "echo"}}},
        )
        run_install(cwd=str(tmp_path))
        config_path = tmp_path / ".vscode" / "mcp.json"
        parsed = json.loads(config_path.read_text())
        server = parsed["servers"]["telescope-mcp"]
        assert server["command"] == sys.executable
        assert "-m" in server["args"]
        assert "telescope_mcp.server" in server["args"]


# =========================================================================
# run_install — Already installed
# =========================================================================


class TestRunInstallAlreadyInstalled:
    """Tests when telescope-mcp is already in config."""

    def test_no_modification(self, tmp_path: Path) -> None:
        """Does not modify config when already installed."""
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        config_path = vscode_dir / "mcp.json"
        original = {
            "servers": {
                "telescope-mcp": {
                    "command": "/old/python",
                    "args": ["-m", "telescope_mcp.server"],
                }
            }
        }
        config_path.write_text(json.dumps(original, indent=2))
        original_text = config_path.read_text()

        run_install(cwd=str(tmp_path))

        # Config should not be changed (but backup is created)
        # The original content should remain since we return early
        # after "already configured" message
        backup = tmp_path / ".vscode" / "mcp.json.bak"
        assert backup.exists()
        # Original config file still has the old python path
        current = json.loads(config_path.read_text())
        assert current["servers"]["telescope-mcp"]["command"] == "/old/python"


# =========================================================================
# run_install — Corrupt JSON
# =========================================================================


class TestRunInstallCorruptJson:
    """Tests for handling corrupt/unparseable mcp.json."""

    def test_writes_fresh_on_corrupt(self, tmp_path: Path) -> None:
        """Writes fresh template when existing config is corrupt."""
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        config_path = vscode_dir / "mcp.json"
        config_path.write_text("{{{invalid json!!!")

        run_install(cwd=str(tmp_path))

        # Should have backed up the corrupt file
        backup = tmp_path / ".vscode" / "mcp.json.bak"
        assert backup.exists()
        assert "{{{invalid" in backup.read_text()

        # Should have written a fresh template
        content = config_path.read_text()
        clean = _strip_jsonc_comments(content)
        parsed = json.loads(clean)
        assert "telescope-mcp" in parsed["servers"]


# =========================================================================
# run_install — JSONC existing config
# =========================================================================


class TestRunInstallJsoncExisting:
    """Tests for merging when existing config has JSONC comments."""

    def test_parses_jsonc_existing(self, tmp_path: Path) -> None:
        """Successfully parses existing config with // comments."""
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        config_path = vscode_dir / "mcp.json"
        config_path.write_text("""\
{
  "servers": {
    "other-server": {
      "command": "/usr/bin/echo",
      "args": [
        "hello",
        // "--verbose",
        "--output", "test"
      ]
    }
  }
}""")

        run_install(cwd=str(tmp_path))

        merged = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        assert "other-server" in merged["servers"]
        assert "telescope-mcp" in merged["servers"]


# =========================================================================
# run_install — Global install
# =========================================================================


class TestRunInstallGlobal:
    """Tests for --global flag (user-level VS Code settings)."""

    def test_global_uses_vscode_dir(self, tmp_path: Path) -> None:
        """Global install targets VS Code user settings directory.

        Mocks _get_global_vscode_dir to use tmp_path for testing.
        """
        global_dir = tmp_path / "Code" / "User"

        with patch(
            "telescope_mcp.cli._get_global_vscode_dir",
            return_value=global_dir,
        ):
            run_install(global_install=True)

        config_path = global_dir / "mcp.json"
        assert config_path.exists()
        content = config_path.read_text()
        clean = _strip_jsonc_comments(content)
        parsed = json.loads(clean)
        assert "telescope-mcp" in parsed["servers"]


# =========================================================================
# main() — CLI dispatch
# =========================================================================


class TestMainDispatch:
    """Tests for CLI entry point dispatch logic."""

    def test_install_subcommand(self, tmp_path: Path) -> None:
        """'install' subcommand calls run_install."""
        with patch("telescope_mcp.cli.run_install") as mock_install:
            with patch("sys.argv", ["telescope-mcp", "install"]):
                result = main()
            mock_install.assert_called_once_with(
                global_install=False,
            )
            assert result == 0

    def test_install_global_flag(self) -> None:
        """'install --global' passes global_install=True."""
        with patch("telescope_mcp.cli.run_install") as mock_install:
            with patch(
                "sys.argv",
                ["telescope-mcp", "install", "--global"],
            ):
                result = main()
            mock_install.assert_called_once_with(
                global_install=True,
            )
            assert result == 0

    def test_no_args_delegates_to_server(self) -> None:
        """No subcommand delegates to server.main().

        When no subcommand is given, telescope-mcp should run the
        MCP server, matching the previous behavior of the
        telescope-mcp entry point.
        """
        with patch("telescope_mcp.server.main") as mock_server:
            with patch("sys.argv", ["telescope-mcp"]):
                main()
            mock_server.assert_called_once()

    def test_server_subcommand_delegates(self) -> None:
        """'server' subcommand delegates to server.main().

        The 'server' subcommand should be stripped from sys.argv
        before delegating so server.parse_args() can parse its
        own args correctly.
        """
        with patch("telescope_mcp.server.main") as mock_server:
            with patch(
                "sys.argv",
                ["telescope-mcp", "server", "--mode", "hardware"],
            ):
                main()
                mock_server.assert_called_once()
                # Verify "server" was stripped from sys.argv
                assert sys.argv == [
                    "telescope-mcp",
                    "--mode",
                    "hardware",
                ]

    def test_server_subcommand_strips_server_arg(self) -> None:
        """'server' is removed from sys.argv before delegation.

        server.main() calls parse_args() which reads sys.argv.
        The 'server' subcommand must be stripped so --dashboard-host
        etc. are parsed correctly by server's argparser.
        """
        with patch("telescope_mcp.server.main") as mock_server:
            with patch(
                "sys.argv",
                [
                    "telescope-mcp",
                    "server",
                    "--dashboard-host",
                    "0.0.0.0",
                ],
            ):
                main()
                mock_server.assert_called_once()
                assert sys.argv == [
                    "telescope-mcp",
                    "--dashboard-host",
                    "0.0.0.0",
                ]
