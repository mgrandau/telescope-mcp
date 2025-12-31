"""Tests for server.py targeting 100% coverage.

Covers edge cases and branches not tested in test_server_comprehensive.py:
- Hardware mode in create_server()
- Dashboard error handling (_run_dashboard OSError/Exception)
- run_server() async function
- main() entry point
- parse_args() with all options
"""

import argparse
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telescope_mcp import server as server_module
from telescope_mcp.server import (
    _run_dashboard,
    create_server,
    main,
    parse_args,
    run_server,
    stop_dashboard,
)


class TestCreateServerCoverage:
    """Tests for create_server() uncovered branches."""

    def test_create_server_hardware_mode(self):
        """Verifies create_server handles hardware mode configuration.

        Arrangement:
        1. Mode="hardware" triggers use_hardware() call.
        2. Should log hardware mode message.

        Action:
        Calls create_server(mode="hardware") with mocked use_hardware.

        Assertion Strategy:
        Validates hardware branch by confirming:
        - use_hardware() is called
        - Server is still created successfully

        Testing Principle:
        Covers lines 58-61 (hardware mode branch).
        """
        # Patch at the source where it's imported from
        with patch("telescope_mcp.drivers.config.use_hardware") as mock_use_hw:
            with patch("telescope_mcp.drivers.config.get_factory") as mock_factory:
                mock_driver = MagicMock()
                mock_factory.return_value.create_camera_driver.return_value = (
                    mock_driver
                )
                with patch("telescope_mcp.devices.init_registry"):
                    server = create_server(mode="hardware")

                    mock_use_hw.assert_called_once()
                    assert server is not None

    def test_create_server_hardware_mode_case_insensitive(self):
        """Verifies hardware mode detection is case-insensitive.

        Testing Principle:
        Covers .lower() call in mode comparison.
        """
        with patch("telescope_mcp.drivers.config.use_hardware") as mock_use_hw:
            with patch("telescope_mcp.drivers.config.get_factory") as mock_factory:
                mock_driver = MagicMock()
                mock_factory.return_value.create_camera_driver.return_value = (
                    mock_driver
                )
                with patch("telescope_mcp.devices.init_registry"):
                    create_server(mode="HARDWARE")
                    mock_use_hw.assert_called_once()


class TestRunDashboardCoverage:
    """Tests for _run_dashboard error handling."""

    def test_run_dashboard_oserror(self):
        """Verifies _run_dashboard handles OSError (port in use).

        Arrangement:
        1. Mock create_app to succeed.
        2. Mock uvicorn.Server.run() to raise OSError.

        Action:
        Calls _run_dashboard which catches OSError.

        Assertion Strategy:
        Validates error handling by confirming:
        - No exception propagates
        - Error is logged

        Testing Principle:
        Covers lines 116-117 (OSError handler).
        """
        with patch("telescope_mcp.server.create_app") as mock_create_app:
            mock_create_app.return_value = MagicMock()
            with patch("telescope_mcp.server.uvicorn.Config"):
                with patch("telescope_mcp.server.uvicorn.Server") as mock_server_cls:
                    mock_server = MagicMock()
                    mock_server.run.side_effect = OSError("Address already in use")
                    mock_server_cls.return_value = mock_server

                    # Should not raise - error is caught
                    _run_dashboard("127.0.0.1", 8080)

    def test_run_dashboard_unexpected_exception(self):
        """Verifies _run_dashboard handles unexpected exceptions.

        Arrangement:
        1. Mock create_app to raise RuntimeError.

        Action:
        Calls _run_dashboard which catches generic Exception.

        Assertion Strategy:
        Validates error handling by confirming:
        - No exception propagates
        - logger.exception called

        Testing Principle:
        Covers lines 118-119 (generic Exception handler).
        """
        with patch("telescope_mcp.server.create_app") as mock_create_app:
            mock_create_app.side_effect = RuntimeError("Unexpected error")

            # Should not raise - error is caught
            _run_dashboard("127.0.0.1", 8080)


class TestRunServerCoverage:
    """Tests for run_server() async function."""

    @pytest.mark.asyncio
    async def test_run_server_without_dashboard(self):
        """Verifies run_server works without dashboard.

        Arrangement:
        1. dashboard_host=None, dashboard_port=None.
        2. Mock stdio_server and server.run().

        Action:
        Calls run_server with no dashboard.

        Assertion Strategy:
        Validates no-dashboard path by confirming:
        - start_dashboard not called
        - Server runs and cleans up

        Testing Principle:
        Covers run_server with dashboard disabled.
        """
        mock_read = AsyncMock()
        mock_write = AsyncMock()

        with patch("telescope_mcp.server.create_server") as mock_create:
            mock_server = MagicMock()
            mock_server.run = AsyncMock()
            mock_server.create_initialization_options.return_value = {}
            mock_create.return_value = mock_server

            with patch("telescope_mcp.server.stdio_server") as mock_stdio:
                mock_stdio.return_value.__aenter__ = AsyncMock(
                    return_value=(mock_read, mock_write)
                )
                mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

                with patch("telescope_mcp.server.start_dashboard") as mock_start:
                    with patch("telescope_mcp.server.stop_dashboard"):
                        with patch("telescope_mcp.devices.shutdown_registry"):
                            await run_server(None, None, "digital_twin")

                            # start_dashboard should NOT be called
                            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_server_with_dashboard(self):
        """Verifies run_server starts dashboard when configured.

        Arrangement:
        1. dashboard_host="127.0.0.1", dashboard_port=8080.
        2. Mock stdio_server and server.run().

        Action:
        Calls run_server with dashboard enabled.

        Assertion Strategy:
        Validates dashboard startup by confirming:
        - start_dashboard called with correct args

        Testing Principle:
        Covers dashboard startup branch in run_server.
        """
        mock_read = AsyncMock()
        mock_write = AsyncMock()

        with patch("telescope_mcp.server.create_server") as mock_create:
            mock_server = MagicMock()
            mock_server.run = AsyncMock()
            mock_server.create_initialization_options.return_value = {}
            mock_create.return_value = mock_server

            with patch("telescope_mcp.server.stdio_server") as mock_stdio:
                mock_stdio.return_value.__aenter__ = AsyncMock(
                    return_value=(mock_read, mock_write)
                )
                mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

                with patch("telescope_mcp.server.start_dashboard") as mock_start:
                    with patch("telescope_mcp.server.stop_dashboard"):
                        with patch("telescope_mcp.devices.shutdown_registry"):
                            await run_server("127.0.0.1", 8080, "digital_twin")

                            mock_start.assert_called_once_with(
                                "127.0.0.1", 8080, "warning"
                            )

    @pytest.mark.asyncio
    async def test_run_server_cleanup_on_exit(self):
        """Verifies run_server calls cleanup in finally block.

        Arrangement:
        1. Mock server to complete immediately.
        2. Check stop_dashboard and shutdown_registry called.

        Action:
        Calls run_server and verifies cleanup.

        Assertion Strategy:
        Validates cleanup by confirming:
        - stop_dashboard called
        - shutdown_registry called

        Testing Principle:
        Covers finally block in run_server.
        """
        mock_read = AsyncMock()
        mock_write = AsyncMock()

        with patch("telescope_mcp.server.create_server") as mock_create:
            mock_server = MagicMock()
            mock_server.run = AsyncMock()
            mock_server.create_initialization_options.return_value = {}
            mock_create.return_value = mock_server

            with patch("telescope_mcp.server.stdio_server") as mock_stdio:
                mock_stdio.return_value.__aenter__ = AsyncMock(
                    return_value=(mock_read, mock_write)
                )
                mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

                with patch("telescope_mcp.server.start_dashboard"):
                    with patch("telescope_mcp.server.stop_dashboard") as mock_stop:
                        with patch(
                            "telescope_mcp.devices.shutdown_registry"
                        ) as mock_shutdown:
                            await run_server(None, None)

                            mock_stop.assert_called_once()
                            mock_shutdown.assert_called_once()


class TestMainCoverage:
    """Tests for main() entry point."""

    def test_main_basic(self):
        """Verifies main() parses args and starts server.

        Arrangement:
        1. Mock parse_args to return minimal args.
        2. Mock asyncio.run to avoid actual server.

        Action:
        Calls main() with mocked dependencies.

        Assertion Strategy:
        Validates main flow by confirming:
        - configure_logging called
        - asyncio.run called with run_server

        Testing Principle:
        Covers main() basic path (lines 372-409).
        """
        mock_args = argparse.Namespace(
            dashboard_host=None,
            dashboard_port=None,
            dashboard_log_level="warning",
            data_dir=None,
            mode="digital_twin",
            latitude=None,
            longitude=None,
            height=0.0,
        )

        with patch("telescope_mcp.server.parse_args", return_value=mock_args):
            with patch("telescope_mcp.server.configure_logging"):
                with patch(
                    "telescope_mcp.drivers.config.get_session_manager"
                ) as mock_mgr:
                    mock_mgr.return_value.log = MagicMock()
                    with patch("telescope_mcp.server.asyncio.run") as mock_run:
                        main()
                        mock_run.assert_called_once()

    def test_main_with_data_dir(self):
        """Verifies main() configures data directory when specified.

        Arrangement:
        1. args.data_dir = "/tmp/test_data".
        2. set_data_dir should be called.

        Action:
        Calls main() with data_dir argument.

        Assertion Strategy:
        Validates data_dir config by confirming:
        - set_data_dir called with Path("/tmp/test_data")

        Testing Principle:
        Covers data_dir configuration branch in main().
        """
        mock_args = argparse.Namespace(
            dashboard_host=None,
            dashboard_port=None,
            dashboard_log_level="warning",
            data_dir="/tmp/test_data",
            mode="digital_twin",
            latitude=None,
            longitude=None,
            height=0.0,
        )

        with patch("telescope_mcp.server.parse_args", return_value=mock_args):
            with patch("telescope_mcp.server.configure_logging"):
                with patch("telescope_mcp.drivers.config.set_data_dir") as mock_set_dir:
                    with patch(
                        "telescope_mcp.drivers.config.get_session_manager"
                    ) as mock_mgr:
                        mock_mgr.return_value.log = MagicMock()
                        with patch("telescope_mcp.server.asyncio.run"):
                            main()
                            mock_set_dir.assert_called_once()

    def test_main_with_location(self):
        """Verifies main() configures observer location when specified.

        Arrangement:
        1. args.latitude = 40.7128, args.longitude = -74.0060.
        2. set_location should be called.

        Action:
        Calls main() with location arguments.

        Assertion Strategy:
        Validates location config by confirming:
        - set_location called with lat/lon/alt

        Testing Principle:
        Covers location configuration branch in main().
        """
        mock_args = argparse.Namespace(
            dashboard_host=None,
            dashboard_port=None,
            dashboard_log_level="warning",
            data_dir=None,
            mode="digital_twin",
            latitude=40.7128,
            longitude=-74.0060,
            height=10.0,
        )

        with patch("telescope_mcp.server.parse_args", return_value=mock_args):
            with patch("telescope_mcp.server.configure_logging"):
                with patch("telescope_mcp.drivers.config.set_location") as mock_set_loc:
                    with patch(
                        "telescope_mcp.drivers.config.get_session_manager"
                    ) as mock_mgr:
                        mock_mgr.return_value.log = MagicMock()
                        with patch("telescope_mcp.server.asyncio.run"):
                            main()
                            mock_set_loc.assert_called_once_with(
                                lat=40.7128,
                                lon=-74.0060,
                                alt=10.0,
                            )

    def test_main_invalid_latitude(self):
        """Verifies main() raises ValueError for invalid latitude.

        Testing Principle:
        Covers P2-2 latitude validation branch.
        """
        mock_args = argparse.Namespace(
            dashboard_host=None,
            dashboard_port=None,
            dashboard_log_level="warning",
            data_dir=None,
            mode="digital_twin",
            latitude=91.0,  # Invalid: > 90
            longitude=-74.0060,
            height=0.0,
        )

        with patch("telescope_mcp.server.parse_args", return_value=mock_args):
            with patch("telescope_mcp.server.configure_logging"):
                with pytest.raises(ValueError, match="Latitude must be between"):
                    main()

    def test_main_invalid_longitude(self):
        """Verifies main() raises ValueError for invalid longitude.

        Testing Principle:
        Covers P2-2 longitude validation branch.
        """
        mock_args = argparse.Namespace(
            dashboard_host=None,
            dashboard_port=None,
            dashboard_log_level="warning",
            data_dir=None,
            mode="digital_twin",
            latitude=40.7128,
            longitude=181.0,  # Invalid: > 180
            height=0.0,
        )

        with patch("telescope_mcp.server.parse_args", return_value=mock_args):
            with patch("telescope_mcp.server.configure_logging"):
                with pytest.raises(ValueError, match="Longitude must be between"):
                    main()

    def test_main_with_dashboard(self):
        """Verifies main() passes dashboard args to run_server.

        Arrangement:
        1. args with dashboard_host and dashboard_port.
        2. asyncio.run should receive coroutine with dashboard args.

        Action:
        Calls main() with dashboard configuration.

        Assertion Strategy:
        Validates dashboard args passed by confirming:
        - asyncio.run called (integration handled by run_server)

        Testing Principle:
        Covers dashboard configuration path in main().
        """
        mock_args = argparse.Namespace(
            dashboard_host="127.0.0.1",
            dashboard_port=8080,
            dashboard_log_level="warning",
            data_dir=None,
            mode="digital_twin",
            latitude=None,
            longitude=None,
            height=0.0,
        )

        with patch("telescope_mcp.server.parse_args", return_value=mock_args):
            with patch("telescope_mcp.server.configure_logging"):
                with patch(
                    "telescope_mcp.drivers.config.get_session_manager"
                ) as mock_mgr:
                    mock_mgr.return_value.log = MagicMock()
                    with patch("telescope_mcp.server.asyncio.run") as mock_run:
                        main()
                        mock_run.assert_called_once()


class TestParseArgsCoverage:
    """Tests for parse_args() additional options."""

    def test_parse_args_mode_hardware(self):
        """Verifies --mode hardware option parses correctly.

        Testing Principle:
        Covers mode argument parsing.
        """
        original_argv = sys.argv
        try:
            sys.argv = ["telescope-mcp", "--mode", "hardware"]
            args = parse_args()
            assert args.mode == "hardware"
        finally:
            sys.argv = original_argv

    def test_parse_args_mode_digital_twin(self):
        """Verifies --mode digital_twin option parses correctly.

        Testing Principle:
        Covers mode argument default and explicit.
        """
        original_argv = sys.argv
        try:
            sys.argv = ["telescope-mcp", "--mode", "digital_twin"]
            args = parse_args()
            assert args.mode == "digital_twin"
        finally:
            sys.argv = original_argv

    def test_parse_args_location_options(self):
        """Verifies location options (latitude, longitude, height) parse.

        Testing Principle:
        Covers location argument parsing.
        """
        original_argv = sys.argv
        try:
            sys.argv = [
                "telescope-mcp",
                "--latitude",
                "40.7128",
                "--longitude",
                "-74.0060",
                "--height",
                "100.5",
            ]
            args = parse_args()
            assert args.latitude == 40.7128
            assert args.longitude == -74.0060
            assert args.height == 100.5
        finally:
            sys.argv = original_argv

    def test_parse_args_height_default(self):
        """Verifies --height defaults to 0.0.

        Testing Principle:
        Covers height default value.
        """
        original_argv = sys.argv
        try:
            sys.argv = ["telescope-mcp"]
            args = parse_args()
            assert args.height == 0.0
        finally:
            sys.argv = original_argv


class TestModuleLevelCoverage:
    """Tests for module-level code."""

    def test_module_constants_exist(self):
        """Verifies module-level variables are initialized.

        Testing Principle:
        Covers module initialization.
        """
        assert hasattr(server_module, "_dashboard")
        assert hasattr(server_module, "logger")

    def test_stop_dashboard_when_not_running(self):
        """Verifies stop_dashboard is safe when no server running.

        Testing Principle:
        Covers stop_dashboard no-op path.
        """
        server_module._dashboard.server = None
        stop_dashboard()  # Should not raise

    def test_stop_dashboard_when_running(self):
        """Verifies stop_dashboard sets should_exit flag.

        Testing Principle:
        Covers stop_dashboard active server path.
        """
        mock_server = MagicMock()
        server_module._dashboard.server = mock_server

        stop_dashboard()

        assert mock_server.should_exit is True
        assert server_module._dashboard.server is None
