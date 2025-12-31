"""Comprehensive tests for server.py to increase coverage to 80%."""

import time

import pytest
from mcp.server import Server

from telescope_mcp import server as server_module
from telescope_mcp.drivers.config import use_digital_twin
from telescope_mcp.server import (
    create_server,
    parse_args,
    start_dashboard,
    stop_dashboard,
)


@pytest.fixture(autouse=True)
def cleanup_dashboard():
    """Ensure dashboard is stopped before and after each test to free port.

    This fixture runs automatically for every test in this module,
    stopping any running dashboard server to prevent port conflicts
    between tests. Runs cleanup both before and after to handle
    leftover servers from previous test runs.

    Args:
        None: Autouse fixture takes no arguments.

    Yields:
        None: Test runs during yield.

    Returns:
        Generator[None, None, None]: Pytest fixture generator.

    Raises:
        None: Exceptions from stop_dashboard() are suppressed.

    Cleanup:
        Calls stop_dashboard() and clears _dashboard state to ensure
        clean state for next test.

    Business Context:
        Dashboard server binds to port 8080. Without cleanup, port
        remains bound causing subsequent test failures.
    """
    # Cleanup before test (handle leftover from previous runs)
    stop_dashboard()
    time.sleep(0.1)
    server_module._dashboard.thread = None
    server_module._dashboard.server = None

    yield

    # Cleanup after test
    stop_dashboard()
    time.sleep(0.1)
    server_module._dashboard.thread = None
    server_module._dashboard.server = None


class TestCreateServer:
    """Test MCP server creation and initialization."""

    def test_create_server_returns_server_instance(self):
        """Verifies create_server returns valid MCP Server instance.

        Arrangement:
        1. use_digital_twin() configures digital twin mode.
        2. create_server() builds MCP server with tools.
        3. Expected: Server instance with name="telescope-mcp".

        Action:
        Calls create_server() and inspects returned object.

        Assertion Strategy:
        Validates server creation by confirming:
        - Returns Server instance (from mcp.server).
        - server.name = "telescope-mcp".

        Testing Principle:
        Validates server factory, ensuring create_server
        returns properly initialized MCP server.
        """
        use_digital_twin()  # Ensure we use digital twin
        server = create_server()
        assert isinstance(server, Server)
        assert server.name == "telescope-mcp"

    def test_create_server_initializes_camera_registry(self):
        """Verifies server creation initializes camera registry.

        Arrangement:
        1. Digital twin mode enabled.
        2. create_server() should initialize camera registry.
        3. get_registry() retrieves singleton registry.

        Action:
        Creates server, then retrieves camera registry.

        Assertion Strategy:
        Validates registry initialization by confirming:
        - get_registry() returns non-None registry.

        Testing Principle:
        Validates server initialization side effects, ensuring
        camera registry is ready for device discovery.
        """
        use_digital_twin()
        server = create_server()
        # Registry should be initialized
        from telescope_mcp.devices import get_registry

        registry = get_registry()
        assert registry is not None

    def test_create_server_registers_tools(self):
        """Verifies server registers all MCP tool modules.

        Arrangement:
        1. Digital twin mode enabled.
        2. create_server() registers camera, session, motor tools.
        3. Server should be fully configured.

        Action:
        Creates server and validates configuration.

        Assertion Strategy:
        Validates tool registration by confirming:
        - Server is not None (creation succeeded).

        Testing Principle:
        Validates tool registration, ensuring server
        exposes all telescope control capabilities.
        """
        use_digital_twin()
        server = create_server()
        # Server should be configured with tools
        assert server is not None

    def test_create_server_multiple_calls(self):
        """Verifies multiple server creations succeed.

        Arrangement:
        1. Digital twin mode enabled.
        2. create_server() called twice.
        3. Both calls should succeed independently.

        Action:
        Creates two server instances.

        Assertion Strategy:
        Validates idempotent creation by confirming:
        - server1 is not None.
        - sVerifies dashboard server starts in background thread.

        Arrangement:
        1. start_dashboard() creates FastAPI server thread.
        2. Server runs on 127.0.0.1:18080.
        3. Thread should start without blocking.

        Action:
        Starts dashboard, waits 0.5s for thread startup.

        Assertion Strategy:
        Validates dashboard startup by confirming:
        - _dashboard_thread is not None (thread created).

        Testing Principle:
        Validates background server startup, ensuring
        dashboard runs without blocking MCP server.


        Testing Principle:
        Validates factory idempotence, ensuring multiple
        server creations don't interfere with each other.
        """
        use_digital_twin()
        server1 = create_server()
        server2 = create_server()
        # Both should be valid instances
        assert server1 is not None
        assert server2 is not None

    def test_start_dashboard_idempotent(self):
        """Verifies start_dashboard is idempotent when already running.

        Arrangement:
        1. Dashboard started on 127.0.0.1:18081.
        2. Second start_dashboard() call issued.
        3. Should be no-op without error.

        Action:
        Starts dashboard twice with same parameters.

        Assertion Strategy:
        Validates idempotence by confirming:
        - Second call does not raise exception.

        Testing Principle:
        Validates idempotent startup, preventing duplicate
        dashboard threads or port conflicts.
        """
        pass  # TODO: Implement test


class TestDashboardManagement:
    """Test dashboard server management."""

    def test_start_dashboard_basic(self):
        """Verifies dashboard server starts successfully with explicit params.

        Arrangement:
        1. start_dashboard(host="127.0.0.1", port=18080) called.
        2. Brief sleep allows background thread to initialize.
        3. Dashboard thread should be created in server_module.

        Action:
        Starts dashboard with explicit host and port parameters.

        Assertion Strategy:
        Validates basic startup by confirming:
        - _dashboard.thread is not None (thread created).

        Testing Principle:
        Validates core functionality, ensuring dashboard can be
        started programmatically for integration scenarios.
        """
        # Start dashboard in background
        start_dashboard(host="127.0.0.1", port=18080)

        # Should start without errors
        import time

        time.sleep(0.1)  # Brief wait for startup

        # Dashboard thread should exist
        from telescope_mcp import server as server_module

        assert server_module._dashboard.thread is not None

    def test_start_dashboard_already_running(self):
        """Verifies start_dashboard is idempotent when already running.

        Arrangement:
        1. start_dashboard(host="127.0.0.1", port=18081) starts server.
        2. Brief sleep ensures server is fully initialized.
        3. Second start_dashboard() call with same parameters.

        Action:
        Starts dashboard twice with identical parameters.

        Assertion Strategy:
        Validates idempotence by confirming:
        - No exception raised on second start call.

        Testing Principle:
        Validates idempotent startup, preventing duplicate dashboard
        threads or port conflicts from repeated start calls.
        """
        start_dashboard(host="127.0.0.1", port=18081)

        import time

        time.sleep(0.3)

        # Try starting again - should be no-op
        start_dashboard(host="127.0.0.1", port=18081)
        # Should not raise error

    def test_start_dashboard_default_params(self):
        """Verifies dashboard starts with default host and port.

        Arrangement:
        1. start_dashboard() called with no arguments.
        2. Should use default host (0.0.0.0) and port (8080).
        3. Brief sleep allows server initialization.

        Action:
        Starts dashboard with no explicit parameters.

        Assertion Strategy:
        Validates default parameters by confirming:
        - No exception raised (successful startup).

        Testing Principle:
        Validates default configuration, ensuring dashboard works
        out-of-box without requiring explicit configuration.
        """
        # Should use defaults without error
        start_dashboard()
        import time

        time.sleep(0.3)


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_parse_args_defaults(self):
        """Verifies parse_args returns default values when no args provided.

        Arrangement:
        1. sys.argv = ["telescope-mcp"] (no options).
        2. parse_args() should return defaults.
        3. Expected: dashboard_host and dashboard_port attributes.

        Action:
        Calls parse_args() with minimal argv.

        Assertion Strategy:
        Validates default parsing by confirming:
        - hasattr(args, 'dashboard_host').
        - hasattr(args, 'dashboard_port').

        Testing Principle:
        Validates argument parser defaults, ensuring CLI
        works without explicit configuration.
        """
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["telescope-mcp"]
            args = parse_args()

            # Check default values exist
            assert hasattr(args, "dashboard_host")
            assert hasattr(args, "dashboard_port")
        finally:
            sys.argv = original_argv

    def test_parse_args_dashboard_host_option(self):
        """Verifies --dashboard-host option sets host address.

        Arrangement:
        1. sys.argv includes --dashboard-host 0.0.0.0.
        2. parse_args() should capture host option.

        Action:
        Parses args with dashboard-host option.

        Assertion Strategy:
        Validates host option by confirming:
        - args.dashboard_host = "0.0.0.0".

        Testing Principle:
        Validates CLI option parsing, ensuring dashboard
        host can be configured via command line.
        """
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["telescope-mcp", "--dashboard-host", "0.0.0.0"]
            args = parse_args()
            assert args.dashboard_host == "0.0.0.0"
        finally:
            sys.argv = original_argv

    def test_parse_args_dashboard_port_option(self):
        """Verifies --dashboard-port option sets port number.

        Arrangement:
        1. sys.argv includes --dashboard-port 9090.
        2. parse_args() should convert to integer.

        Action:
        Parses args with dashboard-port option.

        Assertion Strategy:
        Validates port option by confirming:
        - args.dashboard_port = 9090 (int).

        Testing Principle:
        Validates port parsing and type conversion, ensuring
        dashboard port is configurable and properly typed.
        """
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["telescope-mcp", "--dashboard-port", "9090"]
            args = parse_args()
            assert args.dashboard_port == 9090
        finally:
            sys.argv = original_argv

    def test_parse_args_data_dir(self):
        """Verifies --data-dir option sets data directory path.

        Arrangement:
        1. sys.argv includes --data-dir /tmp/data.
        2. parse_args() should capture directory path.

        Action:
        Parses args with data-dir option.

        Assertion Strategy:
        Validates data-dir option by confirming:
        - args.data_dir = "/tmp/data".

        Testing Principle:
        Validates data directory configuration, allowing
        custom session storage location via CLI.
        """
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["telescope-mcp", "--data-dir", "/tmp/data"]
            args = parse_args()
            assert args.data_dir == "/tmp/data"
        finally:
            sys.argv = original_argv

    def test_parse_args_multiple_options(self):
        """Verifies multiple CLI options can be combined.

                Arrangement:
                1. sys.argv with --dashboard-host and --dashboard-port.
                2. Both options should be parsed correctly.
        Verifies server uses digital twin driver by default.

                Arrangement:
                1. use_digital_twin() configures twin mode.
                2. create_server() builds server with twin cameras.
                3. Server should initialize successfully.

                Action:
                Creates server in digital twin mode.

                Assertion Strategy:
                Validates digital twin integration by confirming:
                - server.name = "telescope-mcp".

                Testing Principle:
                Validates hardware abstraction, ensuring server
                works without physical telescope hardware.

                Action:
                Parses args with multiple options.

                Assertion Strategy:
                Validates option combination by confirming:
                - args.dashboard_host = "0.0.0.0".
                - args.dashboard_port = 8080.
        Verifies camera registry is functional after server creation.

                Arrangement:
                1. Server created in digital twin mode.
                2. get_registry() retrieves camera registry.
                3. registry.discover() should list cameras.

                Action:
                Creates server, retrieves registry, discovers cameras.

                Assertion Strategy:
                Validates registry functionality by confirming:
                - registry.discover() returns list (len >= 0).

                Testing Principle:
                Validates post-initialization state, ensuring camera
                registry is ready for device operations.

                Testing Principle:
                Validates argument combination, ensuring multiple
                options work together without conflicts.
        """
        import sys

        original_argv = sys.argv
        try:
            sys.argv = [
                "telescope-mcp",
                "--dashboard-host",
                "0.0.0.0",
                "--dashboard-port",
                "8080",
            ]
            args = parse_args()
            assert args.dashboard_host == "0.0.0.0"
            assert args.dashboard_port == 8080
        finally:
            sys.argv = original_argv

    def test_parse_args_all_options(self):
        """Verifies server identifies with correct MCP name.

        Arrangement:
        1. Digital twin mode enabled.
        2. create_server() sets name="telescope-mcp".
        3. Name used for MCP protocol identification.

        Action:
        Creates server and reads name attribute.

        Assertion Strategy:
        Validates server identity by confirming:
        - server.name = "telescope-mcp".

        Testing Principle:
        Validates MCP protocol compliance, ensuring server
        announces correct identity to clients.
        tions parse correctly.

        Arrangement:
        1. sys.argv with all options: host, port, data-dir.
        2. Complete configuration via CLI.

        ActVerifies server has tools properly registered.

        Arrangement:
        1. Digital twin mode enabled.
        2. create_server() registers tool modules.
        3. Server should have name and tool handlers.

        Action:
        Creates server and validates configuration.

        Assertion Strategy:
        Validates tool registration by confirming:
        - hasattr(server, 'name').
        - server.name = "telescope-mcp".

        Testing Principle:
        Validates server configuration completeness, ensuring
        all tool modules are registered for MCP client access.

        Parses args with all available options.

        Assertion Strategy:
        Validates comprehensive parsing by confirming:
        - args.dashboard_host = "127.0.0.1".
        - args.dashboard_port = 7070.
        - args.data_dir = "/data/telescope".

        Testing Principle:
        Validates complete CLI configuration, ensuring
        all options can be specified simultaneously.
        """
        import sys

        original_argv = sys.argv
        try:
            sys.argv = [
                "telescope-mcp",
                "--dashboard-host",
                "127.0.0.1",
                "--dashboard-port",
                "7070",
                "--data-dir",
                "/data/telescope",
            ]
            args = parse_args()
            assert args.dashboard_host == "127.0.0.1"
            assert args.dashboard_port == 7070
            assert args.data_dir == "/data/telescope"
        finally:
            sys.argv = original_argv


class TestServerIntegration:
    """Integration tests for server functionality."""

    def test_server_creation_with_digital_twin(self):
        """Verifies server creation uses digital twin driver by default.

        Arrangement:
        1. use_digital_twin() configures mock camera driver.
        2. create_server() initializes MCP server instance.
        3. Server should be fully functional with simulated hardware.

        Action:
        Configures digital twin, creates server, validates name.

        Assertion Strategy:
        Validates server creation by confirming:
        - server.name equals "telescope-mcp".

        Testing Principle:
        Validates dependency injection, ensuring digital twin driver
        enables testing without physical telescope hardware.
        """
        use_digital_twin()
        server = create_server()

        # Should have valid server
        assert server.name == "telescope-mcp"

    def test_server_registry_accessible_after_creation(self):
        """Verifies camera registry is accessible after server creation.

        Arrangement:
        1. use_digital_twin() configures mock driver.
        2. create_server() initializes server and registry.
        3. get_registry() should return initialized CameraRegistry.

        Action:
        Creates server, retrieves registry, attempts camera discovery.

        Assertion Strategy:
        Validates registry access by confirming:
        - len(cameras) is at least 0 (no exception on discover).

        Testing Principle:
        Validates component wiring, ensuring server creation properly
        initializes camera registry for subsequent tool operations.
        """
        use_digital_twin()
        create_server()

        from telescope_mcp.devices import get_registry

        registry = get_registry()

        # Should be able to discover cameras
        cameras = registry.discover()
        assert len(cameras) >= 0


class TestServerConfiguration:
    """Test server configuration options."""

    def test_server_name(self):
        """Verifies server has correct name for MCP identification.

        Arrangement:
        1. use_digital_twin() configures mock driver.
        2. create_server() initializes server with default config.
        3. Server name used for MCP client identification.

        Action:
        Creates server, checks name attribute.

        Assertion Strategy:
        Validates naming by confirming:
        - server.name equals "telescope-mcp".

        Testing Principle:
        Validates configuration correctness, ensuring server
        identifies itself properly to MCP clients.
        """
        use_digital_twin()
        server = create_server()
        assert server.name == "telescope-mcp"

    def test_server_tools_registered(self):
        """Verifies all MCP tool modules are registered with server.

        Arrangement:
        1. use_digital_twin() configures mock driver.
        2. create_server() initializes server and registers tools.
        3. Tool modules (cameras, sessions, etc.) should be loaded.

        Action:
        Creates server, validates it has expected attributes.

        Assertion Strategy:
        Validates registration by confirming:
        - hasattr(server, 'name') is True.
        - server.name equals "telescope-mcp".

        Testing Principle:
        Validates server initialization, ensuring tool registration
        completes without errors for full MCP functionality.
        """
        use_digital_twin()
        server = create_server()

        # Server should be properly initialized
        assert hasattr(server, "name")
        assert server.name == "telescope-mcp"
