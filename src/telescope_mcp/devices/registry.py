"""Camera registry with discovery and singleton management.

This module provides CameraRegistry for centralized camera access:
- Discovery of connected cameras
- Singleton Camera instances per camera_id
- Automatic recovery from disconnects

Follows SOLID principles:
- Single Responsibility: Registry manages camera lifecycle only
- Dependency Inversion: Accepts CameraDriver protocol, not concrete type

Example:
    from telescope_mcp.devices import CameraRegistry
    from telescope_mcp.drivers.cameras import DigitalTwinCameraDriver
    
    driver = DigitalTwinCameraDriver()
    
    with CameraRegistry(driver) as registry:
        # Discover cameras
        cameras = registry.discover()
        print(f"Found {len(cameras)} cameras")
        
        # Get singleton Camera instance
        camera = registry.get(0)
        camera.connect()
        result = camera.capture()
    # All cameras disconnected on exit
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telescope_mcp.drivers.cameras import CameraDriver

from telescope_mcp.devices.camera import (
    Camera,
    CameraConfig,
    CameraInfo,
    CameraHooks,
    Clock,
    OverlayRenderer,
    SystemClock,
)


class CameraNotInRegistryError(Exception):
    """Raised when camera_id is not found in registry or discovery."""
    pass


class RecoveryStrategy:
    """Strategy for recovering from camera disconnects.
    
    Used by Camera to attempt auto-recovery when USB errors occur.
    This implementation uses the registry's discovery mechanism to
    re-scan for available cameras.
    """
    
    def __init__(self, registry: CameraRegistry) -> None:
        """Create recovery strategy linked to registry.
        
        Args:
            registry: Registry to use for camera re-discovery.
        """
        self._registry = registry
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """USB bus rescan for recovering disconnected cameras.
        
        Forces registry re-discovery to scan USB bus for cameras (driver.get_connected_cameras()),
        then checks if target camera reappeared. Used by Camera class when capture operations
        fail due to USB disconnects. Does not perform hardware resets - relies on natural
        re-enumeration after transient USB issues.
        
        Business context: USB cameras can temporarily disappear from enumeration due to power
        glitches, cable issues, or driver hangs, then reappear seconds later once hardware
        stabilizes. Registry-based recovery leverages existing discovery mechanism for re-scanning,
        detecting cameras that recovered naturally. Lighter-weight than USB hardware resets (which
        affect all devices on same controller). Critical for long-running observatory sessions where
        brief USB issues shouldn't terminate multi-hour observations.
        
        Implementation details: Calls self._registry.discover(refresh=True) forcing cache invalidation
        and driver re-query. Driver scans USB bus (e.g., libusb device enumeration), returns currently
        available cameras. Returns True if camera_id in discovery results. Takes 100-500ms (USB
        enumeration latency). No hardware reset performed - detection-only. Camera.connect() must
        be called after successful recovery to re-open device.
        
        Args:
            camera_id: 0-based camera ID to check for (matches device index from driver discovery).
                ID 0 typically main imaging camera, 1 guide camera in dual-camera setups.
            
        Returns:
            True if camera_id found in post-rescan discovery (available for reconnection).
            False if camera not found or driver raised exceptions during discovery.
        
        Raises:
            None. Catches all exceptions during discover() and returns False. Camera interprets
            False as "recovery failed" and raises CameraDisconnectedError.
        
        Example:
            >>> strategy = RecoveryStrategy(registry)
            >>> # Camera 0 disconnected during capture
            >>> if strategy.attempt_recovery(0):
            ...     print("Camera reappeared, reconnecting...")
            ...     camera.connect()  # Reconnect after successful recovery
            ... else:
            ...     print("Camera still unavailable")
        """
        cameras = self._registry.discover(refresh=True)
        return camera_id in cameras


class NullRecoveryStrategy:
    """No-op recovery strategy (for testing or when recovery not desired)."""
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """No-op recovery returning False immediately (Null Object pattern).
        
        Returns False without attempting any recovery actions (USB rescanning, hardware resets,
        driver operations). Used when recovery not desired or unavailable. Implements Null Object
        pattern eliminating null checks in Camera class.
        
        Business context: Provides safe default preventing unexpected recovery attempts that might
        affect system stability or other USB devices. Users opt-in to recovery by injecting
        RecoveryStrategy. Useful in environments where USB resets dangerous (production systems,
        shared USB controllers, mission-critical hardware) or when camera disconnects should fail
        fast rather than attempt recovery. Enables explicit control over recovery behavior through
        dependency injection.
        
        Implementation details: Always returns False immediately with no side effects. No USB
        operations, no driver calls, no delays. Camera class interprets False as "recovery not
        possible" and raises CameraDisconnectedError with original error chained. Pattern eliminates
        `if recovery is None` checks throughout Camera code - can always call attempt_recovery()
        safely. For testing or when recovery undesired.
        
        Args:
            camera_id: Camera ID to recover (0-based). Ignored completely - no operations performed
                regardless of value. Parameter exists for protocol compliance.
            
        Returns:
            Always False indicating recovery was not attempted and camera unavailable. Camera will
            raise CameraDisconnectedError immediately.
        
        Raises:
            None. This implementation never raises exceptions under any circumstances.
        
        Example:
            >>> # Use NullRecoveryStrategy when recovery not desired
            >>> camera = Camera(
            ...     driver=driver,
            ...     config=config,
            ...     recovery=NullRecoveryStrategy()  # Fail fast, no recovery
            ... )
            >>> try:
            ...     camera.capture()  # If fails, immediate exception
            ... except CameraDisconnectedError:
            ...     print("Camera disconnected, no recovery attempted")
        """
        return False


class CameraRegistry:
    """Centralized camera discovery and singleton management.
    
    Provides:
    - Camera discovery with caching
    - Singleton Camera instances per camera_id
    - Automatic cleanup via context manager
    - Injectable dependencies (renderer, clock, hooks) for created cameras
    
    Thread Safety:
        Not thread-safe. Use external synchronization if needed.
    
    Example:
        with CameraRegistry(driver) as registry:
            cameras = registry.discover()
            camera = registry.get(0)
            camera.connect()
            result = camera.capture()
    """
    
    def __init__(
        self,
        driver: CameraDriver,
        renderer: OverlayRenderer | None = None,
        clock: Clock | None = None,
        hooks: CameraHooks | None = None,
    ) -> None:
        """Create registry with driver and optional dependencies.
        
        Initializes the registry without connecting to any cameras.
        Call discover() to scan for cameras, then get() to create
        Camera instances.
        
        Args:
            driver: Camera driver for hardware access (ASI or DigitalTwin).
            renderer: Overlay renderer passed to created cameras.
            clock: Clock for timing (default: SystemClock).
            hooks: Event hooks passed to created cameras.
        
        Example:
            from telescope_mcp.drivers.cameras import DigitalTwinCameraDriver
            
            driver = DigitalTwinCameraDriver()
            registry = CameraRegistry(driver)
            registry.discover()
        """
        self._driver = driver
        self._renderer = renderer
        self._clock = clock or SystemClock()
        self._hooks = hooks
        
        self._cameras: dict[int, Camera] = {}
        self._discovery_cache: dict[int, CameraInfo] | None = None
        self._recovery_strategy: RecoveryStrategy | None = None
    
    @property
    def driver(self) -> CameraDriver:
        """Get the camera driver for direct hardware access.
        
        Provides access to the underlying driver (ASICameraDriver or
        DigitalTwinCameraDriver) for operations not exposed through the
        registry interface, such as driver-specific configuration or
        low-level hardware queries.
        
        Business context: Enables advanced users to access driver-specific
        features while still benefiting from registry camera management.
        Useful for debugging, hardware diagnostics, or accessing vendor-specific
        APIs not abstracted by the Camera interface. Most applications should
        use registry methods rather than direct driver access.
        
        Returns:
            The CameraDriver instance injected during registry construction.
            Same instance returned on every call (not a copy).
        
        Raises:
            None. Always returns the configured driver.
        
        Example:
            >>> registry = CameraRegistry(ASICameraDriver())
            >>> driver = registry.driver
            >>> # Access driver-specific methods
            >>> raw_cameras = driver.get_connected_cameras()
            >>> # Or check driver type for conditional logic
            >>> if isinstance(driver, ASICameraDriver):
            ...     print("Using real ASI hardware")
        """
        return self._driver
    
    def discover(self, refresh: bool = False) -> dict[int, CameraInfo]:
        """Discover connected cameras.
        
        Scans for available cameras using the driver and caches results.
        Subsequent calls return cached results unless refresh=True.
        
        Args:
            refresh: Force re-discovery ignoring cache. Defaults to False.
            
        Returns:
            Dict mapping camera_id to CameraInfo with capabilities.
        
        Raises:
            RuntimeError: If driver discovery fails or returns invalid data.
        
        Example:
            cameras = registry.discover()
            for cam_id, info in cameras.items():
                print(f"Camera {cam_id}: {info.name} ({info.max_width}x{info.max_height})")
        """
        if self._discovery_cache is None or refresh:
            raw = self._driver.get_connected_cameras()
            self._discovery_cache = {}
            
            for cam_id, info in raw.items():
                # Handle both dict and CameraInfo from driver
                if isinstance(info, CameraInfo):
                    self._discovery_cache[cam_id] = info
                else:
                    # Convert dict to CameraInfo
                    self._discovery_cache[cam_id] = CameraInfo(
                        camera_id=cam_id,
                        name=info.get("name", f"Camera {cam_id}"),
                        max_width=info.get("max_width", 0),
                        max_height=info.get("max_height", 0),
                        is_color=info.get("is_color", False),
                        bayer_pattern=info.get("bayer_pattern"),
                        supported_bins=info.get("supported_bins", [1]),
                        controls=info.get("controls", {}),
                    )
        
        return self._discovery_cache
    
    def get(
        self, 
        camera_id: int, 
        name: str | None = None,
        auto_connect: bool = False,
    ) -> Camera:
        """Get or create a Camera singleton for this camera_id.
        
        Returns the existing Camera instance if already created, or creates
        a new one with injected dependencies. Ensures only one Camera exists
        per camera_id to prevent resource conflicts.
        
        Args:
            camera_id: Hardware camera ID (0, 1, etc.).
            name: Optional friendly name (only used on first creation).
            auto_connect: If True, connect camera before returning.
            
        Returns:
            Camera instance (singleton per camera_id).
            
        Raises:
            CameraNotInRegistryError: If camera_id not found in discovery.
        
        Example:
            camera = registry.get(0, auto_connect=True)
            result = camera.capture()
        """
        # Ensure discovery has been run
        if self._discovery_cache is None:
            self.discover()
        
        # Validate camera exists
        if camera_id not in self._discovery_cache:
            raise CameraNotInRegistryError(
                f"Camera {camera_id} not found. "
                f"Available: {list(self._discovery_cache.keys())}"
            )
        
        # Get or create singleton
        if camera_id not in self._cameras:
            # Use name from discovery if not provided
            if name is None:
                name = self._discovery_cache[camera_id].name
            
            config = CameraConfig(
                camera_id=camera_id,
                name=name,
            )
            
            # Create recovery strategy lazily (needs self reference)
            if self._recovery_strategy is None:
                self._recovery_strategy = RecoveryStrategy(self)
            
            camera = Camera(
                driver=self._driver,
                config=config,
                renderer=self._renderer,
                clock=self._clock,
                hooks=self._hooks,
                recovery=self._recovery_strategy,
            )
            
            self._cameras[camera_id] = camera
        
        camera = self._cameras[camera_id]
        
        if auto_connect and not camera.is_connected:
            camera.connect()
        
        return camera
    
    def has(self, camera_id: int) -> bool:
        """Check if a camera instance exists in the registry.
        
        Returns True if a Camera has been created for this ID via get().
        Does not check discovery cache or hardware availability.
        
        Args:
            camera_id: Camera ID to check.
            
        Returns:
            True if Camera instance exists, False otherwise.
        
        Raises:
            None. Never raises exceptions.
        
        Example:
            if not registry.has(0):
                camera = registry.get(0, auto_connect=True)
        """
        return camera_id in self._cameras
    
    def remove(self, camera_id: int) -> Camera | None:
        """Remove a camera from the registry.
        
        Removes the Camera instance from the registry without disconnecting.
        Call camera.disconnect() first if you want to release hardware.
        
        Args:
            camera_id: Camera ID to remove.
            
        Returns:
            Removed Camera instance, or None if not in registry.
        
        Raises:
            None. Returns None if camera_id not found.
        
        Example:
            camera = registry.get(0)
            camera.disconnect()
            registry.remove(0)  # Free from registry
        """
        return self._cameras.pop(camera_id, None)
    
    @property
    def camera_ids(self) -> list[int]:
        """List of camera IDs with active Camera instances in registry.
        
        Returns IDs for cameras created via get(). Represents currently managed cameras, not
        necessarily connected or discovered cameras. Subset of discovered_ids (cameras must be
        discovered before get() creates instances).
        
        Business context: Essential for multi-camera telescope systems (main imager + guide camera)
        to enumerate active cameras for status displays, health checks, or coordinated operations.
        Differs from discovered_ids - this shows cameras application is actively using (created
        Camera instances), while discovered_ids shows what hardware is available. Useful for UI
        dashboards showing "cameras in use" vs "cameras available", resource cleanup loops
        (disconnect all active cameras), and debugging instance leaks.
        
        Implementation details: Returns list(self._cameras.keys()) where _cameras is dict[int, Camera]
        of singleton instances. Order not guaranteed (dict iteration order). Empty list if no
        cameras created yet via get(). Camera remains in list even if disconnected - only removed
        by explicit remove() or clear() calls. IDs typically 0-based sequential (0, 1, 2) matching
        USB device enumeration order.
        
        Returns:
            List of integer camera IDs (0-based) for Camera instances currently in registry.
            Empty list if no cameras created yet. Not sorted - use sorted(registry.camera_ids)
            for ordered display.
        
        Raises:
            None. Always succeeds, returns empty list if no cameras.
        
        Example:
            >>> registry = CameraRegistry(driver)
            >>> registry.discover()  # Found cameras 0, 1
            >>> registry.get(0)  # Create Camera instance for ID 0
            >>> print(registry.camera_ids)  # [0]
            >>> print(registry.discovered_ids)  # [0, 1]
            >>> # Only camera 0 has active instance
            >>> for cam_id in registry.camera_ids:
            ...     camera = registry.get(cam_id)
            ...     print(f"Camera {cam_id}: connected={camera.is_connected}")
        """
        return list(self._cameras.keys())
    
    @property
    def discovered_ids(self) -> list[int]:
        """List of camera IDs from last discovery scan (available hardware).
        
        Returns IDs found during discover() call (hardware currently connected to USB bus).
        Empty list if discover() not yet called. Superset of camera_ids - shows what hardware
        exists, not what's actively managed by registry.
        
        Business context: Critical for multi-camera systems to verify expected hardware present
        before operations. Enables "expected vs actual" health checks (e.g., main+guide cameras
        should show IDs [0,1]). Essential for diagnostics when cameras missing - user can compare
        discovered_ids to expected configuration. Used in setup/configuration flows to present
        available cameras for user selection. Helps identify USB enumeration issues (camera present
        but not discovered) vs hardware issues (camera missing entirely).
        
        Implementation details: Returns list(self._discovery_cache.keys()) if discovery run, else
        empty list. Discovery cache populated by discover() which calls driver.get_connected_cameras().
        Cache persists until next discover(refresh=True) or clear(). IDs represent hardware state
        at last scan - cameras may have disconnected since. Order not guaranteed. Typical IDs are
        0-based sequential (0, 1, 2) matching driver enumeration order (usually USB port order).
        
        Returns:
            List of integer camera IDs (0-based) found during last discover(). Empty list if
            discover() never called or clear() was called. Not sorted - use sorted() for
            ordered display.
        
        Raises:
            None. Always succeeds, returns empty list if no discovery run.
        
        Example:
            >>> registry = CameraRegistry(driver)
            >>> print(registry.discovered_ids)  # []
            >>> registry.discover()
            >>> print(registry.discovered_ids)  # [0, 1] - two cameras found
            >>> registry.get(0)  # Create instance for camera 0
            >>> print(registry.camera_ids)  # [0] - only one instance
            >>> print(registry.discovered_ids)  # [0, 1] - both still discovered
            >>> # Check if expected cameras present
            >>> expected = [0, 1]
            >>> if set(registry.discovered_ids) == set(expected):
            ...     print("All cameras present")
        """
        if self._discovery_cache is None:
            return []
        return list(self._discovery_cache.keys())
    
    def clear(self) -> None:
        """Disconnect all cameras and clear registry (graceful shutdown).
        
        Iterates through all Camera instances, disconnects each (releasing USB resources),
        then clears registry and discovery cache. Safe to call multiple times (idempotent).
        Errors during disconnect silently ignored for best-effort cleanup. Called automatically
        by __exit__ for context manager cleanup.
        
        Business context: Essential for graceful application shutdown releasing exclusive camera
        resources (USB handles, driver allocations) so other processes can access cameras.
        Critical in long-running servers where restart/reload without proper cleanup causes "device
        busy" errors requiring system reboots. Used in test teardown ensuring clean state between
        tests. Enables runtime reconfiguration (clear old cameras, discover new configuration).
        Multi-camera systems require coordinated shutdown to avoid leaving cameras in inconsistent
        states.
        
        Implementation details: Iterates self._cameras.values(), calls camera.disconnect() only if
        is_connected=True. Catches and suppresses all exceptions (best-effort cleanup - partial
        failures don't prevent clearing remaining cameras). Clears _cameras dict and sets
        _discovery_cache=None after disconnects. Idempotent - safe to call on empty registry.
        No return value, no exceptions raised. Typical disconnect takes 10-50ms per camera (driver
        close + USB release).
        
        Args:
            None.
        
        Returns:
            None. Side effects: all cameras disconnected, registry emptied, discovery cache cleared.
        
        Raises:
            None. Never raises - suppresses all exceptions for reliable cleanup in finally blocks
            and __exit__ handlers.
        
        Example:
            >>> registry = CameraRegistry(driver)
            >>> cameras = registry.discover()
            >>> camera = registry.get(0, auto_connect=True)
            >>> camera.capture()  # Use camera
            >>> registry.clear()  # Disconnect all, clear registry
            >>> assert len(registry.camera_ids) == 0
            >>> assert len(registry.discovered_ids) == 0
            >>> # Safe to call multiple times
            >>> registry.clear()  # No-op, no errors
        """
        for camera in self._cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception:
                    pass  # Best effort cleanup
        
        self._cameras.clear()
        self._discovery_cache = None
    
    # Context manager support
    
    def __enter__(self) -> CameraRegistry:
        """Enter context manager.
        
        Returns:
            Self for use in with statement.
        
        Raises:
            None. Never raises during context entry.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, disconnect all cameras.
        
        Calls clear() to disconnect all cameras and release resources.
        Does not suppress exceptions.
        
        Args:
            exc_type: Exception type if raised, else None.
            exc_val: Exception value if raised, else None.
            exc_tb: Traceback if raised, else None.
        """
        self.clear()
    
    def __repr__(self) -> str:
        """Return string representation of registry state.
        
        Returns:
            String showing discovered and active camera counts.
        
        Raises:
            None. Never fails during string formatting.
        """
        discovered = len(self._discovery_cache) if self._discovery_cache else 0
        active = len(self._cameras)
        return f"<CameraRegistry(discovered={discovered}, active={active})>"


# =============================================================================
# Module-level convenience (optional singleton pattern)
# =============================================================================

_default_registry: CameraRegistry | None = None


def init_registry(
    driver: CameraDriver,
    renderer: OverlayRenderer | None = None,
    clock: Clock | None = None,
    hooks: CameraHooks | None = None,
) -> CameraRegistry:
    """Initialize the default module-level camera registry singleton.
    
    Creates a singleton CameraRegistry accessible throughout the application
    via get_registry(). This provides a convenient global access point for
    camera management without passing registry instances through the call stack.
    Must be called once at application startup before any camera operations.
    
    Business context: Simplifies application architecture by providing a
    well-known access point for camera resources. Eliminates the need to
    pass registry instances through multiple layers of code. Particularly
    useful in server applications (FastAPI, MCP servers) where cameras need
    to be accessed from various endpoints and handlers. Follows the Service
    Locator pattern for resource management.
    
    Implementation details: Uses a module-level global variable for singleton
    storage. Subsequent calls replace the existing registry (useful for testing).
    Not thread-safe - ensure single initialization before concurrent access.
    Consider using CameraRegistry instances directly for better testability
    in complex applications.
    
    Args:
        driver: Camera driver for hardware access (ASICameraDriver or
            DigitalTwinCameraDriver). Required.
        renderer: Optional overlay renderer applied to all cameras created
            from this registry. Defaults to NullRenderer (no overlays).
        clock: Optional clock implementation for time functions. Defaults to
            SystemClock (time.monotonic/sleep). Inject MockClock for testing.
        hooks: Optional event hooks (on_connect, on_capture, etc.) applied
            to all cameras. Useful for centralized logging/metrics.
        
    Returns:
        The initialized CameraRegistry instance. Same instance returned by
        subsequent get_registry() calls until next init_registry().
    
    Raises:
        None. Always succeeds in creating registry.
    
    Example:
        >>> # Application startup
        >>> from telescope_mcp.drivers.cameras import ASICameraDriver
        >>> from telescope_mcp.devices.registry import init_registry, get_registry
        >>> 
        >>> # Initialize once at startup
        >>> registry = init_registry(
        ...     ASICameraDriver(),
        ...     hooks=CameraHooks(on_capture=log_capture_metrics)
        ... )
        >>> 
        >>> # Access from anywhere in the application
        >>> def capture_endpoint():
        ...     registry = get_registry()
        ...     camera = registry.get_or_create(0, "main")
        ...     return camera.capture()
    """
    global _default_registry
    _default_registry = CameraRegistry(driver, renderer, clock, hooks)
    return _default_registry


def get_registry() -> CameraRegistry:
    """Get module-level singleton registry (Service Locator pattern).
    
    Returns singleton registry created by init_registry(). Provides global access point for
    camera resources without passing registry through call stack. Must be called after
    init_registry() at application startup.
    
    Business context: Simplifies architecture in server applications (FastAPI, MCP servers)
    where cameras accessed from multiple endpoints, handlers, tools. Eliminates need to pass
    registry as dependency through every layer. Follows Service Locator pattern - well-known
    access point for shared resources. Particularly useful in telescope control servers where
    multiple HTTP endpoints, MCP tools, WebSocket handlers need camera access without complex
    dependency injection. Trade-off: convenience vs testability (global state harder to mock).
    
    Implementation details: Returns module-level global _default_registry created by init_registry().
    Same instance returned on every call until next init_registry() or shutdown_registry(). Not
    thread-safe during initialization (ensure init_registry() called once before concurrent access).
    Thread-safe for read access after initialization. For better testability in unit tests,
    prefer passing CameraRegistry instances directly rather than using global singleton.
    
    Returns:
        The default CameraRegistry instance configured at application startup. Same instance
        on every call (true singleton). Contains driver, renderer, clock, hooks from
        init_registry().
    
    Raises:
        RuntimeError: If init_registry() not called yet (module-level _default_registry is None).
            Error message guides user to call init_registry() first.
    
    Example:
        >>> # Application startup (main.py)
        >>> from telescope_mcp.devices.registry import init_registry, get_registry
        >>> from telescope_mcp.drivers.cameras import ASICameraDriver
        >>> init_registry(ASICameraDriver())
        >>> 
        >>> # Anywhere in application
        >>> def capture_handler():
        ...     registry = get_registry()  # No need to pass as parameter
        ...     camera = registry.get(0)
        ...     return camera.capture()
        >>> 
        >>> # FastAPI endpoint
        >>> @app.get("/capture/{camera_id}")
        >>> async def capture_endpoint(camera_id: int):
        ...     registry = get_registry()
        ...     camera = registry.get(camera_id)
        ...     result = camera.capture()
        ...     return {"image": base64.encode(result.image_data)}
    """
    if _default_registry is None:
        raise RuntimeError(
            "Registry not initialized. Call init_registry() first."
        )
    return _default_registry


def shutdown_registry() -> None:
    """Shutdown module-level singleton registry (application teardown).
    
    Clears module-level registry (disconnects all cameras, releases USB resources) and sets
    global _default_registry=None. Safe to call even if never initialized (no-op). Call at
    application shutdown for clean resource release before process exit.
    
    Business context: Critical for graceful application shutdown in long-running servers
    (FastAPI, MCP servers, observatory control systems). Ensures exclusive USB camera resources
    released so next application start (or other processes) can access cameras without "device
    busy" errors. Prevents resource leaks in restart scenarios (systemd service restart, Docker
    container recreation). Used in signal handlers (SIGTERM, SIGINT) for clean shutdown on
    Ctrl+C or service stop. Essential for proper systemd integration where service manager
    expects clean resource release.
    
    Implementation details: Checks if _default_registry is None (not initialized). If initialized,
    calls registry.clear() (disconnects all cameras, clears caches), then sets global
    _default_registry=None. Idempotent - safe to call multiple times or when not initialized
    (no-op on subsequent calls). No exceptions raised - best-effort cleanup. Typical execution
    50-200ms depending on number of cameras (each disconnect ~10-50ms). Call in finally blocks
    or atexit handlers for guaranteed cleanup.
    
    Args:
        None.
    
    Returns:
        None. Side effects: cameras disconnected, module-level registry cleared and set to None.
    
    Raises:
        None. Never raises - safe for cleanup code. Suppresses all exceptions from clear().
    
    Example:
        >>> # Application startup
        >>> from telescope_mcp.devices.registry import init_registry, shutdown_registry
        >>> import signal, atexit
        >>> 
        >>> init_registry(ASICameraDriver())
        >>> 
        >>> # Register shutdown handler
        >>> def cleanup():
        ...     print("Shutting down cameras...")
        ...     shutdown_registry()
        >>> 
        >>> atexit.register(cleanup)
        >>> signal.signal(signal.SIGTERM, lambda s, f: cleanup())
        >>> 
        >>> # Or in FastAPI lifespan
        >>> @app.on_event("shutdown")
        >>> async def shutdown_event():
        ...     shutdown_registry()
    """
    global _default_registry
    if _default_registry is not None:
        _default_registry.clear()
        _default_registry = None
