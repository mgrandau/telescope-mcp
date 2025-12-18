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
        """Attempt to recover a disconnected camera.
        
        Forces registry re-discovery to scan USB bus for cameras,
        then checks if the target camera is now available.
        
        Args:
            camera_id: ID of the camera to recover (0-indexed).
            
        Returns:
            True if camera is available after recovery attempt,
            False if camera was not found during re-discovery.
        
        Raises:
            None. Handles all driver errors by returning False.
        
        Note:
            This triggers a USB rescan which may take 100-500ms.
            The camera instance will need to call connect() again.
        """
        cameras = self._registry.discover(refresh=True)
        return camera_id in cameras


class NullRecoveryStrategy:
    """No-op recovery strategy (for testing or when recovery not desired)."""
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """Return False without attempting recovery.
        
        Args:
            camera_id: ID of camera to recover (ignored).
        
        Returns:
            Always False, indicating no recovery was attempted.
        
        Raises:
            None. Never raises exceptions.
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
        """List of camera IDs with active instances in registry.
        
        Returns IDs for cameras that have been created via get().
        
        Returns:
            List of integer camera IDs currently in the registry.
        
        Raises:
            None. Returns empty list if no cameras registered.
        """
        return list(self._cameras.keys())
    
    @property
    def discovered_ids(self) -> list[int]:
        """List of camera IDs from last discovery scan.
        
        Returns IDs found during discover() call. Empty list if
        discover() has not been called yet.
        
        Returns:
            List of discovered camera IDs, or empty list.
        
        Raises:
            None. Returns empty list if discovery not yet run.
        """
        if self._discovery_cache is None:
            return []
        return list(self._discovery_cache.keys())
    
    def clear(self) -> None:
        """Disconnect all cameras and clear registry.
        
        Iterates through all Camera instances, disconnects each,
        then clears the registry and discovery cache. Safe to call
        multiple times. Errors during disconnect are silently ignored.
        
        Raises:
            None. Suppresses all exceptions during disconnect.
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
    """Get the default module-level registry.
    
    Returns the singleton registry created by init_registry().
    Must be called after init_registry() or will raise an error.
    
    Returns:
        The default CameraRegistry instance.
    
    Raises:
        RuntimeError: If init_registry() has not been called yet.
    """
    if _default_registry is None:
        raise RuntimeError(
            "Registry not initialized. Call init_registry() first."
        )
    return _default_registry


def shutdown_registry() -> None:
    """Shutdown the default registry, disconnecting all cameras.
    
    Clears the module-level registry and disconnects all cameras.
    Safe to call even if registry was never initialized (no-op).
    Call at application shutdown for clean resource release.
    
    Raises:
        None. Safe to call multiple times or when not initialized.
    """
    global _default_registry
    if _default_registry is not None:
        _default_registry.clear()
        _default_registry = None
