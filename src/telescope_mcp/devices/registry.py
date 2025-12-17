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
    This implementation uses the registry's discovery mechanism.
    """
    
    def __init__(self, registry: CameraRegistry) -> None:
        """Create recovery strategy linked to registry.
        
        Args:
            registry: Registry to use for re-discovery
        """
        self._registry = registry
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """Attempt to recover a disconnected camera.
        
        Forces re-discovery and checks if camera is available.
        
        Args:
            camera_id: ID of camera to recover
            
        Returns:
            True if camera is available after recovery attempt
        """
        cameras = self._registry.discover(refresh=True)
        return camera_id in cameras


class NullRecoveryStrategy:
    """No-op recovery strategy (for testing or when recovery not desired)."""
    
    def attempt_recovery(self, camera_id: int) -> bool:
        """Always returns False (no recovery attempted)."""
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
        
        Args:
            driver: Camera driver for hardware access
            renderer: Overlay renderer for created cameras
            clock: Clock for timing (default: SystemClock)
            hooks: Event hooks for created cameras
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
        """Get the camera driver."""
        return self._driver
    
    def discover(self, refresh: bool = False) -> dict[int, CameraInfo]:
        """Discover connected cameras.
        
        Uses cached results unless refresh=True.
        
        Args:
            refresh: Force re-discovery (ignore cache)
            
        Returns:
            Dict mapping camera_id to CameraInfo
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
        
        Args:
            camera_id: Hardware camera ID
            name: Optional friendly name (only used on first creation)
            auto_connect: If True, connect camera before returning
            
        Returns:
            Camera instance (singleton per camera_id)
            
        Raises:
            CameraNotInRegistryError: If camera_id not in discovery cache
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
        """Check if a camera is in the registry.
        
        Args:
            camera_id: Camera ID to check
            
        Returns:
            True if camera exists in registry
        """
        return camera_id in self._cameras
    
    def remove(self, camera_id: int) -> Camera | None:
        """Remove a camera from the registry.
        
        Does NOT disconnect the camera - call camera.disconnect() first
        if needed.
        
        Args:
            camera_id: Camera ID to remove
            
        Returns:
            Removed camera or None if not found
        """
        return self._cameras.pop(camera_id, None)
    
    @property
    def camera_ids(self) -> list[int]:
        """List of camera IDs currently in registry."""
        return list(self._cameras.keys())
    
    @property
    def discovered_ids(self) -> list[int]:
        """List of discovered camera IDs."""
        if self._discovery_cache is None:
            return []
        return list(self._discovery_cache.keys())
    
    def clear(self) -> None:
        """Disconnect all cameras and clear registry.
        
        Safe to call multiple times.
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
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, disconnect all cameras."""
        self.clear()
    
    def __repr__(self) -> str:
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
    """Initialize the default registry.
    
    Call once at application startup.
    
    Args:
        driver: Camera driver
        renderer: Optional overlay renderer
        clock: Optional clock implementation
        hooks: Optional event hooks
        
    Returns:
        The initialized registry
    """
    global _default_registry
    _default_registry = CameraRegistry(driver, renderer, clock, hooks)
    return _default_registry


def get_registry() -> CameraRegistry:
    """Get the default registry.
    
    Raises:
        RuntimeError: If registry not initialized
    """
    if _default_registry is None:
        raise RuntimeError(
            "Registry not initialized. Call init_registry() first."
        )
    return _default_registry


def shutdown_registry() -> None:
    """Shutdown the default registry, disconnecting all cameras."""
    global _default_registry
    if _default_registry is not None:
        _default_registry.clear()
        _default_registry = None
