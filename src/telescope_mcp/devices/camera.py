"""Logical camera device with driver injection.

This module provides a hardware-agnostic Camera class that accepts
a driver (real ASI or digital twin) via dependency injection.

Example:
    from telescope_mcp.devices import Camera, CameraConfig
    from telescope_mcp.drivers.cameras import ASICameraDriver
    
    driver = ASICameraDriver()
    config = CameraConfig(camera_id=0, name="Main Camera")
    
    with Camera(driver, config) as cam:
        result = cam.capture(exposure_us=100_000)
        print(f"Captured {len(result.image_data)} bytes")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telescope_mcp.drivers.cameras import CameraDriver, CameraInstance


@dataclass
class CameraConfig:
    """Configuration for camera initialization.
    
    Attributes:
        camera_id: Hardware camera ID (0, 1, etc.)
        name: Optional friendly name for logging
        default_gain: Default gain value (0-100 typical)
        default_exposure_us: Default exposure in microseconds
    """
    camera_id: int
    name: str | None = None
    default_gain: int = 50
    default_exposure_us: int = 100_000


@dataclass
class CameraInfo:
    """Information about a connected camera.
    
    Returned by Camera.connect() with details about the camera's
    capabilities and current state.
    """
    camera_id: int
    name: str
    max_width: int
    max_height: int
    is_color: bool
    bayer_pattern: str | None
    supported_bins: list[int]
    controls: dict[str, dict]  # control_name -> {min, max, default, value}
    
    @classmethod
    def from_driver_info(cls, info: dict, controls: dict) -> CameraInfo:
        """Create CameraInfo from driver response dicts."""
        return cls(
            camera_id=info.get("camera_id", 0),
            name=info.get("name", "Unknown"),
            max_width=info.get("max_width", 0),
            max_height=info.get("max_height", 0),
            is_color=info.get("is_color", False),
            bayer_pattern=info.get("bayer_pattern"),
            supported_bins=info.get("supported_bins", [1]),
            controls=controls,
        )


@dataclass
class CaptureResult:
    """Result of a frame capture.
    
    Contains the image data and metadata about the capture settings.
    """
    image_data: bytes
    timestamp: datetime
    exposure_us: int
    gain: int
    width: int = 0
    height: int = 0
    format: str = "jpeg"
    metadata: dict = field(default_factory=dict)


class CameraError(Exception):
    """Base exception for camera operations."""
    pass


class CameraNotConnectedError(CameraError):
    """Raised when operation requires connected camera."""
    pass


class CameraAlreadyConnectedError(CameraError):
    """Raised when connecting to already-connected camera."""
    pass


class Camera:
    """Logical camera device with injected driver.
    
    Provides a clean interface for camera operations independent of
    the underlying hardware. Accepts a driver (ASI SDK or digital twin)
    via dependency injection.
    
    Can be used as a context manager for automatic disconnect:
    
        with Camera(driver, config) as cam:
            result = cam.capture()
    
    Or manually:
    
        cam = Camera(driver, config)
        cam.connect()
        try:
            result = cam.capture()
        finally:
            cam.disconnect()
    
    Attributes:
        config: Camera configuration
        is_connected: Whether camera is currently connected
        info: Camera info (None if not connected)
    """
    
    def __init__(self, driver: CameraDriver, config: CameraConfig) -> None:
        """Create camera with injected driver and config.
        
        Args:
            driver: Camera driver (ASI or digital twin)
            config: Camera configuration including ID and defaults
        """
        self._driver = driver
        self._config = config
        self._instance: CameraInstance | None = None
        self._info: CameraInfo | None = None
        self._current_gain: int = config.default_gain
        self._current_exposure_us: int = config.default_exposure_us
    
    @property
    def config(self) -> CameraConfig:
        """Get camera configuration."""
        return self._config
    
    @property
    def is_connected(self) -> bool:
        """Check if camera is currently connected."""
        return self._instance is not None
    
    @property
    def info(self) -> CameraInfo | None:
        """Get camera info (None if not connected)."""
        return self._info
    
    def connect(self) -> CameraInfo:
        """Connect to camera and return camera info.
        
        Returns:
            CameraInfo with camera capabilities and controls
            
        Raises:
            CameraAlreadyConnectedError: If already connected
            CameraError: If connection fails
        """
        if self._instance is not None:
            raise CameraAlreadyConnectedError(
                f"Camera {self._config.camera_id} is already connected"
            )
        
        try:
            self._instance = self._driver.open(self._config.camera_id)
            
            # Get camera info from driver
            driver_info = self._instance.get_info()
            driver_controls = self._instance.get_controls()
            
            self._info = CameraInfo.from_driver_info(driver_info, driver_controls)
            
            # Apply default settings
            self._instance.set_control("Gain", self._config.default_gain)
            self._instance.set_control("Exposure", self._config.default_exposure_us)
            self._current_gain = self._config.default_gain
            self._current_exposure_us = self._config.default_exposure_us
            
            return self._info
            
        except Exception as e:
            self._instance = None
            self._info = None
            raise CameraError(f"Failed to connect to camera {self._config.camera_id}: {e}") from e
    
    def disconnect(self) -> None:
        """Disconnect from camera and release resources.
        
        Safe to call even if not connected (no-op).
        """
        if self._instance is not None:
            try:
                self._instance.close()
            except Exception:
                pass  # Best effort cleanup
            finally:
                self._instance = None
                self._info = None
    
    def capture(
        self,
        exposure_us: int | None = None,
        gain: int | None = None,
    ) -> CaptureResult:
        """Capture a frame with optional override settings.
        
        Args:
            exposure_us: Override exposure time (microseconds), or None for current
            gain: Override gain value, or None for current
            
        Returns:
            CaptureResult with image data and metadata
            
        Raises:
            CameraNotConnectedError: If not connected
            CameraError: If capture fails
        """
        if self._instance is None:
            raise CameraNotConnectedError("Camera is not connected")
        
        # Apply overrides if provided
        effective_exposure = exposure_us if exposure_us is not None else self._current_exposure_us
        effective_gain = gain if gain is not None else self._current_gain
        
        # Update settings if changed
        if effective_gain != self._current_gain:
            self.set_control("Gain", effective_gain)
        if effective_exposure != self._current_exposure_us:
            self.set_control("Exposure", effective_exposure)
        
        try:
            image_data = self._instance.capture(effective_exposure)
            
            return CaptureResult(
                image_data=image_data,
                timestamp=datetime.now(timezone.utc),
                exposure_us=effective_exposure,
                gain=effective_gain,
                width=self._info.max_width if self._info else 0,
                height=self._info.max_height if self._info else 0,
                format="jpeg",
                metadata={
                    "camera_id": self._config.camera_id,
                    "camera_name": self._config.name or self._info.name if self._info else "Unknown",
                },
            )
        except Exception as e:
            raise CameraError(f"Capture failed: {e}") from e
    
    def set_control(self, name: str, value: int) -> None:
        """Set a camera control value.
        
        Args:
            name: Control name (e.g., "Gain", "Exposure", "WB_R")
            value: Control value to set
            
        Raises:
            CameraNotConnectedError: If not connected
            CameraError: If setting control fails
        """
        if self._instance is None:
            raise CameraNotConnectedError("Camera is not connected")
        
        try:
            self._instance.set_control(name, value)
            
            # Track common controls
            if name == "Gain":
                self._current_gain = value
            elif name == "Exposure":
                self._current_exposure_us = value
                
        except Exception as e:
            raise CameraError(f"Failed to set {name}={value}: {e}") from e
    
    def get_control(self, name: str) -> int:
        """Get current value of a camera control.
        
        Args:
            name: Control name (e.g., "Gain", "Exposure")
            
        Returns:
            Current control value
            
        Raises:
            CameraNotConnectedError: If not connected
            CameraError: If getting control fails
        """
        if self._instance is None:
            raise CameraNotConnectedError("Camera is not connected")
        
        try:
            result = self._instance.get_control(name)
            return result.get("value", 0)
        except Exception as e:
            raise CameraError(f"Failed to get {name}: {e}") from e
    
    # Context manager support
    
    def __enter__(self) -> Camera:
        """Enter context manager, connect to camera."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, disconnect from camera."""
        self.disconnect()
    
    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        name = self._config.name or f"camera_{self._config.camera_id}"
        return f"<Camera({name}, {status})>"
