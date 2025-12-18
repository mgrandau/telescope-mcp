"""ZWO ASI Camera 2 SDK.

This module provides the path to the ASI Camera 2 shared library for use with
the `zwoasi` Python package.

SDK Version: 1.40
Source: https://www.zwoastro.com/software/

Usage:
    import zwoasi as asi
    from telescope_mcp.drivers.asi_sdk import get_sdk_library_path
    
    asi.init(get_sdk_library_path())

Installation (udev rules for USB access):
    sudo cp asi.rules /etc/udev/rules.d/99-asi.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
"""

import platform
from pathlib import Path

SDK_VERSION = "1.40"

# Map platform.machine() to SDK library subdirectory
_ARCH_MAP = {
    "x86_64": "x64",
    "AMD64": "x64",
    # Future: add ARM support if needed
    # "aarch64": "armv8",
    # "armv7l": "armv7",
}


def get_sdk_library_path() -> str:
    """Get the path to the ASI Camera 2 shared library for the current architecture.
    
    Returns:
        Full path to libASICamera2.so.{version}
        
    Raises:
        RuntimeError: If the current architecture is not supported
    """
    machine = platform.machine()
    arch_dir = _ARCH_MAP.get(machine)
    
    if arch_dir is None:
        supported = list(_ARCH_MAP.keys())
        raise RuntimeError(
            f"Unsupported architecture: {machine}. "
            f"Supported architectures: {supported}"
        )
    
    sdk_dir = Path(__file__).parent
    lib_path = sdk_dir / arch_dir / f"libASICamera2.so.{SDK_VERSION}"
    
    if not lib_path.exists():
        raise RuntimeError(
            f"ASI SDK library not found at {lib_path}. "
            f"Please ensure the SDK is properly installed."
        )
    
    return str(lib_path)


def get_udev_rules_path() -> str:
    """Get the path to the udev rules file for ASI camera USB permissions.
    
    Returns the filesystem path to the asi.rules file which configures Linux
    udev to grant USB access to ZWO ASI cameras without requiring root privileges.
    Use this path when installing camera drivers on Linux systems.
    
    Business context: Essential for Linux installations where cameras would
    otherwise require root/sudo access. Allows normal users to access ASI cameras
    by configuring proper USB device permissions. Part of the camera driver setup
    process documented in installation guides. Without proper udev rules, cameras
    may not be accessible or may require running applications as root (security risk).
    
    Returns:
        Absolute path to asi.rules file as string. This file should be copied to
        /etc/udev/rules.d/ and udev reloaded for camera access to work properly.
    
    Raises:
        None. Always returns the path (file existence not validated).
    
    Example:
        >>> rules_path = get_udev_rules_path()
        >>> print(f"Copy {rules_path} to /etc/udev/rules.d/")
        >>> # In setup script:
        >>> import shutil
        >>> shutil.copy(get_udev_rules_path(), "/etc/udev/rules.d/asi.rules")
        >>> # Then reload: sudo udevadm control --reload-rules
    """
    return str(Path(__file__).parent / "asi.rules")
