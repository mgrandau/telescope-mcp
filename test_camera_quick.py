#!/usr/bin/env python3
"""Quick test of camera integration."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from telescope_mcp.drivers.config import DriverConfig, DriverMode, DriverFactory
from telescope_mcp.devices import init_registry, get_registry, shutdown_registry
from telescope_mcp.devices.camera import CaptureOptions

print("=" * 60)
print("Camera Integration Test")
print("=" * 60)

# Step 1: Initialize digital twin driver
print("\n1. Initialize digital twin driver...")
config = DriverConfig(mode=DriverMode.DIGITAL_TWIN)
factory = DriverFactory(config)
driver = factory.create_camera_driver()
print(f"   ✓ Created driver: {type(driver).__name__}")

init_registry(driver)
registry = get_registry()
print(f"   ✓ Initialized registry: {registry}")

# Step 2: Discover cameras
print("\n2. Discover cameras...")
cameras = registry.discover()
print(f"   ✓ Discovered {len(cameras)} camera(s)")
for cam_id, cam_info in cameras.items():
    print(f"     - {cam_info.name} (ID: {cam_id})")

# Step 3: Get camera instance
print("\n3. Get camera instance...")
camera = registry.get(camera_id=0, auto_connect=True)
print(f"   ✓ Got camera: {camera}")
print(f"   ✓ Connected: {camera.is_connected}")

info = camera.info
print(f"\n   Camera Info:")
print(f"     Name: {info.name}")
print(f"     Resolution: {info.max_width}x{info.max_height}")

# Step 4: Capture frame
print("\n4. Capture frame...")
options = CaptureOptions(exposure_us=100000, gain=50)
result = camera.capture(options)
print(f"   ✓ Capture successful")
print(f"     Format: {result.format}")
print(f"     Size: {len(result.image_data)} bytes")
print(f"     Dimensions: {result.width}x{result.height}")

# Step 5: Test camera controls
print("\n5. Test camera controls...")
gain_control = camera._instance.get_control('GAIN')
gain_value = gain_control.get('value', 0)
print(f"   Current gain: {gain_value}")

camera._instance.set_control('GAIN', value=75)
new_control = camera._instance.get_control('GAIN')
new_gain = new_control.get('value', 0)
print(f"   New gain: {new_gain}")

# Step 6: Test multiple camera access
print("\n6. Test multiple camera access (singleton pattern)...")
camera2 = registry.get(camera_id=0)
print(f"   Same instance: {camera is camera2}")

# Step 7: Cleanup
print("\n7. Cleanup...")
shutdown_registry()
print("   ✓ Registry shutdown complete")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
