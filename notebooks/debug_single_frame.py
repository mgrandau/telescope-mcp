#!/usr/bin/env python3
"""Debug script: Capture single frame and save to disk for inspection."""

import numpy as np
import zwoasi as asi
from pathlib import Path
import cv2

# Initialize SDK
from telescope_mcp.drivers.asi_sdk import get_sdk_library_path
sdk_path = get_sdk_library_path()
asi.init(sdk_path)

# Get camera info
num_cameras = asi.get_num_cameras()
print(f"Found {num_cameras} camera(s)")

if num_cameras == 0:
    print("No cameras found!")
    exit(1)

# Open camera 0
camera = asi.Camera(0)
info = camera.get_camera_property()
print(f"Camera: {info['Name']}")
print(f"Resolution: {info['MaxWidth']}x{info['MaxHeight']}")
print(f"IsColorCam: {info['IsColorCam']}")

width = info['MaxWidth']
height = info['MaxHeight']
is_color = info['IsColorCam']

# Configure camera
exposure_us = 100_000  # 100ms
gain = 50

camera.set_control_value(asi.ASI_GAIN, gain)
camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 80)

output_dir = Path("/home/mark/src/telescope-mcp/notebooks/data")
output_dir.mkdir(exist_ok=True)

print("\n=== Testing RAW8 ===")
camera.set_roi(width=width, height=height, bins=1, image_type=asi.ASI_IMG_RAW8)
buffer_raw8 = bytearray(width * height)
camera.start_video_capture()
try:
    camera.capture_video_frame(buffer_=buffer_raw8, timeout=2000)
    arr_raw8 = np.frombuffer(buffer_raw8, dtype=np.uint8)
    print(f"RAW8 data: {len(buffer_raw8)} bytes (expected {width*height})")
    print(f"RAW8 stats: min={arr_raw8.min()}, max={arr_raw8.max()}, mean={arr_raw8.mean():.1f}")
    
    img_raw8 = arr_raw8.reshape((height, width))
    cv2.imwrite(str(output_dir / "debug_raw8.png"), img_raw8)
    print(f"Saved: {output_dir}/debug_raw8.png")
finally:
    camera.stop_video_capture()

print("\n=== Testing RGB24 ===")
camera.set_roi(width=width, height=height, bins=1, image_type=asi.ASI_IMG_RGB24)
buffer_rgb24 = bytearray(width * height * 3)
camera.start_video_capture()
try:
    camera.capture_video_frame(buffer_=buffer_rgb24, timeout=2000)
    arr_rgb24 = np.frombuffer(buffer_rgb24, dtype=np.uint8)
    print(f"RGB24 data: {len(buffer_rgb24)} bytes (expected {width*height*3})")
    print(f"RGB24 stats: min={arr_rgb24.min()}, max={arr_rgb24.max()}, mean={arr_rgb24.mean():.1f}")
    
    img_rgb24 = arr_rgb24.reshape((height, width, 3))
    cv2.imwrite(str(output_dir / "debug_rgb24.png"), img_rgb24)
    print(f"Saved: {output_dir}/debug_rgb24.png")
finally:
    camera.stop_video_capture()

camera.close()
print("\n✓ Done - check notebooks/data/ for images")
