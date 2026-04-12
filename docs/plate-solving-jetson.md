# Plate Solving Service — Jetson Orin Nano

## Summary

Add a local plate-solving service to telescope-mcp, designed to run on an
NVIDIA Jetson Orin Nano 8GB as a network-accessible solver. Eliminates
dependency on astrometry.net cloud API, reduces solve times from 10-30s
to under 1s, and enables continuous pointing verification.

## Architecture

```
Telescope Laptop (existing, unchanged)
    |
    | sends image over local network / Tailscale
    v
Jetson Orin Nano (plate-solve service)
    |
    | solves in <1s, returns RA/Dec
    v
Telescope Laptop receives result, continues workflow
```

The Jetson runs as a dedicated plate-solving appliance alongside the
existing telescope control laptop. No changes to the current setup —
just swap the solver endpoint from astrometry.net to the local Jetson.

Later phase: migrate full telescope control (KStars/Ekos/INDI) to the
Jetson, replacing the laptop entirely.

## Fisheye Finder Camera Use Case

Primary use case is solving images from the fisheye camera mounted on
the scope as a digital finder/spotter:

- Fisheye gives 120-180 deg FOV — tons of stars, very reliable solves
- Continuous solving: capture frame every N seconds, solve, verify pointing
- Drift detection without guiding — if solved position shifts from target, correct automatically
- Goto verification — slew, solve fisheye, confirm on target, start imaging
- Never interrupts main imaging camera

### Fisheye Distortion Handling

Fisheye lenses have massive barrel distortion. Two approaches:

1. **Undistort first** — OpenCV lens calibration matrix, one-time setup,
   apply correction before solving
2. **Center crop** — crop center of fisheye frame to ~30-40 deg FOV,
   minimal distortion, standard solvers handle it fine (easiest path)

## Solver Backends

### Tetra3 (recommended for main scope)
- Python-based, hash pattern matching
- Sub-second solves on modest hardware
- Best for known FOV ranges (main imaging camera)
- Generate database per FOV range at setup time

### ASTAP (recommended for fisheye)
- Handles wide-field and distortion better
- Star catalogs at different depths (H18 wide, G18 narrow)
- Runs on ARM Linux
- 1-3 second solves typical

### Astrometry.net local (fallback)
- Handles SIP distortion terms natively
- Heavier weight, more dependencies
- Good for blind solves when FOV is unknown

## Proposed API

```
POST /solve
{
    "image": <base64 FITS/JPEG>,
    "lens": "fisheye" | "standard",   // triggers undistort if fisheye
    "fov_min": 0.5,                   // degrees, optional
    "fov_max": 2.0,                   // degrees, optional
    "focal_length": 1000,             // mm, optional
    "pixel_size": 3.76                // microns, optional
}

Response:
{
    "success": true,
    "ra": 180.234,          // degrees
    "dec": 45.678,          // degrees
    "rotation": 12.3,       // degrees
    "fov": 1.2,             // degrees
    "solve_time_ms": 340,
    "stars_detected": 47,
    "stars_matched": 31
}
```

If focal_length + pixel_size provided, FOV is calculated automatically.
If nothing provided, broader search (slower but still local).

## MCP Tool Integration

New telescope-mcp tool:

```python
plate_solve(
    image: str,              # base64 image data or file path
    solver: str = "auto",    # "tetra3" | "astap" | "astrometry" | "auto"
    lens: str = "standard",  # "standard" | "fisheye"
    fov_hint: float = None,  # degrees, optional
    remote_url: str = None,  # Jetson URL, if None use local solver
) -> dict
```

Supports both local solving (if running on Jetson) and remote solving
(send image to Jetson service over network/Tailscale).

## Hardware Requirements

| Component | Spec | Cost |
|-----------|------|------|
| Jetson Orin Nano | 8GB, 1024 CUDA cores, 40 TOPS | ~$250 |
| Storage | MicroSD or NVMe SSD | ~$30 |
| Power | 5V 4A barrel jack | ~$15 |
| Star catalogs | H18/G18 for ASTAP, or Tetra3 database | Free |
| Total | | ~$300 |

## Advantages Over Current Setup

| | Cloud (astrometry.net) | Jetson Local |
|---|---|---|
| Solve time | 10-30s + upload | <1s |
| Network required | Yes | No |
| Works in field | Cell signal needed | Fully offline |
| Continuous solving | Impractical (rate limits) | Every frame if needed |
| Cost | Free but slow | $300 one-time |

## Implementation Phases

### Phase 1: Plate-solve service (FastAPI on Jetson)
- HTTP endpoint for remote solving
- Tetra3 backend for standard lenses
- ASTAP backend for fisheye
- Catalog setup and FOV configuration

### Phase 2: telescope-mcp integration
- New `plate_solve` MCP tool
- Support local and remote solver backends
- Solver result logging in observation sessions

### Phase 3: Continuous pointing verification
- Fisheye camera loop: capture → solve → verify → correct
- Drift detection and automatic correction
- Dashboard integration for real-time pointing status

### Phase 4: Full telescope control migration (optional)
- Move KStars/Ekos/INDI to Jetson
- Laptop no longer needed at scope
- Jetson as complete telescope control + solving appliance
