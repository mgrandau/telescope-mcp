# Focusing the Main Imaging Camera (ASI482MC)

## Overview

The main imaging camera (Camera 1: ASI482MC, 1920×1080, 1600mm focal length) requires manual focusing using the Crayford focuser knob. This procedure uses the telescope-mcp web dashboard's live MJPEG stream to achieve focus remotely from a phone or laptop.

## Prerequisites

- Telescope set up outside and powered on
- telescope-mcp web server running (`python -m telescope_mcp.web`)
- Clear sky with visible stars
- Phone or laptop connected to the same network as the telescope controller

## Equipment

- **Camera:** ASI482MC (Camera 1) at 1600mm focal length, 0.748 arcsec/pixel
- **Focuser:** Crayford focuser with manual knob
- **Stream endpoint:** `/stream/main` (or `/stream/1`)

## Procedure

### 1. Open the Main Camera Stream

On your phone or laptop browser, navigate to the telescope dashboard and open the main camera stream:

```
http://<telescope-host>:8080/stream/main
```

Or use the dashboard at `http://<telescope-host>:8080/` which embeds both camera streams.

**Tip:** For focusing, use a moderate exposure and higher gain to get a responsive preview:

```
http://<telescope-host>:8080/stream/main?exposure_us=300000&gain=80&fps=10
```

### 2. Point at a Bright Star

Use the motor controls (dashboard buttons or API) to slew to a bright star. A magnitude 1-3 star works well — bright enough to see even when defocused.

### 3. Assess Current Focus

Look at the star image on the stream:

| What you see | Focus state |
|---|---|
| Large fuzzy disk / donut | Very far out of focus |
| Soft blob, no sharp edges | Moderately out of focus |
| Small tight point | Near or at focus |
| Airy disk with diffraction rings (if seeing allows) | Excellent focus |

### 4. Adjust the Crayford Focuser

With the stream visible on your phone:

1. **Turn the Crayford focuser knob slowly** — small adjustments only
2. **Watch the star on the stream** — it should shrink as you approach focus
3. **Go past focus intentionally** — turn until the star starts getting bigger again, then reverse
4. **Split the difference** — find the point where the star is smallest and tightest
5. **Check multiple stars** if visible in the field — they should all be tight points when focused

**Important:** The stream has some latency (depends on exposure time and network). Make small adjustments, then **wait** for the stream to update before adjusting again. Don't rush it.

### 5. Verify Focus

Once you think you have focus:

- Stars should appear as tight points across the field of view
- Edge-of-field stars may show some coma or astigmatism — that's optical, not focus
- If stars look elongated uniformly, that's likely tracking/wind, not focus

### 6. Lock the Focuser (if applicable)

If your Crayford focuser has a lock screw or tension adjustment, secure it so the focus doesn't shift during the session from cable drag or gravity as the telescope moves.

## Tips

- **Temperature changes shift focus.** On cold nights, metal contracts and focus can drift. Check focus every 30-60 minutes, especially during the first hour as the telescope acclimates to outdoor temperature.
- **Use the finder camera for rough pointing, main camera for fine focus.** The finder (`/stream/finder`) has a 150° all-sky view — great for confirming you're in the right part of the sky before switching attention to the main camera.
- **Higher gain = faster feedback for focusing.** Crank the gain up during focusing (you can lower it for actual imaging afterward). You want to see the star response quickly.
- **Avoid touching the telescope tube.** Body heat and vibration from touching the tube will distort the image temporarily.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| No stars visible on stream | Camera not connected, lens cap on, or pointing at clouds | Check `/api/cameras`, remove lens cap, check sky |
| Stars visible but won't come to focus | Focuser at mechanical limit | Reverse direction, check backfocus distance |
| Stream is very laggy | Long exposure time or slow network | Reduce `exposure_us`, increase `fps`, check wifi signal |
| Stars drift out of frame during focusing | Tracking not active or wind | Work quickly, re-center as needed |
| Focus shifts after a few minutes | Temperature change or focuser slip | Tighten Crayford tension, re-focus periodically |

## Session Log: February 13, 2026

First successful use of this procedure. Main camera was out of focus at session start. Streamed `/stream/main` to phone, adjusted the Crayford focuser knob while watching the star images on the phone screen. Stars came to focus and the session proceeded normally. Cloud cover: 0% (10 PM – 2 AM CT).
