# Data Storage Architecture

## Overview

All telescope data uses **ASDF (Advanced Scientific Data Format)** as the single source of truth. ASDF files are self-contained packages that include images, metadata, telemetry, and observability logs - everything needed to understand and reproduce a session.

## Core Principle: Sessions, Not Just Observations

**Key insight:** Not all telescope activity is an "observation." We introduce the concept of a **session**—a time-bounded period of telescope activity that gets its own ASDF file.

### Session Types

| Type | Description | Example |
|------|-------------|---------|
| `observation` | Scientific data collection | Imaging M31 for 2 hours |
| `alignment` | Calibration procedures | Plate solving for RA/Dec offset |
| `experiment` | Testing and development | Trying new exposure settings |
| `maintenance` | System checks | Focusing, collimation |
| `idle` | Background system logs | Logs when nothing else is active |

Every session produces an ASDF file. This solves the "where do logs go when there's no observation?" problem.

---

## ASDF File Structure

```
session_20251214_220000.asdf
├── meta/
│   ├── session_type: "observation"
│   ├── session_id: "obs_m31_20251214"
│   ├── start_time: "2025-12-14T22:00:00Z"
│   ├── end_time: "2025-12-14T23:45:00Z"
│   ├── target: "M31"
│   └── location: {lat: 45.5, lon: -122.6, alt: 100}
│
├── cameras/
│   ├── main/
│   │   ├── info: {model: "ASI482MC", sensor_size: [11.13, 6.26], ...}
│   │   ├── settings: {gain: 200, exposure_us: 312000, ...}
│   │   └── frames: [ndarray...]  # or references to external files
│   └── spotter/
│       ├── info: {model: "ASI120MC-S", lens: "all-sky-150", ...}
│       └── frames: [ndarray...]
│
├── telemetry/
│   ├── mount_position: [{time, ra, dec, alt, az}, ...]
│   ├── temperature: [{time, sensor, value}, ...]
│   └── focus_position: [{time, value}, ...]
│
├── calibration/
│   ├── dark_frames: [ndarray...]
│   ├── flat_frames: [ndarray...]
│   └── plate_solve_results: [{ra, dec, rotation, scale}, ...]
│
└── observability/
    ├── logs: [
    │     {time, level, source, message, context},
    │     ...
    │   ]
    ├── metrics: {
    │     frames_captured: 564,
    │     errors: 3,
    │     warnings: 12,
    │     duration_seconds: 6300
    │   }
    └── events: [
          {time, event: "tracking_lost", details: {...}},
          {time, event: "cloud_detected", details: {...}},
          ...
        ]
```

---

## Session Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                     Session Manager                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   start_session(type="observation", target="M31")           │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────────┐                                       │
│   │ Active Session  │◄──── log(), capture(), telemetry()    │
│   │ (in-memory)     │                                       │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼ end_session()                                  │
│   ┌─────────────────┐                                       │
│   │ Write ASDF      │──── session_20251214_220000.asdf      │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼ (optional)                                     │
│   ┌─────────────────┐                                       │
│   │ Update Catalog  │──── catalog.parquet                   │
│   └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Handling Non-Observation Logging

### Problem
Logs happen even when no observation is active:
- System startup/shutdown
- Manual focusing
- Alignment procedures
- Failed observation attempts

### Solution: Always Have an Active Session

```python
class SessionManager:
    def __init__(self):
        self._active_session = None
        self._ensure_idle_session()

    def _ensure_idle_session(self):
        """Create idle session if nothing else is active."""
        if self._active_session is None:
            self._active_session = Session(
                session_type="idle",
                auto_rotate=True,  # New file every hour
            )

    def start_session(self, session_type: str, **kwargs) -> Session:
        """Start a new session, closing any existing one."""
        if self._active_session:
            self._active_session.close()

        self._active_session = Session(session_type=session_type, **kwargs)
        return self._active_session

    def end_session(self) -> Path:
        """End current session, return to idle."""
        path = self._active_session.close()
        self._ensure_idle_session()
        return path

    def log(self, level: str, message: str, **context):
        """Log to current session (always exists)."""
        self._active_session.log(level, message, **context)
```

### Session Type Transitions

```
                    start_session("alignment")
         ┌──────────────────────────────────────────┐
         │                                          ▼
    ┌────┴────┐                              ┌─────────────┐
    │  IDLE   │                              │  ALIGNMENT  │
    └────┬────┘                              └──────┬──────┘
         ▲                                          │
         │              end_session()               │
         └──────────────────────────────────────────┘

    start_session("observation", target="M31")
         ┌──────────────────────────────────────────┐
         │                                          ▼
    ┌────┴────┐                              ┌─────────────┐
    │  IDLE   │                              │ OBSERVATION │
    └────┬────┘                              └──────┬──────┘
         ▲                                          │
         │              end_session()               │
         └──────────────────────────────────────────┘
```

---

## Strengths & Weaknesses

### ✅ Strengths

| Aspect | Benefit |
|--------|---------|
| **Complete provenance** | Every log entry tied to a session with full context |
| **No orphan logs** | Idle sessions catch everything between observations |
| **Portable archives** | Share one file → recipient gets data + logs + context |
| **Reproducibility** | Re-run analysis with exact conditions recorded |
| **Unified schema** | Same structure for observations, alignment, experiments |
| **ASDF extensibility** | Add new data types without breaking readers |

### ❌ Weaknesses

| Concern | Mitigation |
|---------|------------|
| **Idle session bloat** | Auto-rotate hourly; compress; retention policy |
| **No real-time log access** | Dual-write: stream to console + buffer for ASDF |
| **Cross-session queries slow** | Build Parquet/SQLite index (see below) |
| **Large files** | Store frames as external references, not embedded |
| **Learning curve** | ASDF less familiar than FITS to some astronomers |

---

## Index Backends for Fast Queries

ASDF files are the source of truth, but querying across thousands of files is slow. Build a derived index:

### Option 1: Parquet (Best for ML pipelines)

```python
import polars as pl

# Build catalog from ASDF files
def build_catalog(data_dir: Path) -> pl.DataFrame:
    records = []
    for asdf_path in data_dir.glob("**/*.asdf"):
        with asdf.open(asdf_path) as af:
            records.append({
                "session_id": af["meta"]["session_id"],
                "session_type": af["meta"]["session_type"],
                "start_time": af["meta"]["start_time"],
                "target": af["meta"].get("target"),
                "error_count": af["observability"]["metrics"]["errors"],
                "asdf_path": str(asdf_path),
            })
    return pl.DataFrame(records)

# Query: "Find all failed alignment sessions"
catalog = pl.scan_parquet("catalog/*.parquet")
failed = catalog.filter(
    (pl.col("session_type") == "alignment") &
    (pl.col("error_count") > 0)
).collect()

# Open specific ASDF files for details
for row in failed.iter_rows(named=True):
    with asdf.open(row["asdf_path"]) as af:
        print(af["observability"]["logs"])
```

### Option 2: SQLite (Best for simple queries)

```python
import sqlite3

def build_index(data_dir: Path, db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            session_type TEXT,
            start_time TEXT,
            target TEXT,
            error_count INTEGER,
            asdf_path TEXT
        )
    """)
    # ... populate from ASDF files
```

### Option 3: DuckDB (SQL on Parquet)

```python
import duckdb

# Query Parquet files with SQL
result = duckdb.sql("""
    SELECT session_id, target, error_count, asdf_path
    FROM 'catalog/*.parquet'
    WHERE session_type = 'observation'
      AND target LIKE 'M%'
    ORDER BY start_time DESC
    LIMIT 10
""").fetchall()
```

---

## Example: Complete Session Flow

```python
from telescope_mcp.data import SessionManager
from telescope_mcp.drivers import asi_sdk

# Initialize (starts idle session automatically)
sessions = SessionManager(data_dir="/data/telescope")

# System startup logs go to idle session
sessions.log("INFO", "Telescope MCP server started")
sessions.log("INFO", "Cameras initialized", cameras=["ASI482MC", "ASI120MC-S"])

# Start alignment session
alignment = sessions.start_session("alignment", purpose="ra_dec_calibration")
sessions.log("INFO", "Beginning plate solve alignment")

# ... do alignment work ...
spotter_frame = capture_spotter()
main_frames = capture_main_sequence(564)
solve_result = plate_solve(spotter_frame, main_frames[282])

alignment.add_calibration("plate_solve", solve_result)
sessions.log("INFO", "Alignment complete", offset_ra=0.5, offset_dec=-0.3)

# End alignment, creates: alignment_20251214_203000.asdf
alignment_path = sessions.end_session()

# Back to idle session automatically
sessions.log("INFO", "Waiting for observation command")

# Start observation
obs = sessions.start_session(
    "observation",
    target="M31",
    planned_duration_minutes=120,
)
sessions.log("INFO", "Starting M31 observation")

# ... capture frames, telemetry, etc. ...
for i in range(1000):
    frame = capture_frame()
    obs.add_frame("main", frame)
    sessions.log("DEBUG", f"Frame {i} captured", exposure_us=312000)

# End observation, creates: observation_m31_20251214_210000.asdf
obs_path = sessions.end_session()

# Update catalog index
sessions.rebuild_catalog()  # Updates catalog.parquet
```

---

## File Organization

```
/data/telescope/
├── catalog/
│   └── catalog.parquet          # Derived index (rebuildable)
│
├── 2025/
│   └── 12/
│       └── 14/
│           ├── idle_20251214_180000.asdf
│           ├── alignment_20251214_203000.asdf
│           ├── observation_m31_20251214_210000.asdf
│           └── idle_20251214_234500.asdf
│
└── calibration/                  # Shared calibration data
    ├── darks/
    ├── flats/
    └── bias/
```

---

## Summary

| Question | Answer |
|----------|--------|
| Where is truth? | ASDF files |
| What about non-observations? | Everything is a session (idle, alignment, experiment, etc.) |
| How to query across sessions? | Parquet/SQLite/DuckDB index (derived, rebuildable) |
| Real-time logs? | Dual-write: console + ASDF buffer |
| Portability? | ASDF is self-describing; share one file |

**The key abstraction is the session.** Observations are sessions. Alignment is a session. Even "nothing happening" is an idle session. This ensures every log entry has a home.
