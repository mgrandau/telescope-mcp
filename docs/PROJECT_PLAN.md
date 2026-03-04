# Project Plan — telescope-mcp

This is a **historical record** of what was actually built, when, and why. For the philosophy and design intent behind this project, see [🧭 Intent](../README.md#-intent) in the README.

Current state: **v0.1.0** — active development. Cameras fully operational, motors partially wired, position sensing in progress. 99% test coverage.

---

## Phase 1: Foundation & Camera Control (2025-12-13 → 2025-12-18)

**Goal:** Port the telescope from Jupyter notebooks to a proper MCP server — AI agents should be able to discover and control cameras through structured tool calls, not cell-by-cell execution.

Built the entire core architecture in one week: MCP server, device/driver abstraction with DI, camera registry, digital twin, web dashboard with MJPEG streaming, session management with ASDF storage, structured logging, and full test suite.

| Date | Work |
| ---- | ---- |
| 2025-12-13 | Initial commit, motor control, camera tools, position sensing |
| 2025-12-14 | ZWO ASI Camera SDK integration |
| 2025-12-16 | SessionManager for ASDF data storage |
| 2025-12-17 | Layered camera architecture, CameraController, ASI driver, digital twin, observability |
| 2025-12-18 | Web dashboard, documentation, session tool tests |

**Key decisions:**

- MCP over REST API — AI agents call tools natively, no HTTP construction
- Dual control paths — MCP (AI) and HTTP (human) share the same device layer
- Driver injection — `--mode hardware` or `--mode digital_twin` at startup
- Protocol-based DI — `CameraDriver`, `SensorDriver`, `MotorDriver` as Protocols
- ASDF for session storage — handles numpy arrays natively
- Architecture READMEs in every subpackage (9 READMEs total)

**Risk posture:** Medium — hardware integration (USB cameras, SDK bindings) with real physical consequences (motors move things). Digital twin mode derisks development but hardware mode needs careful testing.

**Design discussions (journal):**

- [2025-12-13](journal/2025-12-13.md) — Origin story: MCP over REST (rejected context-switching for AI), dual control paths (rejected MCP-wrapping-HTTP), driver injection (digital twin as first-class runtime, not just test mock), ASDF over JSON/SQLite, AI-optimized documentation philosophy

---

## Phase 2: Hardware Integration (2025-12-19 → 2026-01-04)

**Goal:** Get real hardware working — motors, sensors, and cameras on the physical telescope, not just simulation.

Extended from pure software into physical device control: motor serial protocol, Arduino IMU sensor driver, camera fine-tuning, and extensive hardware debugging through Jupyter notebooks.

| Date | Work | Issues |
| ---- | ---- | ------ |
| 2025-12-19 | Refactoring and motor test improvements | — |
| 2025-12-23–24 | Hardware mode config, dashboard server, numpy version | — |
| 2025-12-26 | Sensor driver tests, dashboard localhost binding | — |
| 2025-12-28–29 | Sensor interfaces, comprehensive tests, camera driver docs | — |
| 2025-12-30 | Web dashboard tests, streaming tests, 100% coverage push | — |
| 2025-12-31 | Coordinate utilities, RAW capture, CoordinateProvider | — |
| 2026-01-01 | 100% coverage on arduino.py, camera.py, web/app.py | — |
| 2026-01-03–04 | Motor control notebooks, Teensy serial, digital twin motor | — |

**Intent evolution:** The digital twin proved more valuable than expected. Originally just a testing aid, it became the primary development mode — you could build and test the full MCP tool chain, web dashboard, and session management without the telescope. This shifted the architecture to treat the twin as a first-class runtime, not a mock.

**Key decisions:**

- Arduino Nano BLE33 Sense for IMU (accelerometer/gyroscope for alt/az)
- Serial protocol for Teensy motor controller (custom command format)
- CoordinateProvider for automatic telescope position injection
- Per-axis motor configuration (inversion flag for altitude)
- 100% coverage target on all non-driver modules

**Risk posture:** High — serial communication with motor controllers, USB camera bandwidth management, IMU calibration. Physical hardware failures can't be caught by tests alone. The dual-camera USB bandwidth issue (#5) was discovered only on real hardware.

---

## Phase 3: Dashboard & Polish (2026-01-31 → 2026-02-16)

**Goal:** Make the telescope usable for real observing sessions — motor controls on the dashboard, camera configuration, sensor display, and reliability fixes.

Focused on the web dashboard becoming a real control surface: wiring motors and sensors to the UI, fixing dual-camera streaming crashes, adding configurable camera defaults, and sensor smoothing.

| Date | Work | Issues |
| ---- | ---- | ------ |
| 2026-01-31 | Localhost binding fix | — |
| 2026-02-14 | Focusing procedure, debugging scripts, CLI install, camera config | #1, #2, #5 |
| 2026-02-15 | Motor zero position, azimuth dashboard, direction fix, sensor fix | #3, #4, #6, #10 |
| 2026-02-16 | Arduino serial fix, dashboard refresh sync, rolling average | #8, #11 |

**Issues resolved:** #1 (remove per-camera controls), #2 (configurable gain/exposure), #3 (altitude motor to dashboard), #4 (set home button), #5 (dual camera streaming), #6 (azimuth to dashboard), #8 (sensor display), #10 (altitude direction), #11 (dashboard refresh sync)

**Intent evolution:** The web dashboard evolved from a monitoring-only view to a full manual control surface. Originally, the intent was "AI controls the telescope, human watches." In practice, you need manual controls for setup, focusing, and quick adjustments — the AI handles the planned observing sequences, the human handles the ad-hoc stuff. Both through the same device layer.

**Key decisions:**

- USB bandwidth fixed at 40% per camera (was crashing with dynamic allocation)
- Rolling average (5-reading window) for sensor smoothing — eliminates ±0.5° IMU jitter
- Dashboard refresh rate synced to sensor read cadence (5s, not 500ms)
- Motor direction configurable per-axis via inversion flag

**Risk posture:** High — changes to motor control and camera streaming affect physical hardware behavior. The dual-camera crash (#5) required real-hardware debugging that tests couldn't simulate.

---

## Phase 4: Code Quality Audit (2026-02-20)

**Goal:** Community readiness — Discord link, code quality assessment for open issues.

Added community link and ran a systematic code audit that identified 17 findings (F1–F17), filed as issues #12–#28.

| Date | Work |
| ---- | ---- |
| 2026-02-20 | Discord community link, code quality audit |

**Issues filed:** #12–#28 (17 findings from automated code analysis)

**Key categories:**
- Security: path traversal in set_data_dir (#12), RA auto-conversion heuristic (#13)
- Architecture: web/app.py god-file (#17), thread-unsafe singletons (#16), unregistered tools (#15)
- Code quality: duplicated constants (#18, #19), TypedDict casing (#25), docstring verbosity (#28)

**Risk posture:** Low — audit and documentation only, no code changes.

---

## Version History

| Version | Date | Highlights |
| ------- | ---- | ---------- |
| **v0.1.0** | **2025-12-13** | **Initial release — MCP server, cameras, motors, sensors, web dashboard, digital twin, 99% coverage** |

No tagged releases yet. Single version under active development.

---

## Roadmap

### Open Issues (17)

**Security & Correctness:**

| Issue | Description |
| ----- | ----------- |
| [#12](https://github.com/mgrandau/telescope-mcp/issues/12) | Path traversal risk in set_data_dir |
| [#13](https://github.com/mgrandau/telescope-mcp/issues/13) | RA auto-conversion silently corrupts values 0-24° |

**Architecture:**

| Issue | Description |
| ----- | ----------- |
| [#15](https://github.com/mgrandau/telescope-mcp/issues/15) | Position/session tools defined but never registered |
| [#16](https://github.com/mgrandau/telescope-mcp/issues/16) | Thread-unsafe global singletons in config |
| [#17](https://github.com/mgrandau/telescope-mcp/issues/17) | web/app.py is a 2492-line god-file |

**Code Quality:**

| Issue | Description |
| ----- | ----------- |
| [#14](https://github.com/mgrandau/telescope-mcp/issues/14) | ASI capture polling timeout |
| [#18](https://github.com/mgrandau/telescope-mcp/issues/18)–[#28](https://github.com/mgrandau/telescope-mcp/issues/28) | Constants duplication, thread safety, docstring verbosity, etc. |

**Features:**

| Issue | Description |
| ----- | ----------- |
| [#7](https://github.com/mgrandau/telescope-mcp/issues/7) | Plate-solving notebook via Astrometry.net |
| [#9](https://github.com/mgrandau/telescope-mcp/issues/9) | Sensor mounting offset calibration |

### TODO.md Phases (Remaining)

The project has a detailed [TODO.md](../TODO.md) with 7 phases. Phases 1 (cameras) and most of Phase 5 (web dashboard) are complete. Key remaining work:

- **Phase 2:** Motor control serial driver (partially complete)
- **Phase 3:** Position sensing calibration and persistence
- **Phase 4:** Star alignment and goto (coordinate conversion)
- **Phase 6:** Configuration and persistence layer
- **Phase 7:** Documentation and hardware-in-loop testing

### Future Ideas (from TODO.md)

- **Virtual Star Parties** — relay MJPEG streams to YouTube Live / Discord for remote observing
- **Meteor Shower Monitoring** — wide-field finder camera for automated streak detection
