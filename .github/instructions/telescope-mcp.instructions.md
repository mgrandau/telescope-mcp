---
applyTo: '**'
---
# telescope-mcp Project Rules

## Python
- Use `python3` not `python`
- Run via PDM: `pdm run python3 ...` or `pdm run pytest`
- Install: `pdm add <pkg>` (not pip)
- Dev deps: `pdm add -dG dev <pkg>`

## MCP Config
When defining settings in server.py or config:
- Show ALL options with defaults, even if disabled
- Comment disabled options, don't omit them
- Users need visibility into available configuration

## Stack
- Python 3.13+ | PDM | FastAPI | MCP
- Tests: pytest + pytest-asyncio
- Lint: ruff | Type: mypy (strict)
- Data: ASDF format for sessions

## Architecture
- `devices/` = logical abstractions (Camera, CameraController, CameraRegistry)
- `drivers/` = hardware implementations (ASI, DigitalTwin)
- `tools/` = MCP tool definitions
- Inject drivers into devices (DI pattern)
