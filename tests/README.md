# tests/ Architecture

> **AI Navigation**: Test suite for telescope_mcp package.
> Entry via `pdm run pytest`. 1095 tests, 99% coverage.

| Aspect | Details |
|--------|---------|
| **Purpose** | Unit, integration, and coverage tests for telescope_mcp |
| **Framework** | pytest + pytest-asyncio + pytest-cov |
| **Run Command** | `pdm run pytest -v` |
| **Coverage** | `pdm run pytest --cov=src/telescope_mcp --cov-report=html` |

---

## 1. Overview

The test suite provides comprehensive coverage of all telescope_mcp components:

- **Unit tests**: Individual function/class behavior
- **Integration tests**: Cross-module interactions
- **Coverage gap tests**: Edge cases for 100% coverage
- **Protocol compliance**: Interface contract verification

**Design Patterns**:
- **Fixtures**: Shared mocks in `conftest.py`
- **Helpers**: Protocol assertions in `helpers.py`
- **Isolation**: Each test file targets specific module

**Key Insight**: Tests avoid real hardware via mock injection and digital twin drivers.

---

## 2. Layout

```
tests/
├── __init__.py                    # Package marker
├── conftest.py                    # Shared fixtures (mock_cv2_module)
├── helpers.py                     # Protocol compliance helpers
│
├── test_server_*.py               # server.py tests
│   ├── test_server_coverage.py    # CLI, main, lifecycle
│   └── test_server_comprehensive.py
│
├── test_data_*.py                 # data/ package tests
│   ├── test_data.py               # Quick tests
│   ├── test_data_session.py       # Session state machine
│   └── test_data_session_manager.py
│
├── test_devices_*.py              # devices/ package tests
│   ├── test_devices_exports.py    # Public API
│   └── test_devices_extended.py   # Camera, Registry, Controller
│
├── test_tools*.py                 # tools/ package tests
│   ├── test_tools.py              # All tool modules
│   ├── test_tools_cameras.py      # Camera tools deep
│   └── test_tools_sessions.py     # Session tools deep
│
├── test_sessions*.py              # Session integration
│   ├── test_sessions.py           # Basic
│   ├── test_sessions_extended.py  # Error handling
│   └── test_sessions_mcp_integration.py
│
├── test_web_app.py                # web/ package tests
├── test_utils_*.py                # utils/ package tests
├── test_config*.py                # drivers/config tests
├── test_observability.py          # Logging, stats
├── test_logging_*.py              # Structured logging
├── test_drivers.py                # Driver implementations
├── test_sensor.py                 # Sensor facade
├── test_coverage_gaps.py          # Edge case coverage
├── test_protocol_compliance.py    # Interface contracts
│
└── drivers/                       # Driver-specific tests
    ├── test_asi_sdk.py            # ASI SDK wrapper
    ├── cameras/
    │   ├── test_asi.py            # ASI driver
    │   └── test_twin.py           # Digital twin
    ├── motors/
    │   └── test_serial_controller.py
    └── sensors/
        ├── test_arduino.py        # Arduino driver
        ├── test_twin.py           # Sensor twin
        └── test_types.py          # Type definitions
```

---

## 3. Public Surface

### 3.1 conftest.py — Shared Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `mock_cv2_module` | function | Mocked cv2 for Python 3.13 compat |

### 3.2 helpers.py — Test Utilities

| Function | Signature | Description |
|----------|-----------|-------------|
| `assert_implements_protocol` | `(instance, Protocol, check_signatures=False)` | Verify protocol compliance |
| `assert_all_implement_protocol` | `(list[instance], Protocol)` | Batch protocol check |

---

## 4. Test File → Module Mapping

| Test File | Target Module | Tests |
|-----------|---------------|-------|
| `test_server_coverage.py` | `server.py` | CLI, main, lifecycle |
| `test_server_comprehensive.py` | `server.py` | Integration |
| `test_data_session.py` | `data/session.py` | Session state |
| `test_data_session_manager.py` | `data/session_manager.py` | Manager |
| `test_devices_exports.py` | `devices/__init__.py` | Public API |
| `test_devices_extended.py` | `devices/*.py` | Camera, Registry |
| `test_tools.py` | `tools/*.py` | All tools |
| `test_tools_cameras.py` | `tools/cameras.py` | Camera tools |
| `test_tools_sessions.py` | `tools/sessions.py` | Session tools |
| `test_sessions_mcp_integration.py` | `tools/sessions.py` | MCP dispatch |
| `test_web_app.py` | `web/app.py` | FastAPI routes |
| `test_utils_image.py` | `utils/image.py` | ImageEncoder |
| `test_utils_init.py` | `utils/__init__.py` | Lazy imports |
| `test_config_comprehensive.py` | `drivers/config.py` | DriverFactory |
| `test_observability.py` | `observability/*.py` | Logging, stats |
| `test_logging_comprehensive.py` | `observability/logging.py` | Structured logs |
| `test_drivers.py` | `drivers/*.py` | Driver impls |
| `test_sensor.py` | `devices/sensor.py` | Sensor facade |
| `test_coverage_gaps.py` | Various | Edge cases |
| `test_protocol_compliance.py` | Various | Protocols |
| `drivers/test_asi_sdk.py` | `drivers/asi_sdk/` | SDK wrapper |
| `drivers/cameras/test_asi.py` | `drivers/cameras/asi.py` | ASI driver |
| `drivers/cameras/test_twin.py` | `drivers/cameras/twin.py` | Twin driver |
| `drivers/motors/test_serial_controller.py` | `drivers/motors/` | Motor serial |
| `drivers/sensors/test_arduino.py` | `drivers/sensors/arduino.py` | Arduino |
| `drivers/sensors/test_twin.py` | `drivers/sensors/twin.py` | Sensor twin |
| `drivers/sensors/test_types.py` | `drivers/sensors/types.py` | Types |

---

## 5. Running Tests

### 5.1 Full Suite

```bash
pdm run pytest -v
```

### 5.2 With Coverage

```bash
pdm run pytest --cov=src/telescope_mcp --cov-report=html --cov-branch
```

### 5.3 Specific Module

```bash
# Single file
pdm run pytest tests/test_server_coverage.py -v

# By pattern
pdm run pytest tests/test_devices_*.py -v

# By marker
pdm run pytest -m "not slow" -v
```

### 5.4 Coverage for Specific Package

```bash
pdm run pytest tests/test_tools*.py \
  --cov=src/telescope_mcp/tools \
  --cov-report=term-missing
```

---

## 6. Test Conventions

### 6.1 Naming

| Pattern | Meaning |
|---------|---------|
| `test_<module>.py` | Tests for `<module>.py` |
| `test_<module>_*.py` | Extended/specialized tests |
| `Test<Class>` | Test class grouping |
| `test_<method>_<scenario>` | Test method |

### 6.2 Fixtures

```python
@pytest.fixture
def mock_registry():
    """Provide mocked CameraRegistry."""
    registry = MagicMock(spec=CameraRegistry)
    registry.discover.return_value = {}
    return registry
```

### 6.3 Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_function()
    assert result == expected
```

### 6.4 Mocking

```python
from unittest.mock import patch, MagicMock

def test_with_mock():
    with patch("telescope_mcp.module.dependency") as mock:
        mock.return_value = expected_value
        result = function_under_test()
        assert result == expected
```

---

## 7. Coverage Configuration

From `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["telescope_mcp"]
branch = true
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    # Hardware SDK paths excluded
]
```

---

## 8. AI Accessibility Map

| Task | Target | Notes |
|------|--------|-------|
| Add test for new function | Matching `test_<module>.py` | Follow naming convention |
| Add test fixture | `conftest.py` | Shared across tests |
| Add protocol assertion | `helpers.py` | Use `assert_implements_protocol` |
| Test new tool | `test_tools*.py` | Check TOOLS + call_tool |
| Test new driver | `tests/drivers/` subdirectory | Match package structure |
| Increase coverage | `test_coverage_gaps.py` | Edge cases |
| Test MCP integration | `test_sessions_mcp_integration.py` | Full dispatch |

---

## 9. Test Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 1095 |
| Coverage | 99% |
| Statements | 2895 |
| Branches | 554 |
| Test Files | 30+ |
| Avg Runtime | ~13s |

---

## 10. Key Test Classes by Domain

### Server Tests
| Class | File | Focus |
|-------|------|-------|
| `TestCreateServerCoverage` | `test_server_coverage.py` | Server factory |
| `TestMainCoverage` | `test_server_coverage.py` | CLI entry |
| `TestRunServerCoverage` | `test_server_coverage.py` | Lifecycle |
| `TestParseArgsCoverage` | `test_server_coverage.py` | Arguments |

### Data Tests
| Class | File | Focus |
|-------|------|-------|
| `TestSessionInitialization` | `test_data_session.py` | Session create |
| `TestSessionLog` | `test_data_session.py` | Logging |
| `TestSessionClose` | `test_data_session.py` | ASDF write |
| `TestSessionManagerInit` | `test_data_session_manager.py` | Manager |

### Device Tests
| Class | File | Focus |
|-------|------|-------|
| `TestCameraCapture` | `test_devices_extended.py` | Capture |
| `TestCameraRegistry` | `test_devices_extended.py` | Discovery |
| `TestCameraController` | `test_devices_extended.py` | Multi-cam |

### Tool Tests
| Class | File | Focus |
|-------|------|-------|
| `TestCameraTools` | `test_tools.py` | All camera tools |
| `TestCameraToolDispatcher` | `test_tools.py` | MCP routing |
| `TestMCPCallToolDispatcher` | `test_sessions_mcp_integration.py` | Session routing |

### Driver Tests
| Class | File | Focus |
|-------|------|-------|
| `TestDigitalTwinCameraDriver` | `test_drivers.py` | Twin driver |
| `TestDigitalTwinSensorDriver` | `test_drivers.py` | Sensor twin |
| `TestArduinoSensorDriver` | `drivers/sensors/test_arduino.py` | Arduino |
