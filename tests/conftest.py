"""Pytest configuration and fixtures for telescope-mcp tests.

This module provides test fixtures and configuration that apply across
all test modules. It handles mocking of hardware dependencies like cv2
to enable testing on systems without physical camera hardware.
"""

import sys
from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """Configure pytest before test collection.

    Note: cv2 mocking is now done per-test-file basis via fixtures.
    The global mock was removed because driver tests need real cv2.imencode
    for JPEG generation testing.

    Args:
        config: Pytest configuration object.
    """
    pass


@pytest.fixture
def mock_cv2_module():
    """Provide a mocked cv2 module for tests that need it.

    This fixture temporarily replaces cv2 in sys.modules with a mock.
    Used by web/app tests that can't use real cv2 due to Python 3.13
    compatibility issues with cv2.typing.

    Yields:
        MagicMock: The mocked cv2 module.
    """
    original_cv2 = sys.modules.get("cv2")

    mock_cv2 = MagicMock()
    mock_cv2.__version__ = "4.12.0"
    mock_cv2.FONT_HERSHEY_SIMPLEX = 0
    mock_cv2.IMWRITE_JPEG_QUALITY = 1
    mock_cv2.imencode = MagicMock(
        return_value=(True, MagicMock(tobytes=lambda: b"\xff\xd8test_jpeg"))
    )
    mock_cv2.putText = MagicMock()

    sys.modules["cv2"] = mock_cv2
    yield mock_cv2

    # Restore original
    if original_cv2 is not None:
        sys.modules["cv2"] = original_cv2
    else:
        del sys.modules["cv2"]
