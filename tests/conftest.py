"""Pytest configuration and fixtures for telescope-mcp tests.

This module provides test fixtures and configuration that apply across
all test modules. It handles mocking of hardware dependencies like cv2
to enable testing on systems without physical camera hardware.
"""

import sys
from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """Configure pytest before test collection begins.

    Hook called by pytest during initialization, before any tests
    are collected or run. Currently a no-op placeholder.

    Business context:
    cv2 mocking was previously done globally here but moved to per-test
    fixtures because driver tests need real cv2.imencode for JPEG validation.
    This hook remains for future global configuration needs.

    Args:
        config: Pytest configuration object with CLI args and settings.

    Returns:
        None.

    Raises:
        None.

    Example:
        # Automatically called by pytest - not called directly.
    """
    pass


@pytest.fixture
def mock_cv2_module():
    """Provide a mocked cv2 module for tests that need it.

    This fixture temporarily replaces cv2 in sys.modules with a mock.
    Used by web/app tests that can't use real cv2 due to Python 3.13
    compatibility issues with cv2.typing.

    Business context:
    Python 3.13 has cv2.typing import crashes. This fixture enables
    testing image-related functionality without loading the real cv2
    module, ensuring tests run reliably in CI environments.

    Args:
        None (pytest fixture with implicit request parameter).

    Yields:
        MagicMock: The mocked cv2 module with version="4.12.0",
            common constants (FONT_HERSHEY_SIMPLEX, IMWRITE_JPEG_QUALITY),
            and stub functions (imencode, putText).

    Raises:
        None. Fixture handles cleanup gracefully.

    Example:
        >>> def test_with_mock_cv2(mock_cv2_module):
        ...     assert mock_cv2_module.__version__ == "4.12.0"
        ...     result, _ = mock_cv2_module.imencode(".jpg", img)
        ...     assert result is True
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
