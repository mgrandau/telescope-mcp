"""Unit tests for telescope_mcp.utils package __init__.py.

Tests the lazy import mechanism and module introspection.
"""

import sys

import pytest


class TestUtilsLazyImports:
    """Tests for lazy import mechanism via __getattr__."""

    def test_getattr_returns_image_encoder(self) -> None:
        """Verify ImageEncoder can be accessed via lazy import."""
        # Force fresh import to test __getattr__
        if "telescope_mcp.utils" in sys.modules:
            # Clear cached globals to force __getattr__ path
            utils_module = sys.modules["telescope_mcp.utils"]
            if "ImageEncoder" in vars(utils_module):
                delattr(utils_module, "ImageEncoder")

        from telescope_mcp.utils import ImageEncoder

        # Should be the Protocol class
        assert hasattr(ImageEncoder, "__protocol_attrs__") or hasattr(
            ImageEncoder, "_is_protocol"
        )

    def test_getattr_returns_cv2_image_encoder(self) -> None:
        """Verify CV2ImageEncoder can be accessed via lazy import."""
        if "telescope_mcp.utils" in sys.modules:
            utils_module = sys.modules["telescope_mcp.utils"]
            if "CV2ImageEncoder" in vars(utils_module):
                delattr(utils_module, "CV2ImageEncoder")

        from telescope_mcp.utils import CV2ImageEncoder

        # Should be the implementation class
        assert callable(CV2ImageEncoder)

    def test_getattr_caches_imports_in_globals(self) -> None:
        """Verify imports are cached after first access."""
        # Clear cache
        if "telescope_mcp.utils" in sys.modules:
            utils_module = sys.modules["telescope_mcp.utils"]
            for name in ("ImageEncoder", "CV2ImageEncoder"):
                if name in vars(utils_module):
                    delattr(utils_module, name)

        import telescope_mcp.utils as utils

        # First access triggers __getattr__
        _ = utils.ImageEncoder
        _ = utils.CV2ImageEncoder

        # Now they should be in module globals (cached)
        assert "ImageEncoder" in vars(utils)
        assert "CV2ImageEncoder" in vars(utils)

    def test_getattr_raises_attribute_error_for_unknown(self) -> None:
        """Verify AttributeError raised for unknown attributes."""
        import telescope_mcp.utils as utils

        with pytest.raises(
            AttributeError, match=r"module 'telescope_mcp.utils' has no attribute 'foo'"
        ):
            _ = utils.foo  # type: ignore[attr-defined]

    def test_getattr_raises_for_private_names(self) -> None:
        """Verify private names raise AttributeError."""
        import telescope_mcp.utils as utils

        with pytest.raises(AttributeError):
            _ = utils._private  # type: ignore[attr-defined]


class TestUtilsDir:
    """Tests for __dir__ introspection."""

    def test_dir_returns_public_api(self) -> None:
        """Verify __dir__ includes lazy-loaded exports."""
        import telescope_mcp.utils as utils

        result = dir(utils)

        assert "ImageEncoder" in result
        assert "CV2ImageEncoder" in result

    def test_dir_includes_standard_attributes(self) -> None:
        """Verify __dir__ includes standard module attributes."""
        import telescope_mcp.utils as utils

        result = dir(utils)

        assert "__all__" in result
        assert "__doc__" in result
        assert "__name__" in result
        assert "__file__" in result

    def test_dir_returns_list(self) -> None:
        """Verify __dir__ returns a list."""
        import telescope_mcp.utils as utils

        result = utils.__dir__()

        assert isinstance(result, list)
        assert len(result) > 0


class TestUtilsAllExport:
    """Tests for __all__ export list."""

    def test_all_contains_expected_exports(self) -> None:
        """Verify __all__ lists public API."""
        import telescope_mcp.utils as utils

        assert "ImageEncoder" in utils.__all__
        assert "CV2ImageEncoder" in utils.__all__

    def test_all_matches_lazy_imports(self) -> None:
        """Verify __all__ exports match lazy import capabilities."""
        import telescope_mcp.utils as utils

        # Every item in __all__ should be importable
        for name in utils.__all__:
            obj = getattr(utils, name)
            assert obj is not None
