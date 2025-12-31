"""Unit tests for telescope_mcp.utils package __init__.py.

Tests the lazy import mechanism and module introspection.
"""

import sys

import pytest


class TestUtilsLazyImports:
    """Tests for lazy import mechanism via __getattr__."""

    def test_getattr_returns_image_encoder(self) -> None:
        """Verifies ImageEncoder Protocol accessible via lazy __getattr__.

        Tests that module-level __getattr__ returns ImageEncoder when
        not yet cached in module globals.

        Business context:
        Lazy import allows package import without cv2 dependency. Tests
        and code that don't need encoding can safely import utils.

        Arrangement:
        1. Clear ImageEncoder from module globals if present.
        2. Forces __getattr__ path on next access.

        Action:
        Import ImageEncoder from telescope_mcp.utils.

        Assertion Strategy:
        Confirms returned object has Protocol attributes (_is_protocol
        or __protocol_attrs__).

        Testing Principle:
        Validates lazy import mechanism, ensuring deferred loading.
        """
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
        """Verifies CV2ImageEncoder implementation accessible via lazy import.

        Tests that __getattr__ returns the concrete encoder class.

        Business context:
        CV2ImageEncoder is the production encoder. Lazy access allows
        conditional use - only loaded when actually instantiated.

        Arrangement:
        1. Clear CV2ImageEncoder from module globals if present.

        Action:
        Import CV2ImageEncoder from telescope_mcp.utils.

        Assertion Strategy:
        Confirms returned object is callable (can be instantiated).

        Testing Principle:
        Validates lazy import mechanism for implementation class.
        """
        if "telescope_mcp.utils" in sys.modules:
            utils_module = sys.modules["telescope_mcp.utils"]
            if "CV2ImageEncoder" in vars(utils_module):
                delattr(utils_module, "CV2ImageEncoder")

        from telescope_mcp.utils import CV2ImageEncoder

        # Should be the implementation class
        assert callable(CV2ImageEncoder)

    def test_getattr_caches_imports_in_globals(self) -> None:
        """Verifies lazy imports are cached in module globals after first access.

        Tests that subsequent access skips __getattr__ overhead.

        Business context:
        Caching avoids repeated import overhead. After first access,
        attribute lookup is O(1) dict access instead of import machinery.

        Arrangement:
        1. Clear both ImageEncoder and CV2ImageEncoder from globals.

        Action:
        Access both attributes to trigger lazy import.

        Assertion Strategy:
        Confirms both names present in vars(module) after access.

        Testing Principle:
        Validates caching optimization, ensuring efficient repeated access.
        """
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
        """Verifies AttributeError raised for undefined attributes.

        Tests standard module behavior for missing attributes.

        Business context:
        Clear errors help catch typos and API misuse. AttributeError
        is the standard Python exception for missing attributes.

        Arrangement:
        1. Import utils module.

        Action:
        Access undefined attribute 'foo'.

        Assertion Strategy:
        Confirms AttributeError raised with module name in message.

        Testing Principle:
        Validates error handling, ensuring Pythonic behavior.
        """
        import telescope_mcp.utils as utils

        with pytest.raises(
            AttributeError, match=r"module 'telescope_mcp.utils' has no attribute 'foo'"
        ):
            _ = utils.foo  # type: ignore[attr-defined]

    def test_getattr_raises_for_private_names(self) -> None:
        """Verifies AttributeError raised for private attribute access.

        Tests that underscore-prefixed names are not exposed.

        Business context:
        Private attributes should not be accessible via __getattr__.
        Protects internal implementation from external use.

        Arrangement:
        1. Import utils module.

        Action:
        Access '_private' attribute.

        Assertion Strategy:
        Confirms AttributeError raised (not in __all__).

        Testing Principle:
        Validates encapsulation, ensuring API boundaries.
        """
        import telescope_mcp.utils as utils

        with pytest.raises(AttributeError):
            _ = utils._private  # type: ignore[attr-defined]


class TestUtilsDir:
    """Tests for __dir__ introspection."""

    def test_dir_returns_public_api(self) -> None:
        """Verifies __dir__ includes lazy-loaded public exports.

        Tests that dir(utils) shows all public API including not-yet-loaded.

        Business context:
        IDE autocomplete relies on __dir__. Users should see ImageEncoder
        and CV2ImageEncoder in completion lists before first access.

        Arrangement:
        1. Import utils module.

        Action:
        Call dir(utils) to get attribute list.

        Assertion Strategy:
        Confirms both ImageEncoder and CV2ImageEncoder in result.

        Testing Principle:
        Validates discoverability, ensuring IDE support.
        """
        import telescope_mcp.utils as utils

        result = dir(utils)

        assert "ImageEncoder" in result
        assert "CV2ImageEncoder" in result

    def test_dir_includes_standard_attributes(self) -> None:
        """Verifies __dir__ includes standard module attributes.

        Tests that __all__, __doc__, __name__, __file__ are listed.

        Business context:
        Standard attributes expected in dir() output for any module.
        Ensures utils behaves like a normal Python module.

        Arrangement:
        1. Import utils module.

        Action:
        Call dir(utils).

        Assertion Strategy:
        Confirms __all__, __doc__, __name__, __file__ all present.

        Testing Principle:
        Validates module completeness, ensuring standard behavior.
        """
        import telescope_mcp.utils as utils

        result = dir(utils)

        assert "__all__" in result
        assert "__doc__" in result
        assert "__name__" in result
        assert "__file__" in result

    def test_dir_returns_list(self) -> None:
        """Verifies __dir__ returns a list type.

        Tests return type contract of __dir__ magic method.

        Business context:
        Python expects __dir__ to return a list. Other return types
        may break introspection tools.

        Arrangement:
        1. Import utils module.

        Action:
        Call utils.__dir__() directly.

        Assertion Strategy:
        Confirms result isinstance(list) and has items.

        Testing Principle:
        Validates type contract, ensuring Python compatibility.
        """
        import telescope_mcp.utils as utils

        result = utils.__dir__()

        assert isinstance(result, list)
        assert len(result) > 0


class TestUtilsAllExport:
    """Tests for __all__ export list."""

    def test_all_contains_expected_exports(self) -> None:
        """Verifies __all__ lists the public API.

        Tests that exported names match expected public interface.

        Business context:
        __all__ controls 'from utils import *' behavior and documents
        the public API contract.

        Arrangement:
        1. Import utils module.

        Action:
        Inspect utils.__all__.

        Assertion Strategy:
        Confirms ImageEncoder and CV2ImageEncoder in __all__.

        Testing Principle:
        Validates API documentation, ensuring export visibility.
        """
        import telescope_mcp.utils as utils

        assert "ImageEncoder" in utils.__all__
        assert "CV2ImageEncoder" in utils.__all__

    def test_all_matches_lazy_imports(self) -> None:
        """Verifies every __all__ export is accessible via getattr.

        Tests that __all__ and __getattr__ are synchronized.

        Business context:
        Mismatch between __all__ and actual exports causes import
        failures. This catches drift between documentation and code.

        Arrangement:
        1. Import utils module.

        Action:
        Iterate __all__ and getattr each name.

        Assertion Strategy:
        Confirms all names resolve to non-None objects.

        Testing Principle:
        Validates consistency, ensuring __all__ is accurate.
        """
        import telescope_mcp.utils as utils

        # Every item in __all__ should be importable
        for name in utils.__all__:
            obj = getattr(utils, name)
            assert obj is not None
