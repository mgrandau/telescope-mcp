"""Utility modules for telescope-mcp.

This package uses lazy imports via __getattr__ to defer cv2 loading until
classes are actually accessed. This avoids Python 3.13 cv2.typing import
errors when the package is imported but cv2 classes aren't used.

Available exports (lazy-loaded):
    ImageEncoder: Protocol for image encoding operations
    CV2ImageEncoder: OpenCV-based implementation

Example:
    from telescope_mcp.utils import ImageEncoder, CV2ImageEncoder
    encoder = CV2ImageEncoder()
"""

__all__ = ["ImageEncoder", "CV2ImageEncoder"]


def __getattr__(name: str) -> type:
    """Lazy import to avoid cv2 issues at package import time.

    Imports are cached in module globals after first access to avoid
    repeated import overhead. This pattern defers cv2 loading until
    an encoder is actually needed.

    Business context: Python 3.13 has cv2.typing import issues that crash
    on import even if cv2 classes aren't used. Lazy loading allows tests
    and code that don't need image encoding to import this package safely.

    Args:
        name: Attribute name being accessed.

    Returns:
        The requested class (ImageEncoder or CV2ImageEncoder).

    Raises:
        AttributeError: If name is not a public export.

    Example:
        >>> from telescope_mcp import utils
        >>> # cv2 not loaded yet
        >>> encoder_class = utils.CV2ImageEncoder  # triggers lazy import
        >>> # cv2 now loaded, encoder_class is the real class
    """
    if name in ("ImageEncoder", "CV2ImageEncoder"):
        from telescope_mcp.utils.image import CV2ImageEncoder, ImageEncoder

        # Cache in module globals to avoid re-import on subsequent access
        globals()["ImageEncoder"] = ImageEncoder
        globals()["CV2ImageEncoder"] = CV2ImageEncoder
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public API for introspection and autocomplete.

    Provides complete list of module exports for IDE autocomplete and
    introspection tools (dir(), help()). Includes lazy-loaded classes
    that haven't been accessed yet plus standard module attributes.

    Business context: Enables IDE code completion to show ImageEncoder
    and CV2ImageEncoder even before they're imported. Critical for
    developer experience with lazy import pattern.

    Args:
        None.

    Returns:
        List of public attribute names: ['ImageEncoder', 'CV2ImageEncoder',
        '__all__', '__doc__', '__name__', '__file__'].

    Raises:
        None. Always succeeds.

    Example:
        >>> import telescope_mcp.utils as utils
        >>> 'ImageEncoder' in dir(utils)
        True
        >>> 'CV2ImageEncoder' in dir(utils)
        True
    """
    return [*__all__, "__all__", "__doc__", "__name__", "__file__"]
