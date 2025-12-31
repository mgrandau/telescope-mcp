"""Image encoding abstractions for dependency injection.

This module provides a Protocol-based interface for image encoding operations,
allowing cv2 to be mocked in tests without sys.modules manipulation. The
CV2ImageEncoder provides the real implementation using OpenCV.

Usage:
    # Production (default)
    encoder = CV2ImageEncoder()
    jpeg_bytes = encoder.encode_jpeg(img, quality=85)

    # Testing
    class MockEncoder:
        def encode_jpeg(self, img, quality=85):
            return b'\xff\xd8mock_jpeg'
    encoder = MockEncoder()

Architecture:
    ImageEncoder (Protocol) <- CV2ImageEncoder (real)
                            <- MockImageEncoder (tests)

The lazy import pattern in CV2ImageEncoder ensures cv2 is only imported
when the encoder is instantiated, not at module import time. This enables
tests to provide mock encoders without cv2 import issues on Python 3.13.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ImageEncoder", "CV2ImageEncoder"]


@runtime_checkable
class ImageEncoder(Protocol):
    """Protocol defining image encoding operations.

    This protocol abstracts OpenCV's image encoding functionality,
    enabling dependency injection for testability. Implementations
    must provide JPEG encoding and text overlay capabilities.

    The protocol is used by the web application's streaming generator
    to encode camera frames as JPEG and render error messages.

    Example:
        >>> class MockEncoder:
        ...     def encode_jpeg(self, img, quality=85):
        ...         return b'\xff\xd8test'
        ...     def put_text(self, img, text, position, scale, color, thickness):
        ...         pass  # No-op for tests
        >>> encoder: ImageEncoder = MockEncoder()
        >>> encoder.encode_jpeg(np.zeros((100, 100), dtype=np.uint8))
        b'\xff\xd8test'
    """

    def encode_jpeg(self, img: NDArray[Any], quality: int = 85) -> bytes:
        """Encode image array as JPEG bytes.

        Business context: Core encoding operation for MJPEG camera streams.
        Each captured frame is encoded to JPEG and yielded to the streaming
        response. Quality/size tradeoff affects bandwidth and latency.

        Args:
            img: NumPy array containing image data (uint8 or uint16).
                Can be grayscale (H, W) or color (H, W, 3) in BGR format.
            quality: JPEG quality 1-100. Higher = better quality,
                larger file. Default 85 is good for streaming.

        Returns:
            JPEG-encoded bytes starting with 0xFFD8 magic bytes.

        Raises:
            ValueError: If quality not in 1-100 range or encoding fails.

        Example:
            >>> encoder: ImageEncoder = CV2ImageEncoder()
            >>> img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            >>> jpeg = encoder.encode_jpeg(img, quality=85)
            >>> jpeg[:2]  # JPEG magic bytes
            b'\\xff\\xd8'
        """
        ...  # pragma: no cover

    def put_text(
        self,
        img: NDArray[Any],
        text: str,
        position: tuple[int, int],
        scale: float,
        color: int | tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw text on image (modifies in place).

        Uses FONT_HERSHEY_SIMPLEX font. For grayscale images, color
        should be an int 0-255. For BGR images, use (B, G, R) tuple.

        Business context: Renders error messages and status text on camera
        stream frames. When capture fails, black placeholder frames with
        error text keep operators informed rather than showing frozen streams.

        Args:
            img: Image array to draw on (modified in place).
            text: Text string to render.
            position: (x, y) position for text baseline start.
            scale: Font scale factor (1.0 = base size).
            color: Text color (int for grayscale, BGR tuple for color).
            thickness: Line thickness in pixels.

        Returns:
            None. Image is modified in place.

        Raises:
            cv2.error: If image dtype/shape incompatible with text rendering.

        Example:
            >>> encoder: ImageEncoder = get_encoder()
            >>> error_frame = np.zeros((480, 640), dtype=np.uint8)
            >>> encoder.put_text(error_frame, "No camera", (100, 240), 1.0, 255, 2)
        """
        ...  # pragma: no cover


class CV2ImageEncoder(ImageEncoder):
    """OpenCV-based image encoder implementation.

    Real implementation using cv2.imencode and cv2.putText. The cv2
    import is deferred to __init__ to allow tests to provide mock
    encoders without triggering cv2 import issues.

    Thread Safety:
        cv2 functions are generally thread-safe for encoding operations.
        Multiple encoders can be used concurrently.

    Example:
        >>> encoder = CV2ImageEncoder()
        >>> img = np.zeros((480, 640), dtype=np.uint8)
        >>> jpeg = encoder.encode_jpeg(img)
        >>> jpeg[:2]
        b'\xff\xd8'
    """

    def __init__(self) -> None:
        """Initialize encoder with lazy cv2 import.

        Imports cv2 on first instantiation. This allows test files
        to avoid importing cv2 entirely by providing mock encoders.

        Business context: The cv2 import is deferred to avoid Python 3.13
        compatibility issues where cv2.typing crashes on import. Tests
        can provide mock encoders without ever loading cv2.

        Args:
            None.

        Returns:
            None. Initializes self._cv2 with imported module.

        Raises:
            ImportError: If cv2/opencv-python-headless not installed.

        Example:
            >>> encoder = CV2ImageEncoder()  # cv2 imported here
            >>> encoder._cv2.__name__
            'cv2'
        """
        import cv2

        self._cv2 = cv2

    def encode_jpeg(self, img: NDArray[Any], quality: int = 85) -> bytes:
        """Encode image array as JPEG bytes using OpenCV.

        Uses cv2.imencode with IMWRITE_JPEG_QUALITY parameter for
        configurable compression. Validates quality range and encoding success.

        Business context: Primary encoding path for web dashboard MJPEG streams.
        Called once per frame at 10-30 fps. Quality 85 balances file size (~50KB
        for 640x480) with visual quality for astronomy preview streams.

        Args:
            img: NumPy image array (grayscale or BGR, uint8/uint16).
            quality: JPEG quality 1-100.

        Returns:
            JPEG bytes starting with 0xFFD8 magic bytes.

        Raises:
            ValueError: If quality not in 1-100 range or encoding fails.

        Example:
            >>> encoder = CV2ImageEncoder()
            >>> img = np.zeros((480, 640), dtype=np.uint8)
            >>> jpeg = encoder.encode_jpeg(img, quality=90)
            >>> len(jpeg) > 0 and jpeg[:2] == b'\\xff\\xd8'
            True
        """
        if not 1 <= quality <= 100:
            raise ValueError(f"quality must be 1-100, got {quality}")
        success, data = self._cv2.imencode(
            ".jpg", img, [self._cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            raise ValueError(
                f"JPEG encoding failed for image shape={img.shape}, dtype={img.dtype}"
            )
        return data.tobytes()

    def put_text(
        self,
        img: NDArray[Any],
        text: str,
        position: tuple[int, int],
        scale: float,
        color: int | tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw text on image using cv2.putText with FONT_HERSHEY_SIMPLEX.

        Renders text directly onto the image array for error overlays,
        status indicators, or frame annotations in camera streams.

        Business context: Used by web dashboard streaming to display error
        messages when camera capture fails ("Camera not found", "Frame error").
        Renders on black placeholder frames so operators see feedback rather
        than frozen/blank streams.

        Args:
            img: Image to draw on (modified in place).
            text: Text to render.
            position: (x, y) baseline position.
            scale: Font scale (1.0 = base size, 0.5 = half).
            color: Text color (int 0-255 for grayscale, BGR tuple for color).
            thickness: Line thickness in pixels.

        Returns:
            None. Image is modified in place.

        Raises:
            cv2.error: If image format incompatible with text rendering.

        Example:
            >>> encoder = CV2ImageEncoder()
            >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> encoder.put_text(img, "Error!", (50, 240), 1.0, (0, 0, 255), 2)
            >>> # img now has red "Error!" text at position (50, 240)
        """
        self._cv2.putText(
            img,
            text,
            position,
            self._cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )
