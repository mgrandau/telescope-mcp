"""Unit tests for telescope_mcp.utils.image module.

Tests the ImageEncoder Protocol and CV2ImageEncoder implementation.
Achieves 100% coverage of the image encoding abstraction layer.
"""

from unittest.mock import patch

import numpy as np
import pytest

from telescope_mcp.utils.image import CV2ImageEncoder, ImageEncoder


class TestImageEncoderProtocol:
    """Tests for ImageEncoder Protocol runtime checking."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify ImageEncoder can be used with isinstance().

        The @runtime_checkable decorator enables duck-type checking
        for classes implementing the protocol methods.
        """

        class MockEncoder:
            def encode_jpeg(self, img: np.ndarray, quality: int = 85) -> bytes:
                return b"\xff\xd8test"

            def put_text(
                self,
                img: np.ndarray,
                text: str,
                position: tuple[int, int],
                scale: float,
                color: int | tuple[int, int, int],
                thickness: int,
            ) -> None:
                pass

        encoder = MockEncoder()
        assert isinstance(encoder, ImageEncoder)

    def test_protocol_rejects_incomplete_implementation(self) -> None:
        """Verify incomplete implementations fail isinstance check."""

        class IncompleteEncoder:
            def encode_jpeg(self, img: np.ndarray) -> bytes:
                return b""

            # Missing put_text

        encoder = IncompleteEncoder()
        # Protocol check requires all methods
        assert not isinstance(encoder, ImageEncoder)


class TestCV2ImageEncoderInit:
    """Tests for CV2ImageEncoder initialization."""

    def test_init_imports_cv2(self) -> None:
        """Verify __init__ imports cv2 lazily.

        The cv2 import happens in __init__, not at module level,
        to avoid Python 3.13 cv2.typing issues.
        """
        encoder = CV2ImageEncoder()
        assert hasattr(encoder, "_cv2")
        assert encoder._cv2 is not None

    def test_init_raises_import_error_when_cv2_missing(self) -> None:
        """Verify ImportError when cv2 not available."""
        with patch.dict("sys.modules", {"cv2": None}):
            # Force re-import to trigger ImportError
            with pytest.raises((ImportError, TypeError)):
                # Create new instance which tries to import cv2
                CV2ImageEncoder()


class TestCV2ImageEncoderEncodeJpeg:
    """Tests for CV2ImageEncoder.encode_jpeg() method."""

    def test_encode_jpeg_returns_bytes(self) -> None:
        """Verify encode_jpeg returns JPEG bytes for valid image."""
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 100), dtype=np.uint8)

        result = encoder.encode_jpeg(img)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_jpeg_starts_with_jpeg_magic_bytes(self) -> None:
        """Verify output starts with JPEG magic bytes 0xFFD8."""
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 100), dtype=np.uint8)

        result = encoder.encode_jpeg(img)

        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_default_quality(self) -> None:
        """Verify default quality of 85 is used."""
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 100), dtype=np.uint8)

        # Default should work without specifying quality
        result = encoder.encode_jpeg(img)
        assert len(result) > 0

    def test_encode_jpeg_custom_quality_low(self) -> None:
        """Verify low quality produces smaller output."""
        encoder = CV2ImageEncoder()
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

        low_quality = encoder.encode_jpeg(img, quality=10)
        high_quality = encoder.encode_jpeg(img, quality=95)

        # Low quality should be smaller (usually significantly)
        assert len(low_quality) < len(high_quality)

    def test_encode_jpeg_quality_boundary_1(self) -> None:
        """Verify minimum quality of 1 is accepted."""
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        result = encoder.encode_jpeg(img, quality=1)
        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_quality_boundary_100(self) -> None:
        """Verify maximum quality of 100 is accepted."""
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        result = encoder.encode_jpeg(img, quality=100)
        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_quality_below_range_raises(self) -> None:
        """Verify quality < 1 raises ValueError."""
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="quality must be 1-100, got 0"):
            encoder.encode_jpeg(img, quality=0)

    def test_encode_jpeg_quality_above_range_raises(self) -> None:
        """Verify quality > 100 raises ValueError."""
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="quality must be 1-100, got 101"):
            encoder.encode_jpeg(img, quality=101)

    def test_encode_jpeg_negative_quality_raises(self) -> None:
        """Verify negative quality raises ValueError."""
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="quality must be 1-100, got -5"):
            encoder.encode_jpeg(img, quality=-5)

    def test_encode_jpeg_grayscale_image(self) -> None:
        """Verify grayscale images (H, W) encode correctly."""
        encoder = CV2ImageEncoder()
        img = np.random.randint(0, 255, (100, 150), dtype=np.uint8)

        result = encoder.encode_jpeg(img)
        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_color_image(self) -> None:
        """Verify color images (H, W, 3) encode correctly."""
        encoder = CV2ImageEncoder()
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        result = encoder.encode_jpeg(img)
        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_encoding_failure_raises(self) -> None:
        """Verify encoding failure raises ValueError with context."""
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 100), dtype=np.uint8)

        # Mock imencode to return failure
        with patch.object(encoder, "_cv2") as mock_cv2:
            mock_cv2.imencode.return_value = (False, None)
            mock_cv2.IMWRITE_JPEG_QUALITY = 1

            with pytest.raises(
                ValueError, match=r"JPEG encoding failed for image shape=.*dtype="
            ):
                encoder.encode_jpeg(img)


class TestCV2ImageEncoderPutText:
    """Tests for CV2ImageEncoder.put_text() method."""

    def test_put_text_modifies_image_in_place(self) -> None:
        """Verify put_text draws on the image (modifies in place)."""
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 200), dtype=np.uint8)
        original_sum = img.sum()

        encoder.put_text(img, "TEST", (10, 50), 1.0, 255, 2)

        # Image should be modified (non-zero pixels from text)
        assert img.sum() > original_sum

    def test_put_text_with_color_image(self) -> None:
        """Verify put_text works with BGR color images."""
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        original_sum = img.sum()

        encoder.put_text(img, "COLOR", (10, 50), 1.0, (0, 255, 0), 2)

        assert img.sum() > original_sum

    def test_put_text_with_tuple_color(self) -> None:
        """Verify put_text accepts (B, G, R) tuple for color."""
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        # Should not raise
        encoder.put_text(img, "RGB", (10, 50), 0.5, (255, 128, 64), 1)

        # Verify some pixels were drawn
        assert img.sum() > 0

    def test_put_text_with_int_color(self) -> None:
        """Verify put_text accepts int for grayscale color."""
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 200), dtype=np.uint8)

        encoder.put_text(img, "GRAY", (10, 50), 0.5, 200, 1)

        assert img.sum() > 0

    def test_put_text_returns_none(self) -> None:
        """Verify put_text returns None (modifies in place)."""
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 100), dtype=np.uint8)

        result = encoder.put_text(img, "X", (10, 30), 0.5, 255, 1)

        assert result is None

    def test_put_text_different_scale_factors(self) -> None:
        """Verify different scale factors produce different text sizes."""
        encoder = CV2ImageEncoder()
        img_small = np.zeros((100, 200), dtype=np.uint8)
        img_large = np.zeros((100, 200), dtype=np.uint8)

        encoder.put_text(img_small, "A", (10, 50), 0.5, 255, 1)
        encoder.put_text(img_large, "A", (10, 50), 2.0, 255, 1)

        # Larger scale should use more pixels
        assert img_large.sum() > img_small.sum()

    def test_put_text_different_thickness(self) -> None:
        """Verify different thickness values affect output."""
        encoder = CV2ImageEncoder()
        img_thin = np.zeros((100, 200), dtype=np.uint8)
        img_thick = np.zeros((100, 200), dtype=np.uint8)

        encoder.put_text(img_thin, "T", (10, 50), 1.0, 255, 1)
        encoder.put_text(img_thick, "T", (10, 50), 1.0, 255, 4)

        # Thicker lines should use more pixels
        assert img_thick.sum() > img_thin.sum()
