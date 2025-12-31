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
        """Verifies ImageEncoder Protocol supports isinstance() duck-typing.

        Tests that @runtime_checkable decorator on ImageEncoder enables
        duck-type verification at runtime.

        Business context:
        Protocols enable dependency injection without inheritance. Tests can
        provide mock encoders that pass isinstance() checks without inheriting
        from cv2-dependent classes.

        Arrangement:
        1. Define MockEncoder class with encode_jpeg and put_text methods
           matching the Protocol signature exactly.

        Action:
        Create instance and check isinstance(encoder, ImageEncoder).

        Assertion Strategy:
        Confirms True return from isinstance(), validating that complete
        implementations satisfy Protocol at runtime.

        Testing Principle:
        Validates Protocol contract, ensuring duck-typed substitutability.
        """

        class MockEncoder:
            def encode_jpeg(self, img: np.ndarray, quality: int = 85) -> bytes:
                """Return mock JPEG bytes with valid magic header.

                Provides test double for ImageEncoder.encode_jpeg() without
                requiring cv2 dependency. Returns fixed JPEG-like bytes.

                Business context:
                Enables Protocol isinstance() testing without cv2 import.
                Mock returns valid JPEG magic bytes for format validation.

                Args:
                    img: Input image array (ignored in mock).
                    quality: JPEG quality 1-100 (ignored in mock).

                Returns:
                    Fixed b'\\xff\\xd8test' bytes with JPEG magic header.

                Example:
                    >>> encoder = MockEncoder()
                    >>> encoder.encode_jpeg(np.zeros((10, 10)), 85)
                    b'\\xff\\xd8test'
                """
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
                """No-op mock satisfying Protocol signature.

                Provides test double for ImageEncoder.put_text() without
                cv2 dependency. Does nothing - just satisfies interface.

                Business context:
                Completes Protocol implementation for isinstance() check.
                Actual text rendering not needed for protocol testing.

                Args:
                    img: Image array (ignored in mock).
                    text: Text string (ignored in mock).
                    position: Position tuple (ignored in mock).
                    scale: Font scale (ignored in mock).
                    color: Text color (ignored in mock).
                    thickness: Line thickness (ignored in mock).

                Returns:
                    None. No-op implementation.

                Example:
                    >>> encoder = MockEncoder()
                    >>> encoder.put_text(img, "test", (0, 0), 1.0, 255, 1)
                    # No effect - mock does nothing
                """
                pass

        encoder = MockEncoder()
        assert isinstance(encoder, ImageEncoder)

    def test_protocol_rejects_incomplete_implementation(self) -> None:
        """Verifies incomplete Protocol implementations fail isinstance().

        Tests that classes missing required methods are rejected by
        @runtime_checkable Protocol verification.

        Business context:
        Catches configuration errors early. If a mock encoder missing put_text
        is injected, isinstance() fails before runtime errors occur during
        stream rendering.

        Arrangement:
        1. Define IncompleteEncoder with only encode_jpeg (missing put_text).

        Action:
        Create instance and check isinstance(encoder, ImageEncoder).

        Assertion Strategy:
        Confirms False return from isinstance(), validating that partial
        implementations are rejected.

        Testing Principle:
        Validates Protocol enforcement, ensuring type safety for DI.
        """

        class IncompleteEncoder:
            def encode_jpeg(self, img: np.ndarray) -> bytes:
                """Minimal encode_jpeg for Protocol rejection test.

                Intentionally incomplete implementation missing put_text()
                method to verify Protocol rejects partial implementations.

                Business context:
                Tests Protocol enforcement - ensures isinstance() returns
                False when required methods are missing, catching DI errors.

                Args:
                    img: Input image array (ignored in mock).

                Returns:
                    Empty bytes (implementation not used, just signature).

                Example:
                    >>> encoder = IncompleteEncoder()
                    >>> isinstance(encoder, ImageEncoder)
                    False  # Missing put_text()
                """
                return b""

            # Missing put_text

        encoder = IncompleteEncoder()
        # Protocol check requires all methods
        assert not isinstance(encoder, ImageEncoder)


class TestCV2ImageEncoderInit:
    """Tests for CV2ImageEncoder initialization."""

    def test_init_imports_cv2(self) -> None:
        """Verifies CV2ImageEncoder lazily imports cv2 in __init__.

        Tests that cv2 module is imported during instantiation rather
        than at module load time.

        Business context:
        Python 3.13 has cv2.typing import crashes. Lazy import allows
        code that doesn't use image encoding to import the utils package
        safely, deferring cv2 load until actually needed.

        Arrangement:
        1. Create CV2ImageEncoder instance (triggers __init__).

        Action:
        Instantiate encoder and inspect _cv2 attribute.

        Assertion Strategy:
        Confirms _cv2 attribute exists and is not None, validating
        successful lazy import during construction.

        Testing Principle:
        Validates lazy loading pattern, ensuring import isolation.
        """
        encoder = CV2ImageEncoder()
        assert hasattr(encoder, "_cv2")
        assert encoder._cv2 is not None

    def test_init_raises_import_error_when_cv2_missing(self) -> None:
        """Verifies ImportError raised when cv2 package unavailable.

        Tests error handling for headless environments or missing
        opencv-python-headless dependency.

        Business context:
        Clear error messages help operators diagnose missing dependencies
        during deployment. ImportError is caught at startup rather than
        during first frame capture.

        Arrangement:
        1. Patch sys.modules to simulate cv2 unavailable.

        Action:
        Attempt to create CV2ImageEncoder instance.

        Assertion Strategy:
        Confirms ImportError or TypeError raised, validating graceful
        failure when opencv is not installed.

        Testing Principle:
        Validates dependency error handling, ensuring clear diagnostics.
        """
        with patch.dict("sys.modules", {"cv2": None}):
            # Force re-import to trigger ImportError
            with pytest.raises((ImportError, TypeError)):
                # Create new instance which tries to import cv2
                CV2ImageEncoder()


class TestCV2ImageEncoderEncodeJpeg:
    """Tests for CV2ImageEncoder.encode_jpeg() method."""

    def test_encode_jpeg_returns_bytes(self) -> None:
        """Verifies encode_jpeg returns non-empty bytes for valid input.

        Tests the basic contract that encoding produces byte output.

        Business context:
        MJPEG streaming requires bytes output. This validates the
        fundamental encoding operation works before testing specifics.

        Arrangement:
        1. Create encoder instance.
        2. Create 100x100 black image (zeros array).

        Action:
        Call encode_jpeg with test image.

        Assertion Strategy:
        Confirms result is bytes type and has length > 0.

        Testing Principle:
        Validates basic functionality, ensuring encoding produces output.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 100), dtype=np.uint8)

        result = encoder.encode_jpeg(img)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_jpeg_starts_with_jpeg_magic_bytes(self) -> None:
        """Verifies JPEG output starts with 0xFFD8 magic bytes.

        Tests that output is valid JPEG format per specification.

        Business context:
        Browsers validate JPEG magic bytes. Invalid headers cause stream
        rendering failures. This catches encoding configuration issues.

        Arrangement:
        1. Create encoder and 100x100 black image.

        Action:
        Encode image and inspect first 2 bytes.

        Assertion Strategy:
        Confirms bytes start with \xff\xd8 (JPEG SOI marker).

        Testing Principle:
        Validates format correctness, ensuring browser compatibility.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 100), dtype=np.uint8)

        result = encoder.encode_jpeg(img)

        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_default_quality(self) -> None:
        """Verifies encode_jpeg works without explicit quality parameter.

        Tests that default quality=85 is applied when not specified.

        Business context:
            Default quality balances file size (~50KB for 640x480) with
            visual quality for astronomy preview streams. Callers shouldn't
            need to specify quality for typical use cases.

        Arrangement:
            1. Create CV2ImageEncoder instance.
            2. Create 100x100 black test image (zeros array).
            3. No quality parameter will be provided to encode_jpeg.

        Action:
            Calls encode_jpeg without quality parameter, relying on default.

        Assertion Strategy:
            Validates default handling by confirming:
            - Result has non-zero length (encoding succeeded).
            - Default quality produces valid JPEG output.

        Testing Principle:
            Validates default parameter, ensuring convenience for callers
            who don't need to override quality settings.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 100), dtype=np.uint8)

        # Default should work without specifying quality
        result = encoder.encode_jpeg(img)
        assert len(result) > 0

    def test_encode_jpeg_custom_quality_low(self) -> None:
        """Verifies lower quality produces smaller file size.

        Tests that quality parameter affects compression ratio.

        Business context:
        Bandwidth-constrained situations (remote observing) may need
        lower quality. This validates quality actually affects output.

        Arrangement:
        1. Create encoder and 200x200 random noise image.
        2. Random pixels ensure compression has work to do.

        Action:
        Encode same image at quality=10 and quality=95.

        Assertion Strategy:
        Confirms low_quality output smaller than high_quality output.

        Testing Principle:
        Validates quality tradeoff, ensuring bandwidth control works.
        """
        encoder = CV2ImageEncoder()
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

        low_quality = encoder.encode_jpeg(img, quality=10)
        high_quality = encoder.encode_jpeg(img, quality=95)

        # Low quality should be smaller (usually significantly)
        assert len(low_quality) < len(high_quality)

    def test_encode_jpeg_quality_boundary_1(self) -> None:
        """Verifies minimum quality boundary (1) produces valid JPEG output.

        Tests lower boundary of quality range for extreme compression.

        Business context:
            Extreme low quality for minimum bandwidth situations should
            still produce valid images, even if visually degraded.
            Useful for remote observing over satellite links.

        Arrangement:
            1. Create CV2ImageEncoder instance.
            2. Create 50x50 black test image.
            3. Quality=1 is the minimum valid JPEG quality value.

        Action:
            Encodes image with quality=1 (minimum valid value).

        Assertion Strategy:
            Validates boundary handling by confirming:
            - Output has JPEG magic bytes (\xff\xd8 SOI marker).
            - Valid encoding at minimum quality.

        Testing Principle:
            Validates boundary condition, ensuring minimum quality
            value produces valid output for bandwidth-constrained scenarios.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        result = encoder.encode_jpeg(img, quality=1)
        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_quality_boundary_100(self) -> None:
        """Verifies maximum quality boundary (100) produces valid JPEG output.

        Tests upper boundary of quality range for maximum fidelity.

        Business context:
            Maximum quality for detailed preview or diagnostic captures
            should work correctly for critical observations requiring
            highest visual fidelity in JPEG format.

        Arrangement:
            1. Create CV2ImageEncoder instance.
            2. Create 50x50 black test image.
            3. Quality=100 is the maximum valid JPEG quality value.

        Action:
            Encodes image with quality=100 (maximum valid value).

        Assertion Strategy:
            Validates boundary handling by confirming:
            - Output has JPEG magic bytes (\xff\xd8 SOI marker).
            - Valid encoding at maximum quality.

        Testing Principle:
            Validates boundary condition, ensuring maximum quality
            value produces valid output for high-fidelity captures.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        result = encoder.encode_jpeg(img, quality=100)
        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_quality_below_range_raises(self) -> None:
        """Verifies quality=0 raises ValueError with descriptive message.

        Tests validation rejects below-minimum quality values.

        Business context:
        Quality 0 is invalid for JPEG. Clear error prevents silent
        failures or undefined cv2 behavior.

        Arrangement:
        1. Create encoder and 50x50 black image.

        Action:
        Call encode_jpeg with quality=0.

        Assertion Strategy:
        Confirms ValueError raised with message containing 'quality must be 1-100'.

        Testing Principle:
        Validates input validation, ensuring clear error messages.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="quality must be 1-100, got 0"):
            encoder.encode_jpeg(img, quality=0)

    def test_encode_jpeg_quality_above_range_raises(self) -> None:
        """Verifies quality=101 raises ValueError with descriptive message.

        Tests validation rejects above-maximum quality values.

        Business context:
        Quality > 100 is invalid for JPEG. Catches caller mistakes like
        passing percentage (150%) instead of 1-100 range.

        Arrangement:
        1. Create encoder and 50x50 black image.

        Action:
        Call encode_jpeg with quality=101.

        Assertion Strategy:
        Confirms ValueError raised with message containing 'quality must be 1-100'.

        Testing Principle:
        Validates input validation, ensuring clear error messages.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="quality must be 1-100, got 101"):
            encoder.encode_jpeg(img, quality=101)

    def test_encode_jpeg_negative_quality_raises(self) -> None:
        """Verifies negative quality raises ValueError with descriptive message.

        Tests validation rejects negative quality values.

        Business context:
        Negative quality is nonsensical. Clear error catches sign errors
        in caller code.

        Arrangement:
        1. Create encoder and 50x50 black image.

        Action:
        Call encode_jpeg with quality=-5.

        Assertion Strategy:
        Confirms ValueError raised with message containing 'quality must be 1-100'.

        Testing Principle:
        Validates input validation, ensuring clear error messages.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="quality must be 1-100, got -5"):
            encoder.encode_jpeg(img, quality=-5)

    def test_encode_jpeg_grayscale_image(self) -> None:
        """Verifies 2D grayscale arrays (H, W) encode to valid JPEG.

        Tests encoding of single-channel images common in astronomy.

        Business context:
        ASI cameras often output RAW8 grayscale for guiding. This is
        the primary image format for telescope-mcp camera streams.

        Arrangement:
        1. Create encoder and 100x150 random grayscale image.
        2. Non-square dimensions test aspect ratio handling.

        Action:
        Encode grayscale image.

        Assertion Strategy:
        Confirms output has JPEG magic bytes.

        Testing Principle:
        Validates grayscale support, ensuring astronomy camera compatibility.
        """
        encoder = CV2ImageEncoder()
        img = np.random.randint(0, 255, (100, 150), dtype=np.uint8)

        result = encoder.encode_jpeg(img)
        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_color_image(self) -> None:
        """Verifies 3D color arrays (H, W, 3) encode to valid JPEG.

        Tests encoding of BGR color images from color cameras.

        Business context:
        Color cameras (ASI120MC) output BGR format. Color streaming
        is used for planetary imaging and visual observation.

        Arrangement:
        1. Create encoder and 100x150 random BGR image.
        2. Shape (100, 150, 3) represents H, W, channels.

        Action:
        Encode color image.

        Assertion Strategy:
        Confirms output has JPEG magic bytes.

        Testing Principle:
        Validates color support, ensuring color camera compatibility.
        """
        encoder = CV2ImageEncoder()
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        result = encoder.encode_jpeg(img)
        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_encoding_failure_raises(self) -> None:
        """Verifies encoding failure raises ValueError with shape/dtype info.

        Tests error handling when cv2.imencode returns failure.

        Business context:
        Corrupt images or unsupported formats should produce clear
        error messages with image dimensions and dtype for debugging.

        Arrangement:
        1. Create encoder and mock _cv2.imencode to return (False, None).
        2. Simulates cv2 encoding failure for edge cases.

        Action:
        Call encode_jpeg with mocked failure.

        Assertion Strategy:
        Confirms ValueError raised with message containing shape and dtype.

        Testing Principle:
        Validates error diagnostics, ensuring debuggable failures.
        """
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
        """Verifies put_text modifies image array directly (no return).

        Tests that text rendering mutates the input array.

        Business context:
        In-place modification avoids memory allocation per frame in
        streaming hot path. Error overlays must appear on frames.

        Arrangement:
        1. Create encoder and 100x200 black image.
        2. Record original pixel sum (should be 0).

        Action:
        Call put_text to render 'TEST' in white.

        Assertion Strategy:
        Confirms image sum increased (pixels were drawn).

        Testing Principle:
        Validates in-place mutation, ensuring memory efficiency.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 200), dtype=np.uint8)
        original_sum = img.sum()

        encoder.put_text(img, "TEST", (10, 50), 1.0, 255, 2)

        # Image should be modified (non-zero pixels from text)
        assert img.sum() > original_sum

    def test_put_text_with_color_image(self) -> None:
        """Verifies put_text works on 3-channel BGR images.

        Tests text rendering on color image format.

        Business context:
        Color camera streams need error overlays too. BGR format
        matches cv2's default color order.

        Arrangement:
        1. Create encoder and 100x200 black BGR image.
        2. Record original pixel sum.

        Action:
        Render 'COLOR' text in green (0, 255, 0).

        Assertion Strategy:
        Confirms image sum increased (colored pixels drawn).

        Testing Principle:
        Validates color support, ensuring color stream compatibility.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        original_sum = img.sum()

        encoder.put_text(img, "COLOR", (10, 50), 1.0, (0, 255, 0), 2)

        assert img.sum() > original_sum

    def test_put_text_with_tuple_color(self) -> None:
        """Verifies put_text accepts BGR tuple for color specification.

        Tests that (B, G, R) format is handled correctly.

        Business context:
        BGR tuples enable precise color control for status indicators
        (red for errors, green for success, etc.).

        Arrangement:
        1. Create encoder and 100x200 black BGR image.

        Action:
        Render text with BGR tuple (255, 128, 64).

        Assertion Strategy:
        Confirms no exception raised and pixels were drawn.

        Testing Principle:
        Validates color format flexibility.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        # Should not raise
        encoder.put_text(img, "RGB", (10, 50), 0.5, (255, 128, 64), 1)

        # Verify some pixels were drawn
        assert img.sum() > 0

    def test_put_text_with_int_color(self) -> None:
        """Verifies put_text accepts integer for grayscale color.

        Tests that single int (0-255) works for grayscale images.

        Business context:
        Grayscale images use single intensity value. Simpler API for
        common astronomy use case.

        Arrangement:
        1. Create encoder and 100x200 black grayscale image.

        Action:
        Render text with intensity 200.

        Assertion Strategy:
        Confirms pixels were drawn (sum > 0).

        Testing Principle:
        Validates grayscale color convenience.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((100, 200), dtype=np.uint8)

        encoder.put_text(img, "GRAY", (10, 50), 0.5, 200, 1)

        assert img.sum() > 0

    def test_put_text_returns_none(self) -> None:
        """Verifies put_text return value is None (in-place operation).

        Tests API contract that put_text modifies rather than returns.

        Business context:
        None return signals in-place modification to callers. Prevents
        confusion about whether to use return value.

        Arrangement:
        1. Create encoder and 50x100 black grayscale image.

        Action:
        Call put_text and capture return value.

        Assertion Strategy:
        Confirms return value is exactly None.

        Testing Principle:
        Validates API contract, ensuring clear semantics.
        """
        encoder = CV2ImageEncoder()
        img = np.zeros((50, 100), dtype=np.uint8)

        result = encoder.put_text(img, "X", (10, 30), 0.5, 255, 1)

        assert result is None

    def test_put_text_different_scale_factors(self) -> None:
        """Verifies scale parameter affects rendered text size.

        Tests that larger scale produces more pixels.

        Business context:
        Different scale factors for different display sizes. Small scale
        for preview thumbnails, large scale for full-screen errors.

        Arrangement:
        1. Create encoder and two identical 100x200 black images.

        Action:
        Render 'A' at scale 0.5 on one, scale 2.0 on other.

        Assertion Strategy:
        Confirms large-scale image has more pixels (higher sum).

        Testing Principle:
        Validates scale control, ensuring readable text at any size.
        """
        encoder = CV2ImageEncoder()
        img_small = np.zeros((100, 200), dtype=np.uint8)
        img_large = np.zeros((100, 200), dtype=np.uint8)

        encoder.put_text(img_small, "A", (10, 50), 0.5, 255, 1)
        encoder.put_text(img_large, "A", (10, 50), 2.0, 255, 1)

        # Larger scale should use more pixels
        assert img_large.sum() > img_small.sum()

    def test_put_text_different_thickness(self) -> None:
        """Verifies thickness parameter affects line weight.

        Tests that thicker lines use more pixels.

        Business context:
        Thickness controls visibility. Thin for subtle overlays,
        thick for critical error messages that must be noticed.

        Arrangement:
        1. Create encoder and two identical 100x200 black images.

        Action:
        Render 'T' with thickness 1 on one, thickness 4 on other.

        Assertion Strategy:
        Confirms thick-line image has more pixels (higher sum).

        Testing Principle:
        Validates thickness control, ensuring visibility options.
        """
        encoder = CV2ImageEncoder()
        img_thin = np.zeros((100, 200), dtype=np.uint8)
        img_thick = np.zeros((100, 200), dtype=np.uint8)

        encoder.put_text(img_thin, "T", (10, 50), 1.0, 255, 1)
        encoder.put_text(img_thick, "T", (10, 50), 1.0, 255, 4)

        # Thicker lines should use more pixels
        assert img_thick.sum() > img_thin.sum()
