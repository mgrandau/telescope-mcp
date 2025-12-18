"""Observability module for telescope-mcp.

Provides structured logging, statistics collection, and metrics
for monitoring telescope operations.

Example:
    from telescope_mcp.observability import get_logger, LogContext

    logger = get_logger(__name__)

    # Simple logging
    logger.info("Camera connected")

    # Structured logging with context
    with LogContext(camera_id=0, operation="capture"):
        logger.info("Starting capture", exposure_us=100000, gain=50)

    # Or use extra dict directly
    logger.info("Frame captured", extra={"width": 1920, "height": 1080})

Statistics Example:
    from telescope_mcp.observability import CameraStats

    # Create and inject via DI
    stats = CameraStats()
    camera = Camera(driver, config, stats=stats)

    # Or use directly
    stats.record_capture(camera_id=0, duration_ms=150, success=True)

    summary = stats.get_summary(camera_id=0)
    print(f"Success rate: {summary.success_rate:.1%}")  # StatsSummary dataclass
"""

from telescope_mcp.observability.logging import (
    LogContext,
    StructuredLogger,
    configure_logging,
    get_logger,
    reset_logging,
)
from telescope_mcp.observability.stats import (
    CameraStats,
    StatsSummary,
)

__all__ = [
    # Logging
    "LogContext",
    "StructuredLogger",
    "configure_logging",
    "get_logger",
    "reset_logging",
    # Statistics
    "CameraStats",
    "StatsSummary",
]
