"""Tests for devices module export integrity.

Verifies that __all__ matches actual exports and no namespace pollution.
This ensures the public API remains stable and intentional.

Example:
    >>> pdm run pytest tests/test_devices_exports.py -v
"""

from __future__ import annotations

import telescope_mcp.devices as devices


class TestDevicesExports:
    """Verify devices module export integrity."""

    def test_all_exports_importable(self) -> None:
        """Every item in __all__ should be importable.

        Ensures __all__ doesn't list non-existent symbols that would
        cause 'from telescope_mcp.devices import *' to fail.

        Args:
            None.

        Returns:
            None. Asserts on import failures.

        Raises:
            AssertionError: If any __all__ item not in module.
        """
        for name in devices.__all__:
            assert hasattr(devices, name), f"__all__ lists '{name}' but not in module"
            # Verify it's not None (actual export)
            obj = getattr(devices, name)
            assert obj is not None, f"'{name}' is None"

    def test_no_namespace_pollution(self) -> None:
        """Public namespace should only contain __all__ items + submodules.

        Verifies only expected names are in the public namespace. Python
        automatically adds submodules to package namespace when imported,
        so camera, controller, registry, sensor are expected.

        Args:
            None.

        Returns:
            None. Asserts on unexpected exports.

        Raises:
            AssertionError: If non-__all__ public names found (except submodules).
        """
        allowed = set(devices.__all__)
        # Python adds submodules to package namespace when they're imported from
        expected_submodules = {
            "camera",
            "camera_controller",
            "camera_registry",
            "sensor",
        }
        allowed.update(expected_submodules)

        actual_public = {name for name in dir(devices) if not name.startswith("_")}

        unexpected = actual_public - allowed
        assert not unexpected, f"Unexpected public names in namespace: {unexpected}"

    def test_all_count_matches_comment(self) -> None:
        """__all__ count should match documented total.

        Verifies the '# Total: N exports' comment at end of __all__
        matches the actual count. Prevents comment drift.

        Args:
            None.

        Returns:
            None. Asserts on count mismatch.

        Raises:
            AssertionError: If count doesn't match 30.
        """
        assert (
            len(devices.__all__) == 30
        ), f"Expected 30 exports (per comment), got {len(devices.__all__)}"

    def test_export_categories(self) -> None:
        """Verify expected symbols exist in each category.

        Spot-checks that key symbols from each category (Camera, Controller,
        Registry, Sensor) are present. Catches accidental removal of
        important exports.

        Args:
            None.

        Returns:
            None. Asserts on missing key exports.

        Raises:
            AssertionError: If any key export missing.
        """
        # Key exports that must exist
        camera_exports = ["Camera", "CameraConfig", "CaptureResult", "CameraInfo"]
        controller_exports = ["CameraController", "SyncCaptureConfig"]
        registry_exports = ["CameraRegistry", "init_registry", "get_registry"]
        sensor_exports = ["Sensor", "SensorConfig", "SensorStatistics"]

        all_key = (
            camera_exports + controller_exports + registry_exports + sensor_exports
        )

        for name in all_key:
            assert name in devices.__all__, f"Key export '{name}' missing from __all__"

    def test_star_import_works(self) -> None:
        """'from telescope_mcp.devices import *' should work without error.

        Verifies the star import mechanism works correctly. This is the
        ultimate test of __all__ integrity.

        Args:
            None.

        Returns:
            None. Asserts on import failure.

        Raises:
            AssertionError: If star import fails or doesn't populate namespace.
        """
        # Create isolated namespace for star import
        namespace: dict[str, object] = {}
        exec("from telescope_mcp.devices import *", namespace)  # noqa: S102

        # Should have all __all__ items
        for name in devices.__all__:
            assert name in namespace, f"Star import missing '{name}'"

    def test_no_circular_import(self) -> None:
        """Module should import without circular import errors.

        Verifies fresh import works. Catches circular dependencies
        between devices submodules.

        Args:
            None.

        Returns:
            None. Asserts on import failure.

        Raises:
            AssertionError: If import fails.
        """
        import importlib
        import sys

        # Remove from cache
        prefix = "telescope_mcp.devices"
        modules_to_remove = [k for k in sys.modules if k.startswith(prefix)]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Fresh import should work
        try:
            importlib.import_module("telescope_mcp.devices")
        except ImportError as e:
            raise AssertionError(f"Circular import detected: {e}") from e
        finally:
            # Restore (re-import to fix any broken state)
            importlib.import_module("telescope_mcp.devices")
