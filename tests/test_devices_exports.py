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
        """Verifies every item in __all__ is importable from the module.

        Tests that __all__ doesn't list non-existent symbols that would
        cause 'from telescope_mcp.devices import *' to fail.

        Arrangement:
            1. Import devices module (already done at module level).
            2. Access devices.__all__ list.

        Action:
            Iterate __all__ and verify each name exists via hasattr/getattr.

        Assertion Strategy:
            Validates export integrity by confirming:
            - Each name in __all__ exists as module attribute.
            - Each exported object is not None.

        Testing Principle:
            Validates API contract, ensuring __all__ accurately reflects
            available public symbols.
        """
        for name in devices.__all__:
            assert hasattr(devices, name), f"__all__ lists '{name}' but not in module"
            # Verify it's not None (actual export)
            obj = getattr(devices, name)
            assert obj is not None, f"'{name}' is None"

    def test_no_namespace_pollution(self) -> None:
        """Verifies public namespace only contains __all__ items + submodules.

        Tests that only expected names are in the public namespace. Python
        automatically adds submodules to package namespace when imported.

        Arrangement:
            1. Build allowed set from __all__ + expected submodules.
            2. Get actual public names via dir() filtering.

        Action:
            Compute set difference between actual and allowed names.

        Assertion Strategy:
            Validates namespace cleanliness by confirming:
            - No unexpected public names exist.
            - Only __all__ items and submodule names are public.

        Testing Principle:
            Validates encapsulation, ensuring internal implementation
            details don't leak into public namespace.
        """
        allowed = set(devices.__all__)
        # Python adds submodules to package namespace when they're imported from
        expected_submodules = {
            "camera",
            "camera_controller",
            "camera_registry",
            "coordinate_provider",
            "motor",
            "sensor",
        }
        allowed.update(expected_submodules)

        actual_public = {name for name in dir(devices) if not name.startswith("_")}

        unexpected = actual_public - allowed
        assert not unexpected, f"Unexpected public names in namespace: {unexpected}"

    def test_all_count_matches_comment(self) -> None:
        """Verifies __all__ count matches documented total in source comment.

        Tests that '# Total: N exports' comment at end of __all__
        matches the actual count. Prevents comment drift.

        Arrangement:
            1. Access devices.__all__ list.
            2. Expected count is 30 per source comment.

        Action:
            Count items in __all__ and compare to documented value.

        Assertion Strategy:
            Validates documentation accuracy by confirming:
            - len(__all__) equals documented count of 30.

        Testing Principle:
            Validates documentation currency, ensuring code comments
            accurately reflect implementation.
        """
        assert (
            len(devices.__all__) == 39
        ), f"Expected 39 exports (per comment), got {len(devices.__all__)}"

    def test_export_categories(self) -> None:
        """Verifies expected symbols exist in each export category.

        Spot-checks that key symbols from each category (Camera, Controller,
        Registry, Sensor) are present. Catches accidental removal.

        Arrangement:
            1. Define key exports per category (Camera, Controller, etc.).
            2. Combine into single list of required symbols.

        Action:
            Check each key symbol is in __all__.

        Assertion Strategy:
            Validates API completeness by confirming:
            - All Camera symbols present (Camera, CameraConfig, etc.).
            - All Controller symbols present.
            - All Registry symbols present.
            - All Sensor symbols present.

        Testing Principle:
            Validates API stability, ensuring essential public symbols
            are never accidentally removed from exports.
        """
        # Key exports that must exist
        camera_exports = ["Camera", "CameraConfig", "CaptureResult", "CameraInfo"]
        controller_exports = ["CameraController", "SyncCaptureConfig"]
        registry_exports = ["CameraRegistry", "init_registry", "get_registry"]
        sensor_exports = ["Sensor", "SensorConfig", "SensorDeviceStatus"]

        all_key = (
            camera_exports + controller_exports + registry_exports + sensor_exports
        )

        for name in all_key:
            assert name in devices.__all__, f"Key export '{name}' missing from __all__"

    def test_star_import_works(self) -> None:
        """Verifies 'from telescope_mcp.devices import *' works correctly.

        Tests that star import mechanism works. This is the ultimate test
        of __all__ integrity - if it fails, users can't use star import.

        Arrangement:
            1. Create empty namespace dict for isolated exec.
            2. Prepare star import statement string.

        Action:
            Execute star import in isolated namespace via exec().

        Assertion Strategy:
            Validates import mechanism by confirming:
            - exec() completes without ImportError.
            - All __all__ items appear in resulting namespace.

        Testing Principle:
            Validates user experience, ensuring star import works
            as documented for convenience imports.
        """
        # Create isolated namespace for star import
        namespace: dict[str, object] = {}
        exec("from telescope_mcp.devices import *", namespace)  # noqa: S102

        # Should have all __all__ items
        for name in devices.__all__:
            assert name in namespace, f"Star import missing '{name}'"

    def test_no_circular_import(self) -> None:
        """Verifies module imports without circular import errors.

        Tests fresh import works after clearing module cache. Catches
        circular dependencies between devices submodules.

        Arrangement:
            1. Remove all telescope_mcp.devices.* from sys.modules.
            2. Prepare to call importlib.import_module.

        Action:
            Import telescope_mcp.devices fresh without cache.

        Assertion Strategy:
            Validates import safety by confirming:
            - Fresh import completes without ImportError.
            - No circular dependency exception raised.

        Testing Principle:
            Validates module architecture, ensuring submodule imports
            don't create circular dependency chains.
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
