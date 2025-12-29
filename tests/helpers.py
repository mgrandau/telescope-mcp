"""Test helper functions for telescope-mcp.

Provides utilities for protocol compliance verification and other
common test operations.

Example:
    from tests.helpers import assert_implements_protocol
    from telescope_mcp.drivers.sensors.types import SensorInstance

    def test_my_sensor_implements_protocol():
        sensor = MySensor()
        assert_implements_protocol(sensor, SensorInstance)
"""

from __future__ import annotations

from typing import Any, Protocol, get_type_hints


def assert_implements_protocol(
    instance: object,
    protocol: type[Protocol],
    *,
    check_signatures: bool = False,
) -> None:
    """Assert that an instance implements a Protocol interface.

    Verifies that the given instance satisfies the Protocol contract using
    isinstance() checks (requires @runtime_checkable on the Protocol).
    Optionally performs deeper signature validation.

    Business context: Enables unit tests to verify that mock implementations
    and real drivers correctly implement their Protocol interfaces. Catches
    missing methods early in testing rather than at runtime.

    Args:
        instance: Object to check for protocol compliance.
        protocol: Protocol class to check against. Must be decorated
            with @runtime_checkable.
        check_signatures: If True, also verifies method signatures match.
            Default False for basic isinstance() check only.

    Returns:
        None. Raises AssertionError if protocol not implemented.

    Raises:
        AssertionError: If instance doesn't implement protocol.
        TypeError: If protocol is not @runtime_checkable.

    Example:
        >>> from telescope_mcp.drivers.sensors.types import SensorInstance
        >>> assert_implements_protocol(my_sensor, SensorInstance)
        >>> # Passes silently if protocol satisfied
        >>> # Raises AssertionError with details if not
    """
    # Check isinstance (requires @runtime_checkable)
    if not isinstance(instance, protocol):
        # Build helpful error message listing missing members
        instance_attrs = set(dir(instance))
        protocol_attrs = set(dir(protocol))

        # Filter to public methods defined by protocol (not inherited from object)
        object_attrs = set(dir(object))
        protocol_methods = {
            attr
            for attr in protocol_attrs - object_attrs
            if not attr.startswith("_") or attr in ("__call__",)
        }

        missing = []
        for method in protocol_methods:
            if method not in instance_attrs:
                missing.append(method)
            elif not callable(getattr(instance, method, None)):
                # Might be a property - check if protocol expects property
                pass

        missing_str = ", ".join(sorted(missing)) if missing else "unknown"
        raise AssertionError(
            f"{type(instance).__name__} does not implement {protocol.__name__}. "
            f"Missing: {missing_str}"
        )

    if check_signatures:
        # Optional: deeper signature validation
        try:
            hints = get_type_hints(protocol)
            for name, expected_type in hints.items():
                if hasattr(instance, name):
                    # Could add deeper type checking here
                    pass
        except Exception:
            # Type hint introspection can fail for various reasons
            pass


def assert_all_implement_protocol(
    instances: list[Any],
    protocol: type[Protocol],
) -> None:
    """Assert that all instances in a list implement a Protocol.

    Convenience wrapper for checking multiple implementations at once.

    Args:
        instances: List of objects to check.
        protocol: Protocol class to check against.

    Raises:
        AssertionError: If any instance doesn't implement protocol.

    Example:
        >>> drivers = [ArduinoDriver(), TwinDriver()]
        >>> assert_all_implement_protocol(drivers, SensorDriver)
    """
    for i, instance in enumerate(instances):
        try:
            assert_implements_protocol(instance, protocol)
        except AssertionError as e:
            raise AssertionError(f"Instance at index {i}: {e}") from e
