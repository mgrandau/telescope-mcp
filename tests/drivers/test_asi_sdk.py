"""Unit tests for ASI SDK path utilities.

Tests the path helper functions in the asi_sdk module without
requiring the full ASI camera driver stack or OpenCV.
"""


class TestASISDKPaths:
    """Tests for ASI SDK path utility functions."""

    def test_get_udev_rules_path_returns_valid_path(self) -> None:
        """Verify get_udev_rules_path returns path to asi.rules file.

        Tests that the udev rules path utility returns a properly
        constructed path string.

        Business context:
            Linux installations need udev rules for USB camera access.
            This function provides the path for copying rules to
            /etc/udev/rules.d/ during installation.

        Arrangement:
            None - function requires no setup.

        Action:
            Call get_udev_rules_path().

        Assertion Strategy:
            Validates path structure by confirming:
            - Returns string type.
            - Path ends with asi.rules.
            - Path contains asi_sdk directory.

        Testing Principle:
            Validates utility function returns properly formed path.
        """
        from telescope_mcp.drivers.asi_sdk import get_udev_rules_path

        rules_path = get_udev_rules_path()

        assert isinstance(rules_path, str)
        assert rules_path.endswith("asi.rules")
        assert "asi_sdk" in rules_path
