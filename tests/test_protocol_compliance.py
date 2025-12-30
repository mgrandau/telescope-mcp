"""Tests for protocol compliance across all drivers and devices.

Verifies that all implementations correctly satisfy their Protocol
interfaces using @runtime_checkable isinstance() checks.

These tests catch missing methods or incorrect signatures early,
ensuring mock implementations and real drivers are interchangeable.
"""

from __future__ import annotations

from tests.helpers import assert_implements_protocol


class TestSensorProtocolCompliance:
    """Verify sensor drivers implement SensorDriver/SensorInstance protocols."""

    def test_digital_twin_driver_implements_protocol(self) -> None:
        """DigitalTwinSensorDriver satisfies SensorDriver protocol.

        Verifies that the digital twin sensor driver correctly implements
        all methods required by the SensorDriver protocol interface.

        Arrangement:
        1. Import DigitalTwinSensorDriver and SensorDriver protocol.
        2. Create driver instance.

        Action:
        Call assert_implements_protocol() with driver and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(driver, SensorDriver) returns True.
        """
        from telescope_mcp.drivers.sensors.twin import DigitalTwinSensorDriver
        from telescope_mcp.drivers.sensors.types import SensorDriver

        driver = DigitalTwinSensorDriver()
        assert_implements_protocol(driver, SensorDriver)

    def test_digital_twin_instance_implements_protocol(self) -> None:
        """DigitalTwinSensorInstance satisfies SensorInstance protocol.

        Verifies that opened digital twin instances implement all
        required SensorInstance methods.

        Arrangement:
        1. Import DigitalTwinSensorDriver and SensorInstance protocol.
        2. Create driver and open sensor instance.

        Action:
        Call assert_implements_protocol() with instance and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(instance, SensorInstance) returns True.
        """
        from telescope_mcp.drivers.sensors.twin import DigitalTwinSensorDriver
        from telescope_mcp.drivers.sensors.types import SensorInstance

        driver = DigitalTwinSensorDriver()
        instance = driver.open()
        try:
            assert_implements_protocol(instance, SensorInstance)
        finally:
            instance.close()


class TestCameraProtocolCompliance:
    """Verify camera drivers implement CameraDriver/CameraInstance protocols."""

    def test_digital_twin_camera_driver_implements_protocol(self) -> None:
        """DigitalTwinCameraDriver satisfies CameraDriver protocol.

        Verifies that the digital twin camera driver correctly implements
        all methods required by the CameraDriver protocol interface.

        Arrangement:
        1. Import DigitalTwinCameraDriver and CameraDriver protocol.
        2. Create driver instance.

        Action:
        Call assert_implements_protocol() with driver and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(driver, CameraDriver) returns True.
        """
        from telescope_mcp.drivers.cameras import CameraDriver
        from telescope_mcp.drivers.cameras.twin import DigitalTwinCameraDriver

        driver = DigitalTwinCameraDriver()
        assert_implements_protocol(driver, CameraDriver)

    def test_digital_twin_camera_instance_implements_protocol(self) -> None:
        """DigitalTwinCameraInstance satisfies CameraInstance protocol.

        Verifies that opened digital twin camera instances implement all
        required CameraInstance methods including context manager support.

        Arrangement:
        1. Import DigitalTwinCameraDriver and CameraInstance protocol.
        2. Create driver and open camera instance.

        Action:
        Call assert_implements_protocol() with instance and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(instance, CameraInstance) returns True.
        """
        from telescope_mcp.drivers.cameras import CameraInstance
        from telescope_mcp.drivers.cameras.twin import DigitalTwinCameraDriver

        driver = DigitalTwinCameraDriver()
        instance = driver.open(0)
        try:
            assert_implements_protocol(instance, CameraInstance)
        finally:
            instance.close()

    def test_asi_camera_driver_implements_protocol(self) -> None:
        """ASICameraDriver satisfies CameraDriver protocol.

        Verifies that the real ASI camera driver correctly implements
        all methods required by the CameraDriver protocol interface.
        Uses mock SDK to avoid hardware dependency.

        Arrangement:
        1. Import ASICameraDriver and CameraDriver protocol.
        2. Create driver instance with mock SDK.

        Action:
        Call assert_implements_protocol() with driver and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(driver, CameraDriver) returns True.
        """
        from unittest.mock import MagicMock

        from telescope_mcp.drivers.cameras import CameraDriver
        from telescope_mcp.drivers.cameras.asi import ASICameraDriver

        # Create mock SDK to avoid hardware dependency
        mock_sdk = MagicMock()
        mock_sdk.get_num_cameras.return_value = 0

        driver = ASICameraDriver(sdk=mock_sdk)
        assert_implements_protocol(driver, CameraDriver)

    def test_asi_camera_instance_implements_protocol(self) -> None:
        """ASICameraInstance satisfies CameraInstance protocol.

        Verifies that opened ASI camera instances implement all
        required CameraInstance methods. Uses mock camera object.

        Arrangement:
        1. Import ASICameraInstance and CameraInstance protocol.
        2. Create instance with mock camera object.

        Action:
        Call assert_implements_protocol() with instance and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(instance, CameraInstance) returns True.
        """
        from unittest.mock import MagicMock

        from telescope_mcp.drivers.cameras import CameraInstance
        from telescope_mcp.drivers.cameras.asi import ASICameraInstance

        # Create mock camera object
        mock_camera = MagicMock()
        mock_camera.get_camera_property.return_value = {
            "Name": "Mock Camera",
            "MaxWidth": 1920,
            "MaxHeight": 1080,
            "PixelSize": 2.9,
            "IsColorCam": True,
            "BitDepth": 16,
        }

        instance = ASICameraInstance(camera_id=0, camera=mock_camera)
        try:
            assert_implements_protocol(instance, CameraInstance)
        finally:
            instance.close()


class TestSerialProtocolCompliance:
    """Verify serial port mock implements SerialPort protocol."""

    def test_mock_serial_implements_protocol(self) -> None:
        """Mock serial port satisfies SerialPort protocol.

        Verifies that the mock serial port used in tests correctly
        implements all SerialPort protocol methods.

        Arrangement:
        1. Import SerialPort protocol.
        2. Define MockSerialPort class with all required methods.
        3. Create mock instance.

        Action:
        Call assert_implements_protocol() with mock and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(mock, SerialPort) returns True.
        """
        from telescope_mcp.drivers.serial import SerialPort

        class MockSerialPort:
            """Minimal mock for protocol compliance test.

            Implements SerialPort protocol interface with stub methods
            to verify protocol type checking works correctly.
            """

            is_open = True
            in_waiting = 0

            def read_until(
                self, expected: bytes = b"\n", size: int | None = None
            ) -> bytes:
                """Read until expected terminator (stub implementation).

                Stub method for protocol compliance testing. Returns fixed
                response to verify SerialPort protocol signature.

                Args:
                    expected: Byte sequence to terminate on.
                    size: Maximum bytes to read.

                Returns:
                    bytes: Fixed test response "test\\r\\n".

                Raises:
                    No exceptions raised. Stub always succeeds.

                Example:
                    >>> mock = MockSerialPort()
                    >>> mock.read_until(b"\\n")  # Returns fixed response
                    b'test\\r\\n'
                """
                return b"test\r\n"

            def readline(self) -> bytes:
                """Read line from serial port (stub implementation).

                Stub method for protocol compliance testing. Returns fixed
                line response to verify SerialPort protocol signature.

                Args:
                    self: MockSerialPort instance (implicit).

                Returns:
                    bytes: Fixed test response "test\\n".

                Raises:
                    No exceptions raised. Stub always succeeds.

                Example:
                    >>> mock = MockSerialPort()
                    >>> mock.readline()  # Returns fixed line
                    b'test\\n'
                """
                return b"test\n"

            def write(self, data: bytes) -> int:
                """Write data to serial port (stub implementation).

                Stub method for protocol compliance testing. Returns length
                of data to verify SerialPort protocol signature.

                Args:
                    data: Bytes to write to serial port.

                Returns:
                    int: Number of bytes written (always len(data)).

                Raises:
                    No exceptions raised. Stub always succeeds.

                Example:
                    >>> mock = MockSerialPort()
                    >>> mock.write(b"test")  # Returns byte count
                    4
                """
                return len(data)

            def reset_input_buffer(self) -> None:
                """Clear input buffer (stub implementation).

                Stub method for protocol compliance testing. No-op that
                verifies SerialPort protocol signature.

                Args:
                    self: MockSerialPort instance (implicit).

                Returns:
                    None: No return value.

                Raises:
                    No exceptions raised. Stub always succeeds.

                Example:
                    >>> mock = MockSerialPort()
                    >>> mock.reset_input_buffer()  # No-op, returns None
                """
                pass

            def close(self) -> None:
                """Close serial port connection (stub implementation).

                Stub method for protocol compliance testing. Sets is_open
                flag to False to verify SerialPort protocol signature.

                Args:
                    self: MockSerialPort instance (implicit).

                Returns:
                    None: No return value.

                Raises:
                    No exceptions raised. Stub always succeeds.

                Example:
                    >>> mock = MockSerialPort()
                    >>> mock.is_open
                    True
                    >>> mock.close()
                    >>> mock.is_open
                    False
                """
                self.is_open = False

        mock = MockSerialPort()
        assert_implements_protocol(mock, SerialPort)


class TestMotorProtocolCompliance:
    """Verify motor controllers implement MotorController protocol."""

    def test_stub_motor_controller_implements_protocol(self) -> None:
        """StubMotorController satisfies MotorController protocol.

        Verifies that the stub motor controller used for testing
        implements all required MotorController methods.

        Arrangement:
        1. Import StubMotorController and MotorController protocol.
        2. Create controller instance.

        Action:
        Call assert_implements_protocol() with controller and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(controller, MotorController) returns True.
        """
        from telescope_mcp.drivers.motors import StubMotorController
        from telescope_mcp.drivers.motors.types import MotorController

        controller = StubMotorController()
        assert_implements_protocol(controller, MotorController)


class TestDeviceProtocolCompliance:
    """Verify device-level protocols (Clock, OverlayRenderer, etc.)."""

    def test_system_clock_implements_protocol(self) -> None:
        """SystemClock satisfies Clock protocol.

        Verifies that the system clock implementation correctly
        implements all Clock protocol methods.

        Arrangement:
        1. Import SystemClock and Clock protocol.
        2. Create clock instance.

        Action:
        Call assert_implements_protocol() with clock and protocol.

        Assertion Strategy:
        Validates protocol compliance by confirming:
        - No assertion error raised.
        - isinstance(clock, Clock) returns True.
        """
        from telescope_mcp.devices.camera import Clock, SystemClock

        clock = SystemClock()
        assert_implements_protocol(clock, Clock)
