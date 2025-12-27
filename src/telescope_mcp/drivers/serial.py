"""Serial communication protocols for hardware drivers.

Provides abstractions over pyserial for dependency injection and testing.
Used by both sensor drivers (Arduino) and motor drivers that communicate
via serial USB connections.

Protocols:
    SerialPort: Abstraction for serial port operations
    PortEnumerator: Abstraction for discovering serial ports

Example:
    # For testing - create mock implementations
    class MockSerialPort:
        is_open = True
        in_waiting = 0

        def read_until(self, expected=b"\\n", size=None):
            return b"test data\\r\\n"

        def write(self, data):
            return len(data)

        def close(self):
            self.is_open = False

    # Inject mock into driver
    instance = SomeDriver._create_with_serial(MockSerialPort())

Testing:
    The protocols enable unit testing without hardware by allowing
    injection of mock serial ports that simulate device responses.

    See tests/drivers/sensors/test_arduino.py for comprehensive examples
    of MockSerialPort and MockPortEnumerator implementations.
"""

from __future__ import annotations

from typing import Protocol


class SerialPort(Protocol):  # pragma: no cover
    """Protocol for serial port operations.

    Abstraction over pyserial.Serial to enable testing without hardware.
    Implement this protocol to create mock serial ports for unit tests.

    The protocol matches the subset of pyserial.Serial interface used
    by telescope drivers.

    Attributes:
        is_open: Whether the port is currently open.
        in_waiting: Number of bytes waiting to be read.

    Example:
        class MockSerial:
            is_open = True
            in_waiting = 0
            _buffer = b""

            def queue_data(self, data: bytes):
                self._buffer += data
                self.in_waiting = len(self._buffer)

            def read_until(self, expected=b"\\n", size=None):
                data = self._buffer
                self._buffer = b""
                self.in_waiting = 0
                return data

            def write(self, data):
                return len(data)

            def close(self):
                self.is_open = False
    """

    @property
    def is_open(self) -> bool:
        """Whether the port is currently open."""
        ...

    @property
    def in_waiting(self) -> int:
        """Number of bytes waiting to be read."""
        ...

    def read_until(self, expected: bytes = b"\n", size: int | None = None) -> bytes:
        """Read until expected byte sequence or timeout.

        Args:
            expected: Byte sequence to read until (default newline).
            size: Maximum bytes to read (None for unlimited).

        Returns:
            Bytes read from port, including terminator if found.
        """
        ...

    def readline(self) -> bytes:
        """Read a line (until \\n).

        Returns:
            Bytes up to and including newline character.
        """
        ...

    def write(self, data: bytes) -> int | None:
        """Write bytes to port.

        Args:
            data: Bytes to write.

        Returns:
            Number of bytes written, or None if write fails.
        """
        ...

    def reset_input_buffer(self) -> None:
        """Clear the input buffer.

        Discards any data waiting to be read.
        """
        ...

    def close(self) -> None:
        """Close the port.

        Releases the serial port resource.
        """
        ...


class PortEnumerator(Protocol):  # pragma: no cover
    """Protocol for enumerating serial ports.

    Abstraction over serial.tools.list_ports for testing.
    Allows injection of mock port lists without scanning hardware.

    Example:
        class MockComPort:
            device = "/dev/ttyACM0"
            description = "Arduino Nano"

        class MockPortEnumerator:
            def comports(self):
                return [MockComPort()]

        driver = SomeDriver._create_with_enumerator(MockPortEnumerator())
    """

    def comports(self) -> list:
        """Return list of available serial ports.

        Returns:
            List of port info objects with 'device' and 'description' attrs.
        """
        ...


__all__ = [
    "SerialPort",
    "PortEnumerator",
]
