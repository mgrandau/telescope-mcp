"""Serial communication protocols for hardware drivers.

Provides abstractions over pyserial for dependency injection and testing.
Used by both sensor drivers (Arduino) and motor drivers that communicate
via serial USB connections.

Protocols:
    SerialPort: Abstraction for serial port operations
    PortEnumerator: Abstraction for discovering serial ports

Functions:
    list_serial_ports: Wrapper for pyserial port enumeration with graceful fallback

Example:
    # For testing - create mock implementations
    class MockSerialPort:
        is_open = True
        in_waiting = 0

        def read_until(self, expected=b"\n", size=None):
            return b"test data\r\n"

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

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SerialPort(Protocol):  # pragma: no cover
    """Protocol for serial port operations.

    Abstraction over pyserial.Serial to enable testing without hardware.
    Implement this protocol to create mock serial ports for unit tests.

    The protocol matches the subset of pyserial.Serial interface used
    by telescope drivers.

    Attributes:
        is_open: Whether the port is currently open.
        in_waiting: Number of bytes waiting to be read.

    Note:
        pyserial's Serial is thread-safe for read/write operations.
        Mock implementations should document their threading guarantees.

    Example:
        class MockSerial:
            is_open = True
            in_waiting = 0
            _buffer = b""

            def queue_data(self, data: bytes):
                self._buffer += data
                self.in_waiting = len(self._buffer)

            def read_until(self, expected=b"\n", size=None):
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
        """Check if serial port connection is currently active.

        Indicates whether the serial port is open and available for I/O.
        Used to verify connection state before operations and detect
        disconnections.

        Business context: Essential for robust error handling in telescope
        hardware drivers. Checking is_open before operations prevents
        cryptic errors and enables graceful reconnection.

        Args:
            No arguments (property accessor).

        Returns:
            True if port is open and ready for communication, False if
            closed or disconnected.

        Raises:
            No exceptions raised. Safe to call in any state.

        Example:
            >>> if serial.is_open:
            ...     serial.write(b"command")
            ... else:
            ...     raise RuntimeError("Port disconnected")
        """
        ...

    @property
    def in_waiting(self) -> int:
        """Get number of bytes available to read without blocking.

        Returns count of bytes in the receive buffer. Used to check for
        available data before reading, enabling non-blocking polling of
        serial devices.

        Business context: Enables efficient polling of Arduino sensors
        without blocking. The background reader thread uses this to check
        for new data at high frequency without stalling.

        Args:
            No arguments (property accessor).

        Returns:
            Number of bytes waiting in receive buffer. Zero if buffer empty.

        Raises:
            No exceptions raised. Returns 0 if port closed.

        Example:
            >>> if serial.in_waiting > 0:
            ...     data = serial.readline()
        """
        ...

    def read_until(self, expected: bytes = b"\n", size: int | None = None) -> bytes:
        """Read bytes until delimiter found or timeout occurs.

        Reads from serial port until the expected byte sequence is found,
        maximum size is reached, or the configured timeout expires.
        Primary method for reading line-oriented serial protocols.

        Business context: Arduino sensors and motor controllers use
        line-based protocols with \\r\\n terminators. This method enables
        reliable message framing regardless of timing variations.

        Args:
            expected: Byte sequence marking end of message. Default b"\\n"
                for newline. Arduino uses b"\\r\\n" typically.
            size: Maximum bytes to read before returning. None (default)
                means read until terminator or timeout.

        Returns:
            Bytes read including the terminator sequence if found.
            May be partial if timeout occurs before terminator.

        Raises:
            SerialException: If port is closed or hardware error occurs.

        Example:
            >>> line = serial.read_until(b\"\\r\\n\")
            >>> values = line.decode().strip().split(\"\\t\")
        """
        ...

    def readline(self) -> bytes:
        """Read a complete line ending with newline character.

        Convenience method equivalent to read_until(b"\\n"). Reads until
        newline character or timeout. Standard method for line-oriented
        serial protocols.

        Business context: Used by Arduino sensor driver for reading
        continuous sensor output where each reading is one line of
        tab-separated values.

        Returns:
            Bytes up to and including newline. May be empty or partial
            if timeout occurs. Decode with .decode().strip() for string.

        Raises:
            SerialException: If port is closed or hardware error occurs.

        Example:
            >>> line = serial.readline()
            >>> if line:
            ...     reading = line.decode().strip()
        """
        ...

    def write(self, data: bytes) -> int | None:
        """Write bytes to serial port for transmission.

        Sends data to the connected device. For telescope hardware, this
        sends commands to Arduino sensors or motor controllers.

        Business context: Enables sending commands (RESET, STATUS,
        CALIBRATE) to Arduino sensors and movement commands to motor
        controllers. Thread-safe in pyserial but should be serialized
        at driver level.

        Args:
            data: Bytes to transmit. Commands typically end with b"\\n".
                Example: b"STATUS\\n" for Arduino status query.

        Returns:
            Number of bytes written if successful. None if write fails
            (pyserial may return None on certain error conditions).

        Raises:
            SerialException: If port is closed or hardware error occurs.
            SerialTimeoutError: If write times out (rare with USB serial).

        Example:
            >>> bytes_sent = serial.write(b\"RESET\\n\")
            >>> if bytes_sent:
            ...     response = serial.read_until(b\"\\r\\n\")
        """
        ...

    def reset_input_buffer(self) -> None:
        """Clear all data from the serial input buffer.

        Discards any bytes waiting to be read. Used before sending
        commands to ensure response is fresh, not stale buffered data.

        Business context: Essential for reliable command-response
        protocols. Arduino sensors continuously output data; clearing
        buffer before commands ensures the response matches the command.

        Returns:
            None.

        Raises:
            SerialException: If port is closed.

        Example:
            >>> serial.reset_input_buffer()  # Clear stale data
            >>> serial.write(b\"STATUS\\n\")  # Send command
            >>> response = serial.read_until()  # Get fresh response
        """
        ...

    def close(self) -> None:
        """Close the serial port and release system resources.

        Terminates the serial connection and releases the port for other
        processes. Should be called when done with the device or during
        cleanup.

        Business context: Proper resource cleanup prevents port locking
        issues common with USB serial devices. Essential for graceful
        shutdown and reconnection scenarios.

        Returns:
            None.

        Raises:
            None. Safe to call multiple times or on already-closed port.

        Example:
            >>> try:
            ...     # Use serial port
            ... finally:
            ...     serial.close()
        """
        ...


@runtime_checkable
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

    def comports(self) -> list[Any]:
        """Enumerate available serial ports on the system.

        Scans system for serial ports (USB, hardware, virtual). Returns
        list of port info objects with device paths and descriptions.
        Used for discovering Arduino sensors and motor controllers.

        Business context: Enables automatic discovery of telescope
        hardware without manual configuration. Users don't need to know
        port names - drivers scan and identify devices by description.

        Returns:
            List of port info objects. Each object has attributes:
            - device: Port path (e.g., "/dev/ttyACM0", "COM3")
            - description: Human-readable description (e.g., "Arduino Nano")
            - hwid: Hardware ID for precise device identification

        Raises:
            None. Returns empty list if no ports found or on error.

        Example:
            >>> ports = enumerator.comports()
            >>> for port in ports:
            ...     if "Arduino" in port.description:
            ...         print(f"Found: {port.device}")
        """
        ...


def list_serial_ports() -> list[Any]:  # pragma: no cover
    """List available serial ports on the system.

    Wrapper around pyserial's list_ports.comports() that handles
    ImportError gracefully. Use this instead of direct pyserial import
    to enable testing without pyserial installed.

    Business context: Provides a single point of control for port
    enumeration. Tests can mock this function rather than dealing
    with pyserial import mechanics.

    Returns:
        List of port info objects with device/description attributes.
        Empty list if pyserial not installed.

    Raises:
        No exceptions raised. Returns empty list on ImportError.

    Example:
        >>> from telescope_mcp.drivers.serial import list_serial_ports
        >>> ports = list_serial_ports()
        >>> for p in ports:
        ...     print(f"{p.device}: {p.description}")
    """
    try:
        import serial.tools.list_ports

        return list(serial.tools.list_ports.comports())
    except ImportError:
        return []


__all__ = [
    "SerialPort",
    "PortEnumerator",
    "list_serial_ports",
]
