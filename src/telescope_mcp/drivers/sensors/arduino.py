"""Arduino Nano BLE33 Sense sensor driver.

Communicates with Arduino over serial USB to read IMU, magnetometer,
temperature, and humidity data.

Hardware: Arduino Nano BLE33 Sense
- LSM9DS1 IMU (accelerometer + magnetometer)
- HTS221 (temperature + humidity)

Serial Protocol:
- Baud rate: 115200
- Output format: aX\\taY\\taZ\\tmX\\tmY\\tmZ\\ttemp\\thumidity\\r\\n
- Commands: RESET, STATUS, CALIBRATE, STOP, START

Example:
    from telescope_mcp.drivers.sensors import ArduinoSensorDriver

    driver = ArduinoSensorDriver()
    sensors = driver.get_available_sensors()

    if sensors:
        instance = driver.open(sensors[0]["port"])
        reading = instance.read()
        print(f"ALT: {reading.altitude:.2f}°, AZ: {reading.azimuth:.2f}°")
        instance.close()

Testing:
    The driver supports dependency injection for testing without hardware:

    from telescope_mcp.drivers.serial import SerialPort
    from telescope_mcp.drivers.sensors.arduino import ArduinoSensorInstance

    # Create a mock serial port
    class MockSerial:
        def __init__(self):
            self.is_open = True
            self._data = b"0.5\\t0.0\\t0.87\\t30.0\\t0.0\\t40.0\\t22.5\\t55.0\\r\\n"

        def read_until(self, terminator):
            return self._data

        # ... implement other SerialPort methods

    # Inject mock into instance
    instance = ArduinoSensorInstance._create_with_serial(MockSerial(), "/dev/mock")
"""

from __future__ import annotations

import math
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from telescope_mcp.drivers.sensors.types import SensorReading
from telescope_mcp.drivers.serial import PortEnumerator, SerialPort
from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class ArduinoSensorInstance:
    """Connection to Arduino Nano BLE33 Sense sensor.

    Reads sensor data via serial USB and provides calculated
    altitude/azimuth from IMU data.

    Supports dependency injection for testing without hardware.
    Use `_create_with_serial()` class method for testing.
    """

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        """Open serial connection to Arduino.

        Args:
            port: Serial port (e.g., /dev/ttyACM0).
            baudrate: Serial baud rate (default 115200).

        Raises:
            RuntimeError: If serial connection fails.
        """
        try:
            import serial as serial_module

            serial_port = serial_module.Serial(port, baudrate=baudrate, timeout=1.0)
        except ImportError:
            raise RuntimeError("pyserial not installed. Run: pdm add pyserial")
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {port}: {e}")

        self._init_with_serial(serial_port, port)

        # Start background reader and wait for first reading
        self._start_reader()
        time.sleep(0.5)

        logger.info("Arduino sensor connected", port=port)

    @classmethod
    def _create_with_serial(
        cls,
        serial_port: SerialPort,
        port_name: str = "/dev/mock",
        start_reader: bool = False,
    ) -> ArduinoSensorInstance:
        """Create instance with injected serial port (for testing).

        Args:
            serial_port: Mock or real serial port implementing SerialPort protocol.
            port_name: Port name for identification.
            start_reader: Whether to start background reader thread.

        Returns:
            Configured ArduinoSensorInstance.

        Example:
            class MockSerial:
                is_open = True
                in_waiting = 0
                def read_until(self, expected=b"\\n", size=None):
                    return b"0.5\\t0.0\\t0.87\\t30\\t0\\t40\\t22.5\\t55\\r\\n"
                # ... other methods

            instance = ArduinoSensorInstance._create_with_serial(MockSerial())
        """
        instance = cls.__new__(cls)
        instance._init_with_serial(serial_port, port_name)
        if start_reader:
            instance._start_reader()
        return instance

    def _init_with_serial(self, serial_port: SerialPort, port_name: str) -> None:
        """Initialize instance state with given serial port.

        Args:
            serial_port: Serial port to use for communication.
            port_name: Port name for identification.
        """
        self._serial: SerialPort = serial_port
        self._port = port_name
        self._is_open = True

        # Latest readings (updated by background thread)
        self._accelerometer: dict[str, float] = {}
        self._magnetometer: dict[str, float] = {}
        self._temperature: float = 0.0
        self._humidity: float = 0.0
        self._raw_values: str = ""
        self._last_update: datetime | None = None

        # Calibration parameters
        self._cal_alt_scale = 1.0
        self._cal_alt_offset = 0.0
        self._cal_az_scale = 1.0
        self._cal_az_offset = 0.0

        # Tilt calibration (from notebook)
        self._tilt_m = 1.0  # Scale factor
        self._tilt_b = 0.0  # Offset

        # Background reader state
        self._stop_reading = False
        self._reader_thread: threading.Thread | None = None

    def _start_reader(self) -> None:
        """Start the background reader thread."""
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def parse_line(self, line: str) -> bool:
        """Parse a single line of sensor data.

        Updates internal state if line is valid sensor data.
        Useful for testing without background thread.

        Args:
            line: Tab-separated sensor values string.

        Returns:
            True if line was valid sensor data, False otherwise.

        Example:
            instance = ArduinoSensorInstance._create_with_serial(mock_serial)
            instance.parse_line("0.5\\t0.0\\t0.87\\t30.0\\t0.0\\t40.0\\t22.5\\t55.0")
            reading = instance.read()
        """
        line = line.strip()
        if not line:
            return False

        # Skip command responses
        if line.startswith(("INFO:", "OK:", "ERROR:", "CMD:", "===", "---")):
            return False

        values = line.split("\t")

        try:
            if len(values) == 8:
                # Full format: aX, aY, aZ, mX, mY, mZ, temp, humidity
                self._accelerometer = {
                    "aX": float(values[0]),
                    "aY": float(values[1]),
                    "aZ": float(values[2]),
                }
                self._magnetometer = {
                    "mX": float(values[3]),
                    "mY": float(values[4]),
                    "mZ": float(values[5]),
                }
                self._temperature = float(values[6])
                self._humidity = float(values[7])
                self._raw_values = line
                self._last_update = datetime.now(UTC)
                return True

            elif len(values) == 6:
                # Legacy format: aX, aZ, aY, mX, mZ, mY (note different order)
                self._accelerometer = {
                    "aX": float(values[0]),
                    "aZ": float(values[1]),
                    "aY": float(values[2]),
                }
                self._magnetometer = {
                    "mX": float(values[3]),
                    "mZ": float(values[4]),
                    "mY": float(values[5]),
                }
                self._raw_values = line
                self._last_update = datetime.now(UTC)
                return True

        except (ValueError, UnicodeDecodeError):
            pass  # Skip malformed lines

        return False

    def _read_loop(self) -> None:
        """Background thread to continuously read sensor data."""
        while not self._stop_reading and self._is_open:
            try:
                line = self._serial.read_until(b"\r\n").decode().strip()
                self.parse_line(line)
            except Exception as e:
                if self._is_open:
                    logger.warning("Sensor read error", error=str(e))

    def get_info(self) -> dict:
        """Get sensor information.

        Returns:
            Dict with sensor type, port, and capabilities.
        """
        return {
            "type": "arduino_ble33",
            "name": "Arduino Nano BLE33 Sense",
            "port": self._port,
            "has_accelerometer": True,
            "has_magnetometer": True,
            "has_temperature": True,
            "has_humidity": True,
            "sample_rate_hz": 10.0,
        }

    def _calculate_altitude(self) -> float:
        """Calculate altitude from accelerometer data.

        Uses tilt calculation from accelerometer X, Y, Z values.

        Returns:
            Altitude in degrees (0-90).
        """
        ax = self._accelerometer.get("aX", 0)
        ay = self._accelerometer.get("aY", 0)
        az = self._accelerometer.get("aZ", 0)

        if ay == 0 and az == 0:
            return 0.0

        # Tilt calculation (from digikey article)
        tilt_rad = math.atan(ax / math.sqrt(ay * ay + az * az))
        raw_alt = tilt_rad * (180 / math.pi)

        # Apply calibration
        calibrated = self._tilt_m * raw_alt + self._tilt_b

        # Apply transform
        return self._cal_alt_scale * calibrated + self._cal_alt_offset

    def _calculate_azimuth(self) -> float:
        """Calculate azimuth from magnetometer data.

        Uses atan2 of magnetometer Y, X values.

        Returns:
            Azimuth in degrees (0-360).
        """
        mx = self._magnetometer.get("mX", 0)
        my = self._magnetometer.get("mY", 0)

        if mx == 0 and my == 0:
            return 0.0

        # Heading from magnetometer
        heading = math.atan2(my, mx) * (180 / math.pi)

        # Normalize to 0-360
        if heading < 0:
            heading += 360

        # Apply transform
        return (self._cal_az_scale * heading + self._cal_az_offset) % 360

    def read(self) -> SensorReading:
        """Read current sensor values.

        Returns latest values from background reader thread.

        Returns:
            SensorReading with all sensor data.

        Raises:
            RuntimeError: If sensor is closed or no data available.
        """
        if not self._is_open:
            raise RuntimeError("Sensor is closed")

        if not self._accelerometer:
            raise RuntimeError("No sensor data available yet")

        return SensorReading(
            accelerometer=self._accelerometer.copy(),
            magnetometer=self._magnetometer.copy(),
            altitude=self._calculate_altitude(),
            azimuth=self._calculate_azimuth(),
            temperature=self._temperature,
            humidity=self._humidity,
            timestamp=self._last_update or datetime.now(UTC),
            raw_values=self._raw_values,
        )

    def calibrate(
        self,
        true_altitude: float,
        true_azimuth: float,
    ) -> None:
        """Calibrate sensor to known true position.

        Sets calibration transform so that current reading maps to
        the provided true position.

        Args:
            true_altitude: Known true altitude in degrees.
            true_azimuth: Known true azimuth in degrees.
        """
        # Get current calculated values (before transform)
        current_alt = self._calculate_altitude()
        current_az = self._calculate_azimuth()

        # Calculate offsets
        self._cal_alt_offset = true_altitude - current_alt
        self._cal_az_offset = true_azimuth - current_az

        logger.info(
            "Sensor calibrated",
            alt_offset=self._cal_alt_offset,
            az_offset=self._cal_az_offset,
        )

    def set_tilt_calibration(self, slope: float, intercept: float) -> None:
        """Set tilt calibration parameters.

        Linear calibration: corrected = slope * raw + intercept

        Args:
            slope: Scale factor (m in y = mx + b).
            intercept: Offset (b in y = mx + b).
        """
        self._tilt_m = slope
        self._tilt_b = intercept
        logger.info(
            "Tilt calibration set",
            slope=slope,
            intercept=intercept,
        )

    def send_command(
        self,
        command: str,
        wait_response: bool = True,
        timeout: float = 5.0,
    ) -> str:
        """Send command to Arduino and optionally wait for response.

        Args:
            command: Command string (RESET, STATUS, CALIBRATE, STOP, START).
            wait_response: Whether to wait for response.
            timeout: Seconds to wait for response.

        Returns:
            Response string from Arduino (or empty if not waiting).
        """
        # Clear input buffer
        self._serial.reset_input_buffer()

        # Send command
        self._serial.write(f"{command}\n".encode())
        logger.debug("Sent command", command=command)

        if not wait_response:
            return ""

        # Collect response
        response_lines = []
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            if self._serial.in_waiting:
                line = self._serial.readline().decode().strip()
                response_lines.append(line)

                # Check for end markers
                if line.startswith("===") and len(response_lines) > 1:
                    break
                if line.startswith(("OK:", "ERROR:")):
                    break
            time.sleep(0.01)

        response = "\n".join(response_lines)
        logger.debug("Command response", command=command, response=response[:100])
        return response

    def reset(self) -> None:
        """Reset/reinitialize the Arduino sensors."""
        self.send_command("RESET", timeout=5.0)

    def get_status(self) -> dict:
        """Get sensor status from Arduino.

        Returns:
            Dict with parsed status information.
        """
        response = self.send_command("STATUS", timeout=3.0)

        return {
            "connected": self._is_open,
            "type": "arduino_ble33",
            "port": self._port,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "raw_status": response,
            "calibrated": self._cal_alt_offset != 0 or self._cal_az_offset != 0,
        }

    def calibrate_magnetometer(self) -> str:
        """Run Arduino magnetometer calibration routine.

        User should rotate sensor during calibration (~10 seconds).

        Returns:
            Response from Arduino with calibration results.
        """
        return self.send_command("CALIBRATE", timeout=15.0)

    def stop_output(self) -> None:
        """Stop sensor output stream."""
        self.send_command("STOP", wait_response=False)

    def start_output(self) -> None:
        """Resume sensor output stream."""
        self.send_command("START", wait_response=False)

    def close(self) -> None:
        """Close the serial connection."""
        self._stop_reading = True
        self._is_open = False

        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

        if self._serial and self._serial.is_open:
            self._serial.close()

        logger.info("Arduino sensor closed", port=self._port)


class ArduinoSensorDriver:
    """Driver for Arduino Nano BLE33 Sense sensors.

    Discovers Arduino devices on serial ports and creates connections.
    Supports dependency injection for testing without hardware.

    Example:
        driver = ArduinoSensorDriver()
        sensors = driver.get_available_sensors()

        for sensor in sensors:
            print(f"Found: {sensor['name']} on {sensor['port']}")

        if sensors:
            instance = driver.open(sensors[0]["port"])

    Testing:
        # Create mock port enumerator
        class MockComPort:
            device = "/dev/ttyMOCK0"
            description = "Arduino Nano"

        class MockListPorts:
            @staticmethod
            def comports():
                return [MockComPort()]

        driver = ArduinoSensorDriver._create_with_enumerator(MockListPorts())
        sensors = driver.get_available_sensors()
    """

    def __init__(self, baudrate: int = 115200) -> None:
        """Initialize driver with baud rate.

        Args:
            baudrate: Serial baud rate for Arduino communication.
        """
        self._baudrate = baudrate
        self._instance: ArduinoSensorInstance | None = None
        self._port_enumerator: PortEnumerator | None = None
        self._serial_factory: type | None = None

    @classmethod
    def _create_with_enumerator(
        cls,
        port_enumerator: PortEnumerator,
        serial_factory: type | None = None,
        baudrate: int = 115200,
    ) -> ArduinoSensorDriver:
        """Create driver with injected dependencies (for testing).

        Args:
            port_enumerator: Object with comports() method.
            serial_factory: Optional factory to create serial ports.
            baudrate: Serial baud rate.

        Returns:
            Configured ArduinoSensorDriver.
        """
        driver = cls.__new__(cls)
        driver._baudrate = baudrate
        driver._instance = None
        driver._port_enumerator = port_enumerator
        driver._serial_factory = serial_factory
        return driver

    def get_available_sensors(self) -> list[dict]:
        """List available Arduino sensors on serial ports.

        Scans serial ports for potential Arduino devices.

        Returns:
            List of sensor info dicts with id, type, name, port.
        """
        # Use injected enumerator or real pyserial
        if self._port_enumerator is not None:
            ports = self._port_enumerator.comports()
        else:
            try:
                import serial.tools.list_ports
            except ImportError:
                logger.warning("pyserial not installed, cannot scan ports")
                return []
            ports = serial.tools.list_ports.comports()

        sensors = []

        for i, port in enumerate(ports):
            # Check for Arduino-like devices
            desc = port.description.lower()
            if any(x in desc for x in ["arduino", "acm", "usb serial", "ch340"]):
                sensors.append(
                    {
                        "id": i,
                        "type": "arduino_ble33",
                        "name": f"Arduino Sensor ({port.device})",
                        "port": port.device,
                        "description": port.description,
                    }
                )

        logger.debug("Found sensors", count=len(sensors))
        return sensors

    def open(self, sensor_id: int | str = 0) -> ArduinoSensorInstance:
        """Open connection to Arduino sensor.

        Args:
            sensor_id: Either port path string (e.g., /dev/ttyACM0) or
                sensor index from get_available_sensors(). If int, looks
                up the port from available sensors list.

        Returns:
            ArduinoSensorInstance for reading sensor data.

        Raises:
            RuntimeError: If connection fails or sensor not found.
        """
        if self._instance is not None and self._instance._is_open:
            raise RuntimeError("Sensor already open")

        # Resolve sensor_id to port string
        if isinstance(sensor_id, int):
            sensors = self.get_available_sensors()
            if sensor_id < 0 or sensor_id >= len(sensors):
                raise RuntimeError(
                    f"Sensor index {sensor_id} out of range (0-{len(sensors) - 1})"
                )
            port = sensors[sensor_id]["port"]
        else:
            port = sensor_id

        self._instance = ArduinoSensorInstance(port, self._baudrate)
        return self._instance

    def _open_with_serial(
        self,
        serial_port: SerialPort,
        port_name: str = "/dev/mock",
    ) -> ArduinoSensorInstance:
        """Open with injected serial port (for testing).

        Args:
            serial_port: Mock serial port implementing SerialPort protocol.
            port_name: Port name for identification.

        Returns:
            ArduinoSensorInstance configured with mock serial.
        """
        if self._instance is not None and self._instance._is_open:
            raise RuntimeError("Sensor already open")

        self._instance = ArduinoSensorInstance._create_with_serial(
            serial_port, port_name, start_reader=False
        )
        return self._instance

    def close(self) -> None:
        """Close the current sensor instance."""
        if self._instance is not None:
            self._instance.close()
            self._instance = None
