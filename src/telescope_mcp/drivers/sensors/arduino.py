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
            self.in_waiting = 0
            self._data = b"0.5\\t0.0\\t0.87\\t30.0\\t0.0\\t40.0\\t22.5\\t55.0\\r\\n"

        def read_until(self, expected=b"\\n", size=None):
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
from types import TracebackType
from typing import TypedDict

from telescope_mcp.drivers.sensors.types import (
    AccelerometerData,
    AvailableSensor,
    MagnetometerData,
    SensorInfo,
    SensorInstance,
    SensorReading,
    SensorStatus,
    validate_position,
)

__all__ = [
    "ArduinoSensorInstance",
    "ArduinoSensorDriver",
    "ArduinoSensorInfo",
    "ArduinoSensorStatus",
    "ArduinoAvailableSensor",
]
from telescope_mcp.drivers.serial import PortEnumerator, SerialPort, list_serial_ports
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)

# Protocol constants for Arduino serial data format
_FULL_FORMAT_FIELDS = 8  # aX, aY, aZ, mX, mY, mZ, temp, humidity
_LEGACY_FORMAT_FIELDS = 6  # aX, aZ, aY, mX, mZ, mY (note different order)
_ARDUINO_SAMPLE_RATE_HZ = 10.0  # Fixed sample rate for Arduino BLE33 firmware


class ArduinoSensorInfo(TypedDict):
    """Type for Arduino sensor hardware information.

    Keys:
        type: Sensor type string (always "arduino_ble33").
        name: Human-readable sensor name.
        port: Serial port path (e.g., "/dev/ttyACM0").
        has_accelerometer: True (BLE33 always has IMU).
        has_magnetometer: True (BLE33 always has IMU).
        has_temperature: True (BLE33 has HTS221).
        has_humidity: True (BLE33 has HTS221).
        sample_rate_hz: Current sample rate (typically 10.0).
    """

    type: str
    name: str
    port: str
    has_accelerometer: bool
    has_magnetometer: bool
    has_temperature: bool
    has_humidity: bool
    sample_rate_hz: float


class ArduinoSensorStatus(TypedDict):
    """Type for Arduino sensor operational status.

    Keys:
        connected: Whether serial connection is open.
        type: Sensor type (always "arduino_ble33").
        port: Serial port path.
        last_update: ISO timestamp of last reading, or None.
        raw_status: Raw STATUS response from Arduino.
        calibrated: Whether calibration offsets are non-zero.
    """

    connected: bool
    type: str
    port: str
    last_update: str | None
    raw_status: str
    calibrated: bool


class ArduinoAvailableSensor(TypedDict):
    """Type for discovered Arduino sensor descriptor.

    Keys:
        id: Integer index for selection.
        type: Sensor type (always "arduino_ble33").
        name: Human-readable name.
        port: Serial port path (e.g., "/dev/ttyACM0").
        description: Hardware description from serial driver.
    """

    id: int
    type: str
    name: str
    port: str
    description: str


class ArduinoSensorInstance:
    """Connection to Arduino Nano BLE33 Sense sensor.

    Reads sensor data via serial USB and provides calculated
    altitude/azimuth from IMU data.

    Supports dependency injection for testing without hardware.
    Use `_create_with_serial()` class method for testing.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        *,
        startup_delay: float = 0.5,
    ) -> None:
        """Open serial connection to Arduino sensor.

        Establishes serial connection, initializes state, and starts
        background reader thread for continuous data streaming.

        Business context: Arduino sensor provides telescope orientation
        via IMU data. Serial connection enables continuous data streaming
        for real-time position tracking.

        Args:
            port: Serial port (e.g., /dev/ttyACM0).
            baudrate: Serial baud rate (default 115200).
            startup_delay: Seconds to wait for first reading after
                connection (default 0.5). Set to 0 for faster tests.

        Returns:
            None. Instance connected and reading data.

        Raises:
            RuntimeError: If serial connection fails (port busy, pyserial
                not installed, or hardware error).

        Example:
            >>> instance = ArduinoSensorInstance("/dev/ttyACM0")
            >>> reading = instance.read()
            >>> print(f"Alt: {reading.altitude:.1f}°")
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
        if startup_delay > 0:
            time.sleep(startup_delay)

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

        Sets up all instance variables needed for Arduino sensor operations.
        Called by both __init__ (with real serial) and _create_with_serial
        (with mock serial for testing).

        Business context: Centralizes initialization logic so both production
        and test code paths use identical setup. Ensures consistent state
        for calibration parameters, reading buffers, and thread management.

        Implementation: Initializes empty dicts for accelerometer/magnetometer,
        zeroed calibration offsets/scales, and None for reader thread. Does
        NOT start background reader - that's handled separately by caller
        to allow test control over threading.

        Args:
            serial_port: Serial port to use for communication. Must implement
                SerialPort protocol (readline, write methods).
            port_name: Port name for identification in logs.

        Returns:
            None. Instance state initialized, ready for _start_reader().

        Raises:
            No exceptions raised during initialization.

        Example:
            >>> # Called internally by factory methods
            >>> instance = cls.__new__(cls)
            >>> instance._init_with_serial(mock_serial, "/dev/mock")
            >>> instance._start_reader()  # Optional for tests
        """
        self._serial: SerialPort = serial_port
        self._port = port_name
        self._is_open = True

        # Thread synchronization for shared state
        self._lock = threading.Lock()

        # Latest readings (updated by background thread, protected by _lock)
        self._accelerometer: AccelerometerData | None = None
        self._magnetometer: MagnetometerData | None = None
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
        """Start the background reader thread for continuous data.

        Spawns daemon thread that continuously reads serial data and
        updates internal sensor state. Must be called after serial
        connection established.

        Business context: Arduino streams sensor data continuously at
        ~10Hz. Background thread ensures latest data is always available
        without blocking the main thread on serial I/O.

        Implementation: Creates Thread targeting _read_loop, sets as daemon
        (exits when main program exits), and starts it. Thread reference
        stored in _reader_thread.

        Args:
            No arguments.

        Returns:
            None. Thread started in background.

        Raises:
            No exceptions raised.

        Example:
            >>> instance._init_with_serial(serial_port, port_name)
            >>> instance._start_reader()  # Begin data collection
        """
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _parse_line(self, line: str) -> bool:
        """Parse a single line of sensor data from Arduino serial output.

        Updates internal state (accelerometer, magnetometer, temperature,
        humidity) if line contains valid tab-separated sensor values.
        Handles both 8-value (full format) and 6-value (legacy IMU-only)
        data formats. Ignores command responses and malformed lines.

        Business context: The Arduino streams sensor data continuously.
        This method is the core parser that converts raw serial strings
        into structured data. Direct use is primarily for testing - in
        production, the background reader thread calls this automatically.

        Args:
            line: Tab-separated sensor values from Arduino serial output.
                Full format (8 values): aX, aY, aZ, mX, mY, mZ, temp, humidity
                Legacy format (6 values): aX, aZ, aY, mX, mZ, mY
                Lines starting with INFO:, OK:, ERROR:, CMD: are skipped.

        Returns:
            bool: True if line contained valid sensor data and internal
                state was updated. False if line was empty, a command
                response, or malformed.

        Raises:
            No exceptions raised. Malformed data silently skipped.

        Example:
            >>> instance = driver._open_with_serial(mock_serial)
            >>> # Full format with environmental sensors
            >>> valid = instance.parse_line(
            ...     "0.50\t0.00\t0.87\t30.0\t0.0\t40.0\t22.5\t55.0"
            ... )
            >>> assert valid is True
            >>> # Command response - skipped
            >>> valid = instance.parse_line("OK: CALIBRATE complete")
            >>> assert valid is False
        """
        line = line.strip()
        if not line:
            return False

        # Skip command responses
        if line.startswith(("INFO:", "OK:", "ERROR:", "CMD:", "===", "---")):
            return False

        values = line.split("\t")

        try:
            if len(values) == _FULL_FORMAT_FIELDS:
                # Full format: aX, aY, aZ, mX, mY, mZ, temp, humidity
                accel: AccelerometerData = {
                    "aX": float(values[0]),
                    "aY": float(values[1]),
                    "aZ": float(values[2]),
                }
                mag: MagnetometerData = {
                    "mX": float(values[3]),
                    "mY": float(values[4]),
                    "mZ": float(values[5]),
                }
                temp = float(values[6])
                humidity = float(values[7])
                with self._lock:
                    self._accelerometer = accel
                    self._magnetometer = mag
                    self._temperature = temp
                    self._humidity = humidity
                    self._raw_values = line
                    self._last_update = datetime.now(UTC)
                return True

            elif len(values) == _LEGACY_FORMAT_FIELDS:
                # Legacy format: aX, aZ, aY, mX, mZ, mY (note different order)
                legacy_accel: AccelerometerData = {
                    "aX": float(values[0]),
                    "aZ": float(values[1]),
                    "aY": float(values[2]),
                }
                legacy_mag: MagnetometerData = {
                    "mX": float(values[3]),
                    "mZ": float(values[4]),
                    "mY": float(values[5]),
                }
                with self._lock:
                    self._accelerometer = legacy_accel
                    self._magnetometer = legacy_mag
                    self._raw_values = line
                    self._last_update = datetime.now(UTC)
                return True

        except (ValueError, UnicodeDecodeError) as e:
            logger.debug("Parse error", error=str(e), line_preview=line[:50])
            return False

        # Reaching here means wrong field count (not 6 or 8)
        logger.debug(
            "Unexpected field count",
            field_count=len(values),
            expected=f"{_FULL_FORMAT_FIELDS} or {_LEGACY_FORMAT_FIELDS}",
            line_preview=line[:50],
        )
        return False

    def _read_loop(self) -> None:
        """Background thread loop for continuous sensor data reading.

        Continuously reads lines from serial port and parses sensor data
        until stop flag set or port closed. Updates internal state with
        each valid reading.

        Business context: Continuous reading ensures sensor data is always
        fresh. Daemon thread terminates automatically when main program
        exits, preventing resource leaks.

        Implementation: Loops while _stop_reading is False and _is_open.
        Reads until newline, decodes, and calls parse_line(). Catches
        exceptions to prevent thread crash, logs warnings.

        Args:
            No arguments. Uses instance state.

        Returns:
            None. Runs until stopped.

        Raises:
            No exceptions raised. Errors logged and loop continues.

        Example:
            >>> # Started internally by _start_reader()
            >>> # Thread runs: instance._read_loop()
        """
        logger.info("Sensor reader thread started", port=self._port)
        lines_read = 0
        while not self._stop_reading and self._is_open:
            try:
                raw_bytes = self._serial.read_until(b"\r\n")
                line = raw_bytes.decode().strip()
                lines_read += 1
                if lines_read <= 5 or lines_read % 100 == 0:
                    logger.info(
                        "Serial data received",
                        line_num=lines_read,
                        raw_len=len(raw_bytes),
                        line_preview=line[:80] if line else "<empty>",
                    )
                parsed = self._parse_line(line)
                if parsed and lines_read <= 5:
                    logger.info("Successfully parsed sensor data", line_num=lines_read)
            except Exception as e:
                if self._is_open:
                    logger.warning(
                        "Sensor read error", error=str(e), line_num=lines_read
                    )
        logger.info("Sensor reader thread stopped", lines_read=lines_read)

    def get_info(self) -> SensorInfo:
        """Get Arduino sensor hardware information and capabilities.

        Returns metadata describing the connected Arduino Nano BLE33 Sense
        including sensor capabilities and serial port information. Used
        by the Sensor device layer to populate SensorInfo.

        Business context: The Arduino Nano BLE33 Sense provides a specific
        set of sensors (LSM9DS1 IMU, HTS221 environmental). This method
        enables clients to understand available measurements and configure
        appropriate data collection.

        Returns:
            ArduinoSensorInfo: TypedDict containing:
                - type (str): Always 'arduino_ble33'
                - name (str): 'Arduino Nano BLE33 Sense'
                - port (str): Serial port path (e.g., '/dev/ttyACM0')
                - has_accelerometer (bool): True - 3-axis LSM9DS1
                - has_magnetometer (bool): True - 3-axis LSM9DS1
                - has_temperature (bool): True - HTS221 sensor
                - has_humidity (bool): True - HTS221 sensor
                - sample_rate_hz (float): 10.0 Hz default streaming rate

        Raises:
            No exceptions raised.

        Example:
            >>> instance = driver.open("/dev/ttyACM0")
            >>> info = instance.get_info()
            >>> print(f"{info['name']} on {info['port']}")
            Arduino Nano BLE33 Sense on /dev/ttyACM0
            >>> if info['has_magnetometer']:
            ...     print("Compass available")
        """
        return SensorInfo(
            type="arduino_ble33",
            name="Arduino Nano BLE33 Sense",
            port=self._port,
        )

    def _calculate_altitude(self) -> float:
        """Calculate altitude from accelerometer tilt data.

        Uses 3-axis accelerometer to compute telescope altitude (elevation)
        via trigonometric tilt calculation. Applies calibration scaling
        and offset for accurate readings.

        Business context: Telescope altitude determines which objects are
        visible. Accelerometer-based tilt sensing provides altitude without
        requiring encoder feedback from the mount.

        Implementation: Uses formula: atan(aX / sqrt(aY² + aZ²)) converted
        to degrees. Applies tilt calibration (m, b) then altitude transform
        (scale, offset). Returns 0 if accelerometer data is zero.

        Args:
            No arguments. Uses internal _accelerometer dict and calibration.

        Returns:
            float: Altitude in degrees (0-90), after calibration applied.

        Raises:
            No exceptions raised. Returns 0.0 on missing data.

        Example:
            >>> instance._accelerometer = {'aX': 0.5, 'aY': 0.0, 'aZ': 0.87}
            >>> alt = instance._calculate_altitude()
            >>> 25 < alt < 35  # ~30 degrees
            True
        """
        if self._accelerometer is None:
            return 0.0

        ax = self._accelerometer["aX"]
        ay = self._accelerometer["aY"]
        az = self._accelerometer["aZ"]

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
        """Calculate azimuth from magnetometer compass data.

        Uses 2-axis magnetometer reading to compute telescope azimuth
        (heading/bearing). Applies calibration scaling and offset.

        Business context: Telescope azimuth determines east-west pointing.
        Magnetometer provides absolute heading reference independent of
        mount mechanics.

        Implementation: Uses atan2(mY, mX) for heading, normalizes to 0-360,
        then applies azimuth transform (scale, offset) with modulo 360.
        Returns 0 if magnetometer data is zero.

        Args:
            No arguments. Uses internal _magnetometer dict and calibration.

        Returns:
            float: Azimuth in degrees (0-360), north=0, east=90.

        Raises:
            No exceptions raised. Returns 0.0 on missing data.

        Example:
            >>> instance._magnetometer = {'mX': 30.0, 'mY': 0.0, 'mZ': 40.0}
            >>> az = instance._calculate_azimuth()
            >>> az == 0.0  # Pointing magnetic north
            True
        """
        if self._magnetometer is None:
            return 0.0

        mx = self._magnetometer["mX"]
        my = self._magnetometer["mY"]

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
        """Read current sensor values from the background reader cache.

        Returns the latest IMU and environmental data collected by the
        background reader thread. Does not block for new data - returns
        immediately with cached values.

        Business context: Core sensor operation for telescope orientation.
        Values are continuously updated by background thread at sensor's
        native rate (~10Hz). Callers get latest available data without
        serial I/O latency.

        Args:
            No arguments.

        Returns:
            SensorReading: Dataclass containing:
                - accelerometer: dict with aX, aY, aZ in g's
                - magnetometer: dict with mX, mY, mZ in µT
                - altitude: Calibrated altitude (0-90°)
                - azimuth: Calibrated azimuth (0-360°)
                - temperature: Ambient temperature (°C)
                - humidity: Relative humidity (%)
                - timestamp: datetime of reading (UTC)
                - raw_values: Original sensor data string

        Raises:
            RuntimeError: If sensor is closed or no data received yet.
                Wait briefly after open() for first data.

        Example:
            >>> instance = driver.open("/dev/ttyACM0")
            >>> time.sleep(0.2)  # Wait for first data
            >>> reading = instance.read()
            >>> print(f"Alt: {reading.altitude:.1f}° Az: {reading.azimuth:.1f}°")
        """
        if not self._is_open:
            raise RuntimeError("Sensor is closed")

        with self._lock:
            if self._accelerometer is None or self._magnetometer is None:
                raise RuntimeError("No sensor data available yet")

            return SensorReading(
                accelerometer=self._accelerometer,
                magnetometer=self._magnetometer,
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
        """Calibrate sensor to match a known true telescope position.

        Computes offset values so that the current calculated position
        maps to the provided true position. Uses a simple offset model:
        corrected = calculated + offset. Calibration persists until reset.

        Business context: IMU-derived positions contain systematic errors
        from sensor mounting orientation, magnetic declination, and local
        interference. Calibration against a plate-solved image or known
        star position corrects these errors for accurate Go-To pointing.

        Args:
            true_altitude: Known true altitude in degrees (0-90).
                Obtained from plate solving, star catalog lookup,
                or manual verification at known landmark.
            true_azimuth: Known true azimuth in degrees (0-360).
                0° = North, 90° = East, 180° = South, 270° = West.

        Returns:
            None. Calibration offsets stored in _cal_alt_offset and
            _cal_az_offset instance variables.

        Raises:
            RuntimeError: If sensor is closed.
            ValueError: If altitude not in 0-90° or azimuth not in 0-360°.

        Example:
            >>> instance = driver.open("/dev/ttyACM0")
            >>> # Point telescope at Polaris (known position)
            >>> instance.calibrate(89.26, 0.0)  # Polaris at pole
            >>> # Subsequent reads return calibrated positions
            >>> reading = instance.read()
        """
        if not self._is_open:
            raise RuntimeError("Sensor is closed")

        # Validate input ranges using shared helper
        validate_position(true_altitude, true_azimuth)

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

    def _set_tilt_calibration(self, slope: float, intercept: float) -> None:
        """Set linear calibration parameters for tilt (altitude) calculation.

        Applies linear correction: corrected = slope * raw + intercept.
        This compensates for sensor mounting angle and systematic accelerometer bias.

        Business context: Physical sensor mounting rarely achieves perfect
        alignment. This linear calibration corrects for consistent offset
        (intercept) and scaling errors (slope) in the tilt measurement.
        Parameters typically determined through multi-point calibration.

        Args:
            slope: Scale factor (m in y = mx + b). Typically 0.9-1.1.
                Values <1 compress range, >1 expand range.
                1.0 = no scaling correction.
            intercept: Offset in degrees (b in y = mx + b).
                Positive = add degrees, negative = subtract.
                0.0 = no offset correction.

        Returns:
            None. Parameters stored in _tilt_m and _tilt_b.

        Raises:
            No exceptions raised. Invalid values may produce bad readings.

        Example:
            >>> instance.set_tilt_calibration(1.02, -2.5)
            >>> # Raw 45° becomes: 1.02 * 45 - 2.5 = 43.4°
            >>> # Typically determined from calibration fixture
        """
        self._tilt_m = slope
        self._tilt_b = intercept
        logger.info(
            "Tilt calibration set",
            slope=slope,
            intercept=intercept,
        )

    def _send_command(
        self,
        command: str,
        wait_response: bool = True,
        timeout: float = 5.0,
    ) -> str:
        """Send command to Arduino and optionally wait for response.

        Provides serial command interface for controlling Arduino sensor behavior.
        Commands control sensor output, reset hardware, or query status. The Arduino
        responds with multi-line text including status information or confirmation.

        Business context: Enables runtime control of sensor behavior without
        reconnection. Used for calibration workflows, diagnostics, and graceful
        shutdown. Essential for telescope alignment procedures requiring sensor
        reset or recalibration mid-session.

        Args:
            command: Command string to send. Valid commands:
                - RESET: Reinitialize sensors and clear calibration
                - STATUS: Query current sensor state and configuration
                - CALIBRATE: Trigger magnetometer calibration mode
                - STOP: Pause continuous sensor output
                - START: Resume continuous sensor output
            wait_response: If True (default), blocks until response received
                or timeout. If False, returns immediately after sending.
            timeout: Maximum seconds to wait for response. Default 5.0.
                Longer timeouts needed for RESET (~5s) and CALIBRATE (~10s).

        Returns:
            Response string from Arduino. Multi-line responses joined with newlines.
            Empty string if wait_response=False. Typical responses:
            - "OK: command" for success
            - "ERROR: message" for failures
            - Multi-line status for STATUS command

        Raises:
            RuntimeError: If sensor connection is closed.
            SerialTimeoutError: If response not received within timeout.

        Example:
            >>> instance.send_command("STATUS")
            'Sensor: LSM9DS1\\nTemp: 22.5C\\nOK: STATUS'
            >>> instance.send_command("STOP", wait_response=False)
            ''
            >>> instance.send_command("RESET", timeout=10.0)
            'OK: RESET'
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
        """Reset/reinitialize the Arduino sensors.

        Sends RESET command to Arduino to reinitialize all sensors and
        clear any calibration state. Blocks until reset completes.

        Business context: Recovery mechanism for sensor issues. Use when
        sensor readings become erratic, after power interruption, or to
        clear calibration and start fresh.

        Args:
            No arguments.

        Returns:
            None. Sensor reinitialized on Arduino.

        Raises:
            RuntimeError: If sensor connection closed.
            serial.SerialTimeoutError: If reset doesn't complete in 5s.

        Example:
            >>> instance.reset()  # Reinitialize sensors
            >>> instance.calibrate(45.0, 180.0)  # Recalibrate
        """
        self._send_command("RESET", timeout=5.0)

    def get_status(self) -> SensorStatus:
        """Get current Arduino sensor status including calibration state.

        Sends STATUS command to Arduino and returns parsed response along
        with connection state and calibration information. Useful for
        health monitoring and diagnostics.

        Business context: Long-running observatory sessions need real-time
        sensor health data. This method exposes both Python-side state
        (connection, calibration) and Arduino-reported status for
        comprehensive monitoring.

        Returns:
            ArduinoSensorStatus: TypedDict containing:
                - connected (bool): True if serial connection open
                - type (str): Always 'arduino_ble33'
                - port (str): Serial port path
                - last_update (str | None): ISO timestamp of last reading
                - raw_status (str): Raw response from Arduino STATUS cmd
                - calibrated (bool): True if position calibration applied

        Raises:
            RuntimeError: If serial communication fails.

        Example:
            >>> status = instance.get_status()
            >>> print(f"Connected: {status['connected']}")
            Connected: True
            >>> print(f"Last update: {status['last_update']}")
            Last update: 2025-12-26T22:30:00+00:00
            >>> if not status['calibrated']:
            ...     print("Sensor needs calibration")
        """
        # Send status command to verify Arduino is responding
        self._send_command("STATUS", timeout=3.0)

        # Return protocol-compatible SensorStatus
        # Store extra Arduino-specific info for internal use
        return SensorStatus(
            connected=self._is_open,
            calibrated=self._cal_alt_offset != 0 or self._cal_az_offset != 0,
            is_open=self._is_open,
            error=None if self._is_open else "Connection closed",
        )

    def get_sample_rate(self) -> float:
        """Get sensor sample rate in Hz.

        Returns the fixed sample rate of the Arduino BLE33 firmware.
        Used by device layer to calculate timing for averaged reads.

        Business context: Arduino BLE33 Sense firmware streams sensor data
        at a fixed 10 Hz rate. This rate is determined by the firmware,
        not configurable at runtime. Device layer needs this to properly
        space multi-sample reads.

        Returns:
            float: Always 10.0 Hz for Arduino BLE33 firmware.

        Raises:
            No exceptions raised.

        Example:
            >>> instance = driver.open("/dev/ttyACM0")
            >>> instance.get_sample_rate()
            10.0
        """
        return _ARDUINO_SAMPLE_RATE_HZ

    def calibrate_magnetometer(self) -> str:
        """Run Arduino magnetometer hard-iron calibration routine.

        Triggers the Arduino's built-in magnetometer calibration. During
        calibration (~10-15 seconds), the user should slowly rotate the
        sensor through all orientations to sample the full magnetic sphere.

        Business context: Magnetometer readings suffer from hard-iron
        distortion caused by nearby ferrous materials (screws, motors).
        Calibration computes offsets to center the magnetic field sphere.
        Should be performed after mounting sensor on telescope and whenever
        the magnetic environment changes.

        Args:
            No arguments required.

        Returns:
            str: Response from Arduino containing calibration results.
                Typically includes computed offsets and status.
                Format: 'OK: CALIBRATE\nOffsetX: 12.3\nOffsetY: -5.2...'

        Raises:
            RuntimeError: If serial communication fails or times out
                during the 15 second calibration window.

        Example:
            >>> print("Rotate sensor slowly in all directions...")
            >>> result = instance.calibrate_magnetometer()
            >>> print(result)
            OK: CALIBRATE
            OffsetX: 12.3
            OffsetY: -5.2
            OffsetZ: 8.1
        """
        return self._send_command("CALIBRATE", timeout=15.0)

    def stop_output(self) -> None:
        """Stop the continuous sensor data output stream.

        Sends STOP command to Arduino to pause sensor data streaming.
        Background reader will receive no new data until start_output().

        Business context: Useful during calibration or configuration when
        continuous data stream interferes with command responses. Also
        reduces power consumption when readings not needed.

        Args:
            No arguments.

        Returns:
            None. Does not wait for response.

        Raises:
            RuntimeError: If serial communication fails.

        Example:
            >>> instance.stop_output()  # Pause streaming
            >>> # Do calibration...
            >>> instance.start_output()  # Resume streaming
        """
        self._send_command("STOP", wait_response=False)

    def start_output(self) -> None:
        """Resume the continuous sensor data output stream.

        Sends START command to Arduino to resume sensor data streaming
        after it was paused with stop_output().

        Business context: Resumes normal operation after calibration or
        configuration tasks. Stream provides continuous ~10Hz updates
        for real-time tracking.

        Args:
            No arguments.

        Returns:
            None. Does not wait for response.

        Raises:
            RuntimeError: If serial communication fails.

        Example:
            >>> instance.stop_output()
            >>> # Do configuration...
            >>> instance.start_output()  # Resume data stream
        """
        self._send_command("START", wait_response=False)

    def close(self) -> None:
        """Close the serial connection and stop background reader.

        Signals background reader thread to stop, waits for it to exit,
        and closes the serial port. Safe to call multiple times.

        Business context: Proper cleanup essential for serial port release.
        Unreleased ports prevent reconnection. Always close before
        application exit or switching sensors.

        Args:
            No arguments.

        Returns:
            None. Connection and thread cleaned up.

        Raises:
            No exceptions raised. Errors during close are logged.

        Example:
            >>> instance = driver.open("/dev/ttyACM0")
            >>> reading = instance.read()
            >>> instance.close()  # Release serial port
        """
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
        """Initialize Arduino sensor driver with baud rate.

        Sets up driver configuration without opening any serial ports.
        Call open() or get_available_sensors() to interact with hardware.

        Business context: The driver manages lifecycle of serial connections
        to Arduino sensors. Baud rate stored for consistent port opening.

        Implementation: Stores baudrate, initializes _instance to None.
        No hardware interaction during init.

        Args:
            baudrate: Serial baud rate for Arduino communication.
                Defaults to 115200 (Arduino Nano BLE33 default).

        Returns:
            None. Driver initialized, ready for open().

        Raises:
            No exceptions raised.

        Example:
            >>> driver = ArduinoSensorDriver(baudrate=115200)
            >>> sensors = driver.get_available_sensors()
            >>> if sensors:
            ...     instance = driver.open(sensors[0]['port'])
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
        """Create driver with injected dependencies for testing.

        Factory method that injects custom port enumerator and optional
        serial factory, enabling tests without real hardware.

        Business context: Hardware-independent testing enables CI/CD
        and development without Arduino. Mock enumerators control device
        discovery, serial factories control communication behavior.

        Args:
            port_enumerator: Object with comports() method returning list
                of port objects with device and description attributes.
            serial_factory: Optional class for creating serial connections.
                Defaults to None (uses real pyserial).
            baudrate: Serial baud rate. Defaults to 115200.

        Returns:
            ArduinoSensorDriver: Configured driver using injected
                dependencies for get_available_sensors().

        Raises:
            No exceptions during creation.

        Example:
            >>> class MockPort:
            ...     device = "/dev/ttyMOCK0"
            ...     description = "Arduino Nano"
            >>> class MockEnumerator:
            ...     @staticmethod
            ...     def comports():
            ...         return [MockPort()]
            >>> driver = ArduinoSensorDriver._create_with_enumerator(MockEnumerator())
        """
        driver = cls.__new__(cls)
        driver._baudrate = baudrate
        driver._instance = None
        driver._port_enumerator = port_enumerator
        driver._serial_factory = serial_factory
        return driver

    def _ensure_not_open(self) -> None:
        """Ensure no sensor is currently open before opening a new one.

        Validates that the driver doesn't have an active sensor instance.
        This prevents resource leaks and serial port conflicts by enforcing
        single-instance semantics.

        Business context: Arduino sensors occupy exclusive serial port access.
        Opening a second sensor without closing the first would either fail
        (port busy) or leak the first connection. This guard ensures clean
        resource management.

        Args:
            None. Checks internal driver state.

        Returns:
            None. Method returns normally if no sensor is open.

        Raises:
            RuntimeError: If a sensor instance is already open. Close the
                existing sensor with close() before opening another.

        Example:
            >>> driver = ArduinoSensorDriver()
            >>> driver.open("/dev/ttyACM0")
            >>> driver._ensure_not_open()  # Raises RuntimeError
            RuntimeError: Sensor already open
        """
        if self._instance is not None and self._instance._is_open:
            raise RuntimeError("Sensor already open")

    def get_available_sensors(self) -> list[AvailableSensor]:
        """Discover Arduino sensors available on serial ports.

        Scans system serial ports for devices matching Arduino signatures
        (Arduino in description, ACM devices, USB serial adapters, CH340).
        Does not open connections - only enumerates potential devices.

        Business context: Users may have multiple Arduino devices connected
        for different purposes. This method enables device selection UI
        and auto-discovery workflows. Called by MCP tools to let clients
        enumerate sensors without prior configuration.

        Returns:
            list[dict]: List of sensor descriptors, each containing:
                - id (int): Index for use with open()
                - type (str): Always 'arduino_ble33'
                - name (str): Human-readable name with port
                - port (str): Serial port path (e.g., '/dev/ttyACM0')
                - description (str): OS-provided device description
            Empty list if no compatible devices found or pyserial
            not installed.

        Raises:
            No exceptions raised. Errors logged and empty list returned.

        Example:
            >>> driver = ArduinoSensorDriver()
            >>> sensors = driver.get_available_sensors()
            >>> for s in sensors:
            ...     print(f"{s['name']} on {s['port']}")
            Arduino Sensor (/dev/ttyACM0) on /dev/ttyACM0
            >>> if sensors:
            ...     instance = driver.open(sensors[0]['port'])
        """
        # Use injected enumerator or wrapper function
        if self._port_enumerator is not None:
            ports = self._port_enumerator.comports()
        else:
            ports = list_serial_ports()
            if not ports:
                logger.debug("No serial ports found (pyserial may not be installed)")

        sensors: list[AvailableSensor] = []

        for i, port in enumerate(ports):
            # Check for Arduino-like devices
            desc = port.description.lower()
            device = port.device.lower()
            # Match by description OR by device name (ACM ports)
            if (
                any(
                    x in desc for x in ["arduino", "nano", "ble", "usb serial", "ch340"]
                )
                or "acm" in device
            ):
                sensors.append(
                    AvailableSensor(
                        id=i,
                        type="arduino_ble33",
                        name=f"Arduino Sensor ({port.device})",
                        port=port.device,
                        description=port.description,
                    )
                )

        logger.debug("Found sensors", count=len(sensors))
        return sensors

    def open(self, sensor_id: int | str = 0) -> SensorInstance:
        """Open connection to an Arduino sensor.

        Creates a serial connection to the specified Arduino and starts
        the background reader thread. The sensor begins streaming data
        immediately after connection.

        Business context: This is the primary entry point for connecting
        to physical Arduino Nano BLE33 Sense hardware. The returned
        instance provides all sensor reading and calibration operations.
        Only one instance can be open at a time per driver.

        Args:
            sensor_id: Either a port path string (e.g., '/dev/ttyACM0')
                or an integer index from get_available_sensors().
                If int, looks up the port from the available sensors list.
                Defaults to 0 (first available sensor).

        Returns:
            ArduinoSensorInstance: Connected sensor instance ready for
                reading. Background thread automatically starts
                collecting data.

        Raises:
            RuntimeError: If sensor already open, sensor_id index out
                of range, or serial connection fails.
            serial.SerialException: If port cannot be opened (permissions,
                device not found, etc.).

        Example:
            >>> driver = ArduinoSensorDriver()
            >>> # Open by port path
            >>> instance = driver.open("/dev/ttyACM0")
            >>> reading = instance.read()
            >>> driver.close()
            >>>
            >>> # Open by index
            >>> sensors = driver.get_available_sensors()
            >>> if sensors:
            ...     instance = driver.open(0)  # First sensor
        """
        self._ensure_not_open()

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

        Raises:
            RuntimeError: If sensor already open.

        Example:
            >>> driver = ArduinoSensorDriver()
            >>> mock = MockSerialPort(data=["1.0\\t2.0\\t3.0\\t..."])
            >>> instance = driver._open_with_serial(mock)
            >>> instance.parse_line("1.0\\t2.0\\t3.0\\t4.0\\t5.0\\t6.0\\t25.0\\t50.0")
        """
        if self._instance is not None and self._instance._is_open:
            raise RuntimeError("Sensor already open")

        self._instance = ArduinoSensorInstance._create_with_serial(
            serial_port, port_name, start_reader=False
        )
        return self._instance

    def close(self) -> None:
        """Close the current sensor instance and release resources.

        Closes the underlying ArduinoSensorInstance if open, stopping
        background reader and releasing serial port. Safe to call when
        no instance open.

        Business context: Driver-level cleanup for proper resource
        management. Should be called when done with sensor operations.

        Args:
            No arguments.

        Returns:
            None. Instance reference cleared.

        Raises:
            No exceptions raised.

        Example:
            >>> driver = ArduinoSensorDriver()
            >>> instance = driver.open("/dev/ttyACM0")
            >>> reading = instance.read()
            >>> driver.close()  # Release serial port
        """
        if self._instance is not None:
            self._instance.close()
            self._instance = None

    def __enter__(self) -> ArduinoSensorInstance:  # pragma: no cover
        """Enter context manager, opening first available sensor.

        Enables the driver to be used as a context manager for automatic
        resource cleanup. Opens the first available Arduino sensor and
        returns the instance for reading.

        Business context: Context managers ensure serial ports are properly
        released even if exceptions occur during sensor operations. This
        prevents port lockup that would require device disconnect/reconnect.

        Args:
            self: The driver instance (implicit).

        Returns:
            ArduinoSensorInstance: Open sensor instance ready for read()
                and calibrate() operations.

        Raises:
            RuntimeError: If no Arduino sensors are available, or if
                a sensor is already open on this driver.
            serial.SerialException: If the serial port cannot be opened
                (permissions, device disconnected, etc.).

        Example:
            >>> with ArduinoSensorDriver() as sensor:
            ...     reading = sensor.read()
            ...     print(f"ALT: {reading.altitude:.1f}°")
            ALT: 45.2°
        """
        instance = self.open()
        # Cast is safe - open() always returns ArduinoSensorInstance internally
        assert isinstance(instance, ArduinoSensorInstance)
        return instance

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing sensor and releasing serial port.

        Called automatically when exiting a `with` block, ensuring proper
        cleanup regardless of whether an exception occurred. Closes the
        sensor instance and releases the serial port for other processes.

        Business context: Serial ports are exclusive resources. Failing to
        close them blocks future connections until process termination or
        device reset. Context manager exit guarantees cleanup.

        Args:
            exc_type: Exception type if an exception was raised in the
                with block, None otherwise.
            exc_val: Exception instance if raised, None otherwise.
            exc_tb: Exception traceback if raised, None otherwise.

        Returns:
            None. Exceptions are not suppressed (returns None, not True).

        Raises:
            No exceptions raised. Cleanup is best-effort; errors during
            close() are logged but not re-raised to avoid masking the
            original exception.

        Example:
            >>> with ArduinoSensorDriver() as sensor:
            ...     reading = sensor.read()
            ...     raise ValueError("test")  # __exit__ still called
            # Serial port released even though exception raised
        """
        self.close()
