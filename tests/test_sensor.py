"""Tests for Sensor device and drivers.

Tests the high-level Sensor abstraction and DigitalTwin driver.
Arduino driver tests require hardware and are integration tests.
"""

from datetime import UTC, datetime

import pytest

from telescope_mcp.devices.sensor import Sensor, SensorConfig, SensorInfo
from telescope_mcp.drivers.sensors import (
    DigitalTwinSensorConfig,
    DigitalTwinSensorDriver,
    SensorReading,
)


class TestDigitalTwinSensorDriver:
    """Tests for DigitalTwinSensorDriver."""

    def test_get_available_sensors(self) -> None:
        """Driver returns one simulated sensor."""
        driver = DigitalTwinSensorDriver()
        sensors = driver.get_available_sensors()

        assert len(sensors) == 1
        assert sensors[0]["type"] == "digital_twin"
        assert "name" in sensors[0]

    def test_open_creates_instance(self) -> None:
        """Opening driver creates sensor instance."""
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        assert instance is not None
        info = instance.get_info()
        assert info["type"] == "digital_twin"

        driver.close()

    def test_open_twice_raises(self) -> None:
        """Opening already-open sensor raises error."""
        driver = DigitalTwinSensorDriver()
        driver.open()

        with pytest.raises(RuntimeError, match="already open"):
            driver.open()

        driver.close()

    def test_custom_config(self) -> None:
        """Custom configuration affects sensor behavior."""
        config = DigitalTwinSensorConfig(
            initial_altitude=30.0,
            initial_azimuth=90.0,
            temperature=15.0,
            humidity=60.0,
        )
        driver = DigitalTwinSensorDriver(config)
        instance = driver.open()

        reading = instance.read()
        # Values should be close to config (with noise)
        assert 25.0 <= reading.altitude <= 35.0
        assert 85.0 <= reading.azimuth <= 95.0
        assert 14.0 <= reading.temperature <= 16.0
        assert 58.0 <= reading.humidity <= 62.0

        driver.close()


class TestDigitalTwinSensorInstance:
    """Tests for DigitalTwinSensorInstance."""

    def test_read_returns_sensor_reading(self) -> None:
        """Read returns complete SensorReading."""
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        reading = instance.read()

        assert isinstance(reading.accelerometer, dict)
        assert "aX" in reading.accelerometer
        assert "aY" in reading.accelerometer
        assert "aZ" in reading.accelerometer

        assert isinstance(reading.magnetometer, dict)
        assert "mX" in reading.magnetometer
        assert "mY" in reading.magnetometer
        assert "mZ" in reading.magnetometer

        assert isinstance(reading.altitude, float)
        assert isinstance(reading.azimuth, float)
        assert isinstance(reading.temperature, float)
        assert isinstance(reading.humidity, float)
        assert isinstance(reading.timestamp, datetime)
        assert reading.raw_values  # Non-empty string

        driver.close()

    def test_set_position(self) -> None:
        """Setting position affects readings."""
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        instance._set_position(altitude=60.0, azimuth=270.0)
        reading = instance.read()

        # Should be close to set position (with noise)
        assert 55.0 <= reading.altitude <= 65.0
        assert 265.0 <= reading.azimuth <= 275.0

        driver.close()

    def test_calibrate_adjusts_readings(self) -> None:
        """Calibration adjusts readings to match true position."""
        config = DigitalTwinSensorConfig(
            initial_altitude=40.0,
            initial_azimuth=180.0,
            noise_std_alt=0.0,  # No noise for predictable test
            noise_std_az=0.0,
        )
        driver = DigitalTwinSensorDriver(config)
        instance = driver.open()

        # Calibrate to true position
        instance.calibrate(true_altitude=45.0, true_azimuth=200.0)

        reading = instance.read()
        assert abs(reading.altitude - 45.0) < 1.0
        assert abs(reading.azimuth - 200.0) < 1.0

        driver.close()

    def test_reset_clears_calibration(self) -> None:
        """Reset clears calibration offsets."""
        config = DigitalTwinSensorConfig(
            initial_altitude=45.0,
            initial_azimuth=180.0,
            noise_std_alt=0.0,
            noise_std_az=0.0,
        )
        driver = DigitalTwinSensorDriver(config)
        instance = driver.open()

        instance.calibrate(true_altitude=90.0, true_azimuth=0.0)
        instance.reset()

        reading = instance.read()
        # Should be back to initial position
        assert abs(reading.altitude - 45.0) < 1.0
        assert abs(reading.azimuth - 180.0) < 1.0

        driver.close()

    def test_get_status(self) -> None:
        """Get status returns connection and calibration info."""
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        status = instance.get_status()

        assert status["connected"] is True
        assert status["type"] == "digital_twin"
        assert status["calibrated"] is False
        assert "uptime_seconds" in status

        driver.close()

    def test_read_after_close_raises(self) -> None:
        """Reading after close raises error."""
        driver = DigitalTwinSensorDriver()
        instance = driver.open()
        instance.close()

        with pytest.raises(RuntimeError, match="closed"):
            instance.read()

    def test_calibrate_magnetometer(self) -> None:
        """Magnetometer calibration returns offsets."""
        driver = DigitalTwinSensorDriver()
        instance = driver.open()

        result = instance._calibrate_magnetometer()

        assert "offset_x" in result
        assert "offset_y" in result
        assert "offset_z" in result

        driver.close()


class TestSensorDevice:
    """Tests for high-level Sensor device."""

    def test_connect_and_disconnect(self) -> None:
        """Basic connect and disconnect workflow."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert not sensor.connected

        sensor.connect()
        assert sensor.connected
        assert sensor.info is not None

        sensor.disconnect()
        assert not sensor.connected
        assert sensor.info is None

    def test_auto_connect(self) -> None:
        """Auto-connect in config connects on init."""
        driver = DigitalTwinSensorDriver()
        config = SensorConfig(auto_connect=True)
        sensor = Sensor(driver, config)

        assert sensor.connected

        sensor.disconnect()

    def test_connect_twice_raises(self) -> None:
        """Connecting when already connected raises error."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        with pytest.raises(RuntimeError, match="already connected"):
            sensor.connect()

        sensor.disconnect()

    def test_read_without_connect_raises(self) -> None:
        """Reading without connect raises error."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with pytest.raises(RuntimeError, match="not connected"):
            sensor.read()

    def test_read_returns_sensor_reading(self) -> None:
        """Read returns SensorReading from driver."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        reading = sensor.read()

        assert isinstance(reading.altitude, float)
        assert isinstance(reading.azimuth, float)
        assert reading.accelerometer
        assert reading.magnetometer

        sensor.disconnect()

    def test_calibrate(self) -> None:
        """Calibrate sets transform to match true position."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        sensor.calibrate(true_altitude=50.0, true_azimuth=120.0)

        # Verify calibration was applied
        status = sensor.get_status()
        assert status.get("calibrated", False)

        sensor.disconnect()

    def test_calibrate_validates_altitude(self) -> None:
        """Calibrate validates altitude range."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        with pytest.raises(ValueError, match="Altitude must be 0-90"):
            sensor.calibrate(true_altitude=100.0, true_azimuth=180.0)

        sensor.disconnect()

    def test_calibrate_validates_azimuth(self) -> None:
        """Calibrate validates azimuth range."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        with pytest.raises(ValueError, match="Azimuth must be 0-360"):
            sensor.calibrate(true_altitude=45.0, true_azimuth=400.0)

        sensor.disconnect()

    def test_reset(self) -> None:
        """Reset clears sensor state."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        sensor.calibrate(true_altitude=80.0, true_azimuth=300.0)
        sensor.reset()

        status = sensor.get_status()
        assert not status.get("calibrated", True)

        sensor.disconnect()

    def test_get_status(self) -> None:
        """Get status returns comprehensive info."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        status = sensor.get_status()

        assert status["connected"] is True
        assert status["type"] == "digital_twin"
        assert "connect_time" in status
        assert "read_count" in status

        sensor.disconnect()

    def test_statistics(self) -> None:
        """Statistics tracks reads and errors."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        for _ in range(5):
            sensor.read()

        stats = sensor.statistics
        assert stats["read_count"] == 5
        assert stats["error_count"] == 0
        assert stats["uptime_seconds"] is not None

        sensor.disconnect()

    def test_last_reading(self) -> None:
        """Last reading property returns most recent."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)
        sensor.connect()

        assert sensor.last_reading is None

        reading = sensor.read()
        assert sensor.last_reading is reading

        sensor.disconnect()

    def test_context_manager(self) -> None:
        """Context manager connects and disconnects."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        with sensor:
            assert sensor.connected
            reading = sensor.read()
            assert reading is not None

        assert not sensor.connected

    def test_get_available_sensors(self) -> None:
        """Get available sensors from driver."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        sensors = sensor.get_available_sensors()
        assert len(sensors) == 1

    def test_info_property(self) -> None:
        """Info property returns SensorInfo when connected."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert sensor.info is None

        sensor.connect()
        assert sensor.info is not None
        assert isinstance(sensor.info, SensorInfo)
        assert sensor.info.type == "digital_twin"

        sensor.disconnect()

    def test_repr(self) -> None:
        """Repr shows connection state and type."""
        driver = DigitalTwinSensorDriver()
        sensor = Sensor(driver)

        assert "connected=False" in repr(sensor)

        sensor.connect()
        assert "connected=True" in repr(sensor)
        assert "digital_twin" in repr(sensor)

        sensor.disconnect()


class TestSensorReading:
    """Tests for SensorReading dataclass."""

    def test_sensor_reading_fields(self) -> None:
        """SensorReading has all required fields."""
        reading = SensorReading(
            accelerometer={"aX": 0.0, "aY": 0.0, "aZ": 1.0},
            magnetometer={"mX": 30.0, "mY": 0.0, "mZ": 40.0},
            altitude=45.0,
            azimuth=180.0,
            temperature=20.0,
            humidity=50.0,
            timestamp=datetime.now(UTC),
            raw_values="0.0\t0.0\t1.0\t30.0\t0.0\t40.0\t20.0\t50.0",
        )

        assert reading.altitude == 45.0
        assert reading.azimuth == 180.0
        assert reading.accelerometer["aZ"] == 1.0
