/**
 * Telescope Sensors - Arduino Nano BLE33 Sense
 *
 * This sketch reads data from the onboard sensors and outputs them via serial USB.
 * Supports serial commands for reset, status, and calibration.
 *
 * Sensors Used:
 * - LSM9DS1 IMU: Accelerometer (3-axis) + Magnetometer (3-axis)
 * - HTS221: Humidity + Temperature
 *
 * Output Format (tab-separated, newline terminated):
 * aX\taY\taZ\tmX\tmY\tmZ\ttemperature\thumidity\r\n
 *
 * Serial Commands:
 * - RESET     : Reinitialize all sensors
 * - STATUS    : Report sensor status and configuration
 * - CALIBRATE : Run magnetometer calibration (collect samples while rotating)
 * - STOP      : Pause sensor output
 * - START     : Resume sensor output
 *
 * Units:
 * - Accelerometer: g (gravitational acceleration)
 * - Magnetometer: µT (micro-Tesla)
 * - Temperature: °C
 * - Humidity: %RH (relative humidity)
 *
 * Serial Baud Rate: 115200
 *
 * Libraries Required:
 * - Arduino_LSM9DS1 (IMU)
 * - Arduino_HTS221 (Humidity & Temperature)
 *
 * Purpose:
 * The accelerometer data is used to calculate the altitude (ALT) angle of the
 * Optical Tube Assembly (OTA). The magnetometer data determines the azimuth (AZ)
 * direction. Temperature and humidity provide environmental context.
 *
 * @author Mark
 * @date 2025-12-26
 */

#include <Arduino_LSM9DS1.h>
#include <Arduino_HTS221.h>

// Configuration
const unsigned long SAMPLE_INTERVAL_MS = 100;  // 10 Hz sampling rate
const long SERIAL_BAUD_RATE = 115200;
const int LED_PIN = LED_BUILTIN;               // Onboard LED for status
const int WARMUP_SAMPLES = 10;                 // Samples to discard during warmup
const int CALIBRATION_SAMPLES = 100;           // Samples for magnetometer calibration

// Sensor data storage
float accelX, accelY, accelZ;  // Accelerometer values in g
float magX, magY, magZ;        // Magnetometer values in µT
float temperature;              // Temperature in °C
float humidity;                 // Relative humidity in %

// Magnetometer calibration offsets (hard-iron correction)
float magOffsetX = 0.0;
float magOffsetY = 0.0;
float magOffsetZ = 0.0;

// Sensor status
bool imuInitialized = false;
bool htsInitialized = false;
bool outputEnabled = true;

// Command buffer
String commandBuffer = "";

// Timing
unsigned long lastSampleTime = 0;

/**
 * Blink the LED a specified number of times.
 *
 * Args:
 *     times: Number of blinks.
 *     onMs: LED on duration in milliseconds.
 *     offMs: LED off duration in milliseconds.
 */
void blinkLED(int times, int onMs = 100, int offMs = 100) {
    for (int i = 0; i < times; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(onMs);
        digitalWrite(LED_PIN, LOW);
        delay(offMs);
    }
}

/**
 * Initialize the IMU sensor (LSM9DS1).
 *
 * Returns:
 *     true if initialization successful, false otherwise.
 */
bool initIMU() {
    Serial.println("INFO: Initializing LSM9DS1 IMU...");

    // End any existing connection first
    IMU.end();
    delay(100);

    if (!IMU.begin()) {
        Serial.println("ERROR: Failed to initialize LSM9DS1 IMU!");
        imuInitialized = false;
        return false;
    }

    // Warm-up: discard initial readings
    Serial.println("INFO: IMU warm-up...");
    for (int i = 0; i < WARMUP_SAMPLES; i++) {
        while (!IMU.accelerationAvailable()) delay(10);
        IMU.readAcceleration(accelX, accelY, accelZ);
        while (!IMU.magneticFieldAvailable()) delay(10);
        IMU.readMagneticField(magX, magY, magZ);
        delay(50);
    }

    imuInitialized = true;
    Serial.println("OK: IMU initialized");
    return true;
}

/**
 * Initialize the HTS221 humidity/temperature sensor.
 *
 * Returns:
 *     true if initialization successful, false otherwise.
 */
bool initHTS() {
    Serial.println("INFO: Initializing HTS221 sensor...");

    // End any existing connection first
    HTS.end();
    delay(100);

    if (!HTS.begin()) {
        Serial.println("ERROR: Failed to initialize HTS221 sensor!");
        htsInitialized = false;
        return false;
    }

    // Warm-up: discard initial readings
    Serial.println("INFO: HTS warm-up...");
    for (int i = 0; i < WARMUP_SAMPLES; i++) {
        HTS.readTemperature();
        HTS.readHumidity();
        delay(50);
    }

    htsInitialized = true;
    Serial.println("OK: HTS221 initialized");
    return true;
}

/**
 * Initialize all sensors with LED feedback.
 *
 * Performs full initialization sequence:
 * - 3 quick blinks: starting init
 * - Initialize IMU
 * - Initialize HTS
 * - 2 long blinks: success, 5 rapid blinks: failure
 */
void initAllSensors() {
    Serial.println("INFO: Starting sensor initialization sequence...");
    blinkLED(3, 100, 100);  // Starting init

    bool success = true;

    if (!initIMU()) {
        success = false;
    }

    if (!initHTS()) {
        success = false;
    }

    if (success) {
        Serial.println("OK: All sensors initialized successfully");
        blinkLED(2, 300, 200);  // Success pattern
    } else {
        Serial.println("WARNING: Some sensors failed to initialize");
        blinkLED(5, 50, 50);  // Failure pattern
    }
}

/**
 * Report current sensor status and configuration.
 */
void reportStatus() {
    Serial.println("=== SENSOR STATUS ===");
    Serial.print("IMU (LSM9DS1): ");
    Serial.println(imuInitialized ? "OK" : "FAILED");
    Serial.print("HTS221: ");
    Serial.println(htsInitialized ? "OK" : "FAILED");
    Serial.print("Output: ");
    Serial.println(outputEnabled ? "ENABLED" : "PAUSED");
    Serial.print("Sample Rate: ");
    Serial.print(1000 / SAMPLE_INTERVAL_MS);
    Serial.println(" Hz");
    Serial.print("Mag Offsets (X,Y,Z): ");
    Serial.print(magOffsetX, 2);
    Serial.print(", ");
    Serial.print(magOffsetY, 2);
    Serial.print(", ");
    Serial.println(magOffsetZ, 2);

    // Current readings
    Serial.println("--- Current Readings ---");
    Serial.print("Accel (g): ");
    Serial.print(accelX, 2);
    Serial.print(", ");
    Serial.print(accelY, 2);
    Serial.print(", ");
    Serial.println(accelZ, 2);
    Serial.print("Mag (uT): ");
    Serial.print(magX, 2);
    Serial.print(", ");
    Serial.print(magY, 2);
    Serial.print(", ");
    Serial.println(magZ, 2);
    Serial.print("Temp: ");
    Serial.print(temperature, 1);
    Serial.println(" C");
    Serial.print("Humidity: ");
    Serial.print(humidity, 1);
    Serial.println(" %RH");
    Serial.println("=====================");
}

/**
 * Run magnetometer calibration routine.
 *
 * Collects samples while user rotates device, then calculates
 * hard-iron offsets as the center of the min/max envelope.
 */
void runCalibration() {
    Serial.println("=== MAGNETOMETER CALIBRATION ===");
    Serial.println("Rotate the sensor slowly in all directions.");
    Serial.println("Collecting samples...");

    float minX = 99999, maxX = -99999;
    float minY = 99999, maxY = -99999;
    float minZ = 99999, maxZ = -99999;
    float mx, my, mz;

    blinkLED(2, 200, 200);  // Starting calibration

    for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
        // Wait for magnetometer data
        while (!IMU.magneticFieldAvailable()) delay(10);
        IMU.readMagneticField(mx, my, mz);

        // Track min/max
        if (mx < minX) minX = mx;
        if (mx > maxX) maxX = mx;
        if (my < minY) minY = my;
        if (my > maxY) maxY = my;
        if (mz < minZ) minZ = mz;
        if (mz > maxZ) maxZ = mz;

        // Progress indicator
        if (i % 10 == 0) {
            Serial.print(".");
            digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        }

        delay(100);  // 10 Hz sampling
    }

    digitalWrite(LED_PIN, LOW);
    Serial.println();

    // Calculate offsets (center of envelope)
    magOffsetX = (maxX + minX) / 2.0;
    magOffsetY = (maxY + minY) / 2.0;
    magOffsetZ = (maxZ + minZ) / 2.0;

    Serial.println("Calibration complete!");
    Serial.print("New offsets (X,Y,Z): ");
    Serial.print(magOffsetX, 2);
    Serial.print(", ");
    Serial.print(magOffsetY, 2);
    Serial.print(", ");
    Serial.println(magOffsetZ, 2);
    Serial.println("================================");

    blinkLED(3, 300, 100);  // Calibration complete
}

/**
 * Process incoming serial commands.
 *
 * Supported commands:
 * - RESET: Reinitialize all sensors
 * - STATUS: Report sensor status
 * - CALIBRATE: Run magnetometer calibration
 * - STOP: Pause sensor output
 * - START: Resume sensor output
 */
void processCommand(String cmd) {
    cmd.trim();
    cmd.toUpperCase();

    if (cmd == "RESET") {
        Serial.println("CMD: Resetting sensors...");
        initAllSensors();
    }
    else if (cmd == "STATUS") {
        reportStatus();
    }
    else if (cmd == "CALIBRATE") {
        runCalibration();
    }
    else if (cmd == "STOP") {
        outputEnabled = false;
        Serial.println("CMD: Output paused");
    }
    else if (cmd == "START") {
        outputEnabled = true;
        Serial.println("CMD: Output resumed");
    }
    else if (cmd.length() > 0) {
        Serial.print("ERROR: Unknown command: ");
        Serial.println(cmd);
        Serial.println("Available: RESET, STATUS, CALIBRATE, STOP, START");
    }
}

/**
 * Check for and process any incoming serial commands.
 */
void checkSerialCommands() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (commandBuffer.length() > 0) {
                processCommand(commandBuffer);
                commandBuffer = "";
            }
        } else {
            commandBuffer += c;
        }
    }
}

/**
 * Initialize all sensors and serial communication.
 *
 * Sets up:
 * - Serial at 115200 baud
 * - LED for status indication
 * - LSM9DS1 IMU for accelerometer and magnetometer
 * - HTS221 for humidity and temperature
 *
 * Includes warm-up sequence and LED feedback.
 */
void setup() {
    // Initialize LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Initialize serial communication
    Serial.begin(SERIAL_BAUD_RATE);
    while (!Serial) {
        blinkLED(1, 50, 50);  // Blink while waiting for serial
    }

    Serial.println();
    Serial.println("==============================");
    Serial.println("Telescope Sensors v1.1");
    Serial.println("Arduino Nano BLE33 Sense");
    Serial.println("==============================");
    Serial.println("Commands: RESET, STATUS, CALIBRATE, STOP, START");
    Serial.println();

    // Initialize all sensors with full sequence
    initAllSensors();

    Serial.println();
    Serial.println("Ready. Streaming sensor data...");
    Serial.println();
}

/**
 * Main loop - continuously reads sensors and outputs data.
 *
 * Reads accelerometer, magnetometer, temperature, and humidity at the
 * configured sample rate and outputs tab-separated values over serial.
 * Also checks for incoming serial commands.
 */
void loop() {
    // Check for serial commands
    checkSerialCommands();

    unsigned long currentTime = millis();

    // Check if it's time for a new sample
    if (outputEnabled && (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS)) {
        lastSampleTime = currentTime;

        // Read accelerometer data
        if (imuInitialized && IMU.accelerationAvailable()) {
            IMU.readAcceleration(accelX, accelY, accelZ);
        }

        // Read magnetometer data (apply calibration offsets)
        if (imuInitialized && IMU.magneticFieldAvailable()) {
            float rawX, rawY, rawZ;
            IMU.readMagneticField(rawX, rawY, rawZ);
            magX = rawX - magOffsetX;
            magY = rawY - magOffsetY;
            magZ = rawZ - magOffsetZ;
        }

        // Read temperature and humidity
        if (htsInitialized) {
            temperature = HTS.readTemperature();
            humidity = HTS.readHumidity();
        }

        // Output data in tab-separated format
        outputSensorData();
    }
}

/**
 * Output all sensor data as tab-separated values.
 *
 * Format: aX\taY\taZ\tmX\tmY\tmZ\ttemperature\thumidity\r\n
 *
 * This matches the expected format for the Python notebook parsing code.
 */
void outputSensorData() {
    // Accelerometer (3 values)
    Serial.print(accelX, 2);
    Serial.print("\t");
    Serial.print(accelY, 2);
    Serial.print("\t");
    Serial.print(accelZ, 2);
    Serial.print("\t");

    // Magnetometer (3 values)
    Serial.print(magX, 2);
    Serial.print("\t");
    Serial.print(magY, 2);
    Serial.print("\t");
    Serial.print(magZ, 2);
    Serial.print("\t");

    // Temperature
    Serial.print(temperature, 2);
    Serial.print("\t");

    // Humidity
    Serial.println(humidity, 2);  // println adds \r\n
}
