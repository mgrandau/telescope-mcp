// Telescope Control Dashboard JavaScript

const API_BASE = '';

// Position polling - 500ms for responsive feedback during motion
const POSITION_POLL_INTERVAL_MS = 500;
let positionInterval;

// Motor state tracking for start/stop pattern
let motorState = {
    altitude: { moving: false, direction: 0 },
    azimuth: { moving: false, direction: 0 }
};

document.addEventListener('DOMContentLoaded', () => {
    updateStatus('connected', 'Connected');
    startPositionPolling();
    setupMotorControls();

    // Speed slider
    const speedSlider = document.getElementById('speed');
    const speedValue = document.getElementById('speed-value');
    speedSlider.addEventListener('input', () => {
        speedValue.textContent = speedSlider.value;
    });
});

function updateStatus(state, message) {
    const status = document.getElementById('status');
    status.className = 'status ' + state;
    status.textContent = message;
}

// Position polling
function startPositionPolling() {
    updatePosition();
    positionInterval = setInterval(updatePosition, POSITION_POLL_INTERVAL_MS);
}

async function updatePosition() {
    try {
        const response = await fetch(`${API_BASE}/api/position`);
        const data = await response.json();
        // Update ALT/AZ display
        document.getElementById('altitude').textContent = data.altitude.toFixed(1);
        document.getElementById('azimuth').textContent = data.azimuth.toFixed(1);
        // Update RA/Dec display
        document.getElementById('ra').textContent = data.ra_hms || '--';
        document.getElementById('dec').textContent = data.dec_dms || '--';
    } catch (err) {
        console.error('Position update failed:', err);
    }
}

// Motor control - tap for nudge, hold for continuous motion
function setupMotorControls() {
    // Get all direction buttons by their data attributes
    const buttons = document.querySelectorAll('[data-axis][data-direction]');
    buttons.forEach(btn => {
        const axis = btn.dataset.axis;
        const direction = parseInt(btn.dataset.direction);

        // Tap: nudge (quick press and release)
        btn.addEventListener('click', (e) => {
            // Only trigger nudge if not already moving (prevents double action)
            if (!motorState[axis].moving) {
                nudgeMotor(axis, direction);
            }
        });

        // Hold: start continuous motion
        btn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            startMotorHold(axis, direction);
        });

        // Release: stop motion
        btn.addEventListener('mouseup', (e) => {
            stopMotorHold(axis);
        });
        btn.addEventListener('mouseleave', (e) => {
            stopMotorHold(axis);
        });

        // Touch support for mobile
        btn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            startMotorHold(axis, direction);
        });
        btn.addEventListener('touchend', (e) => {
            stopMotorHold(axis);
        });
    });
}

// Nudge motor by small fixed amount (tap gesture)
async function nudgeMotor(axis, direction) {
    const speed = document.getElementById('speed').value;
    try {
        await fetch(`${API_BASE}/api/motor/nudge?axis=${axis}&direction=${direction}&speed=${speed}`, {
            method: 'POST'
        });
    } catch (err) {
        console.error(`Nudge ${axis} failed:`, err);
        updateStatus('error', 'Motor error');
    }
}

// Start continuous motion (hold gesture)
let holdTimers = { altitude: null, azimuth: null };
const HOLD_DELAY_MS = 200; // Delay before starting continuous motion

function startMotorHold(axis, direction) {
    // Clear any existing timer
    if (holdTimers[axis]) {
        clearTimeout(holdTimers[axis]);
    }

    // Start continuous motion after hold delay
    holdTimers[axis] = setTimeout(async () => {
        const speed = document.getElementById('speed').value;
        motorState[axis].moving = true;
        motorState[axis].direction = direction;

        try {
            await fetch(`${API_BASE}/api/motor/start?axis=${axis}&direction=${direction}&speed=${speed}`, {
                method: 'POST'
            });
            updateStatus('connected', `Moving ${axis}...`);
        } catch (err) {
            console.error(`Start ${axis} failed:`, err);
            updateStatus('error', 'Motor error');
            motorState[axis].moving = false;
        }
    }, HOLD_DELAY_MS);
}

// Stop continuous motion (release gesture)
async function stopMotorHold(axis) {
    // Clear hold timer (for quick tap → nudge)
    if (holdTimers[axis]) {
        clearTimeout(holdTimers[axis]);
        holdTimers[axis] = null;
    }

    // Stop motor if it was moving
    if (motorState[axis].moving) {
        motorState[axis].moving = false;
        motorState[axis].direction = 0;

        try {
            await fetch(`${API_BASE}/api/motor/stop?axis=${axis}`, {
                method: 'POST'
            });
            updateStatus('connected', 'Connected');
        } catch (err) {
            console.error(`Stop ${axis} failed:`, err);
            updateStatus('error', 'Stop failed');
        }
    }
}

// Legacy moveAlt/moveAz for compatibility (deprecated)
async function moveAlt(steps) {
    const direction = steps > 0 ? 1 : -1;
    await nudgeMotor('altitude', direction);
}

async function moveAz(steps) {
    const direction = steps > 0 ? 1 : -1;
    await nudgeMotor('azimuth', direction);
}

async function stopMotors() {
    try {
        await fetch(`${API_BASE}/api/motor/stop`, { method: 'POST' });
        motorState.altitude.moving = false;
        motorState.azimuth.moving = false;
        updateStatus('connected', 'Motors stopped');
    } catch (err) {
        console.error('Stop motors failed:', err);
        updateStatus('error', 'Stop failed!');
    }
}

// Goto position
async function gotoPosition() {
    const alt = document.getElementById('goto-alt').value;
    const az = document.getElementById('goto-az').value;

    updateStatus('connected', `Going to Alt:${alt}° Az:${az}°...`);

    try {
        await fetch(`${API_BASE}/api/goto?altitude=${alt}&azimuth=${az}`, {
            method: 'POST'
        });
    } catch (err) {
        console.error('Goto failed:', err);
        updateStatus('error', 'Goto failed');
    }
}

// Camera controls
async function updateFinderSettings() {
    const exposure = document.getElementById('finder-exposure').value;
    const gain = document.getElementById('finder-gain').value;
    await setCameraControl(0, 'ASI_EXPOSURE', exposure * 1000);
    await setCameraControl(0, 'ASI_GAIN', gain);
}

async function updateMainSettings() {
    const exposure = document.getElementById('main-exposure').value;
    const gain = document.getElementById('main-gain').value;
    await setCameraControl(1, 'ASI_EXPOSURE', exposure * 1000);
    await setCameraControl(1, 'ASI_GAIN', gain);
}

async function setCameraControl(cameraId, control, value) {
    try {
        await fetch(`${API_BASE}/api/camera/${cameraId}/control?control=${control}&value=${value}`, {
            method: 'POST'
        });
    } catch (err) {
        console.error('Camera control failed:', err);
        updateStatus('error', 'Camera error');
    }
}
