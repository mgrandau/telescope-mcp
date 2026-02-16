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

    // Hold step size slider (range 1-100 → 0.1°-10.0°)
    const holdStepSlider = document.getElementById('hold-step');
    const holdStepValue = document.getElementById('hold-step-value');
    holdStepSlider.addEventListener('input', () => {
        holdStepValue.textContent = (holdStepSlider.value / 10).toFixed(1);
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

// Motor control - tap for nudge, hold for step move
// Track whether a hold was triggered to suppress the spurious click event
let holdFired = { altitude: false, azimuth: false };

function setupMotorControls() {
    // Get all direction buttons by their data attributes
    const buttons = document.querySelectorAll('[data-axis][data-direction]');
    buttons.forEach(btn => {
        const axis = btn.dataset.axis;
        const direction = parseInt(btn.dataset.direction);

        // Tap: nudge (quick press and release)
        btn.addEventListener('click', (e) => {
            // Suppress click if hold already fired for this press
            if (holdFired[axis]) {
                console.log(`[motor] click suppressed for ${axis} (hold already fired)`);
                holdFired[axis] = false;
                return;
            }
            if (!motorState[axis].moving) {
                console.log(`[motor] tap → nudge ${axis} dir=${direction}`);
                nudgeMotor(axis, direction);
            }
        });

        // Hold: start continuous motion
        btn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            holdFired[axis] = false;
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
            holdFired[axis] = false;
            startMotorHold(axis, direction);
        });
        btn.addEventListener('touchend', (e) => {
            stopMotorHold(axis);
        });
    });

    console.log('[motor] controls initialized — tap=nudge, hold=step');
}

// Show motor feedback message (auto-clears after 3s)
let feedbackTimer = null;
function showMotorFeedback(message) {
    const el = document.getElementById('motor-feedback');
    el.textContent = message;
    el.classList.add('visible');
    if (feedbackTimer) clearTimeout(feedbackTimer);
    feedbackTimer = setTimeout(() => {
        el.classList.remove('visible');
    }, 3000);
}

// Nudge motor by small fixed amount (tap gesture)
async function nudgeMotor(axis, direction) {
    const nudgeDeg = 0.1;
    const dirLabel = direction > 0 ? (axis === 'altitude' ? 'up' : 'right') : (axis === 'altitude' ? 'down' : 'left');
    console.log(`[motor] nudge ${axis} ${dirLabel} ${nudgeDeg}°`);
    showMotorFeedback(`Nudge ${axis} ${dirLabel} ${nudgeDeg}°`);
    try {
        const url = `${API_BASE}/api/motor/nudge?axis=${axis}&direction=${dirLabel}&degrees=${nudgeDeg}`;
        console.log(`[motor] POST ${url}`);
        const resp = await fetch(url, { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
            console.log('[motor] nudge OK:', data);
            showMotorFeedback(`✓ Nudge ${axis} ${dirLabel} ${nudgeDeg}°`);
        } else {
            console.warn('[motor] nudge failed:', resp.status, data);
            showMotorFeedback(`⚠ ${data.detail || 'Nudge failed'}`);
        }
    } catch (err) {
        console.error(`[motor] nudge ${axis} error:`, err);
        showMotorFeedback(`✗ Nudge error: ${err.message}`);
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

    // Start hold move after delay
    holdTimers[axis] = setTimeout(async () => {
        holdFired[axis] = true;  // suppress the click that follows mouseup
        const holdDeg = (document.getElementById('hold-step').value / 10).toFixed(1);
        motorState[axis].moving = true;
        motorState[axis].direction = direction;

        const dirLabel = direction > 0 ? (axis === 'altitude' ? 'up' : 'right') : (axis === 'altitude' ? 'down' : 'left');
        console.log(`[motor] hold → step ${axis} ${dirLabel} ${holdDeg}°`);
        showMotorFeedback(`Hold ${axis} ${dirLabel} ${holdDeg}°`);

        try {
            const url = `${API_BASE}/api/motor/nudge?axis=${axis}&direction=${dirLabel}&degrees=${holdDeg}`;
            console.log(`[motor] POST ${url}`);
            const resp = await fetch(url, { method: 'POST' });
            const data = await resp.json();
            if (resp.ok) {
                console.log('[motor] hold OK:', data);
                showMotorFeedback(`✓ Hold ${axis} ${dirLabel} ${holdDeg}°`);
            } else {
                console.warn('[motor] hold failed:', resp.status, data);
                showMotorFeedback(`⚠ ${data.detail || 'Move failed'}`);
            }
            updateStatus('connected', `Moved ${axis} ${holdDeg}°`);
        } catch (err) {
            console.error(`[motor] hold ${axis} error:`, err);
            showMotorFeedback(`✗ Hold error: ${err.message}`);
            updateStatus('error', 'Motor error');
            motorState[axis].moving = false;
        }
        motorState[axis].moving = false;
    }, HOLD_DELAY_MS);
}

// Stop continuous motion (release gesture)
async function stopMotorHold(axis) {
    // Clear hold timer (for quick tap → nudge)
    if (holdTimers[axis]) {
        clearTimeout(holdTimers[axis]);
        holdTimers[axis] = null;
        console.log(`[motor] hold timer cleared for ${axis} (released before threshold)`);
    }

    // Stop motor if it was moving
    if (motorState[axis].moving) {
        motorState[axis].moving = false;
        motorState[axis].direction = 0;
        console.log(`[motor] stopping ${axis}`);

        try {
            await fetch(`${API_BASE}/api/motor/stop?axis=${axis}`, {
                method: 'POST'
            });
            updateStatus('connected', 'Connected');
        } catch (err) {
            console.error(`[motor] stop ${axis} error:`, err);
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
    console.log('[motor] EMERGENCY STOP');
    showMotorFeedback('🛑 Emergency Stop — all motors');
    try {
        const resp = await fetch(`${API_BASE}/api/motor/stop`, { method: 'POST' });
        const data = await resp.json();
        motorState.altitude.moving = false;
        motorState.azimuth.moving = false;
        const axes = (data.axes || ['altitude', 'azimuth']).join(', ');
        showMotorFeedback(`🛑 Stopped: ${axes}`);
        updateStatus('connected', 'Motors stopped');
    } catch (err) {
        console.error('Stop motors failed:', err);
        showMotorFeedback('✗ Emergency stop failed!');
        updateStatus('error', 'Stop failed!');
    }
}

// Set Home - zero both motor position counters at current location
async function setHome() {
    if (!confirm('Set current position as Home (0,0)?\\n\\nThis zeros both motor position counters at the current physical location.')) {
        return;
    }

    updateStatus('connected', 'Setting home...');
    try {
        const response = await fetch(`${API_BASE}/api/motor/home/set`, { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            updateStatus('connected', '🏠 Home set (0,0)');
        } else {
            updateStatus('error', data.detail || 'Set home failed');
        }
    } catch (err) {
        console.error('Set home failed:', err);
        updateStatus('error', 'Set home failed!');
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

// Camera control - available for MCP tool use, not exposed in UI
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

// RAW capture to ASDF archive
// cameraId: 0=finder, 1=main
// statusId: 'finder' or 'main' (defaults based on cameraId)
async function captureRaw(cameraId, statusId = null) {
    // Determine which status element and frame type to use
    const camName = statusId || (cameraId === 0 ? 'finder' : 'main');
    const statusEl = document.getElementById(`${camName}-capture-status`);

    // Finder always captures as 'light', main uses dropdown
    const frameType = cameraId === 0 ? 'light' : document.getElementById('frame-type').value;

    statusEl.textContent = `Capturing ${frameType}...`;
    statusEl.className = 'capture-status capturing';

    try {
        const response = await fetch(`${API_BASE}/api/camera/${cameraId}/capture?frame_type=${frameType}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.status === 'success') {
            statusEl.textContent = `✓ ${data.camera}/${data.frame_type} #${data.frame_index} → ${data.filename}`;
            statusEl.className = 'capture-status success';
        } else {
            statusEl.textContent = `✗ Error: ${data.error}`;
            statusEl.className = 'capture-status error';
        }
    } catch (err) {
        console.error('Capture failed:', err);
        statusEl.textContent = `✗ Capture failed: ${err.message}`;
        statusEl.className = 'capture-status error';
    }
}
