// Telescope Control Dashboard JavaScript

const API_BASE = '';

// Position polling
let positionInterval;

document.addEventListener('DOMContentLoaded', () => {
    updateStatus('connected', 'Connected');
    startPositionPolling();

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
    positionInterval = setInterval(updatePosition, 1000);
}

async function updatePosition() {
    try {
        const response = await fetch(`${API_BASE}/api/position`);
        const data = await response.json();
        document.getElementById('altitude').textContent = data.altitude.toFixed(1);
        document.getElementById('azimuth').textContent = data.azimuth.toFixed(1);
    } catch (err) {
        console.error('Position update failed:', err);
    }
}

// Motor control
async function moveAlt(steps) {
    const speed = document.getElementById('speed').value;
    try {
        await fetch(`${API_BASE}/api/motor/altitude?steps=${steps}&speed=${speed}`, {
            method: 'POST'
        });
    } catch (err) {
        console.error('Move altitude failed:', err);
        updateStatus('error', 'Motor error');
    }
}

async function moveAz(steps) {
    const speed = document.getElementById('speed').value;
    try {
        await fetch(`${API_BASE}/api/motor/azimuth?steps=${steps}&speed=${speed}`, {
            method: 'POST'
        });
    } catch (err) {
        console.error('Move azimuth failed:', err);
        updateStatus('error', 'Motor error');
    }
}

async function stopMotors() {
    try {
        await fetch(`${API_BASE}/api/motor/stop`, { method: 'POST' });
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
