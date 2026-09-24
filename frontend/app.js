/**
 * HelmetVision AI - Frontend Core Application
 * Real-Time Helmet Detection, Camera Streaming & Deep Learning Model Bridge
 */

(function () {
  'use strict';

  // =========================================================================
  // Application State
  // =========================================================================
  const state = {
    // Media & Camera
    stream: null,
    isCameraActive: false,
    facingMode: 'user', // 'user' (front) or 'environment' (back)
    mediaType: null, // 'webcam', 'video', 'image', 'demo'
    uploadedImage: null,

    // Inference Settings
    inferenceMode: 'simulation', // 'simulation' or 'backend'
    backendUrl: 'http://127.0.0.1:5000/detect',
    confidenceThreshold: 0.65,
    requestIntervalMs: 250,
    autoLogViolations: true,
    audioAlertEnabled: true,
    autoSimEnabled: true,

    // Runtime Metrics & Stats
    stats: {
      totalFrames: 0,
      compliantCount: 0,
      violationCount: 0,
    },
    currentDetections: [],
    lastInferenceTimeMs: 0,
    lastViolationLogTime: 0,
    isProcessingFrame: false,
    simTimer: null,
    simCurrentState: 'safe', // 'safe', 'violation', 'idle'

    // Logs
    incidentLogs: [],

    // Audio Context (Synthesizer)
    audioCtx: null
  };

  // =========================================================================
  // DOM Element References
  // =========================================================================
  const DOM = {
    // Navigation & Telemetry
    systemStatusDot: document.getElementById('system-status-dot'),
    systemStatusText: document.getElementById('system-status-text'),
    modelModeText: document.getElementById('model-mode-text'),
    fpsCounter: document.getElementById('fps-counter'),
    btnAudioToggle: document.getElementById('btn-audio-toggle'),
    audioIconOn: document.getElementById('audio-icon-on'),
    audioIconOff: document.getElementById('audio-icon-off'),
    btnOpenSettings: document.getElementById('btn-open-settings'),

    // Stats
    statTotalFrames: document.getElementById('stat-total-frames'),
    statCompliantCount: document.getElementById('stat-compliant-count'),
    statViolationCount: document.getElementById('stat-violation-count'),
    statComplianceRate: document.getElementById('stat-compliance-rate'),

    // Camera & Viewport
    videoFeed: document.getElementById('webcam-feed'),
    canvas: document.getElementById('detection-canvas'),
    scanline: document.getElementById('scanline'),
    cameraBadge: document.getElementById('camera-badge'),
    cameraPlaceholder: document.getElementById('camera-placeholder'),
    hudResolution: document.getElementById('hud-resolution'),
    hudClock: document.getElementById('hud-clock'),

    // Controls
    btnStartCamera: document.getElementById('btn-start-camera'),
    btnStopCamera: document.getElementById('btn-stop-camera'),
    btnSwitchCamera: document.getElementById('btn-switch-camera'),
    mediaFileInput: document.getElementById('media-file-input'),
    btnCaptureSnapshot: document.getElementById('btn-capture-snapshot'),
    btnQuickStart: document.getElementById('btn-quick-start'),
    btnQuickDemo: document.getElementById('btn-quick-demo'),

    // Status Panel
    statusCardMain: document.getElementById('status-card-main'),
    svgStandby: document.getElementById('svg-standby'),
    svgSafe: document.getElementById('svg-safe'),
    svgViolation: document.getElementById('svg-violation'),
    statusHeadline: document.getElementById('status-headline'),
    statusDescription: document.getElementById('status-description'),
    confidencePercentage: document.getElementById('confidence-percentage'),
    confidenceBarFill: document.getElementById('confidence-bar-fill'),

    // Class Counts
    countHelmet: document.getElementById('count-helmet'),
    countNoHelmet: document.getElementById('count-no-helmet'),
    countPerson: document.getElementById('count-person'),

    // Simulation Widget
    engineStatusBadge: document.getElementById('engine-status-badge'),
    btnSimulateHelmet: document.getElementById('btn-simulate-helmet'),
    btnSimulateViolation: document.getElementById('btn-simulate-violation'),
    btnToggleAutoSim: document.getElementById('btn-toggle-auto-sim'),

    // Logs
    historyTableBody: document.getElementById('history-table-body'),
    logCounter: document.getElementById('log-counter'),
    btnClearLogs: document.getElementById('btn-clear-logs'),
    btnExportCsv: document.getElementById('btn-export-csv'),

    // Settings Modal
    settingsModal: document.getElementById('settings-modal'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    btnCancelSettings: document.getElementById('btn-cancel-settings'),
    btnSaveSettings: document.getElementById('btn-save-settings'),
    modeSim: document.getElementById('mode-sim'),
    modeBackend: document.getElementById('mode-backend'),
    inputApiUrl: document.getElementById('input-api-url'),
    btnTestConnection: document.getElementById('btn-test-connection'),
    connectionStatusMsg: document.getElementById('connection-status-msg'),
    rangeConfidence: document.getElementById('range-confidence'),
    valConfidenceThresh: document.getElementById('val-confidence-thresh'),
    rangeInterval: document.getElementById('range-interval'),
    valIntervalThresh: document.getElementById('val-interval-thresh'),
    chkAutoLog: document.getElementById('chk-auto-log'),
    chkAudioAlert: document.getElementById('chk-audio-alert')
  };

  const ctx = DOM.canvas.getContext('2d');

  // =========================================================================
  // Initialize Application
  // =========================================================================
  function init() {
    setupEventListeners();
    startClockHUD();
    loadSettingsFromStorage();
    renderStatus('standby', 0, 'STANDBY / NO FEED', 'Activate camera or upload footage to monitor safety helmet compliance in real-time.');
    console.log('[HelmetVision AI] Initialized successfully.');
  }

  // =========================================================================
  // Event Listeners
  // =========================================================================
  function setupEventListeners() {
    // Camera controls
    DOM.btnStartCamera.addEventListener('click', startWebcam);
    DOM.btnQuickStart.addEventListener('click', startWebcam);
    DOM.btnStopCamera.addEventListener('click', stopCamera);
    DOM.btnSwitchCamera.addEventListener('click', switchCamera);
    DOM.mediaFileInput.addEventListener('change', handleMediaUpload);
    DOM.btnCaptureSnapshot.addEventListener('click', captureManualSnapshot);
    DOM.btnQuickDemo.addEventListener('click', loadDemoFeed);

    // Audio toggle
    DOM.btnAudioToggle.addEventListener('click', toggleAudio);

    // Simulation controls
    DOM.btnSimulateHelmet.addEventListener('click', () => setManualSimState('safe'));
    DOM.btnSimulateViolation.addEventListener('click', () => setManualSimState('violation'));
    DOM.btnToggleAutoSim.addEventListener('click', toggleAutoSim);

    // History controls
    DOM.btnClearLogs.addEventListener('click', clearLogs);
    DOM.btnExportCsv.addEventListener('click', exportLogsToCsv);

    // Settings Modal
    DOM.btnOpenSettings.addEventListener('click', openSettings);
    DOM.btnCloseModal.addEventListener('click', closeSettings);
    DOM.btnCancelSettings.addEventListener('click', closeSettings);
    DOM.btnSaveSettings.addEventListener('click', saveSettings);
    DOM.btnTestConnection.addEventListener('click', testBackendConnection);

    // Range input live indicators
    DOM.rangeConfidence.addEventListener('input', (e) => {
      DOM.valConfidenceThresh.textContent = `${e.target.value}%`;
    });
    DOM.rangeInterval.addEventListener('input', (e) => {
      DOM.valIntervalThresh.textContent = `${e.target.value} ms`;
    });

    // Window resize handler for canvas bounds
    window.addEventListener('resize', syncCanvasDimensions);

    // Click outside modal to close
    DOM.settingsModal.addEventListener('click', (e) => {
      if (e.target === DOM.settingsModal) closeSettings();
    });
  }

  // =========================================================================
  // Clock & HUD Display
  // =========================================================================
  function startClockHUD() {
    setInterval(() => {
      const now = new Date();
      DOM.hudClock.textContent = now.toTimeString().split(' ')[0];
    }, 1000);
  }

  // =========================================================================
  // Camera & Video Management
  // =========================================================================
  async function startWebcam() {
    try {
      stopCamera();

      DOM.cameraBadge.textContent = 'CONNECTING...';
      DOM.cameraBadge.className = 'badge';

      const constraints = {
        video: {
          facingMode: state.facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      state.stream = await navigator.mediaDevices.getUserMedia(constraints);
      DOM.videoFeed.srcObject = state.stream;
      state.isCameraActive = true;
      state.mediaType = 'webcam';
      state.uploadedImage = null;

      DOM.videoFeed.onloadedmetadata = () => {
        DOM.videoFeed.play();
        syncCanvasDimensions();
        onMediaStreamStarted();
      };

    } catch (err) {
      console.error('[Webcam Error]:', err);
      alert('Unable to access webcam: ' + (err.message || 'Permission denied or no camera device found.'));
      DOM.cameraBadge.textContent = 'ERROR';
      DOM.cameraPlaceholder.classList.remove('hidden');
    }
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach(track => track.stop());
      state.stream = null;
    }
    DOM.videoFeed.pause();
    DOM.videoFeed.srcObject = null;
    DOM.videoFeed.src = '';
    state.isCameraActive = false;
    state.mediaType = null;
    state.uploadedImage = null;

    ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);

    DOM.cameraPlaceholder.classList.remove('hidden');
    DOM.scanline.classList.remove('active');
    DOM.btnStartCamera.disabled = false;
    DOM.btnStopCamera.disabled = true;
    DOM.cameraBadge.textContent = 'STANDBY';
    DOM.cameraBadge.className = 'badge';
    DOM.hudResolution.textContent = 'RES: -- × --';

    DOM.systemStatusDot.classList.remove('active');
    DOM.systemStatusText.textContent = 'STANDBY';

    renderStatus('standby', 0, 'STANDBY / NO FEED', 'Camera stopped. Launch webcam or load demo media to resume helmet monitoring.');
  }

  function switchCamera() {
    state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
    if (state.isCameraActive && state.mediaType === 'webcam') {
      startWebcam();
    } else {
      alert(`Camera orientation set to: ${state.facingMode === 'user' ? 'Front (User)' : 'Rear (Environment)'}`);
    }
  }

  function onMediaStreamStarted() {
    DOM.cameraPlaceholder.classList.add('hidden');
    DOM.scanline.classList.add('active');
    DOM.btnStartCamera.disabled = true;
    DOM.btnStopCamera.disabled = false;

    DOM.cameraBadge.textContent = '● LIVE STREAM';
    DOM.cameraBadge.className = 'badge badge-live';

    DOM.systemStatusDot.classList.add('active');
    DOM.systemStatusText.textContent = 'ONLINE';

    // Update HUD resolution
    const w = DOM.videoFeed.videoWidth || 1280;
    const h = DOM.videoFeed.videoHeight || 720;
    DOM.hudResolution.textContent = `RES: ${w} × ${h}`;

    // Mirroring adjustments
    if (state.facingMode === 'user' && state.mediaType === 'webcam') {
      DOM.videoFeed.classList.remove('unmirror');
    } else {
      DOM.videoFeed.classList.add('unmirror');
    }

    startInferenceLoop();
  }

  function syncCanvasDimensions() {
    const rect = DOM.videoFeed.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      DOM.canvas.width = rect.width;
      DOM.canvas.height = rect.height;
    }
  }

  // =========================================================================
  // Media File Upload Support (Images & Videos)
  // =========================================================================
  function handleMediaUpload(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    stopCamera();

    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');
    const fileUrl = URL.createObjectURL(file);

    if (isVideo) {
      state.mediaType = 'video';
      DOM.videoFeed.src = fileUrl;
      DOM.videoFeed.loop = true;
      DOM.videoFeed.classList.add('unmirror');
      DOM.videoFeed.onloadedmetadata = () => {
        DOM.videoFeed.play();
        syncCanvasDimensions();
        onMediaStreamStarted();
        DOM.cameraBadge.textContent = 'VIDEO FILE';
      };
    } else if (isImage) {
      state.mediaType = 'image';
      const img = new Image();
      img.onload = () => {
        state.uploadedImage = img;
        DOM.cameraPlaceholder.classList.add('hidden');
        DOM.btnStartCamera.disabled = false;
        DOM.btnStopCamera.disabled = false;
        DOM.cameraBadge.textContent = 'IMAGE FILE';
        DOM.cameraBadge.className = 'badge badge-sim';
        DOM.hudResolution.textContent = `RES: ${img.width} × ${img.height}`;
        syncCanvasDimensions();
        renderImageToCanvas(img);
        startInferenceLoop();
      };
      img.src = fileUrl;
    }
  }

  function renderImageToCanvas(img) {
    ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);
    // Draw background image
    const canvasRatio = DOM.canvas.width / DOM.canvas.height;
    const imgRatio = img.width / img.height;
    let drawW = DOM.canvas.width;
    let drawH = DOM.canvas.height;

    ctx.drawImage(img, 0, 0, drawW, drawH);
  }

  // =========================================================================
  // Built-in Demo Video / Simulation Generator
  // =========================================================================
  function loadDemoFeed() {
    stopCamera();
    state.mediaType = 'demo';
    DOM.cameraPlaceholder.classList.add('hidden');
    DOM.scanline.classList.add('active');
    DOM.btnStartCamera.disabled = false;
    DOM.btnStopCamera.disabled = false;
    DOM.cameraBadge.textContent = 'DEMO FEED';
    DOM.cameraBadge.className = 'badge badge-sim';
    DOM.hudResolution.textContent = 'RES: 1280 × 720 (SIM)';

    DOM.systemStatusDot.classList.add('active');
    DOM.systemStatusText.textContent = 'DEMO RUNNING';

    syncCanvasDimensions();
    startInferenceLoop();
  }

  // =========================================================================
  // Detection Inference Loop
  // =========================================================================
  let inferenceIntervalId = null;

  function startInferenceLoop() {
    if (inferenceIntervalId) clearInterval(inferenceIntervalId);

    inferenceIntervalId = setInterval(async () => {
      if (!state.mediaType) return;
      if (state.isProcessingFrame) return;

      state.isProcessingFrame = true;
      const startTime = performance.now();

      try {
        if (state.inferenceMode === 'backend') {
          await runBackendInference();
        } else {
          runSimulationInference();
        }
      } catch (err) {
        console.error('[Inference Error]:', err);
      } finally {
        const elapsed = Math.round(performance.now() - startTime);
        state.lastInferenceTimeMs = elapsed;
        const fps = elapsed > 0 ? Math.round(1000 / Math.max(elapsed, state.requestIntervalMs)) : 30;
        DOM.fpsCounter.textContent = `${elapsed} ms (${fps} FPS)`;
        state.isProcessingFrame = false;
      }

    }, state.requestIntervalMs);
  }

  // =========================================================================
  // Simulation Deep Learning Engine
  // =========================================================================
  let simAngle = 0;

  function runSimulationInference() {
    syncCanvasDimensions();
    const w = DOM.canvas.width;
    const h = DOM.canvas.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    // If static image uploaded, redraw it first
    if (state.mediaType === 'image' && state.uploadedImage) {
      ctx.drawImage(state.uploadedImage, 0, 0, w, h);
    } else if (state.mediaType === 'demo') {
      // Draw simulated camera background graphic
      drawSimulatedSurveillanceBackdrop(w, h);
    }

    simAngle += 0.05;
    // Moving rider position for realistic simulation
    const riderX = (w * 0.35) + Math.sin(simAngle) * 35;
    const riderY = (h * 0.22) + Math.cos(simAngle * 0.8) * 15;
    const riderW = Math.min(w * 0.3, 260);
    const riderH = Math.min(h * 0.65, 420);

    const headX = riderX + (riderW * 0.25);
    const headY = riderY + (riderH * 0.04);
    const headW = riderW * 0.5;
    const headH = riderH * 0.28;

    const detections = [];
    const isHelmet = state.simCurrentState === 'safe';
    const confidence = isHelmet
      ? +(0.92 + Math.sin(simAngle) * 0.05).toFixed(2)
      : +(0.87 + Math.cos(simAngle) * 0.06).toFixed(2);

    // Person/Rider box
    detections.push({
      label: 'Motorcyclist',
      confidence: 0.96,
      box: [riderX, riderY, riderW, riderH],
      color: '#3b82f6',
      type: 'person'
    });

    // Helmet or No-Helmet box
    if (isHelmet) {
      detections.push({
        label: 'Helmet',
        confidence: confidence,
        box: [headX, headY, headW, headH],
        color: '#10b981',
        type: 'helmet'
      });
    } else {
      detections.push({
        label: 'No Helmet',
        confidence: confidence,
        box: [headX, headY, headW, headH],
        color: '#f43f5e',
        type: 'violation'
      });
    }

    state.currentDetections = detections;
    renderBoundingBoxes(detections);
    updateDetectionsMetrics(detections, isHelmet ? 'safe' : 'violation', confidence);
  }

  function drawSimulatedSurveillanceBackdrop(w, h) {
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#0a101f');
    grad.addColorStop(1, '#050810');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Road lane perspective
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.setLineDash([15, 15]);
    ctx.beginPath();
    ctx.moveTo(w * 0.5, h * 0.3);
    ctx.lineTo(w * 0.5, h);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // =========================================================================
  // Live Backend Deep Learning API Bridge (Flask / FastAPI)
  // =========================================================================
  async function runBackendInference() {
    syncCanvasDimensions();
    const w = DOM.canvas.width;
    const h = DOM.canvas.height;
    if (w === 0 || h === 0) return;

    // Grab current frame into temporary canvas to encode image
    const offCanvas = document.createElement('canvas');
    offCanvas.width = 640;
    offCanvas.height = 360;
    const offCtx = offCanvas.getContext('2d');

    if (state.mediaType === 'image' && state.uploadedImage) {
      offCtx.drawImage(state.uploadedImage, 0, 0, 640, 360);
    } else {
      offCtx.drawImage(DOM.videoFeed, 0, 0, 640, 360);
    }

    const base64Data = offCanvas.toDataURL('image/jpeg', 0.75);

    try {
      const response = await fetch(state.backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: base64Data,
          threshold: state.confidenceThreshold
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      ctx.clearRect(0, 0, w, h);

      if (state.mediaType === 'image' && state.uploadedImage) {
        ctx.drawImage(state.uploadedImage, 0, 0, w, h);
      }

      // Expected payload format:
      // {
      //   "detections": [
      //     { "label": "Helmet", "confidence": 0.94, "box": [x, y, width, height] }
      //   ]
      // }
      const detections = (data.detections || []).map(det => {
        const [bx, by, bw, bh] = det.box;
        // Scale box from 640x360 to current canvas resolution
        const scaleX = w / 640;
        const scaleY = h / 360;
        const labelLower = (det.label || '').toLowerCase();
        const isViolation = labelLower.includes('no') || labelLower.includes('violation');
        const isHelmet = labelLower.includes('helmet') && !isViolation;

        return {
          label: det.label,
          confidence: det.confidence,
          box: [bx * scaleX, by * scaleY, bw * scaleX, bh * scaleY],
          color: isViolation ? '#f43f5e' : (isHelmet ? '#10b981' : '#3b82f6'),
          type: isViolation ? 'violation' : (isHelmet ? 'helmet' : 'person')
        };
      });

      state.currentDetections = detections;
      renderBoundingBoxes(detections);

      // Determine overall compliance
      const hasViolation = detections.some(d => d.type === 'violation');
      const hasHelmet = detections.some(d => d.type === 'helmet');
      let statusKey = 'idle';
      let topConfidence = 0;

      if (hasViolation) {
        statusKey = 'violation';
        topConfidence = Math.max(...detections.filter(d => d.type === 'violation').map(d => d.confidence));
      } else if (hasHelmet) {
        statusKey = 'safe';
        topConfidence = Math.max(...detections.filter(d => d.type === 'helmet').map(d => d.confidence));
      }

      updateDetectionsMetrics(detections, statusKey, topConfidence);

    } catch (error) {
      console.warn('[Backend unreachable]: Fallback warning', error);
      DOM.modelModeText.textContent = 'BACKEND OFFLINE';
      DOM.modelModeText.className = 'telemetry-val text-red';
    }
  }

  // =========================================================================
  // Canvas Bounding Box Rendering Engine
  // =========================================================================
  function renderBoundingBoxes(detections) {
    detections.forEach(det => {
      const [x, y, width, height] = det.box;
      const color = det.color || '#06b6d4';

      ctx.save();

      // Translucent Box Fill
      ctx.fillStyle = hexToRgba(color, 0.12);
      drawRoundedRect(ctx, x, y, width, height, 6);
      ctx.fill();

      // Sharp Glowing Border
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      drawRoundedRect(ctx, x, y, width, height, 6);
      ctx.stroke();

      // Reset shadow for text label
      ctx.shadowBlur = 0;

      // Label Pill Background
      const labelText = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = '600 13px Outfit, sans-serif';
      const textMetrics = ctx.measureText(labelText);
      const textW = textMetrics.width + 16;
      const textH = 22;
      const tagY = Math.max(y - textH - 4, 4);

      ctx.fillStyle = color;
      drawRoundedRect(ctx, x, tagY, textW, textH, 4);
      ctx.fill();

      // Label Text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 8, tagY + 15);

      // Corner target brackets for tech HUD aesthetic
      drawCornerBrackets(ctx, x, y, width, height, color);

      ctx.restore();
    });
  }

  function drawRoundedRect(c, x, y, w, h, r) {
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  function drawCornerBrackets(c, x, y, w, h, color) {
    const len = 12;
    c.strokeStyle = color;
    c.lineWidth = 3;

    // Top-Left
    c.beginPath();
    c.moveTo(x, y + len); c.lineTo(x, y); c.lineTo(x + len, y);
    c.stroke();
    // Top-Right
    c.beginPath();
    c.moveTo(x + w - len, y); c.lineTo(x + w, y); c.lineTo(x + w, y + len);
    c.stroke();
    // Bottom-Left
    c.beginPath();
    c.moveTo(x, y + h - len); c.lineTo(x, y + h); c.lineTo(x + len, y + h);
    c.stroke();
    // Bottom-Right
    c.beginPath();
    c.moveTo(x + w - len, y + h); c.lineTo(x + w, y + h); c.lineTo(x + w, y + h - len);
    c.stroke();
  }

  function hexToRgba(hex, alpha) {
    let c = hex.replace('#', '');
    if (c.length === 3) {
      c = c.split('').map(char => char + char).join('');
    }
    const num = parseInt(c, 16);
    return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
  }

  // =========================================================================
  // Metrics & Status Updating
  // =========================================================================
  function updateDetectionsMetrics(detections, statusKey, confidence) {
    state.stats.totalFrames++;

    const helmets = detections.filter(d => d.type === 'helmet').length;
    const violations = detections.filter(d => d.type === 'violation').length;
    const persons = detections.filter(d => d.type === 'person').length;

    DOM.countHelmet.textContent = helmets;
    DOM.countNoHelmet.textContent = violations;
    DOM.countPerson.textContent = persons;

    if (statusKey === 'safe') {
      state.stats.compliantCount++;
      renderStatus('safe', confidence, 'HELMET DETECTED', 'Rider is wearing a protective helmet. Safety protocol satisfied.');
    } else if (statusKey === 'violation') {
      state.stats.violationCount++;
      renderStatus('violation', confidence, 'VIOLATION: NO HELMET', 'Rider detected without a helmet! Safety alert generated.');

      // Synthesize audible violation alarm sound
      playViolationAlarm();

      // Check auto-log throttling (log violation once every 3.5s)
      const now = Date.now();
      if (state.autoLogViolations && (now - state.lastViolationLogTime > 3500)) {
        state.lastViolationLogTime = now;
        recordIncidentLog('No Helmet (Violation)', 'violation', confidence);
      }
    } else {
      renderStatus('standby', 0, 'SCANNING FOR RIDERS', 'Targeting roadway and motorcyclists...');
    }

    // Update cumulative summary stats
    DOM.statTotalFrames.textContent = state.stats.totalFrames.toLocaleString();
    DOM.statCompliantCount.textContent = state.stats.compliantCount.toLocaleString();
    DOM.statViolationCount.textContent = state.stats.violationCount.toLocaleString();

    const totalEvaluated = state.stats.compliantCount + state.stats.violationCount;
    if (totalEvaluated > 0) {
      const rate = Math.round((state.stats.compliantCount / totalEvaluated) * 100);
      DOM.statComplianceRate.textContent = `${rate}%`;
    }
  }

  function renderStatus(type, confidence, headline, description) {
    DOM.statusCardMain.className = `status-card-main status-${type}`;

    DOM.svgStandby.classList.toggle('hidden', type !== 'standby');
    DOM.svgSafe.classList.toggle('hidden', type !== 'safe');
    DOM.svgViolation.classList.toggle('hidden', type !== 'violation');

    DOM.statusHeadline.textContent = headline;
    DOM.statusDescription.textContent = description;

    const confPct = (confidence * 100).toFixed(1);
    DOM.confidencePercentage.textContent = `${confPct}%`;
    DOM.confidenceBarFill.style.width = `${confPct}%`;
  }

  // =========================================================================
  // Simulation Controls & Manual Override
  // =========================================================================
  function setManualSimState(simState) {
    state.simCurrentState = simState;
    if (state.autoSimEnabled) {
      toggleAutoSim(); // disable auto-sim so manual choice persists
    }
    if (!state.mediaType) {
      loadDemoFeed();
    }
  }

  function toggleAutoSim() {
    state.autoSimEnabled = !state.autoSimEnabled;
    DOM.btnToggleAutoSim.classList.toggle('active', state.autoSimEnabled);
    DOM.btnToggleAutoSim.textContent = `Auto Sim: ${state.autoSimEnabled ? 'ON' : 'OFF'}`;

    if (state.autoSimEnabled) {
      startAutoSimCycle();
    } else if (state.simTimer) {
      clearInterval(state.simTimer);
    }
  }

  function startAutoSimCycle() {
    if (state.simTimer) clearInterval(state.simTimer);
    state.simTimer = setInterval(() => {
      if (!state.autoSimEnabled) return;
      // Cycle between safe (70% probability) and violation (30% probability)
      state.simCurrentState = Math.random() > 0.35 ? 'safe' : 'violation';
    }, 4500);
  }
  startAutoSimCycle();

  // =========================================================================
  // Incident Logging & CSV Export
  // =========================================================================
  function recordIncidentLog(label, type, confidence) {
    // Generate snapshot thumbnail from canvas
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = 64;
    thumbCanvas.height = 48;
    const tCtx = thumbCanvas.getContext('2d');

    // Draw current video or image background
    if (state.mediaType === 'image' && state.uploadedImage) {
      tCtx.drawImage(state.uploadedImage, 0, 0, 64, 48);
    } else {
      tCtx.drawImage(DOM.videoFeed, 0, 0, 64, 48);
    }
    // Overlay canvas
    tCtx.drawImage(DOM.canvas, 0, 0, 64, 48);
    const thumbData = thumbCanvas.toDataURL('image/jpeg', 0.6);

    const logEntry = {
      id: `INC-${String(state.incidentLogs.length + 1).padStart(4, '0')}`,
      timestamp: new Date().toLocaleTimeString(),
      fullDate: new Date().toISOString(),
      label: label,
      type: type,
      confidence: (confidence * 100).toFixed(1) + '%',
      thumbnail: thumbData
    };

    state.incidentLogs.unshift(logEntry);
    if (state.incidentLogs.length > 50) state.incidentLogs.pop(); // limit to 50 in memory

    renderIncidentTable();
  }

  function renderIncidentTable() {
    if (state.incidentLogs.length === 0) {
      DOM.historyTableBody.innerHTML = `
        <tr class="empty-row" id="empty-log-row">
          <td colspan="7">No detection incidents recorded yet. Start camera to record events.</td>
        </tr>`;
      DOM.logCounter.textContent = '0 events logged';
      return;
    }

    DOM.logCounter.textContent = `${state.incidentLogs.length} events logged`;

    DOM.historyTableBody.innerHTML = state.incidentLogs.map(item => `
      <tr>
        <td><code>${item.id}</code></td>
        <td>${item.timestamp}</td>
        <td><img src="${item.thumbnail}" class="log-thumb" alt="Incident snapshot"></td>
        <td><strong>${item.label}</strong></td>
        <td>Rider / Motorcyclist</td>
        <td><code>${item.confidence}</code></td>
        <td>
          <span class="pill-badge ${item.type === 'violation' ? 'pill-violation' : 'pill-safe'}">
            ${item.type === 'violation' ? '⚠ VIOLATION' : '✓ COMPLIANT'}
          </span>
        </td>
      </tr>
    `).join('');
  }

  function clearLogs() {
    state.incidentLogs = [];
    renderIncidentTable();
  }

  function exportLogsToCsv() {
    if (state.incidentLogs.length === 0) {
      alert('No logged incidents to export.');
      return;
    }

    let csvContent = 'data:text/csv;charset=utf-8,ID,Timestamp,Result,Classification,Confidence,Status\n';
    state.incidentLogs.forEach(row => {
      csvContent += `"${row.id}","${row.timestamp}","${row.label}","Rider","${row.confidence}","${row.type === 'violation' ? 'Violation' : 'Compliant'}"\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `HelmetDetection_Logs_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function captureManualSnapshot() {
    recordIncidentLog(
      state.simCurrentState === 'violation' ? 'Manual Capture (Violation)' : 'Manual Capture (Helmet)',
      state.simCurrentState === 'violation' ? 'violation' : 'safe',
      0.95
    );
    alert('Surveillance snapshot captured and added to incident log.');
  }

  // =========================================================================
  // Web Audio Synthesizer (Violation Alarm)
  // =========================================================================
  function toggleAudio() {
    state.audioAlertEnabled = !state.audioAlertEnabled;
    DOM.chkAudioAlert.checked = state.audioAlertEnabled;
    DOM.audioIconOn.classList.toggle('hidden', !state.audioAlertEnabled);
    DOM.audioIconOff.classList.toggle('hidden', state.audioAlertEnabled);
    if (state.audioAlertEnabled) {
      playViolationAlarm();
    }
  }

  function playViolationAlarm() {
    if (!state.audioAlertEnabled) return;

    try {
      if (!state.audioCtx) {
        const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
        state.audioCtx = new AudioCtxClass();
      }

      if (state.audioCtx.state === 'suspended') {
        state.audioCtx.resume();
      }

      const now = state.audioCtx.currentTime;
      const osc = state.audioCtx.createOscillator();
      const gain = state.audioCtx.createGain();

      osc.type = 'sawtooth';
      // Futuristic 2-tone alarm: 880Hz down to 440Hz
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.exponentialRampToValueAtTime(440, now + 0.18);

      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);

      osc.connect(gain);
      gain.connect(state.audioCtx.destination);

      osc.start(now);
      osc.stop(now + 0.2);
    } catch (e) {
      // Audio autoplay restrictions or unsupported
      console.warn('[Audio Alert Notice]:', e);
    }
  }

  // =========================================================================
  // Settings Modal & Local Storage
  // =========================================================================
  function openSettings() {
    DOM.inputApiUrl.value = state.backendUrl;
    DOM.rangeConfidence.value = Math.round(state.confidenceThreshold * 100);
    DOM.valConfidenceThresh.textContent = `${DOM.rangeConfidence.value}%`;
    DOM.rangeInterval.value = state.requestIntervalMs;
    DOM.valIntervalThresh.textContent = `${state.requestIntervalMs} ms`;
    DOM.chkAutoLog.checked = state.autoLogViolations;
    DOM.chkAudioAlert.checked = state.audioAlertEnabled;

    if (state.inferenceMode === 'backend') {
      DOM.modeBackend.checked = true;
    } else {
      DOM.modeSim.checked = true;
    }

    DOM.settingsModal.classList.remove('hidden');
  }

  function closeSettings() {
    DOM.settingsModal.classList.add('hidden');
  }

  function saveSettings() {
    state.inferenceMode = DOM.modeBackend.checked ? 'backend' : 'simulation';
    state.backendUrl = DOM.inputApiUrl.value.trim() || 'http://127.0.0.1:5000/detect';
    state.confidenceThreshold = Number(DOM.rangeConfidence.value) / 100;
    state.requestIntervalMs = Number(DOM.rangeInterval.value);
    state.autoLogViolations = DOM.chkAutoLog.checked;
    state.audioAlertEnabled = DOM.chkAudioAlert.checked;

    DOM.modelModeText.textContent = state.inferenceMode === 'backend' ? 'LIVE BACKEND' : 'SIMULATED DL';
    DOM.modelModeText.className = state.inferenceMode === 'backend' ? 'telemetry-val highlight-cyan' : 'telemetry-val';
    DOM.engineStatusBadge.textContent = state.inferenceMode === 'backend' ? 'Backend API Active' : 'Simulation Active';

    // Persist to localStorage
    const saved = {
      inferenceMode: state.inferenceMode,
      backendUrl: state.backendUrl,
      confidenceThreshold: state.confidenceThreshold,
      requestIntervalMs: state.requestIntervalMs,
      autoLogViolations: state.autoLogViolations,
      audioAlertEnabled: state.audioAlertEnabled
    };
    try {
      localStorage.setItem('helmetvision_settings', JSON.stringify(saved));
    } catch (_) {}

    // Restart inference loop with new interval
    startInferenceLoop();
    closeSettings();
  }

  function loadSettingsFromStorage() {
    try {
      const data = localStorage.getItem('helmetvision_settings');
      if (data) {
        const parsed = JSON.parse(data);
        if (parsed.inferenceMode) state.inferenceMode = parsed.inferenceMode;
        if (parsed.backendUrl) state.backendUrl = parsed.backendUrl;
        if (parsed.confidenceThreshold) state.confidenceThreshold = parsed.confidenceThreshold;
        if (parsed.requestIntervalMs) state.requestIntervalMs = parsed.requestIntervalMs;
        if (typeof parsed.autoLogViolations === 'boolean') state.autoLogViolations = parsed.autoLogViolations;
        if (typeof parsed.audioAlertEnabled === 'boolean') state.audioAlertEnabled = parsed.audioAlertEnabled;
      }
    } catch (_) {}

    DOM.modelModeText.textContent = state.inferenceMode === 'backend' ? 'LIVE BACKEND' : 'SIMULATED DL';
    DOM.engineStatusBadge.textContent = state.inferenceMode === 'backend' ? 'Backend API Active' : 'Simulation Active';
  }

  async function testBackendConnection() {
    const url = DOM.inputApiUrl.value.trim();
    DOM.connectionStatusMsg.textContent = 'Pinging backend endpoint...';
    DOM.connectionStatusMsg.style.color = '#38bdf8';

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);

      // Simple OPTIONS or POST ping
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ping: true }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (res.ok || res.status === 400 || res.status === 404) {
        DOM.connectionStatusMsg.textContent = `Backend online! (Server replied HTTP ${res.status})`;
        DOM.connectionStatusMsg.style.color = '#34d399';
      } else {
        DOM.connectionStatusMsg.textContent = `Backend reached with error HTTP ${res.status}`;
        DOM.connectionStatusMsg.style.color = '#f59e0b';
      }
    } catch (err) {
      DOM.connectionStatusMsg.textContent = `Unable to connect: ${err.message}. Ensure your Python backend is running on ${url}`;
      DOM.connectionStatusMsg.style.color = '#fb7185';
    }
  }

  // Self-execute initialization on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
