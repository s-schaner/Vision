/*
 * VolleySense CalibMap core client logic
 * This file intentionally contains extensive inline documentation so that
 * implementers can understand the math-heavy portions (homography, Kalman,
 * aspect-correct drawing) without external references.
 */

import { BALL_PROMPT, HUMAN_PROMPT } from './llm_prompts.js';

const video = document.getElementById('video');
const videoInput = document.getElementById('video-input');
const playToggle = document.getElementById('play-toggle');
const snapFrameBtn = document.getElementById('snap-frame');
const timecodeEl = document.getElementById('timecode');
const videoOverlay = document.getElementById('video-overlay');
const calibrationCanvas = document.getElementById('calibration-canvas');
const resetCalibrationBtn = document.getElementById('reset-calibration');
const confirmHBtn = document.getElementById('confirm-h');
const pickBallBtn = document.getElementById('pick-ball');
const pickNetBtn = document.getElementById('pick-net');
const pickPlayerBtn = document.getElementById('pick-player');
const positionBallBtn = document.getElementById('position-ball');
const positionPlayerBtn = document.getElementById('position-player');
const fitModeSelect = document.getElementById('fit-mode');
const homographyStatus = document.getElementById('homography-status');
const rmsStatus = document.getElementById('rms-status');
const serviceStatus = document.getElementById('service-status');
const courtCanvas = document.getElementById('court-canvas');
const courtDropTarget = document.getElementById('court-drop-target');
const llmBaseInput = document.getElementById('llm-base');
const llmModelInput = document.getElementById('llm-model');
const frameFpsInput = document.getElementById('frame-fps');
const frameResizeWInput = document.getElementById('frame-resize-w');
const frameResizeHInput = document.getElementById('frame-resize-h');
const frameJpegQInput = document.getElementById('frame-jpegq');
const frameStartInput = document.getElementById('frame-start');
const frameDurationInput = document.getElementById('frame-duration');
const generateBallBtn = document.getElementById('generate-ball');
const generatePlayersBtn = document.getElementById('generate-players');
const stopAnalysisBtn = document.getElementById('stop-analysis');
const ballJsonInput = document.getElementById('ball-json-input');
const humansJsonInput = document.getElementById('humans-json-input');
const downloadBallLink = document.getElementById('download-ball');
const downloadHumansLink = document.getElementById('download-humans');
const telemetryTime = document.getElementById('telemetry-time');
const telemetryBall = document.getElementById('telemetry-ball');
const telemetryBallFilt = document.getElementById('telemetry-ball-filt');
const telemetryBallZ = document.getElementById('telemetry-ball-z');
const splitterVertical = document.getElementById('splitter-vertical');
const splitterHorizontal = document.getElementById('splitter-horizontal');
const panelLeft = document.getElementById('panel-left');
const panelRight = document.getElementById('panel-right');
const playerSection = document.querySelector('.player-section');
const calibrationSection = document.querySelector('.calibration-section');

const overlayConfig = {
  courtWidthM: 9,
  courtLengthM: 18,
  marginFt: 20,
  netHeightMen: 2.43,
  netHeightWomen: 2.24,
};
const FT_TO_M = 0.3048;
const overlayMetersWidth = overlayConfig.courtWidthM + 2 * overlayConfig.marginFt * FT_TO_M;
const overlayMetersLength = overlayConfig.courtLengthM + 2 * overlayConfig.marginFt * FT_TO_M;
const overlayScale = {
  sx: 1000 / overlayMetersWidth,
  sy: 1000 / overlayMetersLength,
};
const overlayOriginMeters = {
  x: overlayConfig.marginFt * FT_TO_M,
  y: overlayConfig.marginFt * FT_TO_M,
};

const state = {
  videoFile: null,
  snappedFrame: null, // ImageBitmap for calibration
  offscreenCanvas: document.createElement('canvas'),
  fitTransform: { sx: 1, sy: 1, dx: 0, dy: 0 },
  mode: 'idle',
  dragging: null,
  calibration: {
    imagePoints: [], // {x,y, overlay:{x,y}}
    H: null,
    Hinv: null,
    rms: null,
    netPosts: [],
    netHeight: overlayConfig.netHeightMen,
    ball: null, // {center:{x,y}, radius, ground:{x,y}, overlay:{x,y,z}}
    player: null, // {box:{x,y,w,h}, foot:{x,y}, overlay:{x,y}}
  },
  tracks: {
    ball: null,
    ballFiltered: null,
    humans: null,
  },
  analysisAbort: false,
};

const DPR = window.devicePixelRatio || 1;

const calibCtx = calibrationCanvas.getContext('2d');
const overlayCtx = videoOverlay.getContext('2d');
const courtCtx = courtCanvas.getContext('2d');

function resizeCanvasToDisplaySize(canvas) {
  const { clientWidth, clientHeight } = canvas;
  const width = Math.max(1, Math.floor(clientWidth * DPR));
  const height = Math.max(1, Math.floor(clientHeight * DPR));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function computeFitTransform(canvas, imgWidth, imgHeight, mode) {
  const cw = canvas.width / DPR;
  const ch = canvas.height / DPR;
  if (!imgWidth || !imgHeight || !cw || !ch) {
    return { sx: 1, sy: 1, dx: 0, dy: 0, scale: 1 };
  }
  const scaleX = cw / imgWidth;
  const scaleY = ch / imgHeight;
  const scale = mode === 'cover' ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY);
  const drawW = imgWidth * scale;
  const drawH = imgHeight * scale;
  const dx = (cw - drawW) / 2;
  const dy = (ch - drawH) / 2;
  return { sx: scale, sy: scale, dx, dy, drawW, drawH };
}

function imageToCanvas(pt) {
  return {
    x: (pt.x * state.fitTransform.sx + state.fitTransform.dx) * DPR,
    y: (pt.y * state.fitTransform.sy + state.fitTransform.dy) * DPR,
  };
}

function canvasToImage(x, y) {
  const px = x / DPR;
  const py = y / DPR;
  const { sx, sy, dx, dy } = state.fitTransform;
  if (!sx || !sy) return null;
  const ix = (px - dx) / sx;
  const iy = (py - dy) / sy;
  const img = state.snappedFrame;
  if (!img) return null;
  if (ix < 0 || iy < 0 || ix > img.width || iy > img.height) return null;
  return { x: ix, y: iy };
}

function drawCalibration() {
  resizeCanvasToDisplaySize(calibrationCanvas);
  calibCtx.save();
  calibCtx.scale(DPR, DPR);
  calibCtx.clearRect(0, 0, calibrationCanvas.width / DPR, calibrationCanvas.height / DPR);
  const frame = state.snappedFrame;
  if (!frame) {
    calibCtx.restore();
    return;
  }
  const mode = fitModeSelect.value;
  state.fitTransform = computeFitTransform(calibrationCanvas, frame.width, frame.height, mode);
  const { sx, sy, dx, dy } = state.fitTransform;
  calibCtx.drawImage(frame, dx, dy, frame.width * sx, frame.height * sy);

  calibCtx.lineWidth = 1.5;
  calibCtx.strokeStyle = 'rgba(255,255,255,0.65)';
  calibCtx.fillStyle = 'rgba(255,255,255,0.65)';
  state.calibration.imagePoints.forEach((pt, idx) => {
    const canvasPt = imageToCanvas(pt);
    if (!canvasPt) return;
    calibCtx.beginPath();
    calibCtx.arc(canvasPt.x / DPR, canvasPt.y / DPR, 5, 0, Math.PI * 2);
    calibCtx.fill();
    calibCtx.fillText(`#${idx} (${pt.overlay?.x?.toFixed(1) ?? '?'},${pt.overlay?.y?.toFixed(1) ?? '?'})`, canvasPt.x / DPR + 8, canvasPt.y / DPR - 8);
  });

  if (state.calibration.H) {
    drawCalibrationGrid();
  }

  drawNet();
  drawBall();
  drawPlayer();
  calibCtx.restore();
}

function drawCalibrationGrid() {
  const { H } = state.calibration;
  if (!H) return;
  calibCtx.save();
  calibCtx.lineWidth = 1;
  calibCtx.strokeStyle = 'rgba(102, 194, 255, 0.6)';
  const overlaySteps = 10;
  for (let i = 0; i <= overlaySteps; i++) {
    const u = (i / overlaySteps) * 1000;
    drawOverlaySegment({ x: u, y: 0 }, { x: u, y: 1000 });
    drawOverlaySegment({ x: 0, y: u }, { x: 1000, y: u });
  }
  calibCtx.restore();
}

function drawOverlaySegment(p1, p2) {
  const a = projectOverlayToCanvas(p1);
  const b = projectOverlayToCanvas(p2);
  if (!a || !b) return;
  calibCtx.beginPath();
  calibCtx.moveTo(a.x / DPR, a.y / DPR);
  calibCtx.lineTo(b.x / DPR, b.y / DPR);
  calibCtx.stroke();
}

function drawNet() {
  const { netPosts } = state.calibration;
  if (netPosts.length < 2) return;
  calibCtx.save();
  calibCtx.strokeStyle = 'rgba(255, 120, 120, 0.8)';
  calibCtx.lineWidth = 2;
  const a = imageToCanvas(netPosts[0]);
  const b = imageToCanvas(netPosts[1]);
  if (a && b) {
    calibCtx.beginPath();
    calibCtx.moveTo(a.x / DPR, a.y / DPR);
    calibCtx.lineTo(b.x / DPR, b.y / DPR);
    calibCtx.stroke();
  }
  calibCtx.restore();
}

function drawBall() {
  const ball = state.calibration.ball;
  if (!ball) return;
  const center = imageToCanvas(ball.center);
  if (!center) return;
  calibCtx.save();
  calibCtx.strokeStyle = 'rgba(255, 230, 120, 0.9)';
  calibCtx.lineWidth = 2;
  calibCtx.beginPath();
  calibCtx.arc(center.x / DPR, center.y / DPR, ball.radius * state.fitTransform.sx, 0, Math.PI * 2);
  calibCtx.stroke();
  if (ball.ground) {
    const ground = imageToCanvas(ball.ground);
    if (ground) {
      calibCtx.strokeStyle = 'rgba(255, 180, 60, 0.9)';
      calibCtx.beginPath();
      calibCtx.moveTo(ground.x / DPR - 6, ground.y / DPR);
      calibCtx.lineTo(ground.x / DPR + 6, ground.y / DPR);
      calibCtx.moveTo(ground.x / DPR, ground.y / DPR - 6);
      calibCtx.lineTo(ground.x / DPR, ground.y / DPR + 6);
      calibCtx.stroke();
    }
  }
  calibCtx.restore();
}

function drawPlayer() {
  const player = state.calibration.player;
  if (!player) return;
  const { box, foot } = player;
  const tl = imageToCanvas({ x: box.x, y: box.y });
  const br = imageToCanvas({ x: box.x + box.w, y: box.y + box.h });
  if (!tl || !br) return;
  calibCtx.save();
  calibCtx.strokeStyle = 'rgba(138, 180, 255, 0.9)';
  calibCtx.lineWidth = 2;
  calibCtx.strokeRect(tl.x / DPR, tl.y / DPR, (br.x - tl.x) / DPR, (br.y - tl.y) / DPR);
  if (foot) {
    const footCanvas = imageToCanvas(foot);
    if (footCanvas) {
      calibCtx.beginPath();
      calibCtx.arc(footCanvas.x / DPR, footCanvas.y / DPR, 5, 0, Math.PI * 2);
      calibCtx.fillStyle = 'rgba(138, 180, 255, 0.9)';
      calibCtx.fill();
    }
  }
  calibCtx.restore();
}

function resetCalibration() {
  state.calibration.imagePoints = [];
  state.calibration.H = null;
  state.calibration.Hinv = null;
  state.calibration.rms = null;
  state.calibration.netPosts = [];
  state.calibration.ball = null;
  state.calibration.player = null;
  updateCalibrationButtons();
  homographyStatus.classList.remove('pill-ok');
  homographyStatus.classList.add('pill-warn');
  homographyStatus.textContent = 'H: unset';
  rmsStatus.textContent = 'RMS: --';
  drawCalibration();
}

function updateCalibrationButtons() {
  const ready = !!state.snappedFrame;
  pickBallBtn.disabled = !ready;
  pickNetBtn.disabled = !ready;
  pickPlayerBtn.disabled = !ready;
  positionBallBtn.disabled = !(ready && state.calibration.ball && state.calibration.H);
  positionPlayerBtn.disabled = !(ready && state.calibration.player && state.calibration.H);
  confirmHBtn.disabled = !(state.calibration.imagePoints.length >= 4);
  generateBallBtn.disabled = !state.calibration.H;
  generatePlayersBtn.disabled = !state.calibration.H;
}

function addCalibrationPoint(pt, overlay) {
  state.calibration.imagePoints.push({ ...pt, overlay });
  updateCalibrationButtons();
  drawCalibration();
}

function removeNearestCalibrationPoint(pt) {
  const points = state.calibration.imagePoints;
  if (!points.length) return;
  let bestIdx = 0;
  let bestDist = Infinity;
  points.forEach((p, idx) => {
    const dx = p.x - pt.x;
    const dy = p.y - pt.y;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = idx;
    }
  });
  points.splice(bestIdx, 1);
  updateCalibrationButtons();
  drawCalibration();
}

function computeHomography() {
  const pts = state.calibration.imagePoints;
  if (pts.length < 4) return;
  const imagePts = pts.map((p) => ({ x: p.x, y: p.y }));
  const overlayPts = pts.map((p) => p.overlay);
  const { H, rms } = normalizedDLT(imagePts, overlayPts);
  if (!H) {
    alert('Failed to compute homography. Ensure points are not collinear.');
    return;
  }
  state.calibration.H = H;
  state.calibration.Hinv = invert3x3(H);
  state.calibration.rms = rms;
  homographyStatus.classList.remove('pill-warn');
  homographyStatus.classList.add('pill-ok');
  homographyStatus.textContent = 'H: ready';
  rmsStatus.textContent = `RMS: ${rms.toFixed(2)}`;
  drawCalibration();
  updateCalibrationButtons();
}

function normalizedDLT(imagePts, overlayPts) {
  const n = imagePts.length;
  if (overlayPts.length !== n || n < 4) {
    return { H: null, rms: null };
  }
  const normImg = normalizePoints(imagePts);
  const normOv = normalizePoints(overlayPts);
  const A = [];
  for (let i = 0; i < n; i++) {
    const { x: xi, y: yi } = normImg.points[i];
    const { x: ui, y: vi } = normOv.points[i];
    A.push([-xi, -yi, -1, 0, 0, 0, ui * xi, ui * yi, ui]);
    A.push([0, 0, 0, -xi, -yi, -1, vi * xi, vi * yi, vi]);
  }
  const ATA = multiplyMatrices(transpose(A), A);
  const h = smallestEigenVector(ATA);
  if (!h) return { H: null, rms: null };
  const Hnorm = [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], h[8]],
  ];
  const TovInv = invert3x3(normOv.T);
  const Htemp = multiply3x3(TovInv, multiply3x3(Hnorm, normImg.T));
  const scale = Htemp[2][2];
  const H = Htemp.map((row) => row.map((val) => val / scale));
  const rms = computeRMS(H, imagePts, overlayPts);
  return { H, rms };
}

function normalizePoints(points) {
  const n = points.length;
  let cx = 0;
  let cy = 0;
  points.forEach((p) => {
    cx += p.x;
    cy += p.y;
  });
  cx /= n;
  cy /= n;
  let meanDist = 0;
  points.forEach((p) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    meanDist += Math.sqrt(dx * dx + dy * dy);
  });
  meanDist /= n;
  const scale = Math.SQRT2 / (meanDist || 1);
  const T = [
    [scale, 0, -scale * cx],
    [0, scale, -scale * cy],
    [0, 0, 1],
  ];
  const normPts = points.map((p) => {
    const x = scale * (p.x - cx);
    const y = scale * (p.y - cy);
    return { x, y };
  });
  return { points: normPts, T };
}

function computeRMS(H, imagePts, overlayPts) {
  let accum = 0;
  for (let i = 0; i < imagePts.length; i++) {
    const proj = applyHomography(H, imagePts[i]);
    const dx = proj.x - overlayPts[i].x;
    const dy = proj.y - overlayPts[i].y;
    accum += dx * dx + dy * dy;
  }
  return Math.sqrt(accum / imagePts.length);
}

function applyHomography(H, pt) {
  const x = H[0][0] * pt.x + H[0][1] * pt.y + H[0][2];
  const y = H[1][0] * pt.x + H[1][1] * pt.y + H[1][2];
  const w = H[2][0] * pt.x + H[2][1] * pt.y + H[2][2];
  return { x: x / w, y: y / w };
}

function applyHomographyInverse(Hinv, pt) {
  const x = Hinv[0][0] * pt.x + Hinv[0][1] * pt.y + Hinv[0][2];
  const y = Hinv[1][0] * pt.x + Hinv[1][1] * pt.y + Hinv[1][2];
  const w = Hinv[2][0] * pt.x + Hinv[2][1] * pt.y + Hinv[2][2];
  return { x: x / w, y: y / w };
}

function transpose(mat) {
  return mat[0].map((_, i) => mat.map((row) => row[i]));
}

function multiplyMatrices(a, b) {
  const result = Array.from({ length: a.length }, () => new Array(b[0].length).fill(0));
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < b.length; k++) {
      const val = a[i][k];
      for (let j = 0; j < b[0].length; j++) {
        result[i][j] += val * b[k][j];
      }
    }
  }
  return result;
}

function multiply3x3(a, b) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return out;
}

function invert3x3(m) {
  const det =
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (Math.abs(det) < 1e-12) return null;
  const invDet = 1 / det;
  const out = [
    [
      (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet,
      (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet,
      (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet,
    ],
    [
      (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet,
      (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet,
      (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet,
    ],
    [
      (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet,
      (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet,
      (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet,
    ],
  ];
  return out;
}

function solveLinearSystem(M, b) {
  const n = M.length;
  const A = M.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    let maxVal = Math.abs(A[i][i]);
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(A[r][i]) > maxVal) {
        maxVal = Math.abs(A[r][i]);
        pivot = r;
      }
    }
    if (maxVal < 1e-12) return null;
    if (pivot !== i) {
      const tmp = A[i];
      A[i] = A[pivot];
      A[pivot] = tmp;
    }
    const pivotVal = A[i][i];
    for (let j = i; j <= n; j++) {
      A[i][j] /= pivotVal;
    }
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = A[r][i];
      if (!factor) continue;
      for (let j = i; j <= n; j++) {
        A[r][j] -= factor * A[i][j];
      }
    }
  }
  return A.map((row) => row[n]);
}

function smallestEigenVector(M) {
  const n = M.length;
  let v = new Array(n).fill(0).map(() => Math.random() - 0.5);
  normalizeVector(v);
  let lambda = 0;
  for (let iter = 0; iter < 80; iter++) {
    const x = solveLinearSystem(M, v);
    if (!x) return null;
    normalizeVector(x);
    const nextLambda = dot(v, x);
    if (Math.abs(nextLambda - lambda) < 1e-8) {
      v = x;
      break;
    }
    v = x;
    lambda = nextLambda;
  }
  return v;
}

function normalizeVector(v) {
  const norm = Math.sqrt(v.reduce((acc, val) => acc + val * val, 0)) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

async function loadVideo(file) {
  const url = URL.createObjectURL(file);
  video.src = url;
  state.videoFile = file;
  await video.play().catch(() => {});
  video.pause();
  playToggle.disabled = false;
  snapFrameBtn.disabled = false;
  updateCalibrationButtons();
}

function togglePlayback() {
  if (video.paused) {
    video.play();
    playToggle.textContent = 'Pause';
  } else {
    video.pause();
    playToggle.textContent = 'Play';
  }
}

async function grabFrame() {
  if (!video.videoWidth) return;
  const off = state.offscreenCanvas;
  off.width = video.videoWidth;
  off.height = video.videoHeight;
  const ctx = off.getContext('2d');
  ctx.drawImage(video, 0, 0, off.width, off.height);
  const bitmap = await createImageBitmap(off);
  if (state.snappedFrame) state.snappedFrame.close?.();
  state.snappedFrame = bitmap;
  resetCalibration();
  drawCalibration();
  pickBallBtn.disabled = false;
  pickNetBtn.disabled = false;
  pickPlayerBtn.disabled = false;
  positionBallBtn.disabled = true;
  positionPlayerBtn.disabled = true;
}

function handleCalibrationCanvasPointerDown(event) {
  const rect = calibrationCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * DPR;
  const y = (event.clientY - rect.top) * DPR;
  const imgPt = canvasToImage(x, y);
  if (!imgPt) return;
  if (event.shiftKey) {
    removeNearestCalibrationPoint(imgPt);
    return;
  }
  switch (state.mode) {
    case 'ball-center': {
      state.calibration.ball = { center: imgPt, radius: 20, ground: null, overlay: null };
      const radius = parseFloat(prompt('Ball radius in image pixels?', '25'));
      if (!Number.isNaN(radius)) {
        state.calibration.ball.radius = Math.max(1, radius);
      }
      state.mode = 'ball-ground';
      drawCalibration();
      break;
    }
    case 'ball-ground': {
      if (state.calibration.ball) {
        state.calibration.ball.ground = imgPt;
      }
      state.mode = 'idle';
      drawCalibration();
      updateCalibrationButtons();
      break;
    }
    case 'net-first': {
      state.calibration.netPosts = [imgPt];
      state.mode = 'net-second';
      drawCalibration();
      break;
    }
    case 'net-second': {
      state.calibration.netPosts.push(imgPt);
      const current = state.calibration.netHeight ?? overlayConfig.netHeightMen;
      const netHeightInput = prompt('Net height in meters (e.g., 2.43 men / 2.24 women)', `${current}`);
      const netHeight = parseFloat(netHeightInput);
      if (Number.isFinite(netHeight)) {
        state.calibration.netHeight = netHeight;
      }
      state.mode = 'idle';
      drawCalibration();
      break;
    }
    case 'player-first': {
      state.dragging = { type: 'player-box', start: imgPt, current: imgPt };
      state.calibration.player = { box: { x: imgPt.x, y: imgPt.y, w: 0, h: 0 }, foot: null, overlay: null };
      break;
    }
    case 'player-foot': {
      if (state.calibration.player) {
        state.calibration.player.foot = imgPt;
      }
      state.mode = 'idle';
      drawCalibration();
      updateCalibrationButtons();
      break;
    }
    default: {
      if (tryBeginBallDrag(imgPt)) return;
      if (tryBeginPlayerDrag(imgPt)) return;
      if (state.mode === 'idle' && state.snappedFrame) {
        const overlayPrompt = prompt('Overlay coordinates x,y (0..1000) separated by comma', '0,0');
        if (!overlayPrompt) return;
        const [xStr, yStr] = overlayPrompt.split(',');
        const overlay = { x: parseFloat(xStr), y: parseFloat(yStr) };
        if (Number.isFinite(overlay.x) && Number.isFinite(overlay.y)) {
          addCalibrationPoint(imgPt, overlay);
        }
      }
    }
  }
}

function handleCalibrationCanvasPointerMove(event) {
  if (!state.dragging) return;
  const rect = calibrationCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * DPR;
  const y = (event.clientY - rect.top) * DPR;
  const imgPt = canvasToImage(x, y);
  if (!imgPt) return;
  if (state.dragging.type === 'player-box') {
    state.dragging.current = imgPt;
    const start = state.dragging.start;
    const box = state.calibration.player.box;
    box.x = Math.min(start.x, imgPt.x);
    box.y = Math.min(start.y, imgPt.y);
    box.w = Math.abs(start.x - imgPt.x);
    box.h = Math.abs(start.y - imgPt.y);
    drawCalibration();
  } else if (state.dragging.type === 'ball') {
    state.calibration.ball.center = imgPt;
    drawCalibration();
  } else if (state.dragging.type === 'player-move') {
    const { offset } = state.dragging;
    const box = state.calibration.player.box;
    box.x = imgPt.x - offset.x;
    box.y = imgPt.y - offset.y;
    drawCalibration();
  }
}

function handleCalibrationCanvasPointerUp(event) {
  if (!state.dragging) return;
  if (state.dragging.type === 'player-box') {
    state.mode = 'player-foot';
  }
  state.dragging = null;
}

function tryBeginBallDrag(imgPt) {
  const ball = state.calibration.ball;
  if (!ball) return false;
  const dx = imgPt.x - ball.center.x;
  const dy = imgPt.y - ball.center.y;
  const dist2 = dx * dx + dy * dy;
  if (dist2 < Math.pow(ball.radius * 1.2, 2)) {
    state.dragging = { type: 'ball' };
    return true;
  }
  return false;
}

function tryBeginPlayerDrag(imgPt) {
  const player = state.calibration.player;
  if (!player) return false;
  const { box } = player;
  if (imgPt.x >= box.x && imgPt.x <= box.x + box.w && imgPt.y >= box.y && imgPt.y <= box.y + box.h) {
    state.dragging = { type: 'player-move', offset: { x: imgPt.x - box.x, y: imgPt.y - box.y } };
    return true;
  }
  return false;
}

function projectOverlayToCanvas(pt) {
  const Hinv = state.calibration.Hinv;
  if (!Hinv) return null;
  const img = applyHomographyInverse(Hinv, pt);
  return imageToCanvas(img);
}

function projectOverlayToImage(pt) {
  const Hinv = state.calibration.Hinv;
  if (!Hinv) return null;
  return applyHomographyInverse(Hinv, pt);
}

function handleKeydown(event) {
  if (event.code === 'Space') {
    event.preventDefault();
    togglePlayback();
  }
}

function updateTimecode() {
  timecodeEl.textContent = video.currentTime.toFixed(3);
}

function updateOverlay() {
  resizeCanvasToDisplaySize(videoOverlay);
  overlayCtx.save();
  overlayCtx.scale(DPR, DPR);
  overlayCtx.clearRect(0, 0, videoOverlay.width / DPR, videoOverlay.height / DPR);
  const Hinv = state.calibration.Hinv;
  if (!Hinv || !state.tracks.ballFiltered) {
    overlayCtx.restore();
    return;
  }
  const t = video.currentTime;
  const ballSample = interpolateTrack(state.tracks.ballFiltered.track, t);
  if (ballSample) {
    const imgPt = applyHomographyInverse(Hinv, { x: ballSample.x, y: ballSample.y });
    const canvasPt = imageToCanvas(imgPt);
    if (canvasPt) {
      overlayCtx.fillStyle = 'rgba(255, 200, 80, 0.9)';
      overlayCtx.beginPath();
      overlayCtx.arc(canvasPt.x / DPR, canvasPt.y / DPR, 8, 0, Math.PI * 2);
      overlayCtx.fill();
    }
  }
  overlayCtx.restore();
}

function interpolateTrack(track, t) {
  if (!track || !track.length) return null;
  if (t <= track[0].t) return track[0];
  if (t >= track[track.length - 1].t) return track[track.length - 1];
  for (let i = 0; i < track.length - 1; i++) {
    const a = track[i];
    const b = track[i + 1];
    if (t >= a.t && t <= b.t) {
      const alpha = (t - a.t) / (b.t - a.t || 1);
      return {
        idx: Math.round(a.idx + alpha * (b.idx - a.idx)),
        t,
        x: a.x + (b.x - a.x) * alpha,
        y: a.y + (b.y - a.y) * alpha,
        z: a.z + (b.z - a.z) * alpha,
      };
    }
  }
  return null;
}

class KalmanFilter2D {
  constructor({ dt = 1 / 12, q = 1, r = 25 } = {}) {
    this.dt = dt;
    const dt2 = dt * dt / 2;
    this.A = [
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];
    this.B = [
      [dt2],
      [dt2],
      [dt],
      [dt],
    ];
    this.H = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ];
    this.Q = matrixScale(
      [
        [dt2 * dt2, 0, dt2 * dt, 0],
        [0, dt2 * dt2, 0, dt2 * dt],
        [dt2 * dt, 0, dt * dt, 0],
        [0, dt2 * dt, 0, dt * dt],
      ],
      q,
    );
    this.R = matrixScale(
      [
        [1, 0],
        [0, 1],
      ],
      r,
    );
    this.P = matrixScale(identity(4), 1000);
    this.x = [[0], [0], [0], [0]];
    this.initialized = false;
  }
  predict() {
    this.x = multiply(this.A, this.x);
    this.P = addMatrix(multiply(multiply(this.A, this.P), transpose(this.A)), this.Q);
  }
  update(z) {
    if (!this.initialized) {
      this.x = [[z[0]], [z[1]], [0], [0]];
      this.initialized = true;
    }
    const y = subtractVectors([[z[0]], [z[1]]], multiply(this.H, this.x));
    const S = addMatrix(multiply(multiply(this.H, this.P), transpose(this.H)), this.R);
    const K = multiply(multiply(this.P, transpose(this.H)), invertMatrix(S));
    this.x = addMatrix(this.x, multiply(K, y));
    const I = identity(4);
    this.P = multiply(subtractMatrices(I, multiply(K, this.H)), this.P);
  }
  step(z) {
    this.predict();
    this.update(z);
    return { x: this.x[0][0], y: this.x[1][0] };
  }
}

function identity(n) {
  const I = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

function matrixScale(m, s) {
  return m.map((row) => row.map((v) => v * s));
}

function multiply(a, b) {
  const rows = a.length;
  const cols = b[0].length;
  const inner = b.length;
  const out = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const val = a[i][k];
      for (let j = 0; j < cols; j++) {
        out[i][j] += val * b[k][j];
      }
    }
  }
  return out;
}

function addMatrix(a, b) {
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

function subtractMatrices(a, b) {
  return a.map((row, i) => row.map((val, j) => val - b[i][j]));
}

function subtractVectors(a, b) {
  return a.map((row, i) => [row[0] - b[i][0]]);
}

function invertMatrix(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...identity(n)[i]]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    let max = Math.abs(A[i][i]);
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(A[r][i]) > max) {
        max = Math.abs(A[r][i]);
        pivot = r;
      }
    }
    if (max < 1e-12) throw new Error('Matrix not invertible');
    if (pivot !== i) {
      const temp = A[i];
      A[i] = A[pivot];
      A[pivot] = temp;
    }
    const pivotVal = A[i][i];
    for (let j = 0; j < 2 * n; j++) A[i][j] /= pivotVal;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = A[r][i];
      for (let j = 0; j < 2 * n; j++) {
        A[r][j] -= factor * A[i][j];
      }
    }
  }
  return A.map((row) => row.slice(n));
}

function buildCourtBackground(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#1b2032';
  ctx.fillRect(0, 0, width, height);
  ctx.save();
  const padding = 40;
  const w = width - padding * 2;
  const h = height - padding * 2;
  ctx.translate(padding, padding);
  ctx.strokeStyle = 'rgba(200, 220, 255, 0.85)';
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, w, h);
  const netY = h / 2;
  ctx.beginPath();
  ctx.moveTo(0, netY);
  ctx.lineTo(w, netY);
  ctx.strokeStyle = 'rgba(255, 180, 180, 0.85)';
  ctx.stroke();
  const attackOffset = (overlayConfig.courtLengthM / 6) / overlayMetersLength * h;
  ctx.strokeStyle = 'rgba(200, 220, 255, 0.6)';
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(0, netY - attackOffset);
  ctx.lineTo(w, netY - attackOffset);
  ctx.moveTo(0, netY + attackOffset);
  ctx.lineTo(w, netY + attackOffset);
  ctx.stroke();
  ctx.restore();
}

function overlayToCourtPixels(pt, width, height) {
  const padding = 40;
  const w = width - padding * 2;
  const h = height - padding * 2;
  return {
    x: padding + (pt.x / 1000) * w,
    y: padding + (pt.y / 1000) * h,
  };
}

function drawCourt() {
  resizeCanvasToDisplaySize(courtCanvas);
  const width = courtCanvas.width;
  const height = courtCanvas.height;
  courtCtx.save();
  courtCtx.scale(DPR, DPR);
  buildCourtBackground(courtCtx, width / DPR, height / DPR);
  const t = video.currentTime;
  telemetryTime.textContent = t.toFixed(3);
  if (state.tracks.ball) {
    const meas = interpolateTrack(state.tracks.ball.track, t);
    const filt = interpolateTrack(state.tracks.ballFiltered?.track ?? [], t);
    telemetryBall.textContent = meas ? `${meas.x.toFixed(1)},${meas.y.toFixed(1)}` : '--';
    telemetryBallFilt.textContent = filt ? `${filt.x.toFixed(1)},${filt.y.toFixed(1)}` : '--';
    telemetryBallZ.textContent = filt ? `${filt.z.toFixed(2)}m` : '--';
    if (filt) {
      const px = overlayToCourtPixels(filt, width / DPR, height / DPR);
      const radius = 10 + filt.z * 2;
      courtCtx.fillStyle = 'rgba(0, 0, 0, 0.35)';
      courtCtx.beginPath();
      courtCtx.ellipse(px.x, px.y + radius * 0.6, radius * 1.4, radius * 0.4, 0, 0, Math.PI * 2);
      courtCtx.fill();
      courtCtx.fillStyle = 'rgba(255, 210, 120, 0.95)';
      courtCtx.beginPath();
      courtCtx.arc(px.x, px.y - filt.z * 6, radius, 0, Math.PI * 2);
      courtCtx.fill();
    }
  }
  if (state.tracks.humans) {
    const sample = findNearestFrame(state.tracks.humans.humans, t);
    if (sample) {
      sample.list.forEach((p) => {
        const px = overlayToCourtPixels(p, width / DPR, height / DPR);
        courtCtx.fillStyle = 'rgba(120, 180, 255, 0.9)';
        courtCtx.beginPath();
        courtCtx.arc(px.x, px.y, 8, 0, Math.PI * 2);
        courtCtx.fill();
        if (p.id) {
          courtCtx.fillStyle = 'rgba(200, 220, 255, 0.85)';
          courtCtx.fillText(p.id, px.x + 10, px.y - 10);
        }
      });
    }
  }
  courtCtx.restore();
}

function findNearestFrame(track, t) {
  if (!track || !track.length) return null;
  let best = track[0];
  let bestDiff = Math.abs(t - best.t);
  for (const entry of track) {
    const diff = Math.abs(t - entry.t);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = entry;
    }
  }
  return best;
}

async function callLLM({ baseUrl, model, systemPrompt, payload }) {
  serviceStatus.textContent = 'LLM: contacting';
  serviceStatus.classList.remove('pill-ok');
  serviceStatus.classList.add('pill-warn');
  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: payload },
  ];
  try {
    const chatUrl = `${baseUrl.replace(/\/$/, '')}/chat/completions`;
    const body = {
      model,
      messages,
      response_format: { type: 'json_object' },
    };
    const resp = await fetch(chatUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (resp.ok) {
      const data = await resp.json();
      const content = data.choices?.[0]?.message?.content;
      if (content) {
        serviceStatus.textContent = 'LLM: ok';
        serviceStatus.classList.add('pill-ok');
        return JSON.parse(content);
      }
    }
  } catch (err) {
    console.warn('LLM chat/completions failed', err);
  }
  try {
    const respUrl = `${baseUrl.replace(/\/$/, '')}/responses`;
    const body = {
      model,
      input: messages.map((m) => ({ role: m.role, content: m.content })),
      response_format: { type: 'json_object' },
    };
    const resp = await fetch(respUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (resp.ok) {
      const data = await resp.json();
      const content = data.output_text ?? data.response?.[0]?.content ?? data;
      if (typeof content === 'string') {
        serviceStatus.textContent = 'LLM: ok';
        serviceStatus.classList.add('pill-ok');
        return JSON.parse(content);
      }
    }
  } catch (err) {
    console.warn('LLM responses failed', err);
  }
  throw new Error('LLM unavailable');
}

async function sampleFrames(params) {
  const { fps, resizeW, resizeH, jpegQ, start, duration } = params;
  const frames = [];
  if (!state.videoFile) return frames;
  const tempVideo = document.createElement('video');
  tempVideo.src = URL.createObjectURL(state.videoFile);
  tempVideo.muted = true;
  await tempVideo.play().catch(() => {});
  tempVideo.pause();
  const off = document.createElement('canvas');
  off.width = resizeW;
  off.height = resizeH;
  const ctx = off.getContext('2d');
  const steps = Math.max(1, Math.floor(fps * duration));
  for (let i = 0; i <= steps; i++) {
    if (state.analysisAbort) break;
    const t = start + (i / fps);
    tempVideo.currentTime = Math.min(t, tempVideo.duration || t);
    await once(tempVideo, 'seeked');
    ctx.drawImage(tempVideo, 0, 0, resizeW, resizeH);
    const dataUrl = off.toDataURL('image/jpeg', jpegQ);
    frames.push({ idx: i, t, dataUrl });
  }
  tempVideo.remove();
  return frames;
}

function once(target, eventName) {
  return new Promise((resolve) => target.addEventListener(eventName, resolve, { once: true }));
}

async function runAnalysis(type) {
  if (!state.calibration.H) {
    alert('Calibrate the court and confirm H before running analysis.');
    return;
  }
  if (type === 'ball') generateBallBtn.disabled = true;
  if (type === 'humans') generatePlayersBtn.disabled = true;
  const baseUrl = llmBaseInput.value.trim();
  const model = llmModelInput.value.trim();
  const fps = parseFloat(frameFpsInput.value) || 6;
  const resizeW = parseInt(frameResizeWInput.value, 10) || 512;
  const resizeH = parseInt(frameResizeHInput.value, 10) || 288;
  const jpegQ = parseFloat(frameJpegQInput.value) || 0.7;
  const start = parseFloat(frameStartInput.value) || 0;
  const duration = parseFloat(frameDurationInput.value) || 5;
  state.analysisAbort = false;
  stopAnalysisBtn.disabled = false;
  try {
    const frames = await sampleFrames({ fps, resizeW, resizeH, jpegQ, start, duration });
    const results = [];
    for (const frame of frames) {
      if (state.analysisAbort) break;
      const hints = buildCalibrationHints(frame);
      let result;
      if (baseUrl && model) {
        try {
          result = await callLLM({
            baseUrl,
            model,
            systemPrompt: type === 'ball' ? BALL_PROMPT : HUMAN_PROMPT,
            payload: JSON.stringify(hints),
          });
        } catch (err) {
          console.warn('LLM unavailable, falling back to heuristic', err);
          result = heuristicResult(type, frame);
        }
      } else {
        result = heuristicResult(type, frame);
      }
      if (result) results.push(result);
    }
    if (type === 'ball') {
      const meta = { overlay_frame: 'EXTENDED_COURT_+20FT', units: { x: '0..1000', y: '0..1000', z: 'm' }, fps };
      const track = applyKalman(results, fps);
      state.tracks.ball = { meta, track: results };
      state.tracks.ballFiltered = { meta, track };
      updateDownloadLinks();
    } else {
      const meta = { overlay_frame: 'EXTENDED_COURT_+20FT', units: { x: '0..1000', y: '0..1000' }, classes: ['player'] };
      state.tracks.humans = { meta, humans: results };
      updateDownloadLinks();
    }
  } finally {
    stopAnalysisBtn.disabled = true;
    state.analysisAbort = false;
    if (type === 'ball') generateBallBtn.disabled = false;
    if (type === 'humans') generatePlayersBtn.disabled = false;
  }
}

function buildCalibrationHints(frame) {
  const { H, netPosts, netHeight, ball, player } = state.calibration;
  return {
    idx: frame.idx,
    t: frame.t,
    frame: frame.dataUrl,
    homography: true,
    H,
    net_posts: netPosts,
    net_height_m: netHeight,
    ball_anchor_img: ball?.center ?? null,
    ball_ground_img: ball?.ground ?? null,
    ball_ref_radius_px: ball?.radius ?? null,
    player_box_img: player?.box ?? null,
    player_foot_img: player?.foot ?? null,
  };
}

function heuristicResult(type, frame) {
  if (type === 'ball') {
    const { ball } = state.calibration;
    if (!ball || !state.calibration.H) return null;
    const overlay = applyHomography(state.calibration.H, ball.center);
    const radiusRef = ball.radius || 25;
    const z = Math.min(10, Math.max(0, (radiusRef / 30) * 2));
    return { idx: frame.idx, t: frame.t, x: overlay.x, y: overlay.y, z };
  }
  if (type === 'humans') {
    const { player } = state.calibration;
    if (!player || !state.calibration.H) return null;
    const overlay = applyHomography(state.calibration.H, player.foot ?? { x: player.box.x + player.box.w / 2, y: player.box.y + player.box.h });
    return { idx: frame.idx, t: frame.t, list: [{ id: 'P1', x: overlay.x, y: overlay.y, conf: 0.5 }] };
  }
  return null;
}

function applyKalman(results, fps) {
  if (!results.length) return results;
  const filter = new KalmanFilter2D({ dt: 1 / fps, q: 0.3, r: 16 });
  const smoothed = [];
  results.forEach((sample) => {
    const { x, y } = filter.step([sample.x, sample.y]);
    smoothed.push({ ...sample, x, y });
  });
  return smoothed;
}

function updateDownloadLinks() {
  if (state.tracks.ball) {
    const blob = new Blob([JSON.stringify(state.tracks.ball, null, 2)], { type: 'application/json' });
    downloadBallLink.href = URL.createObjectURL(blob);
  }
  if (state.tracks.humans) {
    const blob = new Blob([JSON.stringify(state.tracks.humans, null, 2)], { type: 'application/json' });
    downloadHumansLink.href = URL.createObjectURL(blob);
  }
  drawCourt();
  updateOverlay();
}

function handleBallJsonFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      state.tracks.ball = data;
      if (data.track) {
        const fps = data.meta?.fps || 12;
        state.tracks.ballFiltered = { meta: data.meta, track: applyKalman(data.track, fps) };
      }
      updateDownloadLinks();
    } catch (err) {
      alert('Invalid ball JSON');
    }
  };
  reader.readAsText(file);
}

function handleHumansJsonFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      state.tracks.humans = data;
      updateDownloadLinks();
    } catch (err) {
      alert('Invalid humans JSON');
    }
  };
  reader.readAsText(file);
}

function setupDragDrop() {
  courtDropTarget.addEventListener('dragover', (event) => {
    event.preventDefault();
    courtDropTarget.classList.add('drag-hover');
  });
  courtDropTarget.addEventListener('dragleave', () => {
    courtDropTarget.classList.remove('drag-hover');
  });
  courtDropTarget.addEventListener('drop', (event) => {
    event.preventDefault();
    courtDropTarget.classList.remove('drag-hover');
    const files = event.dataTransfer.files;
    for (const file of files) {
      if (file.name.includes('ball')) handleBallJsonFile(file);
      else handleHumansJsonFile(file);
    }
  });
}

function setupSplitters() {
  let dragging = null;
  splitterVertical.addEventListener('pointerdown', (event) => {
    dragging = { type: 'vertical', startX: event.clientX, initialWidth: panelLeft.getBoundingClientRect().width };
    splitterVertical.setPointerCapture(event.pointerId);
  });
  splitterHorizontal.addEventListener('pointerdown', (event) => {
    dragging = { type: 'horizontal', startY: event.clientY, initialHeight: playerSection.getBoundingClientRect().height };
    splitterHorizontal.setPointerCapture(event.pointerId);
  });
  function onPointerMove(event) {
    if (!dragging) return;
    if (dragging.type === 'vertical') {
      const delta = event.clientX - dragging.startX;
      const totalWidth = panelLeft.parentElement.getBoundingClientRect().width;
      const newWidth = Math.min(totalWidth - 200, Math.max(200, dragging.initialWidth + delta));
      panelLeft.style.width = `${(newWidth / totalWidth) * 100}%`;
    } else if (dragging.type === 'horizontal') {
      const delta = event.clientY - dragging.startY;
      const totalHeight = panelLeft.getBoundingClientRect().height;
      const newHeight = Math.min(totalHeight - 200, Math.max(200, dragging.initialHeight + delta));
      playerSection.style.flex = 'none';
      playerSection.style.height = `${newHeight}px`;
      calibrationSection.style.flex = '1 1 auto';
    }
    drawCalibration();
    drawCourt();
  }
  function onPointerUp(event) {
    if (!dragging) return;
    if (dragging.type === 'vertical') splitterVertical.releasePointerCapture(event.pointerId);
    if (dragging.type === 'horizontal') splitterHorizontal.releasePointerCapture(event.pointerId);
    dragging = null;
  }
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
}

function setupButtons() {
  videoInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) loadVideo(file);
  });
  playToggle.addEventListener('click', togglePlayback);
  snapFrameBtn.addEventListener('click', grabFrame);
  resetCalibrationBtn.addEventListener('click', resetCalibration);
  confirmHBtn.addEventListener('click', computeHomography);
  pickBallBtn.addEventListener('click', () => {
    state.mode = 'ball-center';
  });
  pickNetBtn.addEventListener('click', () => {
    state.mode = 'net-first';
  });
  pickPlayerBtn.addEventListener('click', () => {
    state.mode = 'player-first';
  });
  positionBallBtn.addEventListener('click', () => {
    if (!state.calibration.ball || !state.calibration.H) return;
    const overlay = applyHomography(state.calibration.H, state.calibration.ball.center);
    const zInput = prompt('Ball height (meters)', `${state.calibration.ball.overlay?.z ?? 0.5}`);
    const z = Number.parseFloat(zInput) || 0;
    state.calibration.ball.overlay = { x: overlay.x, y: overlay.y, z };
  });
  positionPlayerBtn.addEventListener('click', () => {
    if (!state.calibration.player || !state.calibration.H) return;
    const overlay = applyHomography(state.calibration.H, state.calibration.player.foot ?? { x: state.calibration.player.box.x + state.calibration.player.box.w / 2, y: state.calibration.player.box.y + state.calibration.player.box.h });
    state.calibration.player.overlay = overlay;
  });
  generateBallBtn.addEventListener('click', () => runAnalysis('ball'));
  generatePlayersBtn.addEventListener('click', () => runAnalysis('humans'));
  stopAnalysisBtn.addEventListener('click', () => {
    state.analysisAbort = true;
  });
  ballJsonInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleBallJsonFile(file);
  });
  humansJsonInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleHumansJsonFile(file);
  });
  fitModeSelect.addEventListener('change', drawCalibration);
}

function init() {
  setupButtons();
  setupSplitters();
  setupDragDrop();
  calibrationCanvas.addEventListener('pointerdown', handleCalibrationCanvasPointerDown);
  calibrationCanvas.addEventListener('pointermove', handleCalibrationCanvasPointerMove);
  calibrationCanvas.addEventListener('pointerup', handleCalibrationCanvasPointerUp);
  window.addEventListener('keydown', handleKeydown);
  video.addEventListener('timeupdate', () => {
    updateTimecode();
    updateOverlay();
    drawCourt();
  });
  window.addEventListener('resize', () => {
    drawCalibration();
    drawCourt();
    updateOverlay();
  });
  drawCalibration();
  drawCourt();
  updateOverlay();
}

init();

export { applyHomography, applyHomographyInverse, normalizedDLT };
