/* ==========================================================================
   Gaussian Density Explorer
   An independent (zero-correlation) bivariate normal f(x,y), rendered as a
   density heatmap, with sample points, a reference point c, a plain
   epsilon-ball, highlighted k-nearest-neighbors, and a "closeness-ball":
   the radius r such that P(Ball(c,r)) = (k+1/2)/N, found by numerically
   integrating the density in polar coordinates around c and bisecting on r.
   ========================================================================== */

const CANVAS_RES = 640;     // internal pixel resolution of the main canvas
const HEATMAP_RES = 160;    // low-res offscreen heatmap, upscaled for speed
const MAX_N = 2000;

const state = {
  mx: 0, my: 0,                 // fixed (no longer sliders)
  sigmaX: 1.2, sigmaY: 1.2,      // fixed (no longer sliders)
  N: 300,
  k: 20,
  cx: -1, cy: -1,
  eps: 1.0,
  vis: {
    heatmap: true, points: true, refPoint: true,
    epsilonBall: false, epsNeighborhood: false,
    closenessBall: false, knn: false,
    origin: true, grid: false,
  },
};

// Epsilon ball/neighborhood color: orange, chosen to sit outside the
// viridis heatmap palette (purple -> blue -> teal -> green -> yellow) AND
// to read clearly apart from the closeness-ball/k-NN magenta below.
const EPS_COLOR = '#ff8c2b';
const POINT_FILL = '#888c96'; // solid grey, no outline

let heatmapDirty = true;

/* ==========================================================================
   Gaussian math
   ========================================================================== */

function pdf(x, y) {
  const { mx, my, sigmaX, sigmaY } = state;
  const dx = (x - mx) / sigmaX, dy = (y - my) / sigmaY;
  return Math.exp(-0.5 * (dx * dx + dy * dy)) / (2 * Math.PI * sigmaX * sigmaY);
}

// Probability mass of a disk of radius r centered at (cx,cy), via polar
// quadrature: integral over theta in [0,2pi), rho in [0,r] of
// pdf(cx+rho*cos(theta), cy+rho*sin(theta)) * rho  d(rho) d(theta).
// No closed form exists in general since c need not be the mean and
// sigmaX may differ from sigmaY (so density contours are elliptical).
function ballProbability(cx, cy, r) {
  if (r <= 0) return 0;
  const RADIAL_STEPS = 48, ANGULAR_STEPS = 64;
  const dRho = r / RADIAL_STEPS;
  const dTheta = (2 * Math.PI) / ANGULAR_STEPS;
  let total = 0;
  for (let i = 0; i < RADIAL_STEPS; i++) {
    // midpoint rule in rho
    const rho = (i + 0.5) * dRho;
    let angularSum = 0;
    for (let j = 0; j < ANGULAR_STEPS; j++) {
      const theta = j * dTheta;
      angularSum += pdf(cx + rho * Math.cos(theta), cy + rho * Math.sin(theta));
    }
    total += angularSum * rho * dRho * dTheta;
  }
  return total;
}

// Find r such that ballProbability(cx,cy,r) = target, via bracket + bisection.
function findClosenessRadius(cx, cy, target) {
  if (target <= 0) return 0;
  let rHi = Math.max(state.sigmaX, state.sigmaY) * 0.5;
  const maxR = 40;
  while (ballProbability(cx, cy, rHi) < target && rHi < maxR) rHi *= 1.6;
  let lo = 0, hi = Math.min(rHi, maxR);
  for (let i = 0; i < 28; i++) {
    const mid = (lo + hi) / 2;
    if (ballProbability(cx, cy, mid) < target) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

/* ==========================================================================
   Sampling. Raw standard-normal draws are cached and only regenerated on
   Resample or when N changes; moving mean/sigma sliders just re-transforms
   the same underlying draws, so the point cloud reflows smoothly instead of
   jumping to a new random set on every tweak.
   ========================================================================== */

function randNormal() {
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

let rawZ = [];
function regenerateSamples(n) {
  rawZ = [];
  for (let i = 0; i < n; i++) rawZ.push({ zx: randNormal(), zy: randNormal() });
}
regenerateSamples(state.N);

function samplePositions() {
  return rawZ.map(({ zx, zy }) => ({
    x: state.mx + state.sigmaX * zx,
    y: state.my + state.sigmaY * zy,
  }));
}

/* ==========================================================================
   View mapping: a fixed (non-interactive) square view that auto-fits to the
   current mean/spread, so most samples stay on screen. It only changes when
   mean/sigma sliders change -- there is no pan or zoom.
   ========================================================================== */

function currentView() {
  const halfWidth = Math.min(10, Math.max(2.5, 4 * Math.max(state.sigmaX, state.sigmaY)));
  return { cx: state.mx, cy: state.my, halfWidth };
}

function worldToScreen(view, x, y) {
  const s = CANVAS_RES / 2 / view.halfWidth;
  return {
    x: CANVAS_RES / 2 + (x - view.cx) * s,
    y: CANVAS_RES / 2 - (y - view.cy) * s,
  };
}

function screenToWorld(view, sx, sy) {
  const s = CANVAS_RES / 2 / view.halfWidth;
  return {
    x: view.cx + (sx - CANVAS_RES / 2) / s,
    y: view.cy - (sy - CANVAS_RES / 2) / s,
  };
}

/* ==========================================================================
   Heatmap colormap: a viridis-like gradient. A gamma exponent is applied to
   the normalized density purely to spread the color range across more of
   the view (raw Gaussian density falls off very fast) -- this only affects
   the picture, never the probability integration above.
   ========================================================================== */

const COLOR_STOPS = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];
const GAMMA = 0.45;

function colorForDensity(t) {
  const g = Math.pow(Math.min(1, Math.max(0, t)), GAMMA);
  const n = COLOR_STOPS.length - 1;
  const pos = g * n;
  const i0 = Math.min(n - 1, Math.floor(pos));
  const frac = pos - i0;
  const a = COLOR_STOPS[i0], b = COLOR_STOPS[i0 + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

const heatmapCanvas = document.createElement('canvas');
heatmapCanvas.width = HEATMAP_RES;
heatmapCanvas.height = HEATMAP_RES;
const heatmapCtx = heatmapCanvas.getContext('2d');

function redrawHeatmap(view) {
  const img = heatmapCtx.createImageData(HEATMAP_RES, HEATMAP_RES);
  const maxPdf = 1 / (2 * Math.PI * state.sigmaX * state.sigmaY);
  for (let py = 0; py < HEATMAP_RES; py++) {
    const sy = (py / HEATMAP_RES) * CANVAS_RES;
    for (let px = 0; px < HEATMAP_RES; px++) {
      const sx = (px / HEATMAP_RES) * CANVAS_RES;
      const w = screenToWorld(view, sx, sy);
      const t = pdf(w.x, w.y) / maxPdf;
      const [r, g, b] = colorForDensity(t);
      const idx = (py * HEATMAP_RES + px) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
  }
  heatmapCtx.putImageData(img, 0, 0);
}

/* ==========================================================================
   Main canvas rendering
   ========================================================================== */

const plot = document.getElementById('plot');
plot.width = CANVAS_RES; plot.height = CANVAS_RES;
const ctx = plot.getContext('2d');

function render() {
  const view = currentView();
  if (heatmapDirty) { redrawHeatmap(view); heatmapDirty = false; }

  ctx.clearRect(0, 0, CANVAS_RES, CANVAS_RES);
  ctx.fillStyle = '#0d0e12';
  ctx.fillRect(0, 0, CANVAS_RES, CANVAS_RES);

  if (state.vis.heatmap) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heatmapCanvas, 0, 0, CANVAS_RES, CANVAS_RES);
  }

  if (state.vis.grid) drawGrid(view);
  if (state.vis.origin) drawOrigin(view);

  const samples = samplePositions();

  // k nearest neighbors to c, by Euclidean distance
  const withDist = samples.map((p) => ({
    ...p, d: Math.hypot(p.x - state.cx, p.y - state.cy),
  }));
  withDist.sort((a, b) => a.d - b.d);
  const kClamped = Math.min(state.k, withDist.length);
  const knnSet = new Set(withDist.slice(0, kClamped));
  const epsSet = new Set(state.vis.epsNeighborhood ? withDist.filter((p) => p.d <= state.eps) : []);

  if (state.vis.points) {
    for (const p of withDist) {
      if (state.vis.knn && knnSet.has(p)) continue;        // drawn separately, purple
      if (state.vis.epsNeighborhood && epsSet.has(p)) continue; // drawn separately, yellow
      const s = worldToScreen(view, p.x, p.y);
      ctx.beginPath();
      ctx.arc(s.x, s.y, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = POINT_FILL;
      ctx.fill();
    }
  }

  if (state.vis.epsNeighborhood) {
    for (const p of epsSet) {
      const s = worldToScreen(view, p.x, p.y);
      ctx.beginPath();
      ctx.arc(s.x, s.y, 3.6, 0, Math.PI * 2);
      ctx.fillStyle = EPS_COLOR;
      ctx.fill();
    }
  }

  if (state.vis.knn) {
    for (const p of withDist.slice(0, kClamped)) {
      const s = worldToScreen(view, p.x, p.y);
      ctx.beginPath();
      ctx.arc(s.x, s.y, 3.6, 0, Math.PI * 2);
      ctx.fillStyle = '#b980ff';
      ctx.fill();
    }
  }

  let closenessRadius = null;
  const target = (state.k + 0.5) / Math.max(1, state.N);
  if (state.vis.closenessBall) {
    closenessRadius = findClosenessRadius(state.cx, state.cy, target);
    drawCircle(view, state.cx, state.cy, closenessRadius, '#b980ff', 2);
  }

  if (state.vis.epsilonBall) {
    drawCircle(view, state.cx, state.cy, state.eps, EPS_COLOR, 2);
  }

  if (state.vis.refPoint) {
    const s = worldToScreen(view, state.cx, state.cy);
    ctx.beginPath();
    ctx.arc(s.x, s.y, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = '#ff4d4d';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#1a1a1a';
    ctx.stroke();
  }

  updateReadout(withDist, kClamped, target, closenessRadius);
}

function drawGrid(view) {
  ctx.strokeStyle = '#22252e';
  ctx.lineWidth = 1;
  const step = niceStep(view.halfWidth / 4);
  const startX = Math.ceil((view.cx - view.halfWidth) / step) * step;
  for (let x = startX; x <= view.cx + view.halfWidth; x += step) {
    const s1 = worldToScreen(view, x, view.cy - view.halfWidth);
    const s2 = worldToScreen(view, x, view.cy + view.halfWidth);
    ctx.beginPath(); ctx.moveTo(s1.x, s1.y); ctx.lineTo(s2.x, s2.y); ctx.stroke();
  }
  const startY = Math.ceil((view.cy - view.halfWidth) / step) * step;
  for (let y = startY; y <= view.cy + view.halfWidth; y += step) {
    const s1 = worldToScreen(view, view.cx - view.halfWidth, y);
    const s2 = worldToScreen(view, view.cx + view.halfWidth, y);
    ctx.beginPath(); ctx.moveTo(s1.x, s1.y); ctx.lineTo(s2.x, s2.y); ctx.stroke();
  }
}

function niceStep(approx) {
  const pow10 = Math.pow(10, Math.floor(Math.log10(approx)));
  const frac = approx / pow10;
  const niceFrac = frac < 1.5 ? 1 : frac < 3.5 ? 2 : frac < 7.5 ? 5 : 10;
  return niceFrac * pow10;
}

function drawOrigin(view) {
  const s = worldToScreen(view, 0, 0);
  const arm = 9;
  ctx.strokeStyle = '#0a0a0a';
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.moveTo(s.x - arm, s.y); ctx.lineTo(s.x + arm, s.y);
  ctx.moveTo(s.x, s.y - arm); ctx.lineTo(s.x, s.y + arm);
  ctx.stroke();
  ctx.strokeStyle = '#e7e9ee';
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  ctx.moveTo(s.x - arm, s.y); ctx.lineTo(s.x + arm, s.y);
  ctx.moveTo(s.x, s.y - arm); ctx.lineTo(s.x, s.y + arm);
  ctx.stroke();
}

function drawCircle(view, cx, cy, r, color, width) {
  const scale = CANVAS_RES / 2 / view.halfWidth;
  const s = worldToScreen(view, cx, cy);
  ctx.beginPath();
  ctx.arc(s.x, s.y, r * scale, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function updateReadout(withDist, kClamped, target, closenessRadius) {
  const fC = pdf(state.cx, state.cy);
  const knnDist = withDist.length ? withDist[kClamped - 1].d : NaN;
  const lines = [
    `c = (${state.cx.toFixed(2)}, ${state.cy.toFixed(2)})`,
    `density f(c) = ${fC.toFixed(4)}`,
    `k-th NN distance ≈ ${knnDist.toFixed(3)}`,
    `target P = (k+½)/N = ${target.toFixed(4)}`,
  ];
  if (closenessRadius !== null) {
    lines.push(`closeness radius r* ≈ ${closenessRadius.toFixed(3)}`);
    lines.push(`achieved P(ball) ≈ ${ballProbability(state.cx, state.cy, closenessRadius).toFixed(4)}`);
  }
  document.getElementById('readout').textContent = lines.join('\n');
}

/* ==========================================================================
   UI wiring
   ========================================================================== */

function bindSlider(id, stateKey, decimals, onChange) {
  const input = document.getElementById(id);
  const out = document.getElementById(id + '-val');
  const update = () => {
    const val = parseFloat(input.value);
    state[stateKey] = val;
    out.textContent = val.toFixed(decimals);
    onChange();
  };
  input.addEventListener('input', update);
  update();
}

bindSlider('cx', 'cx', 2, () => render());
bindSlider('cy', 'cy', 2, () => render());
bindSlider('eps', 'eps', 2, () => render());
bindSlider('k', 'k', 0, () => render());

bindSlider('N', 'N', 0, () => {
  regenerateSamples(Math.round(state.N));
  const kMax = Math.max(1, Math.min(100, state.N - 1));
  const kInput = document.getElementById('k');
  kInput.max = kMax;
  if (state.k > kMax) {
    state.k = kMax;
    kInput.value = kMax;
    document.getElementById('k-val').textContent = kMax.toFixed(0);
  }
  render();
});

document.getElementById('resample').addEventListener('click', () => {
  regenerateSamples(Math.round(state.N));
  render();
});

document.getElementById('reset-point').addEventListener('click', () => {
  state.cx = -1; state.cy = -1;
  document.getElementById('cx').value = -1;
  document.getElementById('cy').value = -1;
  document.getElementById('cx-val').textContent = '-1.00';
  document.getElementById('cy-val').textContent = '-1.00';
  render();
});

const visMap = {
  'vis-heatmap': 'heatmap', 'vis-points': 'points', 'vis-ref': 'refPoint',
  'vis-eps': 'epsilonBall', 'vis-eps-neigh': 'epsNeighborhood',
  'vis-closeness': 'closenessBall', 'vis-knn': 'knn',
  'vis-origin': 'origin', 'vis-grid': 'grid',
};
Object.entries(visMap).forEach(([id, key]) => {
  document.getElementById(id).addEventListener('change', (e) => {
    state.vis[key] = e.target.checked;
    render();
  });
});

// Click-to-place reference point
plot.addEventListener('pointerdown', (ev) => {
  const rect = plot.getBoundingClientRect();
  const sx = ((ev.clientX - rect.left) / rect.width) * CANVAS_RES;
  const sy = ((ev.clientY - rect.top) / rect.height) * CANVAS_RES;
  const w = screenToWorld(currentView(), sx, sy);
  state.cx = w.x; state.cy = w.y;
  document.getElementById('cx').value = w.x.toFixed(2);
  document.getElementById('cy').value = w.y.toFixed(2);
  document.getElementById('cx-val').textContent = w.x.toFixed(2);
  document.getElementById('cy-val').textContent = w.y.toFixed(2);
  render();
});

// Mobile panel toggle
const panel = document.getElementById('panel');
document.getElementById('panel-toggle').addEventListener('click', () => {
  panel.classList.toggle('open');
});

/* ==========================================================================
   Photo / video capture
   ========================================================================== */

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

document.getElementById('capture-photo').addEventListener('click', () => {
  plot.toBlob((blob) => {
    if (blob) downloadBlob(blob, `gaussian-${Date.now()}.png`);
  }, 'image/png');
});

const videoBtn = document.getElementById('capture-video');
const canCaptureVideo = !!(plot.captureStream && window.MediaRecorder);

if (!canCaptureVideo) {
  videoBtn.disabled = true;
  videoBtn.title = 'Video recording is not supported in this browser (works in Chrome/Firefox/Edge; limited on iOS Safari).';
  videoBtn.textContent = '⏺ Record Video (unsupported)';
} else {
  let mediaRecorder = null;
  let recordedChunks = [];

  videoBtn.addEventListener('click', () => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      const stream = plot.captureStream(30);
      let mimeType = 'video/webm;codecs=vp9';
      if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'video/webm';
      try {
        mediaRecorder = new MediaRecorder(stream, { mimeType });
      } catch (err) {
        alert('Could not start video recording: ' + err.message);
        return;
      }
      recordedChunks = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        downloadBlob(blob, `gaussian-${Date.now()}.webm`);
        videoBtn.textContent = '⏺ Record Video';
        videoBtn.classList.remove('recording');
      };
      mediaRecorder.start();
      videoBtn.textContent = '⏹ Stop Recording';
      videoBtn.classList.add('recording');
    } else {
      mediaRecorder.stop();
    }
  });
}

render();
