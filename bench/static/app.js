'use strict';

const $ = id => document.getElementById(id);
const canvas = $('cv'), photoContext = canvas.getContext('2d');
const guides = $('guides'), context = guides.getContext('2d');
let drawnPhoto = null;
const roomPresets = [];
const materialCatalogue = [];
const state = { image: null, photo: null, photoUrl: null, roomId: null, quad: [], autoPlane: false, boundary: [], boundaryClosed: false, exclusions: [], aiExclusions: [], tool: 'corners', selectedCorner: -1, selectedBoundary: -1, drag: null, painting: null, material: null, customFaces: [], resultUrl: null, compare: false, request: null, version: 0, photoLoad: 0, loadingPhoto: false, textureLoad: 0, loadingTextures: false, detectingFloor: false, busy: false };

function status(message, kind = 'ready') {
  $('status-text').textContent = message;
  $('status').dataset.state = kind;
}

function setBusy(busy) {
  state.busy = busy;
  $('render-overlay').hidden = !busy;
  $('render-overlay').querySelector('strong').textContent = state.detectingFloor ? 'Finding the floor' : 'Creating your preview';
  $('render-overlay').querySelector('span:last-child').textContent = state.detectingFloor ? 'Preparing an outline you can adjust.' : 'Keeping the room’s light and perspective.';
  $('go').disabled = busy || state.loadingPhoto || state.loadingTextures;
  $('go').setAttribute('aria-busy', String(busy || state.loadingPhoto || state.loadingTextures));
  $('render-label').textContent = busy ? state.detectingFloor ? 'Finding the floor…' : 'Creating preview…' : state.loadingPhoto ? 'Opening room…' : state.loadingTextures ? 'Opening textures…' : 'Create preview';
  $('detect-floor').disabled = !state.photo || busy || state.loadingPhoto || state.loadingTextures;
}

function invalidate(changed = true) {
  if (changed) window.dispatchEvent(new Event('projectchange'));
  state.version += 1;
  if (state.request) state.request.abort();
  state.request = null;
  state.detectingFloor = false;
  setBusy(false);
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  state.resultUrl = null;
  $('result').removeAttribute('src');
  $('download').removeAttribute('href');
  $('download').hidden = true;
  $('compare-view').disabled = true;
  setView(false);
  status(state.loadingPhoto ? 'Opening your photo…' : state.loadingTextures ? 'Opening your texture images…' : state.quad.length === 4 ? 'Ready to preview' : 'Select the four floor corners', state.loadingPhoto || state.loadingTextures ? 'busy' : 'ready');
}

function fitImage() {
  if (!state.image) return;
  const stage = $('stage'), style = getComputedStyle(stage);
  const width = stage.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const height = stage.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
  const points = !state.compare && state.tool === 'corners' ? state.quad : [];
  const left = Math.min(0, ...points.map(p => p[0])), top = Math.min(0, ...points.map(p => p[1]));
  const right = Math.max(canvas.width, ...points.map(p => p[0])), bottom = Math.max(canvas.height, ...points.map(p => p[1]));
  const scale = Math.min(width / (right - left), height / (bottom - top));
  const frame = $('image-frame');
  frame.style.width = `${Math.floor(canvas.width * scale)}px`;
  frame.style.height = `${Math.floor(canvas.height * scale)}px`;
  frame.style.left = `${parseFloat(style.paddingLeft) + (width - (right - left) * scale) / 2 - left * scale}px`;
  frame.style.top = `${parseFloat(style.paddingTop) + (height - (bottom - top) * scale) / 2 - top * scale}px`;
  draw();
}

function outline(points, fill, stroke, width, closed = true) {
  if (!points.length) return;
  context.beginPath();
  context.moveTo(...points[0]);
  points.slice(1).forEach(point => context.lineTo(...point));
  if (closed) context.closePath();
  context.fillStyle = fill;
  context.fill();
  context.strokeStyle = stroke;
  context.lineWidth = width;
  context.stroke();
}

function drawGrid(quad, scale) {
  const [a, b, c, d] = quad;
  const dx1 = b[0] - c[0], dx2 = d[0] - c[0], dx3 = a[0] - b[0] + c[0] - d[0];
  const dy1 = b[1] - c[1], dy2 = d[1] - c[1], dy3 = a[1] - b[1] + c[1] - d[1];
  const det = dx1 * dy2 - dx2 * dy1;
  const g = (dx3 * dy2 - dx2 * dy3) / det, h = (dx1 * dy3 - dx3 * dy1) / det;
  const project = (u, v) => [
    ((b[0] - a[0] + g * b[0]) * u + (d[0] - a[0] + h * d[0]) * v + a[0]) / (g * u + h * v + 1),
    ((b[1] - a[1] + g * b[1]) * u + (d[1] - a[1] + h * d[1]) * v + a[1]) / (g * u + h * v + 1),
  ];
  context.strokeStyle = '#fffaf079'; context.lineWidth = 0.8 * scale;
  for (let n = 1; n < 8; n++) {
    const t = n / 8;
    context.beginPath(); context.moveTo(...project(t, 0)); context.lineTo(...project(t, 1));
    context.moveTo(...project(0, t)); context.lineTo(...project(1, t)); context.stroke();
  }
}

function draw() {
  if (!state.image) return;
  if (drawnPhoto !== state.image) { photoContext.drawImage(state.image, 0, 0, canvas.width, canvas.height); drawnPhoto = state.image; }
  const stage = $('stage'), ratio = window.devicePixelRatio || 1;
  if (guides.width !== Math.round(stage.clientWidth * ratio)) guides.width = Math.round(stage.clientWidth * ratio);
  if (guides.height !== Math.round(stage.clientHeight * ratio)) guides.height = Math.round(stage.clientHeight * ratio);
  context.resetTransform(); context.clearRect(0, 0, guides.width, guides.height);
  guides.hidden = state.compare;
  if (state.compare) return;
  const bounds = canvas.getBoundingClientRect(), stageBounds = stage.getBoundingClientRect();
  const scale = canvas.width / Math.max(1, bounds.width);
  context.setTransform(ratio * bounds.width / canvas.width, 0, 0, ratio * bounds.height / canvas.height, ratio * (bounds.left - stageBounds.left), ratio * (bounds.top - stageBounds.top));
  if (state.tool === 'corners' && state.quad.length === 4) outline(state.quad, '#f9edd011', '#fff3d8', 1.5 * scale);
  if (state.tool === 'corners' && validFloor()) drawGrid(state.quad, scale);
  outline(state.boundary, state.boundaryClosed ? '#79927228' : 'transparent', '#52704b', 2 * scale, state.boundaryClosed);
  [...state.aiExclusions, ...state.exclusions].forEach(poly => outline(poly, '#ae604930', '#d69883', scale));
  if (state.painting) outline(state.painting, '#ae604947', '#ffcfb3', 1.5 * scale);
  const points = state.tool === 'outline' ? state.boundary : state.tool === 'corners' ? state.quad : [];
  const selected = state.tool === 'outline' ? state.selectedBoundary : state.selectedCorner;
  points.forEach(([x, y], index) => {
    context.beginPath();
    context.arc(x, y, (selected === index ? 11 : points.length > 12 ? 4 : 9) * scale, 0, Math.PI * 2);
    context.fillStyle = selected === index ? '#98623e' : '#fffaf0';
    context.fill();
    context.strokeStyle = '#98623e';
    context.lineWidth = 1.5 * scale;
    context.stroke();
    context.fillStyle = selected === index ? '#fff' : '#6c432b';
    context.font = `600 ${10 * scale}px system-ui`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    if (points.length <= 12 || selected === index || (index === 0 && !state.boundaryClosed)) context.fillText(String(index + 1), x, y);
  });
}

function updateEditor() {
  syncOutlinePlane();
  $('corner-count').textContent = state.tool === 'outline' ? `${state.boundary.length} points` : `${state.quad.length}/4`;
  $('step-photo').classList.toggle('complete', Boolean(state.image));
  $('step-floor').classList.toggle('complete', validFloor() && (!state.boundary.length || state.boundaryClosed));
  $('step-material').classList.toggle('complete', Boolean(state.material || state.customFaces.length));
  $('undo').disabled = state.tool === 'outline' ? !state.boundary.length : !state.quad.length && !state.exclusions.length && !state.aiExclusions.length;
  $('reset').disabled = !state.quad.length && !state.boundary.length && !state.exclusions.length && !state.aiExclusions.length;
  $('detect-floor').disabled = !state.photo || state.busy || state.loadingPhoto || state.loadingTextures;
  $('finish-outline').hidden = state.tool !== 'outline' || state.boundaryClosed;
  $('finish-outline').disabled = state.boundary.length < 3;
  if (state.compare) {
    $('hint-title').textContent = 'A new look, in the same room';
    $('hint-text').textContent = 'Move the slider to compare. Switch to Edit floor to refine the area.';
  } else if (state.tool === 'outline') {
    $('hint-title').textContent = state.boundaryClosed ? 'Your floor outline is ready' : 'Trace every bend in the floor';
    $('hint-text').textContent = state.boundaryClosed ? 'Drag any outline point to refine the edge. Use Perspective to align the material, then create a preview.' : 'Click around the visible floor, adding a point at each bend. Choose Finish outline, press Enter, or click the first point to close it.';
  } else if (state.tool === 'protect') {
    $('hint-title').textContent = 'Keep objects in the room';
    $('hint-text').textContent = 'Drag a closed outline around a chair, table or rug. The enclosed area keeps the original photo.';
  } else if (state.quad.length < 4) {
    const next = ['far-left', 'far-right', 'near-right', 'near-left'][state.quad.length];
    $('hint-title').textContent = `Set perspective corner ${state.quad.length + 1}: ${next}`;
    $('hint-text').textContent = 'Use Detect floor for an editable suggestion, or Floor outline to trace the area. Perspective aligns the material using four corners.';
  } else {
    $('hint-title').textContent = 'Your floor is ready';
    $('hint-text').textContent = 'Align the guide lines with the floor edges; corners may sit outside the photo. Use Floor outline to trace every bend in the visible floor.';
  }
  draw();
}

function setView(compare) {
  state.compare = compare && Boolean(state.resultUrl);
  $('result-layer').hidden = !state.compare;
  $('compare-controls').hidden = !state.compare;
  $('edit-view').setAttribute('aria-pressed', String(!state.compare));
  $('compare-view').setAttribute('aria-pressed', String(state.compare));
  if (state.drag === null && !state.painting) fitImage();
  updateEditor();
}

function setTool(tool) {
  state.tool = tool;
  setView(false);
  $('corners-tool').setAttribute('aria-pressed', String(tool === 'corners'));
  $('outline-tool').setAttribute('aria-pressed', String(tool === 'outline'));
  $('protect-tool').setAttribute('aria-pressed', String(tool === 'protect'));
  updateEditor();
}

function position(event) {
  const bounds = canvas.getBoundingClientRect();
  const point = [Math.round((event.clientX - bounds.left) * canvas.width / bounds.width), Math.round((event.clientY - bounds.top) * canvas.height / bounds.height)];
  return state.tool !== 'corners' ? [Math.max(0, Math.min(canvas.width - 1, point[0])), Math.max(0, Math.min(canvas.height - 1, point[1]))] : point;
}

guides.addEventListener('pointerdown', event => {
  if (!state.image || state.compare || event.button !== 0) return;
  const bounds = canvas.getBoundingClientRect();
  if (state.tool !== 'corners' && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) return;
  event.preventDefault();
  guides.focus({ preventScroll: true });
  guides.setPointerCapture(event.pointerId);
  const point = position(event);
  if (state.tool === 'protect') {
    invalidate();
    state.painting = [point];
  } else if (state.tool === 'outline') {
    const hitRadius = 18 * canvas.width / bounds.width;
    const distances = state.boundary.map(p => Math.hypot(p[0] - point[0], p[1] - point[1]));
    const nearest = distances.indexOf(Math.min(...distances));
    const hit = nearest >= 0 && distances[nearest] <= hitRadius ? nearest : -1;
    if (!state.boundaryClosed && hit === 0 && state.boundary.length >= 3) finishFloorOutline();
    else if (hit >= 0) { state.selectedBoundary = hit; state.drag = hit; }
    else if (!state.boundaryClosed) { invalidate(); state.boundary.push(point); state.selectedBoundary = state.boundary.length - 1; }
  } else {
    const hitRadius = 18 * canvas.width / canvas.getBoundingClientRect().width;
    const hit = state.quad.findIndex(p => Math.hypot(p[0] - point[0], p[1] - point[1]) <= hitRadius);
    if (hit >= 0) { state.selectedCorner = hit; state.drag = hit; }
    else if (state.quad.length < 4) { invalidate(); state.autoPlane = false; state.quad.push(point); state.selectedCorner = state.quad.length - 1; }
    if (state.quad.length === 4) status('Ready to preview');
  }
  updateEditor();
});

guides.addEventListener('pointermove', event => {
  if (state.painting) {
    const point = position(event), previous = state.painting.at(-1);
    if (Math.hypot(point[0] - previous[0], point[1] - previous[1]) > 3 * canvas.width / canvas.getBoundingClientRect().width) state.painting.push(point);
    draw();
  } else if (state.drag !== null) {
    invalidate();
    const points = state.tool === 'outline' ? state.boundary : state.quad;
    if (state.tool === 'corners') state.autoPlane = false;
    points[state.drag] = position(event);
    draw();
  }
});

function finishPointer(event) {
  if (state.painting) {
    if (state.painting.length >= 3 && event.type !== 'pointercancel') state.exclusions.push(state.painting);
    else status('Drag an outline around the object to protect it.', 'error');
    state.painting = null;
  }
  state.drag = null;
  fitImage();
  updateEditor();
}
guides.addEventListener('pointerup', finishPointer);
guides.addEventListener('pointercancel', finishPointer);
guides.addEventListener('keydown', event => {
  if (!state.compare && state.tool === 'outline' && event.key === 'Enter') { event.preventDefault(); finishFloorOutline(); return; }
  const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  const selected = state.tool === 'outline' ? state.selectedBoundary : state.selectedCorner;
  if (state.compare || state.tool === 'protect' || selected < 0 || !directions[event.key]) return;
  event.preventDefault();
  invalidate();
  const amount = event.shiftKey ? 10 : 1;
  const points = state.tool === 'outline' ? state.boundary : state.quad;
  if (state.tool === 'corners') state.autoPlane = false;
  const point = points[selected], direction = directions[event.key];
  const x = point[0] + direction[0] * amount, y = point[1] + direction[1] * amount;
  points[selected] = state.tool === 'outline' ? [Math.max(0, Math.min(canvas.width - 1, x)), Math.max(0, Math.min(canvas.height - 1, y))] : [x, y];
  fitImage();
  updateEditor();
});

$('corners-tool').onclick = () => setTool('corners');
$('outline-tool').onclick = () => setTool('outline');
$('protect-tool').onclick = () => setTool('protect');
function syncOutlinePlane() {
  if (!state.autoPlane || state.boundary.length < 3) return;
  const xs = state.boundary.map(p => p[0]), ys = state.boundary.map(p => p[1]);
  const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
  state.quad = [[left, top], [right, top], [right, bottom], [left, bottom]];
}
function finishFloorOutline() {
  if (state.boundary.length < 3 || state.boundaryClosed) return;
  invalidate(); state.boundaryClosed = true;
  if (!validFloor()) state.autoPlane = true;
  state.selectedBoundary = -1;
  fitImage(); updateEditor(); status('Floor outline ready. Adjust Perspective if needed, then create a preview.');
}
$('finish-outline').onclick = finishFloorOutline;
$('edit-view').onclick = () => setView(false);
$('compare-view').onclick = () => setView(true);
function setComparison(value) {
  value = Math.round(Math.max(0, Math.min(100, value)));
  $('compare-slider').value = value;
  $('image-frame').style.setProperty('--compare', `${value}%`);
}
$('compare-slider').oninput = event => setComparison(Number(event.target.value));
let comparisonPointer = null;
function compareAt(event) {
  const bounds = $('image-frame').getBoundingClientRect();
  setComparison((event.clientX - bounds.left) / bounds.width * 100);
}
$('result-layer').addEventListener('pointerdown', event => {
  if (!state.compare || event.button !== 0) return;
  event.preventDefault();
  comparisonPointer = event.pointerId;
  event.currentTarget.setPointerCapture(event.pointerId);
  compareAt(event);
});
$('result-layer').addEventListener('pointermove', event => {
  if (state.compare && event.pointerId === comparisonPointer) compareAt(event);
});
['pointerup', 'pointercancel'].forEach(type => $('result-layer').addEventListener(type, () => { comparisonPointer = null; }));
$('reset').onclick = () => { invalidate(); state.quad = []; state.autoPlane = false; state.boundary = []; state.boundaryClosed = false; state.exclusions = []; state.aiExclusions = []; state.selectedCorner = -1; state.selectedBoundary = -1; setTool('corners'); status('Select the four floor corners'); };
$('undo').onclick = () => { invalidate(); if (state.tool === 'outline') { if (!state.boundaryClosed) state.boundary.pop(); state.boundaryClosed = false; state.selectedBoundary = -1; } else if (state.exclusions.length) state.exclusions.pop(); else if (state.aiExclusions.length) state.aiExclusions.pop(); else { state.autoPlane = false; state.quad.pop(); state.selectedCorner = -1; } fitImage(); updateEditor(); status(state.tool === 'outline' ? 'Continue tracing, then finish the outline.' : state.quad.length === 4 ? 'Ready to preview' : 'Select the four floor corners'); };

async function detectFloor() {
  let controller = null, version = state.version;
  try {
    if (!state.photo || state.loadingPhoto || state.loadingTextures) throw new Error('Wait for your photo and textures to finish opening.');
    invalidate(); version = state.version;
    controller = new AbortController(); state.request = controller;
    state.detectingFloor = true; setBusy(true); status('Finding the floor…', 'busy');
    const form = new FormData(); form.append('photo', state.photo, 'room-photo');
    const response = await fetch('/detect-floor', { method: 'POST', body: form, signal: controller.signal });
    if (!response.ok) {
      let detail = ''; try { const body = await response.json(); if (typeof body.detail === 'string') detail = body.detail; } catch {}
      throw new Error(detail || `Floor detection failed (${response.status}). Try again or select Floor outline manually.`);
    }
    const result = await response.json();
    if (version !== state.version) return;
    invalidate();
    Object.assign(state, { boundary: result.floor_boundary, boundaryClosed: true, aiExclusions: result.exclude, selectedBoundary: -1 });
    let estimated = false;
    if (!validFloor()) {
      // The plane is a true-size floor rectangle, so its dimensions replace the inputs.
      Object.assign(state, { quad: result.plane.quad, autoPlane: false, selectedCorner: -1 });
      $('floorW').value = (result.plane.floor_mm[0] / 1000).toFixed(2);
      $('floorD').value = (result.plane.floor_mm[1] / 1000).toFixed(2);
      window.dispatchEvent(new Event('floorchange'));
      estimated = true;
    }
    setTool('outline');
    const perspective = !estimated ? 'Your Perspective corners are kept.' : result.plane.source === 'vertical-lines' ? 'Perspective was estimated from vertical edges; check it in Perspective.' : 'No vertical edges were found, so Perspective is a guess; drag its corners along lines on the floor.';
    status(`Floor detected · ${result.seconds.toFixed(1)}s. Review rugs and edges. ${perspective}`, 'success');
  } catch (error) { if (version === state.version && error.name !== 'AbortError') status(error.message || 'The floor could not be detected. Use Floor outline to select it manually.', 'error'); }
  finally { if (controller && state.request === controller) { state.request = null; state.detectingFloor = false; setBusy(false); } }
}
$('detect-floor').onclick = detectFloor;

async function normalizeUpload(blob) {
  if (!/\.(heic|heif)$/i.test(blob.name || '') && !/^image\/hei[cf](?:-sequence)?$/i.test(blob.type)) return blob;
  const form = new FormData();
  form.append('file', blob, blob.name || 'photo.heic');
  const response = await fetch('/image', { method: 'POST', body: form });
  if (!response.ok) throw new Error(`This HEIC/HEIF image could not be opened (${response.status}). Try another image.`);
  return response.blob();
}

async function loadPhoto(blob, name, preset, token) {
  blob = await normalizeUpload(blob);
  if (token !== state.photoLoad) return false;
  const url = URL.createObjectURL(blob), image = new Image();
  try { image.src = url; await image.decode(); }
  catch { URL.revokeObjectURL(url); throw new Error('This photo could not be opened. Try another JPG, PNG, WebP or HEIC image.'); }
  if (token !== state.photoLoad) { URL.revokeObjectURL(url); return false; }
  invalidate();
  if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
  Object.assign(state, { image, photo: blob, photoUrl: url, roomId: preset?.id || null, quad: [], autoPlane: false, boundary: [], boundaryClosed: false, exclusions: [], aiExclusions: [], selectedCorner: -1, selectedBoundary: -1 });
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  $('image-frame').hidden = false;
  $('empty-state').hidden = true;
  $('room-name').textContent = name;
  $('photo-kind').textContent = preset ? preset.generated ? 'Illustrative room' : 'Sample room' : 'Your photo';
  $('photo-credit').hidden = !preset;
  if (preset) {
    const scale = ([x, y]) => [x * canvas.width / preset.size[0], y * canvas.height / preset.size[1]];
    state.quad = preset.quad.map(scale);
    state.boundary = (preset.floor_boundary || []).map(scale);
    state.boundaryClosed = state.boundary.length >= 3;
    state.exclusions = preset.exclude.map(poly => poly.map(scale));
    [$('floorW').value, $('floorD').value] = preset.floor_m;
    $('photo-credit').textContent = preset.generated ? 'Illustrative room · Floor dimensions are estimates.' : 'Sample photo via Unsplash';
    if (preset.credit) $('photo-credit').href = preset.credit;
    else $('photo-credit').removeAttribute('href');
  }
  updateRooms();
  window.dispatchEvent(new Event('floorchange'));
  setTool(state.boundaryClosed ? 'outline' : 'corners');
  fitImage();
  status(preset ? 'Sample room ready. Create your first preview.' : 'Photo loaded. Select the four floor corners.');
  return true;
}

async function loadRoom(preset, renderNow = false) {
  const token = ++state.photoLoad;
  let loaded = false;
  $('photo').value = '';
  state.loadingPhoto = true;
  invalidate();
  try {
    status(`Opening ${preset.name.toLowerCase()}…`, 'busy');
    const response = await fetch(preset.image);
    if (!response.ok) throw new Error('The sample room could not be loaded. Try uploading your own photo.');
    if (token !== state.photoLoad) return;
    loaded = await loadPhoto(await response.blob(), preset.name, preset, token);
  } catch (error) { if (token === state.photoLoad) { status(error.message, 'error'); if (!state.image) $('empty-state').hidden = false; } }
  finally { if (token === state.photoLoad) { state.loadingPhoto = false; setBusy(state.busy); } }
  if (loaded && renderNow && token === state.photoLoad) await createPreview();
}
function loadDemo(renderNow = false) {
  if (roomPresets.length) return loadRoom(roomPresets[0], renderNow);
}
function updateRooms() {
  document.querySelectorAll('[data-room]').forEach(card => card.setAttribute('aria-pressed', String(card.dataset.room === state.roomId)));
}
async function loadRoomLibrary() {
  try {
    const response = await fetch('/static/rooms/presets.json');
    if (!response.ok) throw new Error('The sample rooms could not be loaded. Try uploading your own photo.');
    roomPresets.push(...await response.json());
    roomPresets.forEach(preset => {
      const card = document.createElement('button');
      card.className = 'room-card'; card.dataset.room = preset.id; card.setAttribute('aria-pressed', 'false');
      card.innerHTML = `<img alt="" src="${preset.thumbnail}" width="280" height="187"><span class="selected-mark"><svg><use href="#i-check"/></svg></span><strong>${preset.name}</strong>`;
      card.setAttribute('aria-label', `Try the ${preset.name.toLowerCase()}`);
      card.onclick = () => loadRoom(preset, true);
      $('rooms').append(card);
    });
    if (!state.photo && state.photoLoad === 0) await loadDemo();
  } catch (error) { status(error.message, 'error'); if (!state.image) $('empty-state').hidden = false; }
}
$('demo').onclick = () => loadDemo(true);
$('empty-demo').onclick = () => loadDemo(true);
$('photo').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  const token = ++state.photoLoad;
  state.loadingPhoto = true;
  invalidate();
  status('Opening your photo…');
  try { await loadPhoto(file, file.name.replace(/\.[^.]+$/, ''), false, token); }
  catch (error) { if (token === state.photoLoad) status(error.message, 'error'); }
  finally { if (token === state.photoLoad) { state.loadingPhoto = false; setBusy(state.busy); event.target.value = ''; } }
};

function updateMaterials() {
  document.querySelectorAll('.material-card').forEach(card => card.setAttribute('aria-pressed', String(card.dataset.material === state.material?.id)));
  document.querySelector('.custom-upload').classList.toggle('selected', state.customFaces.length > 0);
  const source = $('material-source'), material = state.material;
  source.hidden = !material?.source;
  if (material?.source) {
    source.querySelector('strong').textContent = `${material.publisher} · ${material.name}`;
    source.querySelector('span').textContent = `${material.size.join(' × ')} mm preview preset · ${material.license}`;
    source.querySelector('a').href = material.source;
  }
  updateEditor();
}
function selectMaterial(material, renderNow = true) {
  state.textureLoad += 1; state.loadingTextures = false; $('faces').value = '';
  invalidate(); state.material = material; state.customFaces = [];
  $('custom-note').textContent = 'Add one or more tile images';
  [$('tileW').value, $('tileD').value] = material.size;
  $('layout').value = material.layout;
  $('groutMm').value = material.gap; $('groutColour').value = material.grout; $('colour-label').textContent = material.grout.toUpperCase();
  window.dispatchEvent(new Event('floorchange'));
  updateMaterials();
  if (renderNow && state.image && state.quad.length === 4) createPreview();
}
function addMaterialCard(material, container) {
  const card = document.createElement('button');
  card.className = 'material-card'; card.dataset.material = material.id; card.setAttribute('aria-pressed', String(material === state.material));
  card.innerHTML = `<img alt="" src="${material.faces[0]}"><span class="selected-mark"><svg><use href="#i-check"/></svg></span><strong>${material.name}</strong><small>${material.publisher}<br>${material.size.join(' × ')} mm</small>`;
  card.setAttribute('aria-label', `${material.name}, ${material.size.join(' by ')} millimeters preview preset, ${material.finish}, ${material.license}`);
  card.onclick = () => selectMaterial(material);
  container.append(card);
}
async function loadMaterialCatalogue() {
  try {
    const response = await fetch('/static/materials/catalogue.json');
    if (!response.ok) throw new Error('Catalogue unavailable');
    materialCatalogue.push(...await response.json());
    materialCatalogue.forEach(material => addMaterialCard(material, $('catalogue')));
    if (state.textureLoad === 0) selectMaterial(materialCatalogue.find(material => material.id === 'concrete'), Boolean(state.resultUrl));
  } catch {
    $('catalogue-status').textContent = 'The sample textures could not be loaded. Reload the page or upload your own texture.';
  }
}

const pngBlob = tile => new Promise(resolve => tile.toBlob(resolve, 'image/png'));
$('faces').onchange = async event => {
  const files = [...event.target.files];
  if (!files.length) return;
  const token = ++state.textureLoad;
  state.loadingTextures = true;
  invalidate();
  try {
    const faces = await Promise.all(files.map(async file => {
      file = await normalizeUpload(file);
      const image = new Image(), url = URL.createObjectURL(file);
      try { image.src = url; await image.decode(); const tile = document.createElement('canvas'); tile.width = Math.min(1024, image.naturalWidth); tile.height = Math.max(1, Math.round(image.naturalHeight * tile.width / image.naturalWidth)); tile.getContext('2d').drawImage(image, 0, 0, tile.width, tile.height); return new File([await pngBlob(tile)], 'custom-tile.png', { type: 'image/png' }); }
      finally { URL.revokeObjectURL(url); }
    }));
    if (token !== state.textureLoad) return;
    invalidate();
    state.customFaces = faces; state.material = null;
    window.dispatchEvent(new Event('floorchange'));
    $('custom-note').textContent = `${faces.length} texture ${faces.length === 1 ? 'image' : 'images'} selected`;
    updateMaterials();
    status('Custom texture ready. Set the tile size, then create a preview.');
  } catch { if (token === state.textureLoad) status('A texture could not be opened. Choose JPG, PNG, WebP or HEIC images.', 'error'); }
  finally { if (token === state.textureLoad) { state.loadingTextures = false; setBusy(state.busy); event.target.value = ''; } }
};

function clearErrors() {
  document.querySelectorAll('.input-error').forEach(node => node.remove());
  document.querySelectorAll('[aria-invalid]').forEach(input => { input.removeAttribute('aria-invalid'); input.removeAttribute('aria-describedby'); });
}

function validateSettings() {
  clearErrors();
  const values = {}, inputs = ['floorW', 'floorD', 'tileW', 'tileD', 'groutMm'];
  let invalid = null;
  inputs.forEach(id => {
    const input = $(id), value = Number(input.value);
    if (input.value === '' || !Number.isFinite(value) || value < Number(input.min) || value > Number(input.max)) {
      input.setAttribute('aria-invalid', 'true');
      input.setAttribute('aria-describedby', `${id}-error`);
      const error = document.createElement('p'); error.id = `${id}-error`; error.className = 'input-error'; error.textContent = `Use ${input.min}–${input.max}.`;
      input.parentElement.append(error); invalid ||= input;
    }
    values[id] = value;
  });
  if (invalid) { if (invalid.closest('details')) invalid.closest('details').open = true; invalid.focus(); throw new Error('Check the highlighted dimensions, then try again.'); }
  return values;
}

function validFloor() {
  const q = state.quad;
  if (q.length !== 4) return false;
  const turns = q.map((p, i) => { const next = q[(i + 1) % 4], after = q[(i + 2) % 4]; return (next[0] - p[0]) * (after[1] - next[1]) - (next[1] - p[1]) * (after[0] - next[0]); });
  return turns.every(value => value > 0) && Math.abs(q.reduce((sum, p, i) => sum + p[0] * q[(i + 1) % 4][1] - p[1] * q[(i + 1) % 4][0], 0)) > 100;
}

async function createPreview() {
  let controller = null, version = state.version;
  try {
    if (state.loadingPhoto) throw new Error('Wait for the room photo to finish opening.');
    if (state.loadingTextures) throw new Error('Wait for the texture images to finish opening.');
    if (!state.photo) throw new Error('Upload a room photo, or use Try demo to get started.');
    if (state.boundary.length && !state.boundaryClosed) throw new Error('Finish the floor outline before creating a preview.');
    if (!validFloor()) throw new Error('Select four floor corners clockwise: far-left, far-right, near-right, near-left.');
    if (state.boundaryClosed && state.boundary.some(([x, y]) => state.quad.some(([ax, ay], index) => {
      const [bx, by] = state.quad[(index + 1) % 4];
      return (bx - ax) * (y - ay) - (by - ay) * (x - ax) < -2 * Math.hypot(bx - ax, by - ay);
    }))) throw new Error('The floor outline extends beyond the perspective plane. Move the Perspective corners outside the outline, then try again.');
    if (!state.material && !state.customFaces.length) throw new Error('Choose a material or upload your own texture.');
    const values = validateSettings();
    invalidate(false);
    version = state.version;
    controller = new AbortController();
    state.request = controller;
    setBusy(true); status('Rendering your selected material…', 'busy');
    const started = performance.now(), form = new FormData();
    form.append('photo', state.photo, 'room-photo');
    const faces = state.customFaces.length ? state.customFaces : await Promise.all(state.material.faces.map(async (url, index) => {
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error('The material images could not be loaded. Please try another material.');
      return new File([await response.blob()], `sample-${index}.jpg`, { type: 'image/jpeg' });
    }));
    if (version !== state.version) return;
    faces.forEach(face => form.append('faces', face));
    const colour = $('groutColour').value.match(/[a-f\d]{2}/gi).map(value => parseInt(value, 16)).reverse();
    form.append('payload', JSON.stringify({ quad: state.quad, floor_boundary: state.boundaryClosed ? state.boundary : null, exclude: [...state.aiExclusions, ...state.exclusions], tile_mm: [values.tileW, values.tileD], floor_mm: [values.floorW * 1000, values.floorD * 1000], grout_mm: values.groutMm, grout_bgr: colour, layout: $('layout').value, auto: $('auto').checked ? 'colour-deviation' : 'none' }));
    const response = await fetch('/render', { method: 'POST', body: form, signal: controller.signal });
    if (!response.ok) { let detail = ''; try { const body = await response.json(); if (typeof body.detail === 'string') detail = body.detail; } catch {} throw new Error(`Preview failed (${response.status}). ${detail || 'Please try again or choose another photo.'}`); }
    const blob = await response.blob();
    if (version !== state.version) return;
    const url = URL.createObjectURL(blob);
    $('result').src = url;
    try { await $('result').decode(); } catch { URL.revokeObjectURL(url); throw new Error('The preview image could not be opened. Please try again.'); }
    if (version !== state.version) { URL.revokeObjectURL(url); return; }
    state.resultUrl = url;
    $('download').href = url; $('download').hidden = false;
    $('compare-view').disabled = false;
    setComparison(50);
    setView(true);
    status(`Preview ready · ${((performance.now() - started) / 1000).toFixed(1)}s`, 'success');
  } catch (error) { if (version === state.version && error.name !== 'AbortError') status(error.message || 'The preview could not be created. Please try again.', 'error'); }
  finally { if (controller && state.request === controller) { state.request = null; setBusy(false); } }
}

$('go').onclick = createPreview;
['floorW', 'floorD', 'tileW', 'tileD', 'groutMm', 'groutColour', 'layout', 'auto'].forEach(id => $(id).addEventListener('input', () => { clearErrors(); invalidate(); if (id === 'groutColour') $('colour-label').textContent = $('groutColour').value.toUpperCase(); }));
new ResizeObserver(fitImage).observe($('stage'));
updateMaterials();
const studioReady = Promise.all([loadMaterialCatalogue(), loadRoomLibrary()]);
