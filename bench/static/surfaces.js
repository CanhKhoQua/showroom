'use strict';

window.showroomSurfaces = (() => {
  const fields = ['floorW', 'floorD', 'tileW', 'tileD', 'layout', 'groutMm', 'groutColour', 'tileRotation', 'surfaceFinish'];
  let pointLabel = 1;
  let addingKind = null;
  let lastByKind = {};

  function current() {
    return state.surfaces.find(surface => surface.id === state.activeSurfaceId);
  }

  function captureSurface(surface) {
    return structuredClone({
      id: surface.id, name: surface.name, kind: surface.kind,
      quad: state.quad, boundary: state.boundary, boundary_closed: state.boundaryClosed,
      auto_plane: state.autoPlane, exclusions: state.exclusions, ai_exclusions: state.aiExclusions,
      material_id: state.material?.id || null, custom_faces: state.customFaces,
      settings: { ...Object.fromEntries(fields.map(id => [id, $(id).value])), auto: $('auto').checked },
      selection_points: state.selectionPoints,
    });
  }

  function store() {
    const index = state.surfaces.findIndex(surface => surface.id === state.activeSurfaceId);
    if (index >= 0) state.surfaces[index] = captureSurface(state.surfaces[index]);
  }

  function update() {
    const kind = current()?.kind || 'floor';
    const areas = state.surfaces.filter(surface => surface.kind === kind);
    $('surface-select').replaceChildren(...areas.map(surface => {
      const option = document.createElement('option');
      option.value = surface.id; option.textContent = surface.name;
      return option;
    }));
    $('surface-select').value = state.activeSurfaceId || '';
    $('surface-picker').hidden = areas.length < 2;
    $('surface-name').value = current()?.name || '';
    $('floor-tab').setAttribute('aria-pressed', String(kind === 'floor'));
    $('wall-tab').setAttribute('aria-pressed', String(kind === 'wall'));
    $('add-floor').hidden = kind !== 'floor';
    $('add-wall').hidden = kind !== 'wall';
    $('preview-all').hidden = state.surfaces.length < 2;
    $('remove-surface').disabled = state.surfaces.length < 2;
    $('surface-depth-label').textContent = current()?.kind === 'wall' ? 'Height' : 'Depth';
    $('detect-floor').hidden = current()?.kind === 'wall';
    setBusy(state.busy);
  }

  function apply(surface) {
    lastByKind[surface.kind] = surface.id;
    $('new-area-form').hidden = true;
    state.textureLoad += 1;
    setPointLabel(1);
    Object.assign(state, {
      quad: structuredClone(surface.quad), boundary: structuredClone(surface.boundary),
      boundaryClosed: surface.boundary_closed, autoPlane: surface.auto_plane,
      exclusions: structuredClone(surface.exclusions), aiExclusions: structuredClone(surface.ai_exclusions),
      material: materialCatalogue.find(item => item.id === surface.material_id) || null,
      customFaces: [...surface.custom_faces], selectionPoints: structuredClone(surface.selection_points || []),
      selectedCorner: -1, selectedBoundary: -1,
    });
    fields.forEach(id => { $(id).value = surface.settings[id] ?? (id === 'surfaceFinish' ? 'matte' : id === 'tileRotation' ? '0' : $(id).value); });
    $('auto').checked = surface.settings.auto === true;
    $('colour-label').textContent = $('groutColour').value.toUpperCase();
    $('custom-note').textContent = state.customFaces.length ? `${state.customFaces.length} texture images selected` : 'Add one or more tile images';
    $('faces').value = '';
    updateMaterials(); update(); setTool('select'); fitImage();
    window.dispatchEvent(new Event('floorchange'));
  }

  function reset() {
    lastByKind = {};
    setPointLabel(1);
    state.selectionPoints = [];
    state.activeSurfaceId = 'floor-1';
    state.surfaces = [captureSurface({ id: 'floor-1', name: 'Floor', kind: 'floor' })];
    update();
  }

  function add(kind, areaName = null) {
    if (!state.photo || state.busy || state.loadingPhoto || state.loadingTextures) return;
    if (state.surfaces.length >= 12) { status('A project can contain up to 12 surfaces.', 'error'); return; }
    if (!areaName && state.surfaces.some(surface => surface.kind === kind)) {
      addingKind = kind;
      $('new-area-form').hidden = false;
      $('new-area-name').value = '';
      $('new-area-name').focus();
      return;
    }
    store(); invalidate();
    let number = 1;
    while (state.surfaces.some(surface => surface.id === `${kind}-${number}`)) number += 1;
    const name = areaName || (kind === 'wall' ? 'Wall' : 'Floor');
    state.activeSurfaceId = `${kind}-${number}`;
    const surface = captureSurface({ id: state.activeSurfaceId, name, kind });
    Object.assign(surface, { quad: [], boundary: [], boundary_closed: false, auto_plane: false,
      exclusions: [], ai_exclusions: [], selection_points: [] });
    if (kind === 'wall') Object.assign(surface.settings, { floorW: '3', floorD: '2.4' });
    state.surfaces.push(surface); apply(surface);
    status(`${name}: click inside the surface, then align its four Perspective corners.`);
  }

  function restore(surfaces, activeId) {
    if (!surfaces?.length) { reset(); return; }
    state.surfaces = structuredClone(surfaces);
    for (const surface of state.surfaces) {
      const title = surface.kind === 'floor' ? 'Floor' : 'Wall';
      if (surface.id === `${surface.kind}-1` && surface.name === `${title} 1`) surface.name = title;
    }
    state.activeSurfaceId = surfaces.some(surface => surface.id === activeId) ? activeId : surfaces[0].id;
    apply(current());
  }

  async function refine() {
    let controller = null, version = state.version;
    try {
      if (!state.selectionPoints.some(point => point[2] === 1)) {
        invalidate(); state.boundary = []; state.boundaryClosed = false; state.aiExclusions = [];
        updateEditor(); return;
      }
      const surface = current(), estimate = !validFloor() && surface.kind === 'floor';
      invalidate(); version = state.version;
      controller = new AbortController(); state.request = controller;
      state.selectingSurface = true; setBusy(true); status(`Selecting ${surface.name.toLowerCase()}…`, 'busy');
      const form = new FormData(); form.append('photo', state.photo, 'room-photo');
      form.append('payload', JSON.stringify({ points: state.selectionPoints.map(point => point.slice(0, 2)),
        labels: state.selectionPoints.map(point => point[2]), kind: surface.kind, estimate_plane: estimate }));
      const response = await fetch('/select-surface', { method: 'POST', body: form, signal: controller.signal });
      if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'The area could not be selected. Add another include point.'); }
      const result = await response.json();
      if (version !== state.version) return;
      Object.assign(state, { boundary: result.floor_boundary, boundaryClosed: true, aiExclusions: result.exclude,
        selectedBoundary: -1 });
      if (!validFloor()) {
        if (result.plane) {
          state.quad = result.plane.quad;
          $('floorW').value = (result.plane.floor_mm[0] / 1000).toFixed(2);
          $('floorD').value = (result.plane.floor_mm[1] / 1000).toFixed(2);
        } else {
          const xs = state.boundary.map(point => point[0]), ys = state.boundary.map(point => point[1]);
          state.quad = [[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.min(...ys)],
            [Math.max(...xs), Math.max(...ys)], [Math.min(...xs), Math.max(...ys)]];
        }
        state.autoPlane = false;
        window.dispatchEvent(new Event('floorchange'));
      }
      store(); fitImage(); updateEditor();
      status(`${surface.name} selected · ${result.seconds.toFixed(1)}s. Review the edge and align Perspective before previewing.`, 'success');
    } catch (error) {
      if (version === state.version && error.name !== 'AbortError') status(error.message, 'error');
    } finally {
      if (controller && state.request === controller) { state.request = null; state.selectingSurface = false; setBusy(false); }
    }
  }

  function pick(point) {
    if (!state.photo || state.loadingPhoto || state.loadingTextures) return;
    if (!pointLabel && !state.selectionPoints.some(item => item[2])) { status('First include a point inside the surface.', 'error'); return; }
    if (state.selectionPoints.length >= 64) { status('Undo a point before adding more.', 'error'); return; }
    state.selectionPoints.push([...point, pointLabel]);
    refine();
  }

  function setPointLabel(value) {
    pointLabel = value;
    $('include-point').setAttribute('aria-pressed', String(value === 1));
    $('exclude-point').setAttribute('aria-pressed', String(value === 0));
  }

  $('select-tool').onclick = () => setTool('select');
  $('include-point').onclick = () => setPointLabel(1);
  $('exclude-point').onclick = () => setPointLabel(0);
  $('add-floor').onclick = () => add('floor');
  $('add-wall').onclick = () => add('wall');
  function chooseKind(kind) {
    if (state.busy || state.loadingPhoto || state.loadingTextures) return;
    const surface = state.surfaces.find(area => area.id === lastByKind[kind]) || state.surfaces.find(area => area.kind === kind);
    if (!surface) { add(kind); return; }
    store(); invalidate(); state.activeSurfaceId = surface.id;
    apply(current()); status(`Editing ${current().name.toLowerCase()}. Preview changes only this area.`);
  }
  $('floor-tab').onclick = () => chooseKind('floor');
  $('wall-tab').onclick = () => chooseKind('wall');
  $('surface-name').onchange = () => {
    const name = $('surface-name').value.trim();
    if (!name || state.surfaces.some(area => area.id !== state.activeSurfaceId && area.name.toLowerCase() === name.toLowerCase())) {
      $('surface-name').value = current().name;
      status('Give this area a unique name, such as Wall by the door.', 'error'); return;
    }
    current().name = name; window.dispatchEvent(new Event('projectchange')); update();
  };
  $('new-area-form').onsubmit = event => {
    event.preventDefault();
    const name = $('new-area-name').value.trim();
    if (!name || state.surfaces.some(area => area.name.toLowerCase() === name.toLowerCase())) {
      $('new-area-name').setCustomValidity('Use a different area name.'); $('new-area-name').reportValidity(); return;
    }
    add(addingKind, name);
  };
  $('new-area-name').oninput = () => $('new-area-name').setCustomValidity('');
  $('cancel-add-area').onclick = () => { $('new-area-form').hidden = true; };
  $('surface-select').onchange = () => {
    if (state.busy || state.loadingPhoto || state.loadingTextures) return;
    store(); invalidate(); state.activeSurfaceId = $('surface-select').value;
    apply(current()); status(`Editing ${current().name.toLowerCase()}. Other surfaces keep their settings.`);
  };
  $('remove-surface').onclick = () => {
    if (state.surfaces.length < 2 || state.busy || state.loadingPhoto || state.loadingTextures) return;
    invalidate(); state.surfaces = state.surfaces.filter(surface => surface.id !== state.activeSurfaceId);
    state.activeSurfaceId = state.surfaces[0].id; apply(current());
  };
  $('tileRotation').oninput = () => invalidate();

  if (state.photo) reset();

  return { reset, restore, store, pick, refine, current, update,
    capture: () => { store(); return structuredClone(state.surfaces); } };
})();
