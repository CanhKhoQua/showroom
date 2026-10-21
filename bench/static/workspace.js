'use strict';

window.showroomWorkspace = (() => {
  const fields = ['floorW', 'floorD', 'tileW', 'tileD', 'layout', 'groutMm', 'groutColour'];

  function capture() {
    if (!state.photo || state.loadingPhoto || state.loadingTextures) throw new Error('Wait for the room photo and textures to finish opening.');
    return structuredClone({
      name: $('room-name').textContent, photo: state.photo, room_id: state.roomId,
      quad: state.quad, boundary: state.boundary, boundary_closed: state.boundaryClosed,
      auto_plane: state.autoPlane, exclusions: state.exclusions, ai_exclusions: state.aiExclusions,
      material_id: state.material?.id || null, custom_faces: state.customFaces,
      settings: { ...Object.fromEntries(fields.map(id => [id, $(id).value])), auto: $('auto').checked },
    });
  }

  function imageData(blob) {
    if (typeof blob === 'string') return Promise.resolve(blob);
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('A room image could not be read. Please try saving again.'));
      reader.readAsDataURL(blob);
    });
  }

  async function serialize(snapshot) {
    return { ...snapshot, photo: await imageData(snapshot.photo),
      custom_faces: await Promise.all(snapshot.custom_faces.map(imageData)) };
  }

  async function imageBlob(image) {
    return typeof image === 'string' ? (await fetch(image)).blob() : image;
  }

  async function restore(snapshot) {
    const material = materialCatalogue.find(item => item.id === snapshot.material_id);
    if (snapshot.material_id && !material) throw new Error('The saved material is unavailable. Your saved project has not been changed.');
    const token = ++state.photoLoad;
    state.textureLoad += 1;
    state.loadingPhoto = true;
    invalidate(false);
    try {
      const [photo, ...faces] = await Promise.all([imageBlob(snapshot.photo), ...snapshot.custom_faces.map(imageBlob)]);
      const preset = roomPresets.find(room => room.id === snapshot.room_id);
      if (!await loadPhoto(photo, snapshot.name, preset, token)) return false;
      Object.assign(state, {
        quad: structuredClone(snapshot.quad), boundary: structuredClone(snapshot.boundary),
        boundaryClosed: snapshot.boundary_closed, autoPlane: snapshot.auto_plane,
        exclusions: structuredClone(snapshot.exclusions), aiExclusions: structuredClone(snapshot.ai_exclusions),
        material: material || null, customFaces: faces,
      });
      fields.forEach(id => { if (snapshot.settings[id] !== undefined) $(id).value = snapshot.settings[id]; });
      $('auto').checked = snapshot.settings.auto === true;
      $('colour-label').textContent = $('groutColour').value.toUpperCase();
      $('custom-note').textContent = faces.length ? `${faces.length} texture ${faces.length === 1 ? 'image' : 'images'} selected` : 'Add one or more tile images';
      $('faces').value = '';
      updateMaterials(); setTool(state.boundaryClosed ? 'outline' : 'corners'); fitImage();
      window.dispatchEvent(new Event('floorchange'));
      window.dispatchEvent(new Event('projectchange'));
      status('Saved room loaded. Create a preview to see its selected material.');
      return true;
    } finally {
      if (token === state.photoLoad) { state.loadingPhoto = false; state.loadingTextures = false; setBusy(state.busy); }
    }
  }

  return { capture, serialize, restore };
})();
