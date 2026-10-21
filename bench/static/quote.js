'use strict';

(() => {
  const rooms = [], skus = new Map();
  const money = value => new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND' }).format(value);
  const area = value => new Intl.NumberFormat('en', { maximumFractionDigits: 4 }).format(value);
  let version = 0;

  function message(text, error = false) {
    $('quote-status').textContent = text;
    $('quote-status').classList.toggle('error', error);
  }

  function sourceChanged() {
    $('quote-measured').checked = false;
    $('quote-source').textContent = `Current floor: ${$('floorW').value} × ${$('floorD').value} m · Tile: ${$('tileW').value} × ${$('tileD').value} mm. Adjust these dimensions in the material panel before adding a floor.`;
  }

  function cell(row, text) {
    const td = document.createElement('td');
    td.textContent = text;
    row.append(td);
    return td;
  }

  function renderRooms() {
    $('quote-rooms').replaceChildren();
    rooms.forEach((room, index) => {
      const item = document.createElement('li'), label = document.createElement('button');
      label.type = 'button'; label.className = 'room-open';
      label.textContent = `${room.name} · ${room.size_m.join(' × ')} m · ${room.sku_name}`;
      label.setAttribute('aria-label', `Open ${room.name}`);
      label.onclick = async () => {
        try { await window.showroomWorkspace.restore(room.workspace); }
        catch (error) { message(error.message, true); }
      };
      const remove = document.createElement('button');
      remove.type = 'button'; remove.className = 'text-button'; remove.textContent = 'Remove';
      remove.setAttribute('aria-label', `Remove ${room.name}`);
      remove.onclick = () => {
        rooms.splice(index, 1);
        if (!rooms.some(r => r.sku_name === room.sku_name)) skus.delete(room.sku_name);
        renderRooms(); calculate();
        window.dispatchEvent(new Event('projectchange'));
      };
      item.append(label, remove); $('quote-rooms').append(item);
    });
  }

  async function calculate() {
    const token = ++version;
    $('quote-results').hidden = true; $('quote-print').hidden = true;
    $('quote-total').removeAttribute('data-amount');
    if (!rooms.length) { message('Add a measured floor to start an estimate.'); return; }
    message('Calculating whole-box quantities…');
    try {
      const response = await fetch('/quote', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rooms: rooms.map(({ workspace, ...room }) => room), skus: Object.fromEntries(skus), dimensions_confirmed: true }),
      });
      if (token !== version) return;
      if (!response.ok) throw new Error('Check the room dimensions, packaging and prices, then try again.');
      const result = await response.json();
      if (token !== version) return;
      const body = $('quote-lines').querySelector('tbody'); body.replaceChildren();
      result.lines.forEach(line => {
        const row = document.createElement('tr');
        const title = cell(row, line.sku_name), detail = document.createElement('small');
        detail.textContent = `${line.rooms.join(', ')} · ${line.tile_mm.join(' × ')} mm · ${line.tiles_per_box} tiles/box · ${line.waste_pct}% waste`;
        title.append(detail);
        cell(row, area(line.area_m2)); cell(row, line.tiles); cell(row, line.boxes);
        cell(row, area(line.purchased_m2)); cell(row, money(line.price_per_box)); cell(row, money(line.amount));
        body.append(row);
      });
      $('quote-total').textContent = money(result.total);
      $('quote-total').dataset.amount = result.total;
      $('quote-results').hidden = false; $('quote-print').hidden = false;
      message(`${rooms.length} floor${rooms.length === 1 ? '' : 's'} · ${result.lines.length} tile code${result.lines.length === 1 ? '' : 's'}. Quantities are rounded once per code.`);
    } catch (error) {
      if (token === version) message(`${error.message} Remove and re-add a floor to retry.`, true);
    }
  }

  $('quote-form').onsubmit = event => {
    event.preventDefault();
    const dimensions = ['floorW', 'floorD', 'tileW', 'tileD'].map(id => $(id));
    const invalid = dimensions.find(input => input.value === '' || !input.checkValidity());
    if (invalid) {
      const details = invalid.closest('details');
      if (details) details.open = true;
      invalid.focus();
      message('Enter valid measured floor and tile dimensions above.', true);
      return;
    }
    const name = $('quote-room-name').value.trim(), code = $('quote-sku-name').value.trim();
    if (!name || !code) { message('Enter a room name and product code.', true); return; }
    const [width, depth, tileWidth, tileDepth] = dimensions.map(input => Number(input.value));
    const sku = { tile_mm: [tileWidth, tileDepth], tiles_per_box: Number($('quote-tiles').value), price_per_box: Number($('quote-price').value), waste_pct: Number($('quote-waste').value) };
    const previous = skus.get(code);
    if (previous && JSON.stringify(previous) !== JSON.stringify(sku)) {
      message(`Tile code ${code} already uses different dimensions, packaging, price or waste. Use the same values, or a different code for a different specification.`, true);
      return;
    }
    let workspace;
    try { workspace = window.showroomWorkspace.capture(); }
    catch (error) { message(error.message, true); return; }
    skus.set(code, sku);
    rooms.push({ name, size_m: [width, depth], sku_name: code, workspace });
    $('quote-measured').checked = false;
    $('quote-room-name').value = '';
    renderRooms(); calculate();
    window.dispatchEvent(new Event('projectchange'));
  };

  $('quote-sku-name').addEventListener('change', () => {
    const sku = skus.get($('quote-sku-name').value.trim());
    if (!sku) return;
    $('quote-tiles').value = sku.tiles_per_box;
    $('quote-price').value = sku.price_per_box;
    $('quote-waste').value = sku.waste_pct;
  });
  ['floorW', 'floorD', 'tileW', 'tileD'].forEach(id => $(id).addEventListener('input', sourceChanged));
  window.addEventListener('floorchange', sourceChanged);
  $('quote-print').onclick = () => window.print();
  window.showroomQuote = {
    capture: () => structuredClone({ rooms, skus: Object.fromEntries(skus) }),
    serialize: async snapshot => ({ ...snapshot, rooms: await Promise.all(snapshot.rooms.map(async room => ({ ...room, workspace: await window.showroomWorkspace.serialize(room.workspace) }))) }),
    restore: snapshot => {
      rooms.splice(0, rooms.length, ...snapshot.rooms);
      skus.clear(); Object.entries(snapshot.skus).forEach(([code, sku]) => skus.set(code, sku));
      renderRooms(); return calculate();
    },
  };
  sourceChanged();
})();
