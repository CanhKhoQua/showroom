const assert = require('node:assert/strict');
const {chromium, webkit} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.SHOWROOM_TEST_URL || 'http://127.0.0.1:8100';

async function run(browser) {
  const page = await browser.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', dialog => dialog.accept());
  try {
    await page.goto(base);
    await page.waitForFunction(() => !document.querySelector('#project-save').disabled);
    await page.locator('.finish-settings').evaluate(node => { node.open = true; });
    await page.locator('#surfaceFinish').selectOption('satin');
    await page.locator('#wall-tab').click();
    await page.locator('#surfaceFinish').selectOption('polished');
    await page.evaluate(() => {
      const w = state.image.naturalWidth, h = state.image.naturalHeight;
      state.quad = [[w * .1, h * .1], [w * .5, h * .1], [w * .5, h * .4], [w * .1, h * .4]];
      state.boundary = structuredClone(state.quad); state.boundaryClosed = true;
      showroomSurfaces.store(); updateEditor();
    });
    await page.locator('#floor-tab').click();
    assert.equal(await page.locator('#surfaceFinish').inputValue(), 'satin');
    const saved = await page.evaluate(async () => {
      const workspace = await showroomWorkspace.serialize(showroomWorkspace.capture());
      const response = await fetch('/projects', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: 'Material finish browser check', schema_version: 1, workspace, quote: {rooms: [], skus: {}}})});
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    });
    await page.goto(`${base}/?project=${saved.id}`);
    await page.waitForFunction(() => !document.querySelector('#project-save').disabled);
    assert.equal(await page.locator('#surfaceFinish').inputValue(), 'satin');
    await page.locator('#wall-tab').click();
    assert.equal(await page.locator('#surfaceFinish').inputValue(), 'polished');
    // Binary multipart bodies are omitted from Playwright's request records.
    // Observe the actual FormData at fetch without replacing the HTTP call.
    await page.evaluate(() => {
      const send = window.fetch;
      window.fetch = function(url, options) {
        if (url === '/render') window.sentRenderPayload = JSON.parse(options.body.get('payload'));
        return send.call(this, url, options);
      };
    });
    const response = page.waitForResponse(r => r.url().endsWith('/render'));
    await page.locator('#preview-all').click();
    const rendered = await response;
    assert.equal(rendered.status(), 200);
    assert.deepEqual(await page.evaluate(() => sentRenderPayload.layers.map(layer => layer.surface_finish)), ['satin', 'polished']);
    await page.waitForFunction(() => document.querySelector('#status-text').textContent.startsWith('Preview ready'));
    await page.evaluate(async () => {
      const legacy = showroomWorkspace.capture();
      delete legacy.settings.surfaceFinish;
      for (const area of legacy.surfaces) delete area.settings.surfaceFinish;
      await showroomWorkspace.restore(legacy);
    });
    assert.equal(await page.locator('#surfaceFinish').inputValue(), 'matte');
    assert.deepEqual(errors, []);
    console.log('PASS per-area material finish, server save/reload, whole-room preview and legacy defaults');
  } finally { await page.close(); }
}

(async () => {
  for (const type of [chromium, webkit]) {
    const browser = await type.launch({headless: true});
    try { await run(browser); } finally { await browser.close(); }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
