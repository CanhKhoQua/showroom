const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium, webkit} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.SHOWROOM_TEST_URL || 'http://127.0.0.1:8100';
const photo = process.env.SHOWROOM_TEST_PHOTO || path.resolve(__dirname, '../bench/static/demo-room.jpg');
const checks = JSON.parse(process.env.SHOWROOM_FLOOR_CHECKS || '{"inside":[[1200,850]],"outside":[[100,100]]}');

async function run(browser, mobile) {
  const page = await browser.newPage(mobile ? {viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true} : {viewport: {width: 1440, height: 1100}});
  page.setDefaultTimeout(60000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', dialog => dialog.accept());
  try {
    await page.goto(base);
    await page.waitForFunction(() => !document.querySelector('#project-save').disabled);
    await page.locator('#photo').setInputFiles(photo);
    await page.waitForFunction(name => !state.loadingPhoto && $('room-name').textContent === name, path.parse(photo).name);
    const detected = page.waitForResponse(r => r.url().endsWith('/detect-floor'));
    await page.locator('#detect-floor').click();
    const response = await detected;
    assert.equal(response.status(), 200, await response.text());
    await page.waitForFunction(() => !state.busy);
    const geometry = await response.json();
    const selected = await page.evaluate(({geometry, checks}) => {
      function inside([x, y], polygon) {
        let hit = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
          const [ax, ay] = polygon[i], [bx, by] = polygon[j];
          if ((ay > y) !== (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax) hit = !hit;
        }
        return hit;
      }
      return [...checks.inside, ...checks.outside].map(point => inside(point, geometry.floor_boundary) && !geometry.exclude.some(hole => inside(point, hole)));
    }, {geometry, checks});
    assert.deepEqual(selected, [...checks.inside.map(() => true), ...checks.outside.map(() => false)], 'Detected floor must exclude the checked walls and foreground objects');
    if (checks.max_size_m) assert(geometry.plane.floor_mm.every(mm => mm < checks.max_size_m * 1000), 'Foreground mistakes must not stretch the estimated plane toward the horizon');
    const rendered = page.waitForResponse(r => r.url().endsWith('/render'));
    await page.locator('#go').click();
    assert.equal((await rendered).status(), 200);
    await page.waitForFunction(() => $('status-text').textContent.startsWith('Preview ready'));
    await page.locator('#project-name').fill(`${path.parse(photo).name} - automatic floor check`);
    const saved = page.waitForResponse(r => r.url().endsWith('/projects') && r.request().method() === 'POST');
    await page.locator('#project-copy').click();
    assert.equal((await saved).status(), 201);
    await page.waitForFunction(() => !$('project-save').disabled);
    if (!mobile) {
      const root = path.resolve(__dirname, '../artifacts/floor-detection-check');
      await fs.mkdir(root, {recursive: true});
      await page.locator('.viewer-card').screenshot({path: `${root}/preview.png`});
      await fs.writeFile(`${root}/project.json`, JSON.stringify({id: new URL(page.url()).searchParams.get('project'), url: page.url(), geometry}));
    }
    assert.deepEqual(errors, []);
    console.log(`PASS ${mobile ? 'WebKit mobile' : 'Chromium'}: upload, automatic detection, floor/object checks, preview and save-copy`);
  } finally { await page.close(); }
}

(async () => {
  for (const [type, mobile] of [[chromium, false], [webkit, true]]) {
    const browser = await type.launch({headless: true});
    try { await run(browser, mobile); } finally { await browser.close(); }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
