const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium, webkit } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.SHOWROOM_TEST_URL || 'http://127.0.0.1:8100';
const sourceId = process.env.SHOWROOM_TEST_PROJECT;
const root = path.resolve(__dirname, '..');

async function ready(page, url) {
  page.setDefaultTimeout(45000);
  await page.goto(url);
  await page.waitForFunction(() => !document.querySelector('#project-save').disabled);
}

async function pick(page, x, y, exclude = false) {
  await page.locator('#select-tool').click({timeout:4000});
  await page.locator(exclude ? '#exclude-point' : '#include-point').click();
  await page.locator('#cv').scrollIntoViewIfNeeded();
  const bounds = await page.locator('#cv').boundingBox();
  const response = page.waitForResponse(r => r.url().endsWith('/select-surface'));
  await page.mouse.click(bounds.x + bounds.width * x, bounds.y + bounds.height * y);
  const result = await response;
  assert.equal(result.status(), 200, await result.text());
  await page.waitForFunction(() => !state.busy);
}

async function preview(page, layers) {
  const response = page.waitForResponse(r => r.url().endsWith('/render'));
  await page.locator(layers > 1 ? '#preview-all' : '#go').click();
  const result = await response;
  assert.equal(result.status(), 200, result.status() === 200 ? '' : await result.text());
  await page.waitForFunction(() => document.querySelector('#status-text').textContent.startsWith('Preview ready'));
  assert.equal(await page.evaluate(() => showroomSurfaces.capture().filter(s => s.quad.length).length), layers);
}

(async () => {
  const browser = await chromium.launch({headless:true});
  const errors = [];
  let copyId;
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100},acceptDownloads:true});
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', dialog => dialog.accept());
    await ready(page, sourceId ? `${base}/?project=${sourceId}` : base);
    await pick(page, sourceId ? 1500/3024 : .45, sourceId ? 3400/4032 : .91);
    await pick(page, sourceId ? 1000/3024 : .65, sourceId ? 3950/4032 : .6, true);
    assert.equal(await page.evaluate(() => state.selectionPoints.length), 2);
    await page.locator('#undo').click();
    await page.waitForFunction(() => !state.busy);
    assert.equal(await page.evaluate(() => state.selectionPoints.length), 1);
    await pick(page, sourceId ? 1000/3024 : .65, sourceId ? 3950/4032 : .6, true);
    await page.getByRole('button', {name:/Natural oak,/}).click();
    await page.locator('#floorW').fill('2.71'); await page.locator('#floorD').fill('2.52');
    const floor = await page.evaluate(() => showroomSurfaces.capture()[0]);
    console.log('PASS CPU point selection, exclusion and undo on the actual photo');

    await page.locator('#wall-tab').click();
    await pick(page, sourceId ? 1000/3024 : .08, sourceId ? 1500/4032 : .45);
    await page.getByRole('button', {name:/Ivory limestone,/}).click();
    await page.locator('#floorW').fill('3'); await page.locator('#floorD').fill('2.4');
    await page.locator('#surface-name').fill('Wall by the door');
    await page.locator('#surface-name').press('Tab');
    assert.equal(await page.locator('#surface-depth-label').textContent(), 'Height');
    await page.locator('#floor-tab').click();
    assert.equal(await page.locator('#floorD').inputValue(), '2.52');
    assert.equal(await page.evaluate(() => state.material.id), floor.material_id);
    assert.deepEqual(await page.evaluate(() => state.quad), floor.quad);
    await page.locator('#tileRotation').selectOption('90');
    await page.locator('#wall-tab').click();
    assert.equal(await page.locator('#floorD').inputValue(), '2.4');
    assert.equal(await page.locator('#tileRotation').inputValue(), '0');
    await preview(page, 2);
    console.log('PASS floor and wall retain independent geometry, materials, dimensions and tile direction; both render together');

    await page.locator('#project-name').fill('Floor and wall interaction check');
    const saved = page.waitForResponse(r => r.url().endsWith('/projects') && r.request().method() === 'POST');
    await page.locator('#project-copy').click();
    assert.equal((await saved).status(), 201);
    await page.waitForFunction(() => !document.querySelector('#project-save').disabled);
    copyId = new URL(page.url()).searchParams.get('project');
    assert(copyId && copyId !== sourceId);
    await page.reload();
    await page.waitForFunction(() => !document.querySelector('#project-save').disabled);
    assert.equal(await page.evaluate(()=>state.surfaces.length), 2);
    assert.equal(await page.locator('#surface-select').inputValue(), 'wall-1');
    assert.equal(await page.locator('#surface-name').inputValue(), 'Wall by the door');
    await page.locator('#floor-tab').click();
    assert.equal(await page.locator('#tileRotation').inputValue(), '90');
    assert.equal(await page.evaluate(() => state.selectionPoints.length), 2);
    await preview(page, 2);
    const download = page.waitForEvent('download');
    await page.locator('#download').click();
    await fs.mkdir(`${root}/artifacts/surfaces-check`, {recursive:true});
    await (await download).saveAs(`${root}/artifacts/surfaces-check/preview.png`);
    await page.locator('#stage').screenshot({path:`${root}/artifacts/surfaces-check/desktop.png`});
    await fs.writeFile(`${root}/artifacts/surfaces-check/project.json`,JSON.stringify({id:copyId,url:`http://localhost:8100/?project=${copyId}`}));
    assert.deepEqual(errors, []);
    console.log('PASS save-copy, reload, independent layers and PNG download; original project untouched');
  } finally { await browser.close(); }

  const safari = await webkit.launch({headless:true});
  try {
    const context = await safari.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    const page = await context.newPage(); page.on('pageerror',error=>errors.push(error.message));
    await ready(page, `${base}/?project=${copyId}`);
    await page.locator('#wall-tab').click();
    assert.equal(await page.locator('#floorD').inputValue(),'2.4');
    await page.locator('#select-tool').tap();
    await page.locator('#include-point').tap();
    await page.locator('#cv').scrollIntoViewIfNeeded();
    const bounds = await page.locator('#cv').boundingBox();
    const response = page.waitForResponse(r=>r.url().endsWith('/select-surface'));
    await page.touchscreen.tap(bounds.x+bounds.width*(sourceId?1000/3024:.08),bounds.y+bounds.height*(sourceId?1500/4032:.45));
    assert.equal((await response).status(),200);
    await page.waitForFunction(()=>!state.busy);
    await preview(page,2);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.deepEqual(errors,[]);
    console.log('PASS WebKit iPhone: touch selection, surface switching, layered render and responsive layout');
  } finally {await safari.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
