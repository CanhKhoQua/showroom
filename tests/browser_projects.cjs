const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const { chromium, webkit } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=require('node:path').resolve(__dirname,'..');
const base=process.env.SHOWROOM_TEST_URL || 'http://127.0.0.1:8100';
async function ready(page,url=base) {
  page.setDefaultTimeout(15000);
  page.on('dialog', dialog=>dialog.accept());
  await page.goto(url);
  await page.locator('#project-name').waitFor({timeout:3000});
  await page.waitForFunction(()=>document.querySelector('#project-save').disabled === false);
}
async function add(page,name,width) {
  await page.locator('#floorW').fill(width); await page.locator('#floorD').fill('3');
  await page.locator('#quote-room-name').fill(name); await page.locator('#quote-sku-name').fill('WOOD');
  await page.locator('#quote-tiles').fill('4'); await page.locator('#quote-price').fill('240000');
  await page.locator('#quote-measured').check();
  await page.getByRole('button',{name:'Add floor to estimate'}).click();
  await page.locator('#quote-results').waitFor();
}
async function save(page,copy=false) {
  const response=page.waitForResponse(r=>r.url().includes('/projects')&&['POST','PUT'].includes(r.request().method()));
  await page.locator(copy?'#project-copy':'#project-save').click();
  const r=await response;
  await page.waitForFunction(()=>document.querySelector('#project-save').disabled === false);
  return r;
}
(async()=>{
  const browser=await chromium.launch({headless:true});
  const errors=[];
  try {
    const a=await browser.newPage({viewport:{width:1440,height:1000}});
    a.on('pageerror',e=>errors.push(e.message));
    await ready(a);
    await a.locator('.finish-settings summary').click();
    await a.locator('#tileW').fill('600'); await a.locator('#tileD').fill('600');
    await a.locator('#faces').setInputFiles([`${root}/bench/static/materials/stone-01.jpg`,`${root}/bench/static/materials/stone-02.jpg`]);
    await a.waitForFunction(()=>document.querySelector('#custom-note').textContent==='2 texture images selected');
    await add(a,'Bedroom 1','3');
    await a.locator('[data-room="bedroom"]').click();
    await a.waitForFunction(()=>document.querySelector('#room-name').textContent==='Bedroom'&&document.querySelector('#status-text').textContent.startsWith('Preview ready'));
    await add(a,'Bedroom 2','4');
    await a.waitForFunction(()=>document.querySelector('#quote-total').dataset.amount==='3840000');
    await a.locator('#project-name').fill('Interaction test townhouse');
    const saved=await save(a);
    assert.equal(saved.status(),201);
    const data=await saved.json();
    assert.equal(data.revision,1);
    const url=await a.locator('#project-link').inputValue();
    assert(new URL(url).searchParams.get('project'));
    console.log('PASS server save: two rooms, photos, custom textures, dimensions, quote');
    await a.reload();
    await a.waitForFunction(()=>document.querySelector('#quote-total').dataset.amount==='3840000');
    assert.equal(await a.locator('#project-name').inputValue(),'Interaction test townhouse');
    assert.equal(await a.locator('#room-name').innerText(),'Bedroom');
    assert.equal(await a.locator('#custom-note').innerText(),'2 texture images selected');
    console.log('PASS browser reload restores project and current room');
    const contextB=await browser.newContext();
    const b=await contextB.newPage(); b.on('pageerror',e=>errors.push(e.message));
    await ready(b,url);
    await b.waitForFunction(()=>document.querySelector('#quote-total').dataset.amount==='3840000');
    assert.equal(await b.locator('#quote-rooms li').count(),2);
    await b.getByRole('button',{name:'Open Bedroom 1',exact:true}).click();
    await b.waitForFunction(()=>document.querySelector('#room-name').textContent==='Dining room');
    assert.equal(await b.locator('#floorW').inputValue(),'3');
    assert.equal(await b.locator('#custom-note').innerText(),'2 texture images selected');
    const preview=b.waitForResponse(r=>r.url().endsWith('/render'));
    await b.locator('#go').click(); assert.equal((await preview).status(),200);
    await b.waitForFunction(()=>document.querySelector('#status-text').textContent.startsWith('Preview ready'));
    console.log('PASS independent browser opens the link, selects saved room and renders it');
    await b.locator('#project-name').fill('Interaction test device B');
    assert.equal((await save(b)).status(),200);
    await a.locator('#project-name').fill('Interaction test device A');
    assert.equal((await save(a)).status(),409);
    assert.match(await a.locator('#project-status').innerText(),/another device/i);
    const latest=await a.request.get(`${base}/projects/${data.id}`);
    assert.equal((await latest.json()).name,'Interaction test device B');
    console.log('PASS conflicting device save cannot overwrite newer data');
    const copied=await save(a,true); assert.equal(copied.status(),201);
    const copy=await copied.json(); assert.notEqual(copy.id,data.id);
    console.log('PASS save a copy preserves conflicting local edits');
    await fs.mkdir(`${root}/artifacts/server-project-check`,{recursive:true});
    await fs.writeFile(`${root}/artifacts/server-project-check/fixture.json`,JSON.stringify({url,id:data.id,copyId:copy.id,total:'3840000'}));
    await b.screenshot({path:`${root}/artifacts/server-project-check/desktop.png`,fullPage:true});
    assert.deepEqual(errors,[]);
  } finally {await browser.close();}
  const safari=await webkit.launch({headless:true});
  try {
    const fixture=JSON.parse(await fs.readFile(`${root}/artifacts/server-project-check/fixture.json`,'utf8'));
    const context=await safari.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,deviceScaleFactor:2});
    const page=await context.newPage(); page.on('pageerror',e=>errors.push(e.message));
    await ready(page,fixture.url);
    await page.waitForFunction(()=>document.querySelector('#quote-total').dataset.amount==='3840000');
    assert.equal(await page.locator('#project-name').inputValue(),'Interaction test device B');
    await page.getByRole('button',{name:'Open Bedroom 2',exact:true}).tap();
    await page.waitForFunction(()=>document.querySelector('#room-name').textContent==='Bedroom');
    const response=page.waitForResponse(r=>r.url().endsWith('/render'));
    await page.locator('#go').tap(); assert.equal((await response).status(),200);
    await page.waitForFunction(()=>document.querySelector('#status-text').textContent.startsWith('Preview ready'));
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:`${root}/artifacts/server-project-check/webkit-mobile.png`,fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS WebKit iPhone: shared project, room selection, touch render, responsive layout');
  } finally {await safari.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
