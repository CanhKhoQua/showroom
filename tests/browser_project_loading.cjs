const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=require('node:path').resolve(__dirname,'..');
const base=process.env.SHOWROOM_TEST_URL || 'http://127.0.0.1:8100';
(async()=>{
  const fixture=JSON.parse(await fs.readFile(`${root}/artifacts/server-project-check/fixture.json`,'utf8'));
  const browser=await chromium.launch({headless:true});
  let release;
  try {
    const page=await browser.newPage(); page.setDefaultTimeout(10000);
    page.on('dialog',d=>d.accept());
    const current=await (await page.request.get(`${base}/projects/${fixture.id}`)).json();
    const snapshot={name:'Interaction test slow project',schema_version:1,workspace:current.workspace,quote:current.quote};
    const saved=await (await page.request.post(`${base}/projects`,{data:snapshot})).json();
    for(const endpoint of [`**/projects/${saved.id}`,'**/quote']) {
      let observed;
      const started=new Promise(r=>observed=r);
      const gate=new Promise(r=>release=r);
      await page.route(endpoint,async route=>{observed();await gate;await route.continue();});
      await page.goto(`${base}/?project=${saved.id}`,{waitUntil:'domcontentloaded'});
      await started;
      let blocked=false;
      try {await page.locator('#floorW').fill('5',{timeout:750});}
      catch(error){if(error.name==='TimeoutError')blocked=true;else throw error;}
      assert.equal(blocked,true,`Floor editing must be blocked while ${endpoint} is loading`);
      release();
      await page.waitForFunction(()=>document.querySelector('#project-status').textContent.startsWith('Saved project opened'));
      assert.equal(await page.locator('#floorW').inputValue(),current.workspace.settings.floorW);
      await page.unroute(endpoint);
      console.log(`PASS loading lock for ${endpoint}`);
    }
    await page.locator('#floorW').fill('5');
    assert.match(await page.locator('#project-status').innerText(),/Unsaved changes/);
    assert.equal(await page.locator('#project-name').isEnabled(),true);
    console.log('PASS editor unlocks after loading and tracks new changes');
  } finally {if(release)release();await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
