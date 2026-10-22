const assert = require('node:assert/strict');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.SHOWROOM_TEST_URL || 'http://127.0.0.1:8100';
(async()=>{
  const browser=await chromium.launch({headless:true});
  const failures=[];
  try {
    const page=await browser.newPage();
    let release;
    const barrier=new Promise(resolve=>{release=resolve;});
    await page.route('**/static/surfaces.js',async route=>{await barrier;await route.continue();});
    try {
      await page.goto(base,{waitUntil:'commit'});
      await page.waitForFunction(()=>typeof state!=='undefined'&&state.photo!==null);
      release();
      await page.waitForFunction(()=>!document.querySelector('#project-save').disabled);
      assert.equal(await page.evaluate(()=>showroomSurfaces.capture().length),1);
      assert.equal(await page.locator('#surface-select').inputValue(),'floor-1');
      console.log('PASS delayed surface script still initializes the already loaded photo');
    } catch(error) {release();failures.push(error);console.error('FAIL delayed script:',error.message);}
    finally {await page.close();}

    const upload=await browser.newPage();
    try {
      await upload.goto(base);
      await upload.waitForFunction(()=>!document.querySelector('#project-save').disabled);
      await upload.evaluate(()=>{
        window.originalDecode=Image.prototype.decode;
        Image.prototype.decode=function(){return originalDecode.call(this).then(()=>new Promise(resolve=>{window.releaseTexture=resolve;}));};
      });
      await upload.locator('#faces').setInputFiles(path.resolve(__dirname,'../bench/static/materials/stone-01.jpg'));
      await upload.waitForFunction(()=>state.loadingTextures&&typeof releaseTexture==='function');
      assert.equal(await upload.locator('#wall-tab').isDisabled(),true);
      assert.equal(await upload.locator('#surface-select').isDisabled(),true);
      await upload.evaluate(()=>{Image.prototype.decode=originalDecode;releaseTexture();});
      await upload.waitForFunction(()=>!state.loadingTextures);
      assert.equal(await upload.locator('#wall-tab').isDisabled(),false);
      await upload.locator('#wall-tab').click();
      assert.equal(await upload.evaluate(()=>showroomSurfaces.capture().find(s=>s.id==='floor-1').custom_faces.length),1);
      console.log('PASS pending custom texture stays assigned to its original surface');
    } catch(error) {failures.push(error);console.error('FAIL texture ownership:',error.message);}
    finally {await upload.close();}
  } finally {await browser.close();}
  assert.equal(failures.length,0,failures.map(error=>error.message).join('\n'));
})().catch(error=>{console.error(error);process.exitCode=1;});
