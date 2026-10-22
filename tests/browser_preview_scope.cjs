const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium, webkit} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.SHOWROOM_TEST_URL || 'http://127.0.0.1:8100';
const source = process.env.SHOWROOM_TEST_PROJECT;

async function run(browser, touch = false) {
  const context = await browser.newContext(touch ? {viewport:{width:390,height:844},isMobile:true,hasTouch:true} : {viewport:{width:1440,height:1100}});
  try {
    const page = await context.newPage(); page.setDefaultTimeout(60000);
    const errors=[]; page.on('pageerror',error=>errors.push(error.message));
    page.on('dialog',dialog=>dialog.accept());
    await page.goto(source ? `${base}/?project=${source}` : base);
    await page.waitForFunction(()=>!document.querySelector('#project-save').disabled);
    if (!source) {
      // Compare with the server's JPEG decode, avoiding browser decoder rounding.
      await page.evaluate(async()=>{
        const form=new FormData();form.append('file',state.photo,'room-photo');
        const response=await fetch('/image',{method:'POST',body:form});
        if(!response.ok)throw new Error('Could not prepare the original photo reference');
        const image=await createImageBitmap(await response.blob());
        const canvas=document.createElement('canvas');canvas.id='scope-reference';canvas.hidden=true;
        canvas.width=image.width;canvas.height=image.height;canvas.getContext('2d').drawImage(image,0,0);
        document.body.append(canvas);image.close();
      });
      await page.locator('#wall-tab').click();
      await page.evaluate(()=>{
        const w=state.image.naturalWidth,h=state.image.naturalHeight;
        state.quad=[[w*.1,h*.1],[w*.5,h*.1],[w*.5,h*.45],[w*.1,h*.45]];
        state.boundary=structuredClone(state.quad);state.boundaryClosed=true;
        showroomSurfaces.store();fitImage();updateEditor();
      });
    }
    async function choose(kind) {
      const tab=page.locator(`#${kind}-tab`);
      if (await tab.count()) await tab.click();
      else await page.locator('#surface-select').selectOption(`${kind}-1`);
    }
    async function render(button) {
      const response=page.waitForResponse(r=>r.url().endsWith('/render'));
      await page.locator(button).click();
      const result=await response;assert.equal(result.status(),200);
      await page.waitForFunction(()=>document.querySelector('#status-text').textContent.startsWith('Preview ready'));
      const changed = await page.evaluate((actual)=>{
        const image=document.querySelector('#result'), original=document.querySelector('#scope-reference')||document.querySelector('#cv');
        const result=document.createElement('canvas');result.width=image.naturalWidth;result.height=image.naturalHeight;
        result.getContext('2d').drawImage(image,0,0);
        const points=actual?{floor:[1500,3400],wall:[1000,1500]}:{floor:[original.width*.85,original.height*.92],wall:[original.width*.3,original.height*.2]};
        return Object.fromEntries(Object.entries(points).map(([kind,[x,y]])=>{
          const before=original.getContext('2d').getImageData(Math.round(x),Math.round(y),1,1).data;
          const after=result.getContext('2d').getImageData(Math.round(x),Math.round(y),1,1).data;
          return [kind,Array.from(before).some((value,i)=>value!==after[i])];
        }));
      },Boolean(source));
      if (!touch) {
        const root=path.resolve(__dirname,'../artifacts/preview-scope');
        await fs.mkdir(root,{recursive:true});
        const kind=button==='#preview-all'?'whole-room':await page.evaluate(()=>showroomSurfaces.current().kind);
        await page.locator('.viewer-card').screenshot({path:`${root}/${kind}.png`});
      }
      return changed;
    }
    await choose('floor');
    assert.deepEqual(await render('#go'),{floor:true,wall:false},'Floor preview must leave the wall photo unchanged');
    assert.match(await page.locator('#render-label').textContent(),/preview floor/i);
    await choose('wall');
    assert.deepEqual(await render('#go'),{floor:false,wall:true},'Wall preview must leave the floor photo unchanged');
    assert.match(await page.locator('#render-label').textContent(),/preview wall/i);
    assert.deepEqual(await render('#preview-all'),{floor:true,wall:true},'Whole-room preview must explicitly include both');
    await page.locator('#surface-name').fill('Wall by the door');
    await page.locator('#surface-name').press('Tab');
    await choose('floor');await choose('wall');
    assert.equal(await page.locator('#surface-name').inputValue(),'Wall by the door');
    assert.equal(await page.locator('#wall-tab').getAttribute('aria-pressed'),'true');
    await page.locator('#add-wall').click();
    await page.locator('#new-area-name').fill('Wall behind the sofa');
    await page.locator('#confirm-add-area').click();
    assert.equal(await page.locator('#surface-name').inputValue(),'Wall behind the sofa');
    assert.equal(await page.locator('#surface-select option').count(),2);
    assert.deepEqual(await page.locator('#surface-select option').allTextContents(),['Wall by the door','Wall behind the sofa']);
    await choose('floor');
    assert.deepEqual(await render('#go'),{floor:true,wall:false},'An unfinished wall must not block a floor-only preview');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.deepEqual(errors,[]);
    console.log(`PASS ${touch?'WebKit touch':'Chromium'}: floor-only, wall-only, explicit whole-room preview, named areas and clear tabs`);
  } finally {await context.close();}
}
(async()=>{
  const desktop=await chromium.launch({headless:true});
  try{await run(desktop);}finally{await desktop.close();}
  const mobile=await webkit.launch({headless:true});
  try{await run(mobile,true);}finally{await mobile.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
