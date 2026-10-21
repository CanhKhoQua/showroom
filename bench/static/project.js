'use strict';

(() => {
  let projectId = null, revision = null, loading = true, saving = false, dirty = false, changes = 0;

  function message(text, error = false) {
    $('project-status').textContent = text;
    $('project-status').classList.toggle('error', error);
  }

  function changed() {
    if (loading) return;
    dirty = true; changes += 1;
    if (!saving) message('Unsaved changes. Save the project to keep them on the server.');
  }

  function savedLink(project) {
    const url = new URL(location.href); url.searchParams.set('project', project.id);
    history.replaceState(null, '', url);
    $('project-link').value = project.share_url || url.href;
    $('project-share').hidden = false;
  }

  async function save(copy = false) {
    if (saving || loading) return;
    saving = true;
    $('project-save').disabled = true; $('project-copy').disabled = true;
    const capturedChanges = changes;
    message('Saving the room photos and estimate to the server…');
    try {
      const name = $('project-name').value.trim();
      if (!name) { $('project-name').focus(); throw new Error('Enter a project name.'); }
      const workspace = window.showroomWorkspace.capture(), quote = window.showroomQuote.capture();
      const [savedWorkspace, savedQuote] = await Promise.all([window.showroomWorkspace.serialize(workspace), window.showroomQuote.serialize(quote)]);
      const updating = projectId && !copy;
      const response = await fetch(updating ? `/projects/${projectId}` : '/projects', {
        method: updating ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, schema_version: 1, workspace: savedWorkspace, quote: savedQuote, ...(updating ? { revision } : {}) }),
      });
      if (response.status === 409) throw new Error('This project changed on another device. Reload the saved version, or use Save a copy to keep your current edits.');
      if (!response.ok) throw new Error(`The project could not be saved (${response.status}). Your edits remain in this tab; try saving again.`);
      const project = await response.json();
      projectId = project.id; revision = project.revision; dirty = capturedChanges !== changes;
      savedLink(project);
      message(dirty ? 'The snapshot was saved. Newer edits are still unsaved.' : 'Saved on the server. Open the project link on another device to continue.');
    } catch (error) { message(error.message, true); }
    finally { saving = false; $('project-save').disabled = false; $('project-copy').disabled = false; }
  }

  async function openProject() {
    const id = new URL(location.href).searchParams.get('project');
    try {
      await studioReady;
      if (!id) { message('Save to keep rooms, photos and the estimate on the server.'); return; }
      message('Opening the saved project…');
      const response = await fetch(`/projects/${encodeURIComponent(id)}`, { cache: 'no-store' });
      if (!response.ok) throw new Error('The saved project could not be opened. Check its link and server connection.');
      const project = await response.json();
      if (!await window.showroomWorkspace.restore(project.workspace)) throw new Error('The room changed while opening the project. Reload its link to try again.');
      await window.showroomQuote.restore(project.quote);
      $('project-name').value = project.name;
      projectId = project.id; revision = project.revision;
      savedLink(project);
      message('Saved project opened. Select an added room to reopen its photo and floor settings.');
    } catch (error) { message(error.message, true); }
    finally {
      loading = false; dirty = false;
      $('studio-editor').inert = false;
      $('studio-editor').disabled = false;
      $('studio-editor').setAttribute('aria-busy', 'false');
      $('project-name').disabled = false;
      $('project-save').disabled = false; $('project-copy').disabled = false;
    }
  }

  window.addEventListener('projectchange', changed);
  $('project-name').addEventListener('input', changed);
  $('project-save').onclick = () => save();
  $('project-copy').onclick = () => save(true);
  $('project-copy-link').onclick = async () => {
    try {
      await navigator.clipboard.writeText($('project-link').value);
      message(dirty ? 'Link copied. Save your latest edits before opening it elsewhere.' : 'Project link copied.');
    } catch {
      $('project-link').focus(); $('project-link').select();
      message('The project link is selected. Copy it and open it on another device.');
    }
  };
  window.addEventListener('beforeunload', event => {
    if (dirty || saving) { event.preventDefault(); event.returnValue = ''; }
  });
  openProject();
})();
