    let launched = false;
    let pingTimer = null;
    let mainAlreadyRunning = false;
    let downloadPopupOpen = false;
    let downloadPopupClosed = false;
    let pendingLaunch = false;
    const hasInstalledBefore = localStorage.getItem('installedOnce') === '1';
    const noteOnce = document.getElementById('note');
    if(hasInstalledBefore && noteOnce){ noteOnce.classList.add('hidden'); }

    function startPinging(){
      if (!pingTimer) { pingTimer = setInterval(pingMainServer, 900); }
    }

    async function poll(){
      const res = await fetch('/progress');
      const data = await res.json();
      const status = document.getElementById('status');
      const spinner = document.getElementById('spinner');
      const note = document.getElementById('note');
      const err = document.getElementById('error');

      if (status) {
        status.textContent = data.pct < 100 && data.pct !== -1
          ? 'installing prerequisites. This may take a minute or two.'
          : 'checking…';
      }
      if(spinner){ spinner.classList.toggle('hidden', data.pct >= 100 || data.pct === -1); }
      if(data.pct === -1){
        err.textContent = data.error || 'installation failed';
        err.classList.remove('hidden');
        return;
      }
      if(data.main_running){
        mainAlreadyRunning = true;
      }
      if (data.pct >= 100) {
        localStorage.setItem('installedOnce','1');
        if(note){ note.classList.add('hidden'); }
      }
      const missingModels = Array.isArray(data.models_missing) ? data.models_missing : [];
      if (missingModels.length > 0) {
        showDownloadPopup();
      }
      startPinging();
      if(data.pct >= 100){
        if(!downloadPopupOpen || downloadPopupClosed){
          waitForServer();
        }else{
          pendingLaunch = true;
        }
        return;
      }
      setTimeout(poll, 1000);
    }

    function showDownloadPopup(){
      if(downloadPopupOpen || downloadPopupClosed) return;
      downloadPopupOpen = true;
      const overlay = document.createElement('div');
      overlay.className = 'fixed inset-0 grid place-items-center backdrop-blur-sm';
      overlay.innerHTML = `<div class="sheen glass p-6 rounded-2xl flex flex-col gap-4 max-w-lg w-full">
        <h2 class="text-2xl font-light">Model setup blocked</h2>
        <p class="text-sm opacity-80">This development build cannot distribute model checkpoints until every artifact has an immutable revision, verified SHA-256, and documented license. Existing state-dictionary files can be detected as unsupported user-provided models; legacy pickle models remain disabled.</p>
        <div class="flex justify-end">
          <button id="close-models" class="px-4 py-1 rounded-lg bg-white/90 text-gray-900">close</button>
        </div>
      </div>`;
      document.body.appendChild(overlay);
      const closeBtn = overlay.querySelector('#close-models');
      closeBtn.onclick = () => {
        downloadPopupOpen = false;
        downloadPopupClosed = true;
        overlay.remove();
        if(pendingLaunch){
          pendingLaunch = false;
          waitForServer();
        }
      };
      overlay.tabIndex = 0;
      overlay.focus();
    }

    async function pingMainServer(){
      if (launched) return;
      try{
        await fetch('http://localhost:9876/', {mode:'no-cors'});
        await launchApp();
      }catch(_){ }
    }

    async function waitForServer(){
      startPinging();
      await pingMainServer();
    }

    async function launchApp(){
      if (launched) return;
      launched = true;
      if (pingTimer) { clearInterval(pingTimer); }
      const card = document.getElementById('card');
      const loading = document.getElementById('loading');
      // start a quick fade-out and cue the intro for the app page
      const shell = document.getElementById('installer-shell');
      if(shell){ shell.classList.add('fade-out'); }
      card.style.animation = 'fade .4s ease reverse both';
      loading.classList.remove('hidden');
      // ensure the fade is visible before navigating
      await new Promise(r=>setTimeout(r,260));
      localStorage.setItem('playIntro','1');
      try { navigator.sendBeacon('http://localhost:6060/installer_shutdown'); } catch(_) {
        try { fetch('http://localhost:6060/installer_shutdown', { method:'POST', mode:'no-cors', keepalive:true }); } catch(_){}
      }
      const target = mainAlreadyRunning ? 'http://localhost:9876/?reopen=1' : 'http://localhost:9876/';
      location.replace(target);
    }
    startPinging();
    poll();
