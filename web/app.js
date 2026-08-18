    /* helpers */
    const MAX_TASKS = 50;
    const STORAGE_WARN_GB = 1;
    const STORAGE_BLOCK_GB = 0.5;
    const MEMORY_LIMIT_RATIO = 0.9;
    const MEMORY_LIMIT_MB = 700;
    const FIVE_HOURS_SEC = 5 * 60 * 60;
    const LONG_TRACK_SEC = 15 * 60;

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const modeSwitcher = document.getElementById('mode-switcher');
    const modeSwitcherPill = document.getElementById('mode-switcher-pill');
    const modeChoices = document.getElementById('mode-choices');
    const modesCard = document.getElementById('modes-card');
    const queue     = document.getElementById('queue');
    const clearBtn  = document.getElementById('clear-btn');
    const resumeQueueBtn = document.getElementById('resume-queue-btn');
    const topVig    = document.getElementById('top-vig');
    const botVig    = document.getElementById('bot-vig');
    const bottomPad = document.getElementById('queue-bottom-pad');
    const template  = document.getElementById('item-template');
    const title     = document.getElementById('title');
    const spacer    = document.getElementById('queue-spacer');
    const appShell  = document.getElementById('app-shell');
    const historyBtn = document.getElementById('history-btn');
    const startBtn  = document.getElementById('start-btn');
    const windowCloseBtn = document.getElementById('window-close-btn');
    const windowMinimizeBtn = document.getElementById('window-minimize-btn');
    const windowFullscreenBtn = document.getElementById('window-fullscreen-btn');
    let outputFormatSelect = null;
    let multiStemExportSelect = null;
    let previousFilesRetentionSelect = null;
    let previousFilesRetentionButton = null;
    let previousFilesRetentionMenu = null;
    let previousFilesRetentionLabel = null;
    let previousFilesLimitInput = null;
    let previousFilesWarnInput = null;
    let editorSnapDistanceInput = null;
    let outputFolderInput = null;
    let outputSameAsInput = null;
    let outputFolderChoose = null;
    let outputFolderOpen = null;
    let videoAudioOnly = null;
    let nerdStuffToggle = null;
    let nerdStuffWrap = null;
    let lanAccessHeading = null;
    let lanCopyLocalBtn = null;
    let lanCopyIpBtn = null;
    let lanLocalText = null;
    let lanIpText = null;
    let lanPasscodeEnabled = null;
    let lanPasscodeInput = null;
    let lanPasscodeIndicator = null;
    let lanPasscodeVisibilityBtn = null;
    let lanPasscodeDirty = false;
    let lanPasscodeDraft = '';
    let lanPasscodeVisible = false;
    let lanPasscodeTtl = null;
    let lanPasscodeTtlButton = null;
    let lanPasscodeTtlMenu = null;
    let lanPasscodeTtlLabel = null;
    let lanPasscodeWrap = null;
    let lanPasscodeTtlWrap = null;
    let portStatusText = null;
    let modelsList = null;
    let modelsNote = null;
    let modelsDownloadBtn = null;
    let modelsCtaWrap = null;
    let modelsFolderBtn = null;
    let modelsProgressWrap = null;
    let modelsTotal = null;
    let modelsProgressBar = null;
    let modelsProgressMeta = null;
    let outputFormatButton = null;
    let outputFormatMenu = null;
    let outputFormatLabel = null;
    let multiStemExportButton = null;
    let multiStemExportMenu = null;
    let multiStemExportLabel = null;
    let presetSettingsBtn = null;
    let presetSettingsOverlay = null;
    let boostHarmoniesBackgroundSlider = null;
    let boostHarmoniesBaseSlider = null;
    let boostHarmoniesBackgroundValue = null;
    let boostHarmoniesBaseValue = null;
    let boostGuitarOverlaySlider = null;
    let boostGuitarBaseSlider = null;
    let boostGuitarOverlayValue = null;
    let boostGuitarBaseValue = null;
    let settingsScrollIndicator = null;
    let settingsScrollBody = null;
    let settingsScrollThumb = null;
    let detachAppShellSoftScroll = null;
    const structureCheck = null;
    const structurelessCheck = null;
    let storageBlocked = false;
    let memoryBlocked = false;
    let modelsBlocked = false;
    let lastStorageWarning = 0;
    let lastMemoryWarning = 0;
    let modelStatusPoll = null;
    let releaseStatusPoll = null;
    let activeReleaseOverlay = null;
    let activeReleaseVersion = '';
    let activePortOverlay = null;
    let activeModelOverlay = null;
    let activeHostClosedOverlay = null;
    let lanRuntimeHeartbeat = null;
    let lanRuntimeFailures = 0;
    let lastModelStatus = null;
    let modelPreviewActive = false;
    let modelPreviewTimer = null;
    let nerdStuffExpanded = false;
    let copyResetTimers = new WeakMap();
    let activeModelOverlayUi = null;
    let activeReleaseOverlayUi = null;
    let activeModelToast = null;
    let activeReleaseToast = null;
    let activeModelReminderToast = null;
    let modelToastTimer = null;
    let releaseToastTimer = null;
    let modelDownloadIntentActive = false;
    let modelDownloadObservedActive = false;
    let ffmpegNoticeShown = false;
    let fullscreenTransitionTimer = null;
    let fullscreenResizeTimer = null;
    let tooltipEl = null;
    let tooltipTimer = null;
    let tooltipTarget = null;
    const MODE_TAB_LABELS = {
      single: 'single',
      multi: 'multi',
      presets: 'presets',
    };
    const BS_6S_TOOLTIP = `splits into:\n- bass\n- drums\n- other\n- vocals\n- guitar\n- piano\n- can be slow but provides good quality splits`;
    const DRUMSEP_6S_TOOLTIP = `splits into:\n- crash\n- hh\n- kick\n- ride\n- snare\n- toms\n- slightly cleaner than 4s`;
    const DRUMSEP_4S_TOOLTIP = `splits into:\n- cymbals\n- kick\n- snare\n- toms`;
    const MODEL_LABELS = {
      vocals: 'vocals',
      instrumental: 'instrumental',
      guitar: 'guitar',
      mel_band_karaoke: 'bg vocal',
      denoise: 'denoise',
      bs_roformer_6s: 'full mix',
      htdemucs_ft_drums: 'drums',
      htdemucs_ft_bass: 'bass',
      htdemucs_ft_other: 'other',
      htdemucs_6s: 'full mix faster',
      drumsep_6s: 'drum split - 6',
      drumsep_4s: 'drum split - 4',
    };
    const MODEL_CHECKLIST_ORDER = [
      'vocals',
      'instrumental',
      'guitar',
      'mel_band_karaoke',
      'denoise',
      'bs_roformer_6s',
      'htdemucs_ft_drums',
      'htdemucs_ft_bass',
      'htdemucs_ft_other',
      'htdemucs_6s',
      'drumsep_6s',
      'drumsep_4s',
    ];
    const EXCLUSIVE_STEM_MODES = new Set([
      'guitar',
      'mel_band_karaoke',
      'bs_roformer_6s',
      'htdemucs_ft_drums',
      'htdemucs_ft_bass',
      'htdemucs_ft_other',
      'htdemucs_6s',
      'drumsep_6s',
      'drumsep_4s',
    ]);
    const STEM_TO_REQUIRED_MODELS = {
      vocals: ['vocals'],
      instrumental: ['instrumental'],
      guitar: ['guitar'],
      mel_band_karaoke: ['vocals', 'mel_band_karaoke'],
      bs_roformer_6s: ['bs_roformer_6s'],
      htdemucs_ft_drums: ['htdemucs_ft_drums'],
      htdemucs_ft_bass: ['htdemucs_ft_bass'],
      htdemucs_ft_other: ['guitar', 'htdemucs_ft_other'],
      htdemucs_6s: ['htdemucs_6s'],
      drumsep_6s: ['drumsep_6s'],
      drumsep_4s: ['drumsep_4s'],
      all_stems: ['vocals', 'instrumental', 'mel_band_karaoke', 'bs_roformer_6s', 'drumsep_6s'],
      denoise: ['denoise'],
      boost_harmonies: ['vocals', 'mel_band_karaoke'],
    };
    const MODE_TAB_OPTIONS = {
      single: [
        { kind: 'stem', id: 'vocals', label: 'vocal', tooltip: 'recommended vocal split' },
        { kind: 'stem', id: 'instrumental', label: 'instrumental', tooltip: 'recommended instrumental split' },
        { kind: 'stem', id: 'htdemucs_ft_drums', label: 'drums', tooltip: 'fast drums' },
        { kind: 'stem', id: 'htdemucs_ft_bass', label: 'bass', tooltip: 'fast bass' },
        { kind: 'stem', id: 'guitar', label: 'guitar', tooltip: 'medium guitar split' },
        { kind: 'stem', id: 'htdemucs_ft_other', label: 'other', tooltip: 'other stem with guitar removed first' },
      ],
      multi: [
        { kind: 'stem', id: 'bs_roformer_6s', label: 'full mix', tooltip: BS_6S_TOOLTIP },
        { kind: 'stem', id: 'htdemucs_6s', label: 'full mix faster', tooltip: 'faster full mix split' },
        { kind: 'stem', id: 'drumsep_4s', label: 'drum split - 4', tooltip: DRUMSEP_4S_TOOLTIP },
        { kind: 'stem', id: 'drumsep_6s', label: 'drum split - 6', tooltip: DRUMSEP_6S_TOOLTIP },
      ],
      presets: [
        { kind: 'preset', id: 'all_stems', label: 'all stems', tooltip: 'full stem graph' },
        { kind: 'preset', id: 'denoise', label: 'denoise', tooltip: 'denoise' },
        { kind: 'preset', id: 'mel_band_karaoke', label: 'bg vocal', tooltip: 'background vocal split' },
        { kind: 'preset', id: 'boost_harmonies', label: 'boost harmonies', tooltip: 'boost harmonies' },
      ],
    };
    const MODE_TO_TAB = {
      vocals: 'single',
      instrumental: 'single',
      guitar: 'single',
      mel_band_karaoke: 'presets',
      htdemucs_ft_drums: 'single',
      htdemucs_ft_bass: 'single',
      htdemucs_ft_other: 'single',
      bs_roformer_6s: 'multi',
      htdemucs_6s: 'multi',
      drumsep_6s: 'multi',
      drumsep_4s: 'multi',
      all_stems: 'presets',
      denoise: 'presets',
      boost_harmonies: 'presets',
    };
    const PRESET_CONFIGS = {
      boost_harmonies: {
        label: 'boost harmonies',
        stem: 'boost_harmonies',
        overlayKey: 'boost_harmonies_background_vocals_gain_db',
        baseKey: 'boost_harmonies_base_song_gain_db',
        overlayLabel: 'background vocals',
        defaultOverlayGain: 3,
        defaultBaseGain: -3,
      },
    };
    const QUEUE_STEM_MENU_GROUPS = ['single', 'multi', 'presets'].map((tabKey) => ({
      key: tabKey,
      label: MODE_TAB_LABELS[tabKey] || tabKey,
      options: (MODE_TAB_OPTIONS[tabKey] || [])
        .filter((option) => tabKey === 'presets' ? option.kind === 'preset' : option.kind === 'stem')
        .map((option) => ({
          modeKey: option.id,
          label: option.label,
          stems: [option.id === 'denoise' ? 'preset_denoise' : option.id],
        })),
    }));
    const QUEUE_STEM_OPTION_BY_MODE = new Map(
      QUEUE_STEM_MENU_GROUPS.flatMap((group) => group.options.map((option) => [option.modeKey, option]))
    );
    const QUEUE_STEM_MODE_ORDER = QUEUE_STEM_MENU_GROUPS.flatMap((group) => group.options.map((option) => option.modeKey));
    const OUTPUT_FORMAT_LABELS = {
      same_as_input: 'same as input',
      mp3_320: '320kb mp3',
      mp3_128: '128kb mp3',
      wav: 'wav',
      m4a: 'm4a',
      flac: 'flac',
    };
    const TTL_LABELS = {
      '15m': '15m',
      '1d': '1d',
      '1w': '1w',
    };
    const PREVIOUS_FILES_RETENTION_LABELS = {
      '12h': '12h',
      '1d': '1d',
      '3d': '3d',
      '1w': '1w',
      '2w': '2w',
      '1mo': '1mo',
      '3mo': '3mo',
      '6mo': '6mo',
    };
    const MULTI_STEM_EXPORT_LABELS = {
      zip: 'zip',
      separate: 'separate files',
    };
    const MODEL_SOURCE_URLS = {
      vocals: 'https://huggingface.co/becruily/mel-band-roformer-vocals',
      instrumental: 'https://huggingface.co/becruily/mel-band-roformer-instrumental',
      guitar: 'https://huggingface.co/becruily/mel-band-roformer-guitar',
      mel_band_karaoke: 'https://huggingface.co/becruily/mel-band-roformer-karaoke',
      denoise: 'https://huggingface.co/jarredou/aufr33_MelBand_Denoise',
      bs_roformer_6s: 'https://huggingface.co/jarredou/BS-ROFO-SW-Fixed',
      htdemucs_ft_drums: 'https://github.com/facebookresearch/demucs',
      htdemucs_ft_bass: 'https://github.com/facebookresearch/demucs',
      htdemucs_ft_other: 'https://github.com/facebookresearch/demucs',
      htdemucs_6s: 'https://github.com/facebookresearch/demucs',
      drumsep_6s: 'https://github.com/jarredou/models/releases/tag/aufr33-jarredou_MDX23C_DrumSep_model_v0.1',
      drumsep_4s: 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.5',
    };
    const RELEASES_PAGE_URL = 'https://github.com/Skytheredhead/stemsplat/releases';
    const RELEASE_SESSION_DISMISS_KEY = 'releaseUpdateDismissedThisSession';
    const RELEASE_SNOOZE_STORAGE_KEY = 'releaseUpdateSnooze';
    const EDITOR_TRIM_CLIPBOARD_KEY = 'stemsplat.editorTrimClipboard';
    const EDITOR_SNAP_ENABLED_KEY = 'stemsplat.editorSnapEnabled';
    const EDITOR_TRACK_DRAFTS_KEY = 'stemsplat.editorTrackDrafts';
    const isLanClient = !['127.0.0.1', 'localhost', '::1'].includes(window.location.hostname);

    let startPressed = false;
    let startGeneration = 0;
    let startLock = false;
    let queueStarted = false;
    let activeModeTab = 'single';
    let modesCardHeightTimer = null;
    let modesCardHeightTransitionHandler = null;
    let selectedStemModes = new Set(['vocals']);
    let selectedPresetMode = null;
    let lastStemSelectionAnchor = 'vocals';
    let settingsState = {
      output_root: '',
      structure_mode: 'flat',
      output_format: 'same_as_input',
      multi_stem_export: 'zip',
      previous_files_retention: '1w',
      video_handling: 'audio_only',
      boost_harmonies_background_vocals_gain_db: 3,
      boost_harmonies_base_song_gain_db: -3,
      lan_passcode_enabled: false,
      lan_access_enabled: false,
      lan_passcode_configured: false,
      lan_passcode: '',
      lan_passcode_ttl: '1d',
      editor_snap_distance_ms: 180,
      runtime: null,
    };
    let presetSettingsSaveTimer = null;
    let previousFilesState = [];
    let previousFilesStorageState = null;
    let defaultOutputPath = '';
    let dirPickerInput = null;
    let desktopLaunchAnimated = false;
    let queueContextMenuEl = null;
    let queueGroupExpansionState = {};
    const editorWaveformPayloadCache = new Map();
    const editorWaveformFetches = new Map();

    function applyDesktopShellState(){
      const hasDesktopApi = !!(window.pywebview && window.pywebview.api);
      document.body.classList.toggle('desktop-shell', hasDesktopApi);
      if(settingsBtn){
        settingsBtn.disabled = isLanClient;
        settingsBtn.classList.toggle('dimmed-control', isLanClient);
      }
      if(presetSettingsBtn){
        presetSettingsBtn.disabled = isLanClient;
        presetSettingsBtn.classList.toggle('dimmed-control', isLanClient);
      }
      if(hasDesktopApi && !desktopLaunchAnimated){
        desktopLaunchAnimated = true;
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            document.body.classList.remove('app-launch-pre');
          });
        });
      }
    }

    function currentSelectedModeKey(){
      if(selectedPresetMode){
        return selectedPresetMode;
      }
      if(selectedStemModes.has('vocals') && selectedStemModes.has('instrumental')){
        return 'both_separate';
      }
      return Array.from(selectedStemModes)[0] || 'vocals';
    }

    function setSelectionForModeKey(modeKey){
      if(!modeKey) return;
      if(MODE_TO_TAB[modeKey] === 'presets'){
        selectedPresetMode = modeKey;
        selectedStemModes = new Set();
        return;
      }
      selectedPresetMode = null;
      if(modeKey === 'both_separate'){
        selectedStemModes = new Set(['vocals', 'instrumental']);
        lastStemSelectionAnchor = 'instrumental';
      }else{
        selectedStemModes = new Set([modeKey]);
        lastStemSelectionAnchor = modeKey;
      }
    }

    function visibleStemModeIdsForTab(tabKey = activeModeTab){
      return (MODE_TAB_OPTIONS[tabKey] || [])
        .filter((option) => option.kind === 'stem')
        .map((option) => option.id);
    }

    function stemSelectionAnchorForTab(tabKey = activeModeTab){
      const visibleModeIds = visibleStemModeIdsForTab(tabKey);
      if(visibleModeIds.includes(lastStemSelectionAnchor)){
        return lastStemSelectionAnchor;
      }
      return visibleModeIds.find((modeId) => selectedStemModes.has(modeId)) || visibleModeIds[0] || null;
    }

    function selectStemModeRange(modeKey){
      if(!modeKey) return;
      const visibleModeIds = visibleStemModeIdsForTab();
      const targetIndex = visibleModeIds.indexOf(modeKey);
      if(targetIndex === -1){
        setSelectionForModeKey(modeKey);
        return;
      }
      const anchorMode = stemSelectionAnchorForTab();
      const anchorIndex = anchorMode ? visibleModeIds.indexOf(anchorMode) : -1;
      if(anchorIndex === -1){
        setSelectionForModeKey(modeKey);
        return;
      }
      const start = Math.min(anchorIndex, targetIndex);
      const end = Math.max(anchorIndex, targetIndex);
      selectedPresetMode = null;
      selectedStemModes = new Set(visibleModeIds.slice(start, end + 1));
      lastStemSelectionAnchor = modeKey;
    }

    function ensureSelectionForActiveTab(){
      if(selectedPresetMode && MODE_TO_TAB[selectedPresetMode] === activeModeTab){
        return;
      }
      if(Array.from(selectedStemModes).some((mode) => MODE_TO_TAB[mode] === activeModeTab)){
        return;
      }
      if(selectedPresetMode || selectedStemModes.size){
        return;
      }
      const currentMode = currentSelectedModeKey();
      if(currentMode && MODE_TO_TAB[currentMode] === activeModeTab){
        return;
      }
      const firstOption = (MODE_TAB_OPTIONS[activeModeTab] || [])[0];
      if(firstOption){
        setSelectionForModeKey(firstOption.id);
      }
    }

    function toggleStemModeSelection(modeKey, additive = false){
      if(!modeKey) return;
      selectedPresetMode = null;
      const next = new Set(selectedStemModes);
      if(next.has(modeKey)){
        next.delete(modeKey);
      }else{
        if(!additive){
          setSelectionForModeKey(modeKey);
          return;
        }
        next.add(modeKey);
      }
      selectedStemModes = next;
      lastStemSelectionAnchor = modeKey;
    }

    function positionModeSwitcherPill(){
      if(!modeSwitcher || !modeSwitcherPill) return;
      const activeButton = modeSwitcher.querySelector(`.mode-switch-btn[data-tab="${activeModeTab}"]`) || modeSwitcher.querySelector('.mode-switch-btn.active');
      if(!activeButton) return;
      const switcherRect = modeSwitcher.getBoundingClientRect();
      const buttonRect = activeButton.getBoundingClientRect();
      modeSwitcherPill.style.width = `${buttonRect.width}px`;
      modeSwitcherPill.style.transform = `translateX(${buttonRect.left - switcherRect.left}px)`;
    }

    function updateModeTabUI(){
      const tabs = Array.from(document.querySelectorAll('.mode-switch-btn'));
      tabs.forEach((button) => {
        const active = button.dataset.tab === activeModeTab;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
      if(presetSettingsBtn){
        const visible = !isLanClient && activeModeTab === 'presets' && !!(selectedPresetMode && PRESET_CONFIGS[selectedPresetMode]);
        presetSettingsBtn.hidden = !visible;
        presetSettingsBtn.disabled = !visible || isLanClient;
        presetSettingsBtn.classList.toggle('dimmed-control', visible && isLanClient);
      }
      requestAnimationFrame(positionModeSwitcherPill);
    }

    function updateModeChoiceUI(){
      if(!modeChoices) return;
      modeChoices.querySelectorAll('.stem-choice, .preset-choice').forEach((button) => {
        const mode = button.dataset.mode || button.dataset.preset || '';
        const active = button.dataset.preset ? selectedPresetMode === mode : selectedStemModes.has(mode);
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
    }

    function clearModesCardHeightAnimation(){
      if(modesCardHeightTimer){
        clearTimeout(modesCardHeightTimer);
        modesCardHeightTimer = null;
      }
      if(modesCard && modesCardHeightTransitionHandler){
        modesCard.removeEventListener('transitionend', modesCardHeightTransitionHandler);
        modesCardHeightTransitionHandler = null;
      }
    }

    function finishModesCardHeightAnimation(){
      clearModesCardHeightAnimation();
      if(!modesCard) return;
      modesCard.classList.remove('modes-card-shifting');
      modesCard.classList.remove('modes-card-height-animating');
      modesCard.style.height = '';
    }

    function renderModeChoices(animate = false){
      if(!modeChoices) return;
      let startHeight = 0;
      if(modesCard){
        modesCard.classList.toggle('modes-card-shifting', animate);
        if(animate){
          clearModesCardHeightAnimation();
          startHeight = modesCard.getBoundingClientRect().height;
          modesCard.classList.remove('modes-card-height-animating');
          modesCard.style.height = `${startHeight}px`;
          void modesCard.offsetHeight;
        }else{
          finishModesCardHeightAnimation();
        }
      }
      modeChoices.innerHTML = '';
      (MODE_TAB_OPTIONS[activeModeTab] || []).forEach((option, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `${option.kind === 'preset' ? 'preset-choice' : 'stem-choice'} mode-choice-enter`;
        button.style.setProperty('--mode-item-index', String(index));
        button.dataset.tooltip = option.tooltip || option.label;
        if(option.kind === 'preset'){
          button.dataset.preset = option.id;
          const title = document.createElement('span');
          title.className = 'preset-choice-title';
          title.textContent = option.label;
          button.appendChild(title);
        }else{
          button.dataset.mode = option.id;
          button.textContent = option.label;
        }
        bindDelayedTooltip(button);
        modeChoices.appendChild(button);
      });
      updateModeTabUI();
      updateModeChoiceUI();
      if(animate && modesCard){
        modesCard.style.height = 'auto';
        const endHeight = modesCard.getBoundingClientRect().height;
        modesCard.style.height = `${startHeight}px`;
        void modesCard.offsetHeight;
        modesCard.classList.add('modes-card-height-animating');
        requestAnimationFrame(() => {
          modesCardHeightTransitionHandler = (event) => {
            if(event.target !== modesCard || event.propertyName !== 'height') return;
            finishModesCardHeightAnimation();
          };
          modesCard.addEventListener('transitionend', modesCardHeightTransitionHandler);
          modesCard.style.height = `${endHeight}px`;
        });
        modesCardHeightTimer = window.setTimeout(() => {
          finishModesCardHeightAnimation();
        }, 520);
      }else if(modesCard){
        finishModesCardHeightAnimation();
      }
    }

    function formatGainDb(value){
      const numeric = Number(value || 0);
      const sign = numeric > 0 ? '+' : '';
      return `${sign}${numeric.toFixed(1)} dB`;
    }

    function formatStorageSettingValue(value, fallback){
      const numeric = Number.isFinite(Number(value)) ? Number(value) : Number(fallback);
      return numeric.toFixed(1).replace(/\.0$/, '');
    }

    function coerceStorageSettingValue(value, fallback, minimum){
      const parsed = Number.parseFloat(String(value ?? '').trim());
      if(!Number.isFinite(parsed)) return Number(fallback);
      const clamped = Math.max(minimum, Math.min(1024, parsed));
      return Math.round(clamped * 10) / 10;
    }

    function formatStorageUsage(bytes){
      const numeric = Number(bytes || 0);
      return `${(numeric / (1024 ** 3)).toFixed(1).replace(/\.0$/, '')} GB`;
    }

    function formatDownloadRate(bytesPerSecond){
      const numeric = Number(bytesPerSecond || 0);
      if(!(numeric > 0)) return '';
      return `${(numeric / (1024 * 1024)).toFixed(1)} mb/s`;
    }

    function ensureTooltip(){
      if(tooltipEl) return tooltipEl;
      tooltipEl = document.createElement('div');
      tooltipEl.className = 'hover-tooltip';
      tooltipEl.hidden = true;
      document.body.appendChild(tooltipEl);
      return tooltipEl;
    }

    function hideTooltip(){
      if(tooltipTimer){
        clearTimeout(tooltipTimer);
        tooltipTimer = null;
      }
      tooltipTarget = null;
      if(!tooltipEl) return;
      tooltipEl.classList.remove('visible');
      tooltipEl.hidden = true;
    }

    function positionTooltip(target){
      if(!tooltipEl || !target) return;
      const rect = target.getBoundingClientRect();
      const pad = 12;
      const top = Math.max(pad, rect.top - tooltipEl.offsetHeight - 10);
      let left = rect.left + (rect.width / 2) - (tooltipEl.offsetWidth / 2);
      left = Math.max(pad, Math.min(left, window.innerWidth - tooltipEl.offsetWidth - pad));
      tooltipEl.style.left = `${Math.round(left)}px`;
      tooltipEl.style.top = `${Math.round(top)}px`;
    }

    function showTooltip(target){
      const message = target && target.dataset ? String(target.dataset.tooltip || '').trim() : '';
      if(!message) return;
      const el = ensureTooltip();
      el.textContent = message;
      el.hidden = false;
      positionTooltip(target);
      requestAnimationFrame(() => {
        positionTooltip(target);
        el.classList.add('visible');
      });
    }

    function bindDelayedTooltip(target){
      if(!target || target.dataset.tooltipBound === '1') return;
      target.dataset.tooltipBound = '1';
      const begin = () => {
        const message = String(target.dataset.tooltip || '').trim();
        if(!message) return;
        if(tooltipTimer){
          clearTimeout(tooltipTimer);
        }
        tooltipTarget = target;
        tooltipTimer = setTimeout(() => {
          tooltipTimer = null;
          if(tooltipTarget === target){
            showTooltip(target);
          }
        }, 500);
      };
      target.addEventListener('mouseenter', begin);
      target.addEventListener('mouseleave', hideTooltip);
      target.addEventListener('blur', hideTooltip);
      target.addEventListener('mousedown', hideTooltip);
    }

    function bindChoiceTooltips(){
      document.querySelectorAll('[data-tooltip]').forEach(bindDelayedTooltip);
    }

    function updatePresetRangeVisual(input){
      if(!input) return;
      const min = Number(input.min || -18);
      const max = Number(input.max || 18);
      const value = Number(input.value || 0);
      const pct = max === min ? 50 : ((value - min) / (max - min)) * 100;
      input.style.setProperty('--fill', `${Math.max(0, Math.min(100, pct))}%`);
    }

    function applyPresetSettingsUI(){
      const presetUi = {
        boost_harmonies: {
          overlaySlider: boostHarmoniesBackgroundSlider,
          baseSlider: boostHarmoniesBaseSlider,
          overlayValue: boostHarmoniesBackgroundValue,
          baseValue: boostHarmoniesBaseValue,
        },
      };
      Object.entries(PRESET_CONFIGS).forEach(([mode, config]) => {
        const ui = presetUi[mode];
        if(!ui) return;
        const overlayValue = settingsState[config.overlayKey] ?? config.defaultOverlayGain;
        const baseValue = settingsState[config.baseKey] ?? config.defaultBaseGain;
        if(ui.overlaySlider){
          ui.overlaySlider.value = String(overlayValue);
          updatePresetRangeVisual(ui.overlaySlider);
        }
        if(ui.baseSlider){
          ui.baseSlider.value = String(baseValue);
          updatePresetRangeVisual(ui.baseSlider);
        }
        if(ui.overlayValue){
          ui.overlayValue.textContent = formatGainDb(overlayValue);
        }
        if(ui.baseValue){
          ui.baseValue.textContent = formatGainDb(baseValue);
        }
      });
    }

    function schedulePresetSettingsPersist(patch){
      settingsState = { ...settingsState, ...patch };
      applyPresetSettingsUI();
      if(presetSettingsSaveTimer){
        clearTimeout(presetSettingsSaveTimer);
      }
      presetSettingsSaveTimer = setTimeout(() => {
        presetSettingsSaveTimer = null;
        persistSettings(patch, { showMissingPopup: false });
      }, 140);
    }

    function requiredModelKeysForStems(stems){
      const needed = new Set();
      (Array.isArray(stems) ? stems : []).forEach((stem) => {
        (STEM_TO_REQUIRED_MODELS[stem] || []).forEach((key) => needed.add(key));
      });
      return Array.from(needed);
    }

    function relevantMissingModels(status = lastModelStatus){
      const missing = new Set(Array.isArray(status && status.missing) ? status.missing : []);
      const needed = new Set(requiredModelKeysForStems(selectedStemUnion()));
      (Array.isArray(tasks) ? tasks : []).forEach((task) => {
        const stage = String(task && task.stage || '').toLowerCase();
        if(!task || !Array.isArray(task.stems) || ['done', 'error', 'stopped'].includes(stage)) return;
        requiredModelKeysForStems(task.stems).forEach((key) => needed.add(key));
      });
      return Array.from(needed).filter((key) => missing.has(key));
    }

    async function callDesktopAction(actionName){
      try{
        if(window.pywebview && window.pywebview.api && typeof window.pywebview.api[actionName] === 'function'){
          await window.pywebview.api[actionName]();
          return true;
        }
      }catch(err){
        console.warn(`${actionName} failed`, err);
      }
      return false;
    }

    function beginFullscreenTransition(){
      document.body.classList.add('fullscreen-transition');
      if(fullscreenTransitionTimer){
        clearTimeout(fullscreenTransitionTimer);
      }
      fullscreenTransitionTimer = setTimeout(() => {
        document.body.classList.remove('fullscreen-transition');
        fullscreenTransitionTimer = null;
      }, 560);
    }

    function settleFullscreenTransition(){
      if(fullscreenResizeTimer){
        clearTimeout(fullscreenResizeTimer);
      }
      fullscreenResizeTimer = setTimeout(() => {
        if(fullscreenTransitionTimer){
          clearTimeout(fullscreenTransitionTimer);
          fullscreenTransitionTimer = null;
        }
        document.body.classList.remove('fullscreen-transition');
      }, 180);
    }

    updateStartButton();
    function updateStartButton(){
      if(!startBtn) return;
      const hasItems = queueTaskCount() > 0;
      if(!hasItems || startLock || storageBlocked || memoryBlocked || modelsBlocked){
        startBtn.disabled = true;
        const icon = startBtn.querySelector('svg');
        if(icon){ icon.style.animation = 'none'; }
      }else{
        startBtn.disabled = false;
        const icon = startBtn.querySelector('svg');
        if(icon){ icon.style.animation = ''; }
      }
    }

    function applySettingsUI(){
      if(outputFormatSelect){
        outputFormatSelect.value = settingsState.output_format || 'same_as_input';
      }
      if(outputFormatLabel){
        outputFormatLabel.textContent = OUTPUT_FORMAT_LABELS[settingsState.output_format || 'same_as_input'] || 'same as input';
      }
      if(outputFormatMenu){
        outputFormatMenu.querySelectorAll('.fancy-select-option').forEach((option) => {
          const selected = option.dataset.value === (settingsState.output_format || 'same_as_input');
          option.dataset.selected = selected ? 'true' : 'false';
          option.setAttribute('aria-selected', selected ? 'true' : 'false');
        });
      }
      if(multiStemExportSelect){
        multiStemExportSelect.value = settingsState.multi_stem_export || 'zip';
      }
      if(multiStemExportLabel){
        multiStemExportLabel.textContent = MULTI_STEM_EXPORT_LABELS[settingsState.multi_stem_export || 'zip'] || 'zip';
      }
      if(multiStemExportMenu){
        multiStemExportMenu.querySelectorAll('.fancy-select-option').forEach((option) => {
          const selected = option.dataset.value === (settingsState.multi_stem_export || 'zip');
          option.dataset.selected = selected ? 'true' : 'false';
          option.setAttribute('aria-selected', selected ? 'true' : 'false');
        });
      }
      if(previousFilesRetentionSelect){
        previousFilesRetentionSelect.value = settingsState.previous_files_retention || '1w';
      }
      if(previousFilesRetentionLabel){
        previousFilesRetentionLabel.textContent = PREVIOUS_FILES_RETENTION_LABELS[settingsState.previous_files_retention || '1w'] || '1w';
      }
      if(previousFilesRetentionMenu){
        previousFilesRetentionMenu.querySelectorAll('.fancy-select-option').forEach((option) => {
          const selected = option.dataset.value === (settingsState.previous_files_retention || '1w');
          option.dataset.selected = selected ? 'true' : 'false';
          option.setAttribute('aria-selected', selected ? 'true' : 'false');
        });
      }
      if(previousFilesLimitInput){
        previousFilesLimitInput.value = formatStorageSettingValue(settingsState.previous_files_limit_gb, 10);
      }
      if(previousFilesWarnInput){
        previousFilesWarnInput.value = formatStorageSettingValue(settingsState.previous_files_warn_gb, 8);
      }
      if(editorSnapDistanceInput){
        editorSnapDistanceInput.value = String(Math.max(0, Math.min(2000, Number(settingsState.editor_snap_distance_ms) || 180)));
      }
      if(videoAudioOnly){
        videoAudioOnly.checked = true;
      }
      if(outputSameAsInput){
        outputSameAsInput.checked = !!settingsState.output_same_as_input;
      }
      if(outputFolderInput){
        outputFolderInput.value = settingsState.output_root || '';
        outputFolderInput.classList.toggle('dimmed-control', !!settingsState.output_same_as_input);
      }
      if(outputFolderChoose){
        outputFolderChoose.classList.toggle('dimmed-control', !!settingsState.output_same_as_input);
      }
      if(outputFolderOpen){
        outputFolderOpen.classList.toggle('dimmed-control', !!settingsState.output_same_as_input);
      }
      if(lanPasscodeEnabled){
        lanPasscodeEnabled.checked = !!settingsState.lan_access_enabled;
      }
      if(lanPasscodeInput){
        if(!lanPasscodeDirty){
          lanPasscodeDraft = String(settingsState.lan_passcode || '');
        }
        lanPasscodeInput.value = lanPasscodeDirty ? lanPasscodeDraft : String(settingsState.lan_passcode || '');
        lanPasscodeInput.type = lanPasscodeVisible ? 'text' : 'password';
        lanPasscodeInput.disabled = !settingsState.lan_passcode_enabled;
        lanPasscodeInput.classList.toggle('dimmed-control', !settingsState.lan_passcode_enabled);
      }
      if(lanPasscodeVisibilityBtn){
        lanPasscodeVisibilityBtn.disabled = !settingsState.lan_passcode_enabled || !String(lanPasscodeDraft || '').trim();
        lanPasscodeVisibilityBtn.classList.toggle('visible', lanPasscodeVisible);
      }
      if(lanPasscodeIndicator){
        const enabled = !!settingsState.lan_passcode_enabled;
        const draftPasscode = String(lanPasscodeDirty ? lanPasscodeDraft : (lanPasscodeInput ? lanPasscodeInput.value || '' : '')).trim();
        const hasDraft = !!draftPasscode || !!settingsState.lan_passcode_configured;
        const isPending = enabled && hasDraft && lanPasscodeDirty;
        const isConfirmed = enabled && !!settingsState.lan_passcode_configured && !lanPasscodeDirty;
        lanPasscodeIndicator.disabled = !enabled || !hasDraft;
        lanPasscodeIndicator.classList.toggle('pending', isPending);
        lanPasscodeIndicator.classList.toggle('confirmed', isConfirmed);
      }
      if(lanPasscodeWrap){
        lanPasscodeWrap.classList.toggle('dimmed-control', !settingsState.lan_passcode_enabled);
      }
      if(lanPasscodeTtl){
        lanPasscodeTtl.value = settingsState.lan_passcode_ttl || '1d';
        lanPasscodeTtl.disabled = !settingsState.lan_passcode_enabled;
      }
      if(lanPasscodeTtlLabel){
        lanPasscodeTtlLabel.textContent = TTL_LABELS[settingsState.lan_passcode_ttl || '1d'] || '1d';
      }
      if(lanPasscodeTtlMenu){
        lanPasscodeTtlMenu.querySelectorAll('.fancy-select-option').forEach((option) => {
          const selected = option.dataset.value === (settingsState.lan_passcode_ttl || '1d');
          option.dataset.selected = selected ? 'true' : 'false';
          option.setAttribute('aria-selected', selected ? 'true' : 'false');
        });
      }
      if(lanPasscodeTtlButton){
        lanPasscodeTtlButton.disabled = !settingsState.lan_passcode_enabled;
        lanPasscodeTtlButton.classList.toggle('dimmed-control', !settingsState.lan_passcode_enabled);
      }
      if(lanPasscodeTtlWrap){
        lanPasscodeTtlWrap.classList.toggle('dimmed-control', !settingsState.lan_passcode_enabled);
      }
      if(settingsBtn){
        settingsBtn.disabled = isLanClient;
        settingsBtn.classList.toggle('dimmed-control', isLanClient);
      }
      if(presetSettingsBtn){
        presetSettingsBtn.disabled = isLanClient;
        presetSettingsBtn.classList.toggle('dimmed-control', isLanClient);
      }
      applyRuntimeUI(settingsState.runtime || null);
      applyPresetSettingsUI();
    }

    function closeOutputFormatMenu(){
      if(outputFormatButton){
        outputFormatButton.setAttribute('aria-expanded', 'false');
      }
      if(outputFormatMenu){
        outputFormatMenu.classList.remove('open');
      }
    }

    function closeMultiStemExportMenu(){
      if(multiStemExportButton){
        multiStemExportButton.setAttribute('aria-expanded', 'false');
      }
      if(multiStemExportMenu){
        multiStemExportMenu.classList.remove('open');
      }
    }

    function closeLanTtlMenu(){
      if(lanPasscodeTtlButton){
        lanPasscodeTtlButton.setAttribute('aria-expanded', 'false');
      }
      if(lanPasscodeTtlMenu){
        lanPasscodeTtlMenu.classList.remove('open');
      }
    }

    function closePreviousFilesRetentionMenu(){
      if(previousFilesRetentionButton){
        previousFilesRetentionButton.setAttribute('aria-expanded', 'false');
      }
      if(previousFilesRetentionMenu){
        previousFilesRetentionMenu.classList.remove('open');
      }
    }

    function toggleOutputFormatMenu(forceOpen = null){
      if(!outputFormatButton || !outputFormatMenu) return;
      const shouldOpen = forceOpen === null
        ? outputFormatButton.getAttribute('aria-expanded') !== 'true'
        : !!forceOpen;
      if(shouldOpen){
        closeMultiStemExportMenu();
        closeLanTtlMenu();
        closePreviousFilesRetentionMenu();
      }
      outputFormatButton.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
      outputFormatMenu.classList.toggle('open', shouldOpen);
    }

    function toggleMultiStemExportMenu(forceOpen = null){
      if(!multiStemExportButton || !multiStemExportMenu) return;
      const shouldOpen = forceOpen === null
        ? multiStemExportButton.getAttribute('aria-expanded') !== 'true'
        : !!forceOpen;
      if(shouldOpen){
        closeOutputFormatMenu();
        closeLanTtlMenu();
        closePreviousFilesRetentionMenu();
      }
      multiStemExportButton.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
      multiStemExportMenu.classList.toggle('open', shouldOpen);
    }

    function toggleLanTtlMenu(forceOpen = null){
      if(!lanPasscodeTtlButton || !lanPasscodeTtlMenu || lanPasscodeTtlButton.disabled) return;
      const shouldOpen = forceOpen === null
        ? lanPasscodeTtlButton.getAttribute('aria-expanded') !== 'true'
        : !!forceOpen;
      if(shouldOpen){
        closeOutputFormatMenu();
        closeMultiStemExportMenu();
        closePreviousFilesRetentionMenu();
      }
      lanPasscodeTtlButton.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
      lanPasscodeTtlMenu.classList.toggle('open', shouldOpen);
    }

    function togglePreviousFilesRetentionMenu(forceOpen = null){
      if(!previousFilesRetentionButton || !previousFilesRetentionMenu) return;
      const shouldOpen = forceOpen === null
        ? previousFilesRetentionButton.getAttribute('aria-expanded') !== 'true'
        : !!forceOpen;
      if(shouldOpen){
        closeOutputFormatMenu();
        closeMultiStemExportMenu();
        closeLanTtlMenu();
      }
      previousFilesRetentionButton.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
      previousFilesRetentionMenu.classList.toggle('open', shouldOpen);
    }

    function setNerdStuffExpanded(expanded, { immediate = false } = {}){
      nerdStuffExpanded = !!expanded;
      if(nerdStuffToggle){
        nerdStuffToggle.setAttribute('aria-expanded', nerdStuffExpanded ? 'true' : 'false');
      }
      if(!nerdStuffWrap) return;
      const targetHeight = nerdStuffExpanded ? `${nerdStuffWrap.scrollHeight}px` : '0px';
      if(immediate){
        nerdStuffWrap.classList.toggle('open', nerdStuffExpanded);
        nerdStuffWrap.style.maxHeight = targetHeight;
        return;
      }
      if(nerdStuffExpanded){
        nerdStuffWrap.classList.add('open');
        nerdStuffWrap.style.maxHeight = `${nerdStuffWrap.scrollHeight}px`;
        requestAnimationFrame(() => {
          if(nerdStuffWrap && nerdStuffExpanded){
            nerdStuffWrap.style.maxHeight = `${nerdStuffWrap.scrollHeight}px`;
          }
        });
      }else{
        nerdStuffWrap.style.maxHeight = `${nerdStuffWrap.scrollHeight}px`;
        requestAnimationFrame(() => {
          if(!nerdStuffWrap) return;
          nerdStuffWrap.classList.remove('open');
          nerdStuffWrap.style.maxHeight = '0px';
        });
      }
    }

    function guardClick(el, handler, cooldown = 600){
      if(!el || typeof handler !== 'function') return;
      if(el.__guardedHandler){
        el.removeEventListener('click', el.__guardedHandler);
      }
      let busy = false;
      const wrapped = async (e) => {
        if(busy){
          e.preventDefault();
          e.stopPropagation();
          return;
        }
        busy = true;
        try{
          await handler(e);
        }finally{
          setTimeout(() => { busy = false; }, cooldown);
        }
      };
      el.__guardedHandler = wrapped;
      el.addEventListener('click', wrapped);
    }

    async function loadSettings(){
      try{
        const res = await fetch('/settings');
        if(!res.ok) return;
        const data = await res.json();
        settingsState = { ...settingsState, ...data };
        lanPasscodeDirty = false;
        lanPasscodeDraft = String(settingsState.lan_passcode || '');
        defaultOutputPath = data.output_root || defaultOutputPath;
        applySettingsUI();
      }catch(err){
        console.warn('failed to load settings', err);
      }
    }

    async function persistSettings(patch, {showMissingPopup = true} = {}){
      settingsState = { ...settingsState, ...patch };
      try{
        let res = null;
        for(let attempt = 0; attempt < 2; attempt += 1){
          res = await fetch('/api/settings', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...patch, version: settingsState.version }),
          });
          if(res.status !== 409 || attempt > 0) break;
          await loadSettings();
          settingsState = { ...settingsState, ...patch };
        }
        if(!res || !res.ok){
          if(showMissingPopup) showPopup('settings could not be saved');
          await loadSettings();
          return false;
        }
        const data = await res.json();
        settingsState = { ...settingsState, ...data };
        defaultOutputPath = defaultOutputPath || data.output_root;
        applySettingsUI();
        return true;
      }catch(err){
        console.warn('settings update failed', err);
        if(showMissingPopup) showPopup('settings could not be saved');
        await loadSettings();
        return false;
      }
    }

    async function persistLanConfig({ enabled, passcode = '' }){
      const response = await fetch('/api/lan/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: !!enabled,
          passcode: String(passcode || ''),
          session_ttl: settingsState.lan_passcode_ttl || '1d',
        }),
      });
      if(!response.ok){
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload && payload.detail && payload.detail.message
          ? String(payload.detail.message)
          : 'could not update LAN access');
      }
      const payload = await response.json().catch(() => ({}));
      await loadSettings();
      return payload;
    }

    async function checkStorage(){
      if(!navigator.storage || !navigator.storage.estimate) return true;
      try{
        const {usage = 0, quota = 0} = await navigator.storage.estimate();
        const free = Math.max(0, quota - usage);
        const freeGb = free / (1024 ** 3);
        if(freeGb < STORAGE_BLOCK_GB){
          storageBlocked = true;
          showPopup('storage critically low (<0.5 GB). uploads are blocked');
          updateStartButton();
          return false;
        }
        if(freeGb < STORAGE_WARN_GB){
          const now = Date.now();
          if(now - lastStorageWarning > 10000){
            showPopup('storage low (<1 GB remaining)');
            lastStorageWarning = now;
          }
        }
        storageBlocked = false;
        updateStartButton();
      }catch(_){}
      return !storageBlocked;
    }

    function checkMemory(){
      const perf = performance || window.performance;
      const mem = perf && perf.memory;
      if(!mem) return true;
      const { jsHeapSizeLimit = 0, usedJSHeapSize = 0 } = mem;
      const limit = jsHeapSizeLimit ? jsHeapSizeLimit * MEMORY_LIMIT_RATIO : MEMORY_LIMIT_MB * 1024 * 1024;
      if(usedJSHeapSize > limit){
        memoryBlocked = true;
        const now = Date.now();
        if(now - lastMemoryWarning > 10000){
          showPopup('memory use is high; uploads paused to prevent leaks');
          lastMemoryWarning = now;
        }
        updateStartButton();
        return false;
      }
      memoryBlocked = false;
      updateStartButton();
      return true;
    }

    function estimateDecodeMb(seconds){
      // Roughly 0.34 MB/sec for 44.1kHz stereo float
      return seconds * 0.34;
    }

    function hasMemoryForDuration(seconds){
      if(seconds <= LONG_TRACK_SEC) return true;
      const perf = performance || window.performance;
      const mem = perf && perf.memory;
      if(!mem || !mem.jsHeapSizeLimit) return true;
      const { jsHeapSizeLimit = 0, usedJSHeapSize = 0 } = mem;
      const freeMb = Math.max(0, jsHeapSizeLimit - usedJSHeapSize) / (1024 * 1024);
      const needMb = estimateDecodeMb(seconds) * 1.5; // headroom
      if(freeMb <= needMb){
        showPopup('song is too long for available memory right now');
        return false;
      }
      return true;
    }

    function getAudioDuration(file){
      return new Promise((resolve) => {
        let url = null;
        try{
          const audio = document.createElement('audio');
          audio.preload = 'metadata';
          url = URL.createObjectURL(file);
          const cleanup = () => {
            if(url){ URL.revokeObjectURL(url); url = null; }
            audio.removeAttribute('src');
            audio.load();
          };
          const timeout = setTimeout(() => { cleanup(); resolve(-1); }, 3000);
          audio.onloadedmetadata = () => {
            clearTimeout(timeout);
            const d = audio.duration;
            cleanup();
            resolve(isFinite(d) ? d : -1);
          };
          audio.onerror = () => { clearTimeout(timeout); cleanup(); resolve(-1); };
          audio.src = url;
        }catch(_){
          if(url){ URL.revokeObjectURL(url); }
          resolve(-1);
        }
      });
    }

    const dialogReturnFocus = new WeakMap();

    function dialogFocusableElements(overlay){
      if(!overlay) return [];
      return Array.from(overlay.querySelectorAll('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'))
        .filter((element) => !element.hidden && element.getClientRects().length > 0);
    }

    function activateDialogAccessibility(overlay, preferredFocus){
      if(!overlay) return;
      if(document.activeElement instanceof HTMLElement){
        dialogReturnFocus.set(overlay, document.activeElement);
      }
      requestAnimationFrame(() => {
        const focusTarget = preferredFocus || dialogFocusableElements(overlay)[0];
        if(focusTarget){
          try { focusTarget.focus({ preventScroll: true }); } catch(_) { focusTarget.focus(); }
        }
      });
    }

    function restoreDialogFocus(overlay){
      if(!overlay) return;
      const focusTarget = dialogReturnFocus.get(overlay);
      dialogReturnFocus.delete(overlay);
      if(focusTarget && focusTarget.isConnected && !focusTarget.disabled){
        try { focusTarget.focus({ preventScroll: true }); } catch(_) { focusTarget.focus(); }
      }
    }

    function trapDialogFocus(event){
      const overlay = event.currentTarget;
      if(!overlay || overlay.classList.contains('hidden')) return;
      if(event.key === 'Escape'){
        event.preventDefault();
        if(overlay.id === 'preset-settings-overlay') closePresetSettings();
        else if(overlay.id === 'settings-overlay') closeSettings();
        return;
      }
      if(event.key !== 'Tab') return;
      const focusable = dialogFocusableElements(overlay);
      if(!focusable.length){
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if(event.shiftKey && document.activeElement === first){
        event.preventDefault();
        last.focus();
      }else if(!event.shiftKey && document.activeElement === last){
        event.preventDefault();
        first.focus();
      }
    }

    function installAccessibleListbox(button, menu){
      if(!button || !menu || button.dataset.keyboardReady === 'true') return;
      button.dataset.keyboardReady = 'true';
      const options = Array.from(menu.querySelectorAll('.fancy-select-option'));
      options.forEach((option) => {
        option.setAttribute('role', 'option');
        option.tabIndex = -1;
        if(!option.hasAttribute('aria-selected')) option.setAttribute('aria-selected', 'false');
      });
      const focusOption = (offset, absolute = false) => {
        if(!options.length) return;
        const activeIndex = Math.max(0, options.indexOf(document.activeElement));
        const nextIndex = absolute ? offset : (activeIndex + offset + options.length) % options.length;
        options[Math.max(0, Math.min(options.length - 1, nextIndex))].focus();
      };
      const ensureOpen = () => {
        if(button.getAttribute('aria-expanded') !== 'true') button.click();
      };
      button.addEventListener('keydown', (event) => {
        if(!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        ensureOpen();
        const selectedIndex = options.findIndex((option) => option.getAttribute('aria-selected') === 'true');
        if(event.key === 'Home') focusOption(0, true);
        else if(event.key === 'End') focusOption(options.length - 1, true);
        else if(selectedIndex >= 0) focusOption(Math.max(0, Math.min(options.length - 1, selectedIndex + (event.key === 'ArrowDown' ? 1 : -1))), true);
        else focusOption(event.key === 'ArrowDown' ? 0 : options.length - 1, true);
      });
      menu.addEventListener('keydown', (event) => {
        if(event.key === 'ArrowDown' || event.key === 'ArrowUp'){
          event.preventDefault();
          focusOption(event.key === 'ArrowDown' ? 1 : -1);
        }else if(event.key === 'Home' || event.key === 'End'){
          event.preventDefault();
          focusOption(event.key === 'Home' ? 0 : options.length - 1, true);
        }else if(event.key === 'Enter' || event.key === ' '){
          const activeOption = options.find((option) => option === document.activeElement);
          if(activeOption){
            event.preventDefault();
            activeOption.click();
          }
        }else if(event.key === 'Escape'){
          event.preventDefault();
          menu.classList.remove('open');
          button.setAttribute('aria-expanded', 'false');
          button.focus();
        }
      });
    }

    // settings overlay refs (assigned on load)
    let settingsOverlay = null;
    function openPresetSettings(){
      if(!presetSettingsOverlay) presetSettingsOverlay = document.getElementById('preset-settings-overlay');
      const bg = document.getElementById('preset-overlay-bg');
      const card = document.getElementById('preset-settings-card');
      if(presetSettingsOverlay){
        presetSettingsOverlay.classList.remove('hidden');
        applyPresetSettingsUI();
        activateDialogAccessibility(presetSettingsOverlay, document.getElementById('preset-settings-close'));
        if(bg){
          bg.classList.remove('anim-out');
          bg.classList.remove('anim');
          void bg.offsetWidth;
          bg.classList.add('anim');
        }
        if(card){
          card.classList.remove('settings-card-out');
          card.classList.remove('opacity-0');
          void card.offsetWidth;
          card.classList.add('settings-card-in');
        }
      }
    }

    function closePresetSettings(){
      if(!presetSettingsOverlay) presetSettingsOverlay = document.getElementById('preset-settings-overlay');
      const bg = document.getElementById('preset-overlay-bg');
      const card = document.getElementById('preset-settings-card');
      if(presetSettingsOverlay){
        if(bg){
          bg.classList.remove('anim');
          bg.classList.remove('anim-out');
          void bg.offsetWidth;
          bg.classList.add('anim-out');
        }
        if(card){
          card.classList.remove('settings-card-in');
          card.classList.add('settings-card-out');
        }
        setTimeout(() => {
          if(card){
            card.classList.remove('settings-card-out');
            card.classList.add('opacity-0');
          }
          if(bg){ bg.classList.remove('anim-out'); }
          presetSettingsOverlay.classList.add('hidden');
          restoreDialogFocus(presetSettingsOverlay);
        }, 180);
      }
    }

    async function exportAdjustedPresetMix(task, presetSettings){
      if(!task || !task.id) throw new Error('missing task');
      const res = await fetch(`/api/tasks/${task.id}/preset_mix`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          overlay_gain_db: presetSettings.overlay_gain_db,
          base_song_gain_db: presetSettings.base_song_gain_db,
        }),
      });
      let payload = null;
      try { payload = await res.json(); } catch(_) { payload = null; }
      if(!res.ok){
        const message = payload && payload.detail && payload.detail.message
          ? payload.detail.message
          : 'preset export failed';
        throw new Error(message);
      }
      task.preset_settings = payload && payload.preset_settings ? payload.preset_settings : { ...presetSettings };
      task.can_adjust_preset = !!(payload && payload.can_adjust_preset);
      task.out_dir = payload && payload.out_dir ? payload.out_dir : task.out_dir;
      task.outputs = payload && Array.isArray(payload.outputs) ? payload.outputs : (task.outputs || []);
      saveTasks();
      updateUI();
      return payload;
    }

    function taskPresetMode(task){
      if(task && typeof task.mode === 'string' && PRESET_CONFIGS[task.mode.replace(/^preset_/, '')]){
        return task.mode.replace(/^preset_/, '');
      }
      if(task && Array.isArray(task.stems)){
        return Object.keys(PRESET_CONFIGS).find((mode) => task.stems.includes(PRESET_CONFIGS[mode].stem)) || null;
      }
      return null;
    }

    function openTaskPresetAdjuster(task){
      if(!canAdjustTaskPreset(task)) return;
      const presetMode = taskPresetMode(task);
      const presetConfig = presetMode ? PRESET_CONFIGS[presetMode] : null;
      if(!presetConfig) return;
      const { overlay, card } = createOverlayCard(presetConfig.label);
      card.style.maxWidth = '540px';
      const settings = taskPresetSettings(task);

      const shell = document.createElement('div');
      shell.className = 'preset-slider-shell';
      const heading = document.createElement('div');
      heading.className = 'preset-slider-topline';
      const headingLabel = document.createElement('div');
      headingLabel.className = 'preset-slider-label';
      headingLabel.textContent = presetConfig.label;
      heading.appendChild(headingLabel);
      const stack = document.createElement('div');
      stack.className = 'preset-slider-stack';
      const makeSliderBlock = (labelText, value, role) => {
        const block = document.createElement('div');
        block.className = 'preset-slider-block';
        const topline = document.createElement('div');
        topline.className = 'preset-slider-topline';
        const label = document.createElement('div');
        label.className = 'preset-slider-label';
        label.textContent = labelText;
        const valueLabel = document.createElement('div');
        valueLabel.className = 'preset-slider-value';
        valueLabel.dataset.role = `${role}-value`;
        valueLabel.textContent = formatGainDb(value);
        topline.append(label, valueLabel);
        const row = document.createElement('div');
        row.className = 'preset-slider-row';
        const lower = document.createElement('span');
        lower.className = 'preset-slider-bound'; lower.textContent = '-18';
        const input = document.createElement('input');
        input.className = 'preset-range'; input.dataset.role = `${role}-slider`; input.type = 'range';
        input.min = '-18'; input.max = '18'; input.step = '0.5'; input.value = String(value);
        const upper = document.createElement('span');
        upper.className = 'preset-slider-bound'; upper.textContent = '+18';
        row.append(lower, input, upper);
        block.append(topline, row);
        return block;
      };
      stack.append(
        makeSliderBlock(presetConfig.overlayLabel, settings.overlay_gain_db, 'overlay'),
        makeSliderBlock('base song', settings.base_song_gain_db, 'base'),
      );
      shell.append(heading, stack);
      card.appendChild(shell);

      const actions = document.createElement('div');
      actions.className = 'flex gap-2 justify-end';
      const cancelBtn = makeActionButton('cancel');
      const doneBtn = makeActionButton('done', 'bg-white text-black');
      actions.append(cancelBtn, doneBtn);
      card.appendChild(actions);

      const overlaySlider = shell.querySelector('[data-role="overlay-slider"]');
      const overlayValue = shell.querySelector('[data-role="overlay-value"]');
      const baseSlider = shell.querySelector('[data-role="base-slider"]');
      const baseValue = shell.querySelector('[data-role="base-value"]');

      const sync = () => {
        const overlayGain = Number(overlaySlider.value || settings.overlay_gain_db);
        const base = Number(baseSlider.value || settings.base_song_gain_db);
        updatePresetRangeVisual(overlaySlider);
        updatePresetRangeVisual(baseSlider);
        overlayValue.textContent = formatGainDb(overlayGain);
        baseValue.textContent = formatGainDb(base);
      };
      overlaySlider.addEventListener('input', sync);
      baseSlider.addEventListener('input', sync);
      sync();

      guardClick(cancelBtn, (event) => {
        event.preventDefault();
        closeOverlay(overlay);
      });
      guardClick(doneBtn, async (event) => {
        event.preventDefault();
        if(doneBtn.dataset.busy === '1') return;
        doneBtn.dataset.busy = '1';
        doneBtn.textContent = 'exporting...';
        try{
          const nextSettings = {
            overlay_gain_db: Number(overlaySlider.value || settings.overlay_gain_db),
            base_song_gain_db: Number(baseSlider.value || settings.base_song_gain_db),
          };
          await exportAdjustedPresetMix(task, nextSettings);
          closeOverlay(overlay);
          showPopup(`saved new ${presetConfig.label} export`);
        }catch(err){
          showPopup((err && err.message) || 'preset export failed');
          doneBtn.dataset.busy = '';
          doneBtn.textContent = 'done';
        }
      });
    }

    async function loadPreviousFiles(){
      try{
        const res = await fetch('/api/history', { cache: 'no-store' });
        if(!res.ok){
          previousFilesStorageState = null;
          return [];
        }
        const data = await res.json();
        previousFilesState = Array.isArray(data && data.items) ? data.items : [];
        previousFilesStorageState = data && data.storage ? data.storage : null;
        return previousFilesState;
      }catch(err){
        console.warn('failed to load previous files', err);
        previousFilesStorageState = null;
        return [];
      }
    }

    function formatHistoryDate(value){
      const numeric = Number(value || 0);
      if(!numeric) return '';
      try{
        return new Intl.DateTimeFormat(undefined, {
          month: 'short',
          day: 'numeric',
          hour: 'numeric',
          minute: '2-digit',
        }).format(new Date(numeric * 1000));
      }catch(_){
        return '';
      }
    }

    async function savePreviousFileCopy(entry){
      if(!entry || !entry.id) return;
      try{
        if(window.pywebview && window.pywebview.api && typeof window.pywebview.api.pick_output_folder === 'function'){
          const chosen = await window.pywebview.api.pick_output_folder();
          if(!chosen) return;
          const res = await fetch(`/api/history/${entry.id}/copy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ output_root: chosen }),
          });
          const data = await res.json().catch(() => ({}));
          if(!res.ok){
            showPopup(data?.detail?.message || data?.message || 'could not save files');
            return;
          }
          showPopup('saved copy');
          return;
        }
        const anchor = document.createElement('a');
        anchor.href = `/api/history/${entry.id}/download`;
        anchor.rel = 'noopener';
        anchor.click();
      }catch(err){
        showPopup((err && err.message) || 'could not save files');
      }
    }

    async function reusePreviousFile(entry, stems){
      const res = await fetch(`/api/history/${entry.id}/reuse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stems: stems.join(','),
          output_format: settingsState.output_format || 'same_as_input',
          multi_stem_export: settingsState.multi_stem_export || 'zip',
          video_handling: settingsState.video_handling || 'audio_only',
        }),
      });
      const data = await res.json().catch(() => ({}));
      if(!res.ok){
        throw new Error(data?.detail?.message || data?.message || 'could not reuse song');
      }
      const item = {
        id: data.task_id || data.id || null,
        name: data.name,
        mode: data.mode || null,
        pct: typeof data.pct === 'number' ? data.pct : 0,
        stage: data.stage || 'ready',
        stems: Array.isArray(data.stems) ? data.stems : [],
        out_dir: data.out_dir || null,
        preset_settings: data.preset_settings || null,
        can_adjust_preset: !!data.can_adjust_preset,
        downloaded: false,
        delivery: data.delivery || 'folder',
        autoDownloaded: false,
        frozen: false,
      };
      tasks.push(item);
      createItem(item);
      saveTasks();
      updateUI();
      return item;
    }

    function openPreviousFileReusePicker(entry){
      if(!entry || !entry.id) return;
      const { overlay, card } = createOverlayCard('choose split');
      card.style.maxWidth = '540px';
      const grid = document.createElement('div');
      grid.className = 'grid grid-cols-2 gap-2';
      const options = [
        { label: 'drums', stems: ['htdemucs_ft_drums'], tooltip: 'fast drums' },
        { label: 'bass', stems: ['htdemucs_ft_bass'], tooltip: 'fast bass' },
        { label: 'other', stems: ['htdemucs_ft_other'], tooltip: 'other stem with guitar removed first' },
        { label: 'full mix faster', stems: ['htdemucs_6s'], tooltip: 'faster full mix split' },
        { label: 'vocals', stems: ['vocals'], tooltip: 'vocals' },
        { label: 'instrumental', stems: ['instrumental'], tooltip: 'instrumental' },
        { label: 'guitar', stems: ['guitar'], tooltip: 'guitar' },
        { label: 'bg vocal', stems: ['mel_band_karaoke'], tooltip: 'background vocal split' },
        { label: 'full mix', stems: ['bs_roformer_6s'], tooltip: BS_6S_TOOLTIP },
        { label: 'drum split - 6', stems: ['drumsep_6s'], tooltip: DRUMSEP_6S_TOOLTIP },
        { label: 'drum split - 4', stems: ['drumsep_4s'], tooltip: DRUMSEP_4S_TOOLTIP },
        { label: 'all stems', stems: ['all_stems'], tooltip: 'full stem graph' },
        { label: 'boost harmonies', stems: ['boost_harmonies'], tooltip: 'boost harmonies' },
        { label: 'denoise', stems: ['preset_denoise'], tooltip: 'denoise' },
      ];
      options.forEach((option) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'preset-choice';
        const title = document.createElement('span');
        title.className = 'preset-choice-title';
        title.textContent = option.label;
        btn.appendChild(title);
        btn.dataset.tooltip = option.tooltip;
        bindDelayedTooltip(btn);
        guardClick(btn, async (event) => {
          event.preventDefault();
          try{
            await reusePreviousFile(entry, option.stems);
            closeOverlay(overlay);
            showPopup('song added back to queue');
          }catch(err){
            showPopup((err && err.message) || 'could not reuse song');
          }
        });
        grid.appendChild(btn);
      });
      card.appendChild(grid);
      const closeBtn = makeActionButton('close');
      guardClick(closeBtn, (event) => {
        event.preventDefault();
        closeOverlay(overlay);
      });
      card.appendChild(closeBtn);
    }

    function createPreviousFileRow(entry){
      const row = document.createElement('div');
      row.className = 'previous-history-item';

      const artworkShell = document.createElement('div');
      artworkShell.className = 'artwork-shell';
      const artwork = document.createElement('img');
      artwork.className = 'artwork-img';
      artwork.alt = '';
      artwork.hidden = true;
      const fallback = document.createElement('div');
      fallback.className = 'artwork-fallback';
      fallback.textContent = '♪';
      artwork.onload = () => {
        artwork.hidden = false;
        artworkShell.classList.add('has-image');
      };
      artwork.onerror = () => {
        artwork.hidden = true;
        artworkShell.classList.remove('has-image');
      };
      artwork.src = `${entry.artwork_url}?r=${Date.now()}`;
      artworkShell.append(artwork, fallback);

      const body = document.createElement('div');
      body.className = 'previous-history-body';
      const name = document.createElement('div');
      name.className = 'previous-history-name';
      name.textContent = entry.name || 'untitled';
      name.title = entry.name || '';
      const meta = document.createElement('div');
      meta.className = 'previous-history-meta';
      const labels = document.createElement('div');
      labels.className = 'labels';
      applyLabels(labels, entry.stems || []);
      const date = document.createElement('div');
      date.className = 'previous-history-date';
      date.textContent = formatHistoryDate(entry.finished_at);
      meta.append(labels, date);
      body.append(name, meta);

      const actions = document.createElement('div');
      actions.className = 'previous-history-actions';
      const revealBtn = document.createElement('button');
      revealBtn.type = 'button';
      revealBtn.className = 'history-icon-btn';
      revealBtn.dataset.tooltip = 'open folder';
      revealBtn.innerHTML = `<svg viewBox="0 0 24 24" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h5l2 2h11v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"></path></svg>`;
      bindDelayedTooltip(revealBtn);
      guardClick(revealBtn, async (event) => {
        event.preventDefault();
        const res = await fetch(`/api/history/${entry.id}/reveal`, { method: 'POST' });
        if(!res.ok){
          const payload = await res.json().catch(() => ({}));
          showPopup(payload?.detail?.message || payload?.message || 'could not open folder');
        }
      });
      const saveBtn = document.createElement('button');
      saveBtn.type = 'button';
      saveBtn.className = 'history-icon-btn';
      saveBtn.dataset.tooltip = 'save copy';
      saveBtn.innerHTML = `<svg viewBox="0 0 24 24" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"></path><path d="m7 10 5 5 5-5"></path><path d="M5 21h14"></path></svg>`;
      bindDelayedTooltip(saveBtn);
      guardClick(saveBtn, async (event) => {
        event.preventDefault();
        await savePreviousFileCopy(entry);
      });
      const reuseBtn = document.createElement('button');
      reuseBtn.type = 'button';
      reuseBtn.className = 'history-icon-btn';
      reuseBtn.dataset.tooltip = 'new split';
      reuseBtn.innerHTML = `<svg viewBox="0 0 24 24" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>`;
      bindDelayedTooltip(reuseBtn);
      guardClick(reuseBtn, (event) => {
        event.preventDefault();
        openPreviousFileReusePicker(entry);
      });
      actions.append(revealBtn, saveBtn, reuseBtn);

      row.append(artworkShell, body, actions);
      return row;
    }

    async function openPreviousFilesCard(){
      const { overlay, card } = createOverlayCard('previous files');
      card.style.maxWidth = '760px';
      const shell = document.createElement('div');
      shell.className = 'overlay-scroll-shell';
      const list = document.createElement('div');
      list.className = 'previous-files-list';
      const indicator = document.createElement('div');
      indicator.className = 'overlay-scroll-indicator';
      indicator.setAttribute('aria-hidden', 'true');
      const thumb = document.createElement('div');
      thumb.className = 'overlay-scroll-thumb';
      indicator.appendChild(thumb);
      shell.append(list, indicator);
      card.appendChild(shell);
      const syncScroll = () => updateScrollIndicator(list, indicator, thumb);
      overlay.__cleanupFns = overlay.__cleanupFns || [];
      overlay.__cleanupFns.push(attachSoftWheelScroll(list, syncScroll));
      const items = await loadPreviousFiles();
      if(previousFilesStorageState && (previousFilesStorageState.near_limit || previousFilesStorageState.at_limit)){
        const alert = document.createElement('div');
        alert.className = 'history-storage-alert';
        const prefix = previousFilesStorageState.at_limit ? 'previous files full' : 'previous files nearly full';
        alert.textContent = `${prefix} · ${formatStorageUsage(previousFilesStorageState.usage_bytes)} / ${formatStorageUsage(previousFilesStorageState.limit_bytes)}`;
        list.appendChild(alert);
      }
      if(!items.length){
        const empty = document.createElement('div');
        empty.className = 'history-empty';
        empty.textContent = 'no saved songs yet';
        list.appendChild(empty);
        requestAnimationFrame(syncScroll);
        return;
      }
      items.forEach((entry) => {
        list.appendChild(createPreviousFileRow(entry));
      });
      requestAnimationFrame(syncScroll);
    }

    function openSettings(){
      if(!settingsOverlay) settingsOverlay = document.getElementById('settings-overlay');
      const bg = document.getElementById('overlay-bg');
      const card = document.getElementById('settings-card');
      if(settingsOverlay){
        document.body.classList.add('settings-open');
        settingsOverlay.classList.remove('hidden');
        activateDialogAccessibility(settingsOverlay, document.getElementById('settings-close'));
        if(bg){
          bg.classList.remove('anim-out');
          bg.classList.remove('anim');
          void bg.offsetWidth;
          bg.classList.add('anim'); // IN
        }
        if(card){
          card.classList.remove('settings-card-out');
          card.classList.remove('opacity-0');
          void card.offsetWidth;
          card.classList.add('settings-card-in'); // IN
          if(settingsScrollBody){
            settingsScrollBody.scrollTop = 0;
          }
          requestAnimationFrame(updateSettingsScrollIndicator);
        }
        if(nerdStuffWrap){
          setNerdStuffExpanded(nerdStuffExpanded, { immediate: true });
        }
      }
    }

    function updateSettingsScrollIndicator(){
      const scrollBody = settingsScrollBody || document.getElementById('settings-card-scroll');
      if(!scrollBody || !settingsScrollIndicator || !settingsScrollThumb) return;
      updateScrollIndicator(scrollBody, settingsScrollIndicator, settingsScrollThumb);
    }

    function updateScrollIndicator(scrollBody, indicator, thumb){
      if(!scrollBody || !indicator || !thumb) return;
      const clientHeight = scrollBody.clientHeight || 0;
      const scrollHeight = scrollBody.scrollHeight || 0;
      const maxScroll = Math.max(0, scrollHeight - clientHeight);
      const visible = maxScroll > 4;
      indicator.classList.toggle('visible', visible);
      if(!visible){
        thumb.style.height = '0px';
        thumb.style.transform = 'translateY(0)';
        return;
      }
      const trackHeight = indicator.clientHeight || Math.max(0, clientHeight - 40);
      const thumbHeight = Math.max(40, Math.round((clientHeight / scrollHeight) * trackHeight));
      const thumbTravel = Math.max(0, trackHeight - thumbHeight);
      const thumbOffset = maxScroll > 0 ? (scrollBody.scrollTop / maxScroll) * thumbTravel : 0;
      thumb.style.height = `${thumbHeight}px`;
      thumb.style.transform = `translateY(${thumbOffset}px)`;
    }

    function wantsSoftPanelScroll(){
      if(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches){
        return false;
      }
      return !window.matchMedia || window.matchMedia('(pointer:fine)').matches;
    }

    function attachSoftWheelScroll(scrollBody, updateIndicator = null){
      if(!scrollBody || !wantsSoftPanelScroll()){
        return () => {};
      }
      let raf = null;
      let target = scrollBody.scrollTop || 0;
      const syncTarget = () => {
        if(!raf){
          target = scrollBody.scrollTop || 0;
        }
        if(updateIndicator){
          updateIndicator();
        }
      };
      const step = () => {
        const maxScroll = Math.max(0, (scrollBody.scrollHeight || 0) - (scrollBody.clientHeight || 0));
        target = Math.max(0, Math.min(maxScroll, target));
        const current = scrollBody.scrollTop || 0;
        const delta = target - current;
        if(Math.abs(delta) < 0.5){
          scrollBody.scrollTop = target;
          if(updateIndicator){
            updateIndicator();
          }
          raf = null;
          return;
        }
        scrollBody.scrollTop = current + (delta * 0.16);
        if(updateIndicator){
          updateIndicator();
        }
        raf = requestAnimationFrame(step);
      };
      const onWheel = (event) => {
        if(event.ctrlKey || Math.abs(event.deltaY) <= Math.abs(event.deltaX || 0)){
          return;
        }
        const maxScroll = Math.max(0, (scrollBody.scrollHeight || 0) - (scrollBody.clientHeight || 0));
        if(maxScroll <= 0){
          return;
        }
        event.preventDefault();
        target = Math.max(0, Math.min(maxScroll, target + event.deltaY));
        if(!raf){
          raf = requestAnimationFrame(step);
        }
      };
      scrollBody.addEventListener('scroll', syncTarget, { passive: true });
      scrollBody.addEventListener('wheel', onWheel, { passive: false });
      return () => {
        if(raf){
          cancelAnimationFrame(raf);
          raf = null;
        }
        scrollBody.removeEventListener('scroll', syncTarget);
        scrollBody.removeEventListener('wheel', onWheel);
      };
    }

    function closeSettings(){
      if(!settingsOverlay) settingsOverlay = document.getElementById('settings-overlay');
      const bg = document.getElementById('overlay-bg');
      const card = document.getElementById('settings-card');
      if(settingsOverlay){
        if(bg){
          bg.classList.remove('anim');
          bg.classList.remove('anim-out');
          void bg.offsetWidth;
          bg.classList.add('anim-out'); // OUT
        }
        if(card){
          card.classList.remove('settings-card-in');
          card.classList.remove('fade-in');
          card.classList.add('settings-card-out'); // OUT
        }
        setTimeout(() => {
          if(card){
            card.classList.remove('settings-card-out');
            card.classList.add('opacity-0');
          }
          if(bg){ bg.classList.remove('anim-out'); }
          settingsOverlay.classList.add('hidden');
          document.body.classList.remove('settings-open');
          restoreDialogFocus(settingsOverlay);
          if(modelPreviewActive){
            restoreModelPreview();
          }
        }, 180);
      }
    }

    // --- Progress Smoother Helper ---
    function makeProgressSmoother(bar){
      let raf = null;
      let value = 0;     // currently displayed percent
      let target = 0;    // latest requested percent
      const clamp = p => Math.max(0, Math.min(100, p));
      const tick = () => {
        // critically damped-ish exponential approach toward target
        const delta = target - value;
        // if close enough, snap to target and stop
        if (Math.abs(delta) < 0.08) {
          value = target;
          bar.style.width = value + '%';
          raf = null;
          return;
        }
        // ease toward target; 0.12 controls lag/smoothness
        value += delta * 0.12;
        bar.style.width = value + '%';
        raf = requestAnimationFrame(tick);
      };
      return {
        setImmediate(p){ value = target = clamp(p); bar.style.width = value + '%'; },
        setTarget(p){ target = clamp(p); if(!raf) raf = requestAnimationFrame(tick); },
        stop(){ if(raf){ cancelAnimationFrame(raf); raf = null; } }
      };
    }

    if(localStorage.getItem('playIntro')){
      localStorage.removeItem('playIntro');
      title.style.position = 'absolute';
      title.style.left = '50%';
      title.style.top = '45%';
      title.style.transform = 'translate(-50%, -50%)';
      title.style.opacity = '1';
      setTimeout(() => {
        title.style.transition = 'all 0.5s ease';
        title.style.left = '';
        title.style.top = '';
        title.style.transform = '';
      }, 50);
      setTimeout(() => {
        document.querySelectorAll('#dropzone, .glass.w-64, #queue, #clear-btn').forEach((el,i)=>{
          setTimeout(()=>{el.classList.add('fade-in')}, i*25);
        });
      }, 600);
    } else {
      title.classList.add('fade-in');
    }

    function quarantineStorageValue(storage, key){
      try{
        const value = storage.getItem(key);
        if(value !== null){
          storage.setItem(`${key}.corrupt.${Date.now()}`, value.slice(0, 65536));
        }
        storage.removeItem(key);
      }catch(_error){}
    }

    function safeStorageJson(storage, key, fallback){
      try{
        const raw = storage.getItem(key);
        if(raw === null) return fallback;
        return JSON.parse(raw);
      }catch(_error){
        quarantineStorageValue(storage, key);
        return fallback;
      }
    }

    safeStorageJson(localStorage, 'tasks', []);
    try{ localStorage.removeItem('tasks'); }catch(_error){}
    let tasks = [];
    // Helper: is a task finished? (completed or stopped)
    function isFinished(t){
      return t && (t.pct >= 100 || t.stage === 'stopped');
    }
    function saveTasks(){}

    function setQueuePausedUi(paused){
      if(resumeQueueBtn){ resumeQueueBtn.hidden = !paused; }
    }

    if(resumeQueueBtn){
      guardClick(resumeQueueBtn, async () => {
        const response = await fetch('/api/queue/resume', { method: 'POST' });
        if(!response.ok){ showPopup('could not resume queue'); return; }
        setQueuePausedUi(false);
        showPopup('queue resumed');
      });
      fetch('/api/queue/status').then((response) => response.json()).then((data) => setQueuePausedUi(!!data.paused)).catch(() => {});
    }

    function makeUiGroupKey(prefix = 'group'){
      if(window.crypto && typeof window.crypto.randomUUID === 'function'){
        return `${prefix}-${window.crypto.randomUUID()}`;
      }
      return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function queueLeafRows(){
      return Array.from(queue.querySelectorAll('.item-row'));
    }

    function queueDisplayEntries(){
      return Array.from(queue.children || []);
    }

    function stemsIdentityKey(stems){
      return normalizeStemList(stems).join('|');
    }

    function taskRowIdentity(task){
      if(!task) return '';
      if(task.id) return `id:${task.id}`;
      return `temp:${task.tempKey || ''}:${stemsIdentityKey(task.stems || [])}`;
    }

    const queueAnimatedEntryKeys = new Set();

    function stageQueueEntry(node, key, delayMs = 0){
      if(!node) return;
      const reducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
      if(reducedMotion){
        node.classList.remove('enter-pre');
        node.classList.add('enter-active');
        return;
      }
      if(key){
        if(queueAnimatedEntryKeys.has(key)) return;
        queueAnimatedEntryKeys.add(key);
      }
      node.style.setProperty('--queue-enter-delay', `${Math.max(0, Number(delayMs) || 0)}ms`);
      node.classList.remove('enter-active');
      node.classList.add('enter-pre');
      requestAnimationFrame(() => {
        node.classList.add('enter-active');
      });
    }

    function queueTaskCount(){
      return queueDisplayEntries().length;
    }

    function queueSongGroupKey(task){
      if(task && task.tempKey) return `song:${task.tempKey}`;
      if(task && task.id) return `song-id:${task.id}`;
      return '';
    }

    function songDisplayTitle(name){
      let text = String(name || '').trim();
      if(!text) return 'untitled';
      text = text.replace(/\.[^.]+$/u, '').trim();
      text = text.replace(/^\s*\d{1,3}\s*[-._)\]]+\s*/u, '');
      text = text.replace(/^\s*\d{1,3}\s+/u, '');
      text = text.replace(/\s*[[(](?:from|feat\.?|ft\.?|official|lyrics?|audio|video)[^)\]]*[\])]\s*$/iu, '');
      text = text.replace(/\s+/g, ' ').trim();
      return text || String(name || '').replace(/\.[^.]+$/u, '').trim() || 'untitled';
    }

    function megaCardTitle(tasksList, limit = 35){
      const seen = new Set();
      const names = [];
      (Array.isArray(tasksList) ? tasksList : []).forEach((task) => {
        const title = songDisplayTitle(task && task.name);
        if(!title || seen.has(title)) return;
        seen.add(title);
        names.push(title);
      });
      if(!names.length) return 'songs';
      let result = '';
      for(const title of names){
        const candidate = result ? `${result}, ${title}` : title;
        if(candidate.length > limit){
          break;
        }
        result = candidate;
      }
      if(!result){
        result = names[0].slice(0, limit).trim();
      }
      if(result.length < names.join(', ').length){
        result = `${result}${result.endsWith('…') ? '' : '…'}`;
      }
      return result;
    }

    function groupExpansionKey(kind, key){
      return `${kind}:${key}`;
    }

    function isGroupExpanded(kind, key){
      return !!queueGroupExpansionState[groupExpansionKey(kind, key)];
    }

    function setGroupExpanded(kind, key, expanded){
      queueGroupExpansionState[groupExpansionKey(kind, key)] = !!expanded;
    }

    function truncateFilename(name, max = 25){
      const text = String(name || '');
      if(text.length <= max) return text;
      const dotIndex = text.lastIndexOf('.');
      if(dotIndex > 0 && dotIndex < text.length - 1){
        const ext = text.slice(dotIndex);
        const reserve = max - ext.length - 1;
        if(reserve > 6){
          return `${text.slice(0, reserve)}…${ext}`;
        }
      }
      return `${text.slice(0, Math.max(1, max - 1))}…`;
    }

    function applyFilename(el, name){
      if(!el) return;
      const full = String(name || '');
      el.textContent = truncateFilename(full, 25);
      el.title = full;
    }

    function artworkUrl(taskId){
      return taskId ? `/api/tasks/${encodeURIComponent(taskId)}/artwork` : '';
    }

    function setArtworkLoading(row, isLoading){
      if(!row) return;
      const shell = row.querySelector('.artwork-shell');
      const loading = row.querySelector('.artwork-loading');
      if(shell){
        shell.classList.toggle('is-loading', !!isLoading);
      }
      if(loading){
        loading.hidden = !isLoading;
      }
    }

    function shouldDimArtwork(task){
      if(!task) return false;
      const stage = String(task.stage || '').toLowerCase();
      if(stage === 'ready' || stage === 'done' || stage === 'stopped' || stage === 'error') return false;
      if(stage === 'queued') return true;
      return typeof task.pct === 'number' ? (task.pct >= 0 && task.pct < 100) : false;
    }

    function applyArtwork(row, task){
      if(!row) return;
      const shell = row.querySelector('.artwork-shell');
      const img = row.querySelector('.artwork-img');
      if(!shell || !img) return;
      if(row.__artworkRetryTimer){
        clearTimeout(row.__artworkRetryTimer);
        row.__artworkRetryTimer = null;
      }
      shell.classList.toggle('is-dim', shouldDimArtwork(task));
      const taskId = task && task.id ? String(task.id) : '';
      if(!taskId){
        row.__artworkResolved = false;
        row.__artworkAttempts = 0;
        setArtworkLoading(row, true);
        img.hidden = true;
        img.removeAttribute('src');
        img.dataset.taskId = '';
        shell.classList.remove('has-image');
        return;
      }
      if(img.dataset.taskId === taskId && img.getAttribute('src') && !img.hidden){
        setArtworkLoading(row, false);
        return;
      }
      if(img.dataset.taskId !== taskId){
        row.__artworkResolved = false;
        row.__artworkAttempts = 0;
      }
      img.dataset.taskId = taskId;
      setArtworkLoading(row, !row.__artworkResolved);
      img.hidden = true;
      img.onload = () => {
        if(img.dataset.taskId !== taskId) return;
        if(row.__artworkRetryTimer){
          clearTimeout(row.__artworkRetryTimer);
          row.__artworkRetryTimer = null;
        }
        row.__artworkResolved = true;
        row.__artworkAttempts = 0;
        img.hidden = false;
        shell.classList.add('has-image');
        setArtworkLoading(row, false);
      };
      img.onerror = () => {
        if(img.dataset.taskId !== taskId) return;
        row.__artworkAttempts = Number(row.__artworkAttempts || 0) + 1;
        img.hidden = true;
        img.removeAttribute('src');
        shell.classList.remove('has-image');
        if(row.__artworkAttempts >= 6){
          row.__artworkResolved = true;
          setArtworkLoading(row, false);
          return;
        }
        setArtworkLoading(row, true);
        row.__artworkRetryTimer = setTimeout(() => {
          if(img.dataset.taskId !== taskId) return;
          img.src = `${artworkUrl(taskId)}?v=${encodeURIComponent(taskId)}&r=${Date.now()}`;
        }, 900);
      };
      img.src = `${artworkUrl(taskId)}?v=${encodeURIComponent(taskId)}`;
    }

    function applyRowState(row, task){
      if(!row || !task) return;
      applyArtwork(row, task);
      const labels = row.querySelector('.labels');
      if(labels){
        const stemCount = labels.childElementCount || 0;
        const normalized = String(task.stage || '').toLowerCase();
        const unlocked = normalized === 'ready' && Number(task.pct || 0) === 0 && !task.frozen;
        labels.classList.toggle('hidden', stemCount <= 0);
        labels.classList.toggle('labels-unlocked', stemCount > 0 && unlocked);
      }
      bindTaskEditorButton(row.querySelector('.task-editor-btn'), task);
      bindTaskPresetButton(row.querySelector('.task-preset-btn'), task);
    }

    function isBoostHarmoniesTask(task){
      return !!(task && Array.isArray(task.stems) && task.stems.includes('boost_harmonies'));
    }

    function taskPresetSettings(task){
      const presetMode = taskPresetMode(task);
      const presetConfig = presetMode ? PRESET_CONFIGS[presetMode] : PRESET_CONFIGS.boost_harmonies;
      const source = task && task.preset_settings && typeof task.preset_settings === 'object' ? task.preset_settings : {};
      return {
        overlay_gain_db: Number.isFinite(Number(source.overlay_gain_db)) ? Number(source.overlay_gain_db) : presetConfig.defaultOverlayGain,
        base_song_gain_db: Number.isFinite(Number(source.base_song_gain_db)) ? Number(source.base_song_gain_db) : presetConfig.defaultBaseGain,
      };
    }

    function canAdjustTaskPreset(task){
      if(isLanClient) return false;
      if(!task || !task.id) return false;
      if(String(task.stage || '').toLowerCase() !== 'done') return false;
      if(typeof task.can_adjust_preset === 'boolean') return task.can_adjust_preset;
      return !!taskPresetMode(task);
    }

    function bindTaskPresetButton(btn, task){
      if(!btn) return;
      bindDelayedTooltip(btn);
      const visible = canAdjustTaskPreset(task);
      btn.hidden = !visible;
      btn.disabled = !visible;
      if(!visible) return;
      guardClick(btn, (event) => {
        event.preventDefault();
        event.stopPropagation();
        openTaskPresetAdjuster(task);
      });
    }

    function taskPendingEntry(task){
      if(!task) return null;
      if(task.tempKey){
        return pendingItems.find((pending) => pending && pending.tempKey === task.tempKey) || null;
      }
      if(task.id){
        return pendingItems.find((pending) => pending && pending.id === task.id) || null;
      }
      return null;
    }

    function canOpenTaskEditor(task){
      if(task && task.disableEditor) return false;
      return !!(task && (task.id || taskPendingEntry(task)));
    }

    function taskClipState(task){
      const startMs = Math.max(0, Number(task && task.clip_start_ms) || 0);
      const endRaw = Number(task && task.clip_end_ms);
      return {
        startMs,
        endMs: Number.isFinite(endRaw) && endRaw > 0 ? endRaw : null,
        enabled: !!(task && task.clip_enabled),
      };
    }

    function cloneClipState(clip){
      const source = clip && typeof clip === 'object' ? clip : {};
      return {
        startMs: Math.max(0, Math.round(Number(source.startMs) || 0)),
        endMs: Number.isFinite(Number(source.endMs)) && Number(source.endMs) > 0 ? Math.round(Number(source.endMs)) : null,
        enabled: !!source.enabled,
      };
    }

    function applyClipState(task, clip){
      if(!task || !clip) return;
      task.clip_start_ms = Math.max(0, Math.round(Number(clip.startMs) || 0));
      task.clip_end_ms = Number.isFinite(Number(clip.endMs)) && Number(clip.endMs) > 0
        ? Math.round(Number(clip.endMs))
        : null;
      task.clip_enabled = !!clip.enabled;
    }

    function taskGroupMembers(task){
      if(!task) return [];
      if(task.tempKey){
        const grouped = tasks.filter((candidate) => candidate && candidate.tempKey === task.tempKey);
        if(grouped.length) return grouped;
      }
      return [task];
    }

    function syncPendingClipState(task, clip){
      const pending = taskPendingEntry(task);
      if(!pending) return;
      pending.clip_start_ms = Math.max(0, Math.round(Number(clip.startMs) || 0));
      pending.clip_end_ms = Number.isFinite(Number(clip.endMs)) && Number(clip.endMs) > 0
        ? Math.round(Number(clip.endMs))
        : null;
      pending.clip_enabled = !!clip.enabled;
    }

    function syncPendingClipStateToGroup(task, clip){
      if(task && task.tempKey){
        pendingItems.forEach((pending) => {
          if(!pending || pending.tempKey !== task.tempKey) return;
          syncPendingClipState({ tempKey: pending.tempKey, id: pending.id }, clip);
          pending.clip_start_ms = Math.max(0, Math.round(Number(clip.startMs) || 0));
          pending.clip_end_ms = Number.isFinite(Number(clip.endMs)) && Number(clip.endMs) > 0 ? Math.round(Number(clip.endMs)) : null;
          pending.clip_enabled = !!clip.enabled;
        });
        return;
      }
      syncPendingClipState(task, clip);
    }

    function editorTimeLabel(ms){
      const safe = Math.max(0, Math.round(Number(ms) || 0));
      const totalSeconds = Math.floor(safe / 1000);
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      const millis = Math.floor((safe % 1000) / 10);
      return `${minutes}:${String(seconds).padStart(2, '0')}.${String(millis).padStart(2, '0')}`;
    }

    function editorRulerLabel(ms){
      const safe = Math.max(0, Math.round(Number(ms) || 0));
      const totalSeconds = Math.floor(safe / 1000);
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const seconds = totalSeconds % 60;
      if(hours > 0){
        return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
      }
      return `${minutes}:${String(seconds).padStart(2, '0')}`;
    }

    function editorTaskStorageKey(task){
      return String((task && (task.id || task.tempKey || task.name)) || 'editor-task');
    }

    function editorReadStorageJson(key, fallback){
      const parsed = safeStorageJson(localStorage, key, fallback);
      return parsed && typeof parsed === 'object' ? parsed : fallback;
    }

    function editorReadTrackDraftStore(){
      return editorReadStorageJson(EDITOR_TRACK_DRAFTS_KEY, {});
    }

    function editorWriteTrackDraftStore(store){
      try{
        localStorage.setItem(EDITOR_TRACK_DRAFTS_KEY, JSON.stringify(store || {}));
      }catch(_){}
    }

    function editorLoadTrackDrafts(task){
      const prefix = `${editorTaskStorageKey(task)}::`;
      const store = editorReadTrackDraftStore();
      const drafts = {};
      Object.entries(store).forEach(([key, value]) => {
        if(!key.startsWith(prefix) || !value || typeof value !== 'object') return;
        drafts[key.slice(prefix.length)] = cloneClipState(value);
      });
      return drafts;
    }

    function editorPersistTrackDrafts(task, tracks){
      const prefix = `${editorTaskStorageKey(task)}::`;
      const store = editorReadTrackDraftStore();
      Object.keys(store).forEach((key) => {
        if(key.startsWith(prefix)){
          delete store[key];
        }
      });
      (Array.isArray(tracks) ? tracks : []).forEach((track) => {
        if(!track || track.key === 'source') return;
        store[`${prefix}${track.key}`] = cloneClipState(track.clip);
      });
      editorWriteTrackDraftStore(store);
    }

    function editorRestoreTrackDrafts(task, snapshot){
      const prefix = `${editorTaskStorageKey(task)}::`;
      const store = editorReadTrackDraftStore();
      Object.keys(store).forEach((key) => {
        if(key.startsWith(prefix)){
          delete store[key];
        }
      });
      Object.entries(snapshot || {}).forEach(([trackKey, clip]) => {
        store[`${prefix}${trackKey}`] = cloneClipState(clip);
      });
      editorWriteTrackDraftStore(store);
    }

    function editorReadClipboard(){
      const clip = editorReadStorageJson(EDITOR_TRIM_CLIPBOARD_KEY, null);
      return clip ? cloneClipState(clip) : null;
    }

    function editorWriteClipboard(clip){
      try{
        localStorage.setItem(EDITOR_TRIM_CLIPBOARD_KEY, JSON.stringify(cloneClipState(clip)));
      }catch(_){}
    }

    function editorSnapEnabled(){
      return localStorage.getItem(EDITOR_SNAP_ENABLED_KEY) !== 'false';
    }

    function editorSetSnapEnabled(enabled){
      try{
        localStorage.setItem(EDITOR_SNAP_ENABLED_KEY, enabled ? 'true' : 'false');
      }catch(_){}
    }

    function editorWaveformCacheKey(taskId, output, points){
      return `${String(taskId || '')}::${String(output || 'source')}::${Math.max(0, Math.round(Number(points) || 0))}`;
    }

    function cachedEditorWaveformPayload(taskId, output = '', points = 1280){
      return editorWaveformPayloadCache.get(editorWaveformCacheKey(taskId, output, points)) || null;
    }

    const EDITOR_WAVEFORM_POINT_LEVELS = [
      640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
      2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168,
      7680, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 18432, 20480,
      22528, 24576, 26624, 28672, 30720, 32768, 36864, 40960, 45056, 49152, 53248, 57344,
      61440, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072,
    ];

    function bestCachedEditorWaveformPayload(taskId, output = '', targetPoints = 1280){
      const desired = Math.max(640, Math.min(131072, Math.round(Number(targetPoints) || 1280)));
      const preferred = EDITOR_WAVEFORM_POINT_LEVELS
        .filter((value) => value <= desired)
        .sort((a, b) => b - a);
      for(const points of preferred){
        const payload = cachedEditorWaveformPayload(taskId, output, points);
        if(payload) return payload;
      }
      for(const points of EDITOR_WAVEFORM_POINT_LEVELS){
        const payload = cachedEditorWaveformPayload(taskId, output, points);
        if(payload) return payload;
      }
      return null;
    }

    async function ensureEditorWaveformPayload(taskId, output = '', points = 1280){
      const normalizedPoints = Math.max(640, Math.min(131072, Math.round(Number(points) || 1280)));
      const cacheKey = editorWaveformCacheKey(taskId, output, normalizedPoints);
      if(editorWaveformPayloadCache.has(cacheKey)){
        return editorWaveformPayloadCache.get(cacheKey);
      }
      if(editorWaveformFetches.has(cacheKey)){
        return editorWaveformFetches.get(cacheKey);
      }
      const requested = String(output || '').trim();
      const url = requested
        ? `/api/tasks/${encodeURIComponent(taskId)}/waveform?output=${encodeURIComponent(requested)}&points=${normalizedPoints}`
        : `/api/tasks/${encodeURIComponent(taskId)}/waveform?points=${normalizedPoints}`;
      const fetchPromise = fetch(url)
        .then((response) => response.ok ? response.json() : null)
        .then((payload) => {
          if(payload && typeof payload === 'object'){
            editorWaveformPayloadCache.set(cacheKey, payload);
          }
          return payload;
        })
        .catch(() => null)
        .finally(() => {
          editorWaveformFetches.delete(cacheKey);
        });
      editorWaveformFetches.set(cacheKey, fetchPromise);
      return fetchPromise;
    }

    function primeTaskEditorAssets(task){
      if(!task || !task.id) return;
      void ensureEditorWaveformPayload(task.id, '', 1280);
    }

    function bindTaskEditorButton(btn, task){
      if(!btn) return;
      bindDelayedTooltip(btn);
      const host = btn.closest('.item-row, .queue-group-parent-row');
      if(host && host.dataset.forceHideEditor === '1'){
        btn.hidden = true;
        btn.disabled = true;
        return;
      }
      const visible = canOpenTaskEditor(task);
      btn.hidden = !visible;
      btn.disabled = !visible;
      if(!visible) return;
      guardClick(btn, (event) => {
        event.preventDefault();
        event.stopPropagation();
        openTaskEditor(task);
      });
    }

    async function openTaskEditor(task){
      if(!task) return;
      const pending = taskPendingEntry(task);
      const { overlay, card } = createOverlayCard('editor');
      card.classList.add('editor-card', 'editor-nle-card');
      card.style.maxWidth = '1560px';
      card.style.width = 'min(1560px, calc(100vw - 48px))';
      card.style.minHeight = '0';
      card.style.padding = '0';
      card.style.borderRadius = '16px';

      const titleEl = card.querySelector('h3');
      titleEl.className = 'editor-title';
      titleEl.textContent = '';
      titleEl.hidden = true;

      const editorShell = document.createElement('div');
      editorShell.className = 'editor-shell';
      card.appendChild(editorShell);

      const cancelBtn = makeActionButton('cancel', 'bg-white/10 text-white');
      const saveBtn = makeActionButton('save', 'bg-white text-black');

      const sourceOriginalClip = cloneClipState(taskClipState(task));
      const originalTrackDrafts = editorLoadTrackDrafts(task);
      const snapDistanceMs = () => Math.max(0, Math.min(2000, Number(settingsState.editor_snap_distance_ms) || 0));

      let snapEnabled = editorSnapEnabled();
      let playheadMs = sourceOriginalClip.startMs || 0;
      let activeObjectUrl = '';
      let activeTrackKey = 'source';
      let loadedPreviewTrackKey = '';
      let closingEditor = false;
      let dragState = null;
      let clickLockUntil = 0;
      let contextMenuEl = null;
      let playbackFrame = 0;
      let dismissLockUntil = 0;
      let editorToolMode = 'select';
      let waveZoom = 1;
      let currentVariant = null;
      let layoutRefs = {};

      const editorVariants = [
        {
          number: 1,
          name: 'precision cut',
          pack: 'wire',
          layout: 'stacked',
          buttonShape: 'tile',
          trackStyle: 'glass',
          maxWidth: '1240px',
          waveBg: 'rgba(8, 18, 23, 0.88)',
          waveFill: 'rgba(142, 216, 255, 0.82)',
          waveFillDim: 'rgba(142, 216, 255, 0.56)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(16, 31, 38, 0.94), rgba(10, 21, 27, 0.98))',
            '--editor-shell-border': 'rgba(194,229,245,.16)',
            '--editor-shell-shadow': '0 30px 84px rgba(0,0,0,.38)',
            '--editor-title-strong': '#f8fcff',
            '--editor-title-muted': 'rgba(237,247,252,.82)',
            '--editor-accent': '#8ED8FF',
            '--editor-accent-strong': '#F5FCFF',
            '--editor-accent-soft': 'rgba(142,216,255,.18)',
            '--editor-accent-soft-strong': 'rgba(142,216,255,.28)',
            '--editor-panel-bg': 'linear-gradient(160deg, rgba(255,255,255,.08), rgba(255,255,255,.04))',
            '--editor-panel-border': 'rgba(255,255,255,.10)',
            '--editor-panel-shadow': '0 18px 36px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.08)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03))',
            '--editor-track-border': 'rgba(255,255,255,.10)',
            '--editor-track-shadow': '0 18px 28px rgba(0,0,0,.20)',
            '--editor-selection-bg': 'rgba(142,216,255,.12)',
            '--editor-selection-border': 'rgba(142,216,255,.26)',
            '--editor-marker': 'rgba(247,252,255,.98)',
            '--editor-chip-bg': 'rgba(255,255,255,.05)',
            '--editor-chip-border': 'rgba(255,255,255,.08)',
            '--editor-button-bg': 'rgba(255,255,255,.07)',
            '--editor-button-border': 'rgba(255,255,255,.10)',
            '--editor-button-hover-bg': 'rgba(255,255,255,.12)',
            '--editor-button-hover-border': 'rgba(255,255,255,.22)',
            '--editor-button-active-bg': 'rgba(142,216,255,.18)',
            '--editor-button-active-border': 'rgba(142,216,255,.42)',
            '--editor-kicker': 'rgba(226,239,246,.72)',
            '--editor-wave-radius': '16px',
          },
        },
        {
          number: 2,
          name: 'split deck',
          pack: 'solid',
          layout: 'duo',
          buttonShape: 'pill',
          trackStyle: 'slab',
          maxWidth: '1320px',
          waveBg: 'rgba(11, 26, 32, 0.92)',
          waveFill: 'rgba(173, 227, 255, 0.86)',
          waveFillDim: 'rgba(124, 198, 232, 0.54)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(14, 29, 35, 0.96), rgba(8, 18, 22, 0.98))',
            '--editor-shell-border': 'rgba(178,223,241,.18)',
            '--editor-shell-shadow': '0 30px 80px rgba(0,0,0,.40)',
            '--editor-title-strong': '#f3fbff',
            '--editor-title-muted': 'rgba(226,243,250,.84)',
            '--editor-accent': '#B6EAFF',
            '--editor-accent-strong': '#F4FDFF',
            '--editor-accent-soft': 'rgba(182,234,255,.20)',
            '--editor-panel-bg': 'linear-gradient(155deg, rgba(255,255,255,.11), rgba(255,255,255,.045))',
            '--editor-panel-border': 'rgba(197,230,242,.14)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(222,245,255,.10), rgba(72,124,144,.08))',
            '--editor-track-border': 'rgba(189,226,240,.18)',
            '--editor-track-shadow': '0 18px 36px rgba(0,0,0,.22)',
            '--editor-selection-bg': 'rgba(182,234,255,.16)',
            '--editor-selection-border': 'rgba(182,234,255,.30)',
            '--editor-marker': 'rgba(244,253,255,.98)',
            '--editor-chip-bg': 'rgba(221,245,255,.07)',
            '--editor-chip-border': 'rgba(192,225,238,.14)',
            '--editor-button-bg': 'rgba(227,246,255,.08)',
            '--editor-button-border': 'rgba(195,226,239,.14)',
            '--editor-button-hover-bg': 'rgba(227,246,255,.15)',
            '--editor-button-hover-border': 'rgba(214,238,248,.24)',
            '--editor-button-active-bg': 'rgba(182,234,255,.24)',
            '--editor-button-active-border': 'rgba(182,234,255,.44)',
            '--editor-kicker': 'rgba(221,238,245,.72)',
            '--editor-wave-radius': '18px',
          },
        },
        {
          number: 3,
          name: 'tape rack',
          pack: 'retro',
          layout: 'deck',
          buttonShape: 'rail',
          trackStyle: 'tape',
          maxWidth: '1360px',
          waveBg: 'rgba(10, 21, 26, 0.94)',
          waveFill: 'rgba(153, 228, 245, 0.80)',
          waveFillDim: 'rgba(113, 187, 207, 0.54)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(12, 24, 30, 0.98), rgba(7, 15, 19, 0.99))',
            '--editor-shell-border': 'rgba(151,211,231,.16)',
            '--editor-shell-shadow': '0 28px 78px rgba(0,0,0,.42)',
            '--editor-title-strong': '#effbfd',
            '--editor-title-muted': 'rgba(217,239,245,.84)',
            '--editor-accent': '#99E0F0',
            '--editor-accent-strong': '#F2FDFF',
            '--editor-panel-bg': 'linear-gradient(150deg, rgba(255,255,255,.055), rgba(255,255,255,.025))',
            '--editor-panel-border': 'rgba(151,211,231,.13)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(122,185,201,.10), rgba(255,255,255,.03))',
            '--editor-track-border': 'rgba(151,211,231,.20)',
            '--editor-selection-bg': 'rgba(153,224,240,.14)',
            '--editor-selection-border': 'rgba(153,224,240,.30)',
            '--editor-marker': 'rgba(241,253,255,.96)',
            '--editor-chip-bg': 'rgba(153,224,240,.08)',
            '--editor-chip-border': 'rgba(153,224,240,.16)',
            '--editor-button-bg': 'rgba(153,224,240,.08)',
            '--editor-button-border': 'rgba(153,224,240,.16)',
            '--editor-button-hover-bg': 'rgba(153,224,240,.16)',
            '--editor-button-hover-border': 'rgba(183,233,245,.28)',
            '--editor-button-active-bg': 'rgba(153,224,240,.24)',
            '--editor-button-active-border': 'rgba(153,224,240,.42)',
            '--editor-kicker': 'rgba(208,232,239,.68)',
            '--editor-wave-radius': '12px',
          },
        },
        {
          number: 4,
          name: 'wave lab',
          pack: 'signal',
          layout: 'focus',
          buttonShape: 'pill',
          trackStyle: 'minimal',
          maxWidth: '1180px',
          waveBg: 'rgba(7, 19, 24, 0.96)',
          waveFill: 'rgba(163, 237, 255, 0.84)',
          waveFillDim: 'rgba(116, 201, 224, 0.52)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(10, 22, 28, 0.98), rgba(6, 15, 20, 0.99))',
            '--editor-shell-border': 'rgba(189,231,243,.14)',
            '--editor-shell-shadow': '0 34px 88px rgba(0,0,0,.44)',
            '--editor-title-strong': '#f5fdff',
            '--editor-title-muted': 'rgba(223,245,250,.84)',
            '--editor-accent': '#B1F1FF',
            '--editor-accent-strong': '#F7FEFF',
            '--editor-panel-bg': 'linear-gradient(160deg, rgba(255,255,255,.04), rgba(255,255,255,.02))',
            '--editor-panel-border': 'rgba(189,231,243,.11)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.018))',
            '--editor-track-border': 'rgba(188,231,242,.10)',
            '--editor-selection-bg': 'rgba(177,241,255,.12)',
            '--editor-selection-border': 'rgba(177,241,255,.30)',
            '--editor-marker': 'rgba(247,254,255,.96)',
            '--editor-chip-bg': 'rgba(177,241,255,.06)',
            '--editor-chip-border': 'rgba(177,241,255,.12)',
            '--editor-button-bg': 'rgba(255,255,255,.04)',
            '--editor-button-border': 'rgba(188,231,242,.12)',
            '--editor-button-hover-bg': 'rgba(177,241,255,.10)',
            '--editor-button-hover-border': 'rgba(188,231,242,.20)',
            '--editor-button-active-bg': 'rgba(177,241,255,.18)',
            '--editor-button-active-border': 'rgba(177,241,255,.34)',
            '--editor-kicker': 'rgba(210,234,240,.66)',
            '--editor-wave-radius': '20px',
          },
        },
        {
          number: 5,
          name: 'clip grid',
          pack: 'solid',
          layout: 'grid',
          buttonShape: 'round',
          trackStyle: 'slab',
          maxWidth: '1380px',
          waveBg: 'rgba(9, 21, 28, 0.92)',
          waveFill: 'rgba(138, 229, 246, 0.84)',
          waveFillDim: 'rgba(106, 184, 202, 0.56)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(13, 28, 35, 0.97), rgba(8, 18, 23, 0.99))',
            '--editor-shell-border': 'rgba(173,223,236,.16)',
            '--editor-shell-shadow': '0 28px 76px rgba(0,0,0,.42)',
            '--editor-title-strong': '#f1fcff',
            '--editor-title-muted': 'rgba(220,243,248,.82)',
            '--editor-accent': '#8DE1F4',
            '--editor-accent-strong': '#F3FDFF',
            '--editor-panel-bg': 'linear-gradient(160deg, rgba(255,255,255,.085), rgba(255,255,255,.035))',
            '--editor-panel-border': 'rgba(173,223,236,.14)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025))',
            '--editor-track-border': 'rgba(173,223,236,.16)',
            '--editor-selection-bg': 'rgba(141,225,244,.14)',
            '--editor-selection-border': 'rgba(141,225,244,.28)',
            '--editor-marker': 'rgba(244,253,255,.97)',
            '--editor-chip-bg': 'rgba(141,225,244,.08)',
            '--editor-chip-border': 'rgba(173,223,236,.15)',
            '--editor-button-bg': 'rgba(141,225,244,.08)',
            '--editor-button-border': 'rgba(173,223,236,.16)',
            '--editor-button-hover-bg': 'rgba(141,225,244,.14)',
            '--editor-button-hover-border': 'rgba(197,235,243,.26)',
            '--editor-button-active-bg': 'rgba(141,225,244,.22)',
            '--editor-button-active-border': 'rgba(141,225,244,.40)',
            '--editor-kicker': 'rgba(214,235,241,.69)',
            '--editor-wave-radius': '18px',
          },
        },
        {
          number: 6,
          name: 'console strip',
          pack: 'retro',
          layout: 'console',
          buttonShape: 'tile',
          trackStyle: 'console',
          maxWidth: '1300px',
          waveBg: 'rgba(7, 18, 24, 0.96)',
          waveFill: 'rgba(121, 215, 232, 0.82)',
          waveFillDim: 'rgba(88, 167, 186, 0.55)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(9, 19, 24, 0.98), rgba(6, 12, 16, 0.99))',
            '--editor-shell-border': 'rgba(140,208,225,.14)',
            '--editor-shell-shadow': '0 34px 92px rgba(0,0,0,.48)',
            '--editor-title-strong': '#eefbfd',
            '--editor-title-muted': 'rgba(208,235,241,.78)',
            '--editor-accent': '#79D7E8',
            '--editor-accent-strong': '#EEFCFF',
            '--editor-panel-bg': 'linear-gradient(160deg, rgba(16,36,42,.96), rgba(9,20,24,.98))',
            '--editor-panel-border': 'rgba(121,215,232,.14)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(17,39,46,.94), rgba(10,21,25,.98))',
            '--editor-track-border': 'rgba(121,215,232,.16)',
            '--editor-selection-bg': 'rgba(121,215,232,.14)',
            '--editor-selection-border': 'rgba(121,215,232,.28)',
            '--editor-marker': 'rgba(241,253,255,.94)',
            '--editor-chip-bg': 'rgba(121,215,232,.07)',
            '--editor-chip-border': 'rgba(121,215,232,.14)',
            '--editor-button-bg': 'rgba(121,215,232,.08)',
            '--editor-button-border': 'rgba(121,215,232,.16)',
            '--editor-button-hover-bg': 'rgba(121,215,232,.14)',
            '--editor-button-hover-border': 'rgba(167,228,238,.24)',
            '--editor-button-active-bg': 'rgba(121,215,232,.24)',
            '--editor-button-active-border': 'rgba(121,215,232,.40)',
            '--editor-kicker': 'rgba(193,224,231,.66)',
            '--editor-wave-radius': '14px',
          },
        },
        {
          number: 7,
          name: 'storyboard',
          pack: 'wire',
          layout: 'story',
          buttonShape: 'pill',
          trackStyle: 'minimal',
          maxWidth: '1280px',
          waveBg: 'rgba(10, 23, 29, 0.92)',
          waveFill: 'rgba(160, 230, 255, 0.82)',
          waveFillDim: 'rgba(110, 193, 220, 0.52)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(16, 31, 38, 0.97), rgba(9, 20, 24, 0.99))',
            '--editor-shell-border': 'rgba(176,226,241,.14)',
            '--editor-shell-shadow': '0 28px 80px rgba(0,0,0,.40)',
            '--editor-title-strong': '#f4fdff',
            '--editor-title-muted': 'rgba(223,245,251,.84)',
            '--editor-accent': '#A6E6FF',
            '--editor-accent-strong': '#F7FDFF',
            '--editor-panel-bg': 'linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.02))',
            '--editor-panel-border': 'rgba(176,226,241,.12)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02))',
            '--editor-track-border': 'rgba(176,226,241,.11)',
            '--editor-selection-bg': 'rgba(166,230,255,.12)',
            '--editor-selection-border': 'rgba(166,230,255,.28)',
            '--editor-marker': 'rgba(247,254,255,.96)',
            '--editor-chip-bg': 'rgba(166,230,255,.06)',
            '--editor-chip-border': 'rgba(176,226,241,.12)',
            '--editor-button-bg': 'rgba(255,255,255,.05)',
            '--editor-button-border': 'rgba(176,226,241,.12)',
            '--editor-button-hover-bg': 'rgba(166,230,255,.10)',
            '--editor-button-hover-border': 'rgba(176,226,241,.22)',
            '--editor-button-active-bg': 'rgba(166,230,255,.19)',
            '--editor-button-active-border': 'rgba(166,230,255,.36)',
            '--editor-kicker': 'rgba(212,236,243,.68)',
            '--editor-wave-radius': '18px',
          },
        },
        {
          number: 8,
          name: 'scout desk',
          pack: 'signal',
          layout: 'splitStack',
          buttonShape: 'round',
          trackStyle: 'glass',
          maxWidth: '1340px',
          waveBg: 'rgba(8, 18, 23, 0.94)',
          waveFill: 'rgba(134, 223, 241, 0.82)',
          waveFillDim: 'rgba(95, 181, 202, 0.54)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(15, 29, 35, 0.97), rgba(8, 18, 22, 0.99))',
            '--editor-shell-border': 'rgba(167,221,236,.15)',
            '--editor-shell-shadow': '0 30px 82px rgba(0,0,0,.42)',
            '--editor-title-strong': '#f0fbff',
            '--editor-title-muted': 'rgba(215,241,247,.82)',
            '--editor-accent': '#86DFF1',
            '--editor-accent-strong': '#F3FDFF',
            '--editor-panel-bg': 'linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.03))',
            '--editor-panel-border': 'rgba(167,221,236,.13)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.025))',
            '--editor-track-border': 'rgba(167,221,236,.14)',
            '--editor-selection-bg': 'rgba(134,223,241,.13)',
            '--editor-selection-border': 'rgba(134,223,241,.28)',
            '--editor-marker': 'rgba(245,253,255,.96)',
            '--editor-chip-bg': 'rgba(134,223,241,.07)',
            '--editor-chip-border': 'rgba(167,221,236,.14)',
            '--editor-button-bg': 'rgba(134,223,241,.08)',
            '--editor-button-border': 'rgba(167,221,236,.16)',
            '--editor-button-hover-bg': 'rgba(134,223,241,.14)',
            '--editor-button-hover-border': 'rgba(194,233,242,.24)',
            '--editor-button-active-bg': 'rgba(134,223,241,.22)',
            '--editor-button-active-border': 'rgba(134,223,241,.40)',
            '--editor-kicker': 'rgba(206,233,240,.68)',
            '--editor-wave-radius': '16px',
          },
        },
        {
          number: 9,
          name: 'pilot deck',
          pack: 'retro',
          layout: 'cockpit',
          buttonShape: 'pill',
          trackStyle: 'console',
          maxWidth: '1260px',
          waveBg: 'rgba(7, 16, 21, 0.96)',
          waveFill: 'rgba(145, 220, 235, 0.82)',
          waveFillDim: 'rgba(100, 171, 188, 0.54)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(10, 20, 25, 0.98), rgba(6, 13, 17, 0.99))',
            '--editor-shell-border': 'rgba(168,220,232,.14)',
            '--editor-shell-shadow': '0 36px 94px rgba(0,0,0,.48)',
            '--editor-title-strong': '#eefbfd',
            '--editor-title-muted': 'rgba(210,236,242,.82)',
            '--editor-accent': '#91DCEB',
            '--editor-accent-strong': '#F1FCFF',
            '--editor-panel-bg': 'linear-gradient(160deg, rgba(18,38,44,.96), rgba(9,18,23,.98))',
            '--editor-panel-border': 'rgba(145,220,235,.14)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(19,42,49,.95), rgba(10,20,24,.98))',
            '--editor-track-border': 'rgba(145,220,235,.16)',
            '--editor-selection-bg': 'rgba(145,220,235,.14)',
            '--editor-selection-border': 'rgba(145,220,235,.28)',
            '--editor-marker': 'rgba(242,253,255,.96)',
            '--editor-chip-bg': 'rgba(145,220,235,.08)',
            '--editor-chip-border': 'rgba(145,220,235,.15)',
            '--editor-button-bg': 'rgba(145,220,235,.08)',
            '--editor-button-border': 'rgba(145,220,235,.16)',
            '--editor-button-hover-bg': 'rgba(145,220,235,.15)',
            '--editor-button-hover-border': 'rgba(187,232,241,.24)',
            '--editor-button-active-bg': 'rgba(145,220,235,.24)',
            '--editor-button-active-border': 'rgba(145,220,235,.42)',
            '--editor-kicker': 'rgba(199,227,234,.67)',
            '--editor-wave-radius': '14px',
          },
        },
        {
          number: 10,
          name: 'broadcast lane',
          pack: 'solid',
          layout: 'broadcast',
          buttonShape: 'tile',
          trackStyle: 'slab',
          maxWidth: '1400px',
          waveBg: 'rgba(9, 22, 28, 0.94)',
          waveFill: 'rgba(186, 235, 255, 0.84)',
          waveFillDim: 'rgba(129, 198, 224, 0.56)',
          theme: {
            '--editor-shell-bg': 'linear-gradient(180deg, rgba(13, 28, 34, 0.97), rgba(7, 16, 20, 0.99))',
            '--editor-shell-border': 'rgba(193,233,245,.16)',
            '--editor-shell-shadow': '0 32px 88px rgba(0,0,0,.44)',
            '--editor-title-strong': '#f4fdff',
            '--editor-title-muted': 'rgba(226,246,252,.84)',
            '--editor-accent': '#C0EEFF',
            '--editor-accent-strong': '#F7FEFF',
            '--editor-panel-bg': 'linear-gradient(155deg, rgba(255,255,255,.10), rgba(255,255,255,.04))',
            '--editor-panel-border': 'rgba(193,233,245,.14)',
            '--editor-track-bg': 'linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.03))',
            '--editor-track-border': 'rgba(193,233,245,.16)',
            '--editor-selection-bg': 'rgba(192,238,255,.14)',
            '--editor-selection-border': 'rgba(192,238,255,.30)',
            '--editor-marker': 'rgba(247,254,255,.98)',
            '--editor-chip-bg': 'rgba(192,238,255,.08)',
            '--editor-chip-border': 'rgba(193,233,245,.16)',
            '--editor-button-bg': 'rgba(192,238,255,.08)',
            '--editor-button-border': 'rgba(193,233,245,.16)',
            '--editor-button-hover-bg': 'rgba(192,238,255,.15)',
            '--editor-button-hover-border': 'rgba(220,244,250,.26)',
            '--editor-button-active-bg': 'rgba(192,238,255,.24)',
            '--editor-button-active-border': 'rgba(192,238,255,.44)',
            '--editor-kicker': 'rgba(220,239,245,.72)',
            '--editor-wave-radius': '18px',
          },
        },
      ];

      currentVariant = editorVariants[0];
      const HISTORY_LIMIT = 20;
      const WAVE_ZOOM_MIN = 1;
      const WAVE_ZOOM_MAX = 128;
      const TIMELINE_BASE_WIDTH = 1600;
      const RULER_HEIGHT = 32;
      const EDITOR_WAVEFORM_POINTS_MIN = 640;
      const EDITOR_WAVEFORM_POINTS_MAX = 131072;
      const EDITOR_WAVEFORM_POINT_STEPS = EDITOR_WAVEFORM_POINT_LEVELS.slice();
      const WAVEFORM_DETAIL_TRANSITION_MS = 180;

      const clampNumber = (value, min, max) => Math.max(min, Math.min(max, value));
      const timelineRenderWidth = (zoomValue = waveZoom) => {
        const numericZoom = Number.isFinite(Number(zoomValue)) ? Number(zoomValue) : 1;
        return Math.max(TIMELINE_BASE_WIDTH, Math.round(TIMELINE_BASE_WIDTH * numericZoom));
      };
      const waveformPointTarget = (zoomValue = waveZoom) => {
        const numericZoom = Number.isFinite(Number(zoomValue)) ? Number(zoomValue) : 1;
        if(numericZoom <= 1.02) return 640;
        if(numericZoom <= 1.08) return 704;
        if(numericZoom <= 1.16) return 768;
        if(numericZoom <= 1.24) return 832;
        if(numericZoom <= 1.34) return 896;
        if(numericZoom <= 1.46) return 960;
        if(numericZoom <= 1.6) return 1024;
        if(numericZoom <= 1.76) return 1152;
        if(numericZoom <= 1.96) return 1280;
        if(numericZoom <= 2.18) return 1408;
        if(numericZoom <= 2.42) return 1536;
        if(numericZoom <= 2.7) return 1664;
        if(numericZoom <= 3.0) return 1792;
        if(numericZoom <= 3.35) return 1920;
        if(numericZoom <= 3.7) return 2048;
        const widthTarget = Math.round(timelineRenderWidth(numericZoom) * 0.76);
        return EDITOR_WAVEFORM_POINT_STEPS.find((value) => value >= widthTarget) || EDITOR_WAVEFORM_POINTS_MAX;
      };
      const resizeHiDPICanvas = (canvas, cssWidth, cssHeight) => {
        if(!canvas) return null;
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        const targetWidth = Math.max(1, Math.round(cssWidth * dpr));
        const targetHeight = Math.max(1, Math.round(cssHeight * dpr));
        if(canvas.width !== targetWidth){
          canvas.width = targetWidth;
        }
        if(canvas.height !== targetHeight){
          canvas.height = targetHeight;
        }
        if(canvas.style.width !== `${Math.round(cssWidth)}px`){
          canvas.style.width = `${Math.round(cssWidth)}px`;
        }
        if(canvas.style.height !== `${Math.round(cssHeight)}px`){
          canvas.style.height = `${Math.round(cssHeight)}px`;
        }
        const ctx = canvas.getContext('2d');
        if(!ctx) return null;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return ctx;
      };

      const appendChildren = (parent, children) => {
        (Array.isArray(children) ? children : []).filter(Boolean).forEach((child) => parent.appendChild(child));
        return parent;
      };

      const makeElement = (tag, className = '', text = '') => {
        const element = document.createElement(tag);
        if(className) element.className = className;
        if(text) element.textContent = text;
        return element;
      };

      const setTooltip = (target, message) => {
        if(!target) return;
        delete target.dataset.tooltip;
        target.removeAttribute('title');
        if(message) target.setAttribute('aria-label', message);
      };

      const svgIcon = (kind, pack = currentVariant ? currentVariant.pack : 'wire') => {
        const wireGroup = '<g fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">';
        const solidOpen = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">';
        const wireOpen = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">';
        const wireEnd = '</g></svg>';
        if(pack === 'solid'){
          if(kind === 'magnet') return `${solidOpen}<path fill="currentColor" d="M8 3h4v6H8zM14 3h4v6h-4z"/><path fill="currentColor" d="M8 10.5a1 1 0 0 1 1 1V18a3 3 0 1 0 6 0v-6.5a1 1 0 0 1 2 0V18a5 5 0 1 1-10 0v-6.5a1 1 0 0 1 1-1Z"/></svg>`;
          if(kind === 'select') return `${solidOpen}<path fill="currentColor" d="m5 4 8.4 3.2-3.2 1.2L16.5 16l-2.5 2.2-6.1-7.2-1.4 3.4z"/></svg>`;
          if(kind === 'razor') return `${solidOpen}<path fill="currentColor" d="M6 7.5A2.5 2.5 0 1 1 8.5 10 2.5 2.5 0 0 1 6 7.5Zm9 9A2.5 2.5 0 1 1 17.5 19 2.5 2.5 0 0 1 15 16.5ZM5.2 18l13-13 1.6 1.6-13 13z"/></svg>`;
          if(kind === 'slip') return `${solidOpen}<path fill="currentColor" d="M6 5h2v14H6zm10 0h2v14h-2zM3 8l3 4-3 4zm18 0-3 4 3 4z"/></svg>`;
          if(kind === 'slide') return `${solidOpen}<path fill="currentColor" d="M4 9h5v6H4zm11 0h5v6h-5zM9 11h6v2H9z"/></svg>`;
          if(kind === 'timeline') return `${solidOpen}<path fill="currentColor" d="M4 5h8v4H4zm5 5h11v4H9zm-5 5h7v4H4z"/></svg>`;
          if(kind === 'monitor') return `${solidOpen}<path fill="currentColor" d="M6 5h12a2 2 0 0 1 2 2v8.5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Zm3 14h6v2H9z"/></svg>`;
          if(kind === 'in') return `${solidOpen}<path fill="currentColor" d="M6 4h4v16H6zm6 1h2v14h-2z"/></svg>`;
          if(kind === 'out') return `${solidOpen}<path fill="currentColor" d="M14 4h4v16h-4zm-4 1h2v14h-2z"/></svg>`;
          if(kind === 'clear') return `${solidOpen}<path fill="currentColor" d="M6 6h12v12H6z" opacity=".35"/><path fill="currentColor" d="m8.4 7 3.6 3.6L15.6 7 17 8.4 13.4 12 17 15.6 15.6 17 12 13.4 8.4 17 7 15.6 10.6 12 7 8.4z"/></svg>`;
          if(kind === 'zoom-in') return `${solidOpen}<path fill="currentColor" d="M10.5 4a6.5 6.5 0 1 1 0 13 6.5 6.5 0 0 1 0-13Zm0 2a.75.75 0 0 0-.75.75V9.75H6.75a.75.75 0 0 0 0 1.5h3V14.25a.75.75 0 0 0 1.5 0v-3h3a.75.75 0 0 0 0-1.5h-3V6.75A.75.75 0 0 0 10.5 6Zm5.9 9.8 3.6 3.6-1.4 1.4-3.6-3.6z"/></svg>`;
          if(kind === 'zoom-out') return `${solidOpen}<path fill="currentColor" d="M10.5 4a6.5 6.5 0 1 1 0 13 6.5 6.5 0 0 1 0-13Zm-3.75 5.75a.75.75 0 0 0 0 1.5h7.5a.75.75 0 0 0 0-1.5Zm9.65 6.05 3.6 3.6-1.4 1.4-3.6-3.6z"/></svg>`;
          if(kind === 'start') return `${solidOpen}<path fill="currentColor" d="M5 5h2v14H5zm4 7 10-6v12z"/></svg>`;
          if(kind === 'jump-in') return `${solidOpen}<path fill="currentColor" d="M6 5h2v14H6zm11 13-8-6 8-6z"/></svg>`;
          if(kind === 'play') return `${solidOpen}<path fill="currentColor" d="M8 5.8v12.4a.8.8 0 0 0 1.2.69l9.2-6.2a.8.8 0 0 0 0-1.38l-9.2-6.2A.8.8 0 0 0 8 5.8Z"/></svg>`;
          if(kind === 'pause') return `${solidOpen}<rect x="7" y="5" width="4" height="14" rx="1.2" fill="currentColor"/><rect x="13" y="5" width="4" height="14" rx="1.2" fill="currentColor"/></svg>`;
          if(kind === 'jump-out') return `${solidOpen}<path fill="currentColor" d="M16 5h2v14h-2zM7 6l8 6-8 6z"/></svg>`;
          if(kind === 'end') return `${solidOpen}<path fill="currentColor" d="M17 5h2v14h-2zM7 6l8 6-8 6z"/></svg>`;
        }
        if(pack === 'retro'){
          if(kind === 'magnet') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="square" stroke-linejoin="miter"><path d="M8 4h4v4H8z"/><path d="M14 4h4v4h-4z"/><path d="M8 9v7a4 4 0 1 0 8 0V9"/></g></svg>`;
          if(kind === 'select') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square" stroke-linejoin="miter"><path d="m5 4 7 6-3 .8L12 18"/><path d="m12 18 3-2"/></g></svg>`;
          if(kind === 'razor') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="square" stroke-linejoin="miter"><path d="M5 19 19 5"/><path d="M7 7h4"/><path d="M13 17h4"/><path d="M6.5 5.5v3"/><path d="M17.5 15.5v3"/></g></svg>`;
          if(kind === 'slip') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square"><path d="M7 6v12"/><path d="M17 6v12"/><path d="M3 9h5"/><path d="M3 15h5"/><path d="M16 9h5"/><path d="M16 15h5"/></g></svg>`;
          if(kind === 'slide') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square"><rect x="4" y="9" width="5" height="6"/><rect x="15" y="9" width="5" height="6"/><path d="M9 12h6"/></g></svg>`;
          if(kind === 'timeline') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square"><path d="M4 7h8"/><path d="M9 12h11"/><path d="M4 17h7"/></g></svg>`;
          if(kind === 'monitor') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square"><rect x="5" y="6" width="14" height="11"/><path d="M9 20h6"/></g></svg>`;
          if(kind === 'in') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="square"><path d="M7 4v16"/><path d="M11 6v12"/></g></svg>`;
          if(kind === 'out') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="square"><path d="M17 4v16"/><path d="M13 6v12"/></g></svg>`;
          if(kind === 'clear') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square"><rect x="7" y="7" width="10" height="10"/><path d="m9 9 6 6"/><path d="m15 9-6 6"/></g></svg>`;
          if(kind === 'zoom-in') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square"><circle cx="10" cy="10" r="5"/><path d="M10 7v6"/><path d="M7 10h6"/><path d="M15 15l5 5"/></g></svg>`;
          if(kind === 'zoom-out') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="square"><circle cx="10" cy="10" r="5"/><path d="M7 10h6"/><path d="M15 15l5 5"/></g></svg>`;
          if(kind === 'start') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="square"><path d="M6 5v14"/><path d="M18 7 10 12l8 5"/></g></svg>`;
          if(kind === 'jump-in') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="square"><path d="M7 5v14"/><path d="m16 18-7-6 7-6"/></g></svg>`;
          if(kind === 'play') return `${wireOpen}<path fill="currentColor" d="M8 5.5v13l9-6.5z"/></svg>`;
          if(kind === 'pause') return `${wireOpen}<g fill="currentColor"><rect x="7" y="5" width="4" height="14"/><rect x="13" y="5" width="4" height="14"/></g></svg>`;
          if(kind === 'jump-out') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="square"><path d="M17 5v14"/><path d="m8 18 7-6-7-6"/></g></svg>`;
          if(kind === 'end') return `${wireOpen}<g fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="square"><path d="M18 5v14"/><path d="M7 7l8 5-8 5"/></g></svg>`;
        }
        if(pack === 'signal'){
          if(kind === 'magnet') return `${wireOpen}${wireGroup}<path d="M8.5 4.5H12V8H8.5z"/><path d="M12 8v3.5a4 4 0 1 0 8 0V8"/><path d="M16.5 4.5H20V8h-3.5"/></g></svg>`;
          if(kind === 'select') return `${wireOpen}${wireGroup}<path d="m5 6 7 4-3 .7L11 18"/></g></svg>`;
          if(kind === 'razor') return `${wireOpen}${wireGroup}<path d="M4.5 18.5 19.5 5.5"/><path d="M8 5.5c1.4 0 2.5 1.1 2.5 2.5"/><path d="M13.5 16c0 1.4 1.1 2.5 2.5 2.5"/></g></svg>`;
          if(kind === 'slip') return `${wireOpen}${wireGroup}<path d="M7 6v12"/><path d="M17 6v12"/><path d="M4 12h5"/><path d="M15 12h5"/></g></svg>`;
          if(kind === 'slide') return `${wireOpen}${wireGroup}<path d="M5 9h4v6H5z"/><path d="M15 9h4v6h-4z"/><path d="M9 12h6"/></g></svg>`;
          if(kind === 'timeline') return `${wireOpen}${wireGroup}<path d="M4 7h7"/><path d="M9 12h11"/><path d="M4 17h8"/></g></svg>`;
          if(kind === 'monitor') return `${wireOpen}${wireGroup}<rect x="5" y="6" width="14" height="11" rx="2"/><path d="M9 20h6"/></g></svg>`;
          if(kind === 'in') return `${wireOpen}${wireGroup}<path d="M8 5v14"/><path d="M12 8h-3"/></g></svg>`;
          if(kind === 'out') return `${wireOpen}${wireGroup}<path d="M16 5v14"/><path d="M12 8h3"/></g></svg>`;
          if(kind === 'clear') return `${wireOpen}${wireGroup}<path d="M7 7h10v10H7z"/><path d="m9 9 6 6"/><path d="m15 9-6 6"/></g></svg>`;
          if(kind === 'zoom-in') return `${wireOpen}${wireGroup}<circle cx="10.5" cy="10.5" r="5"/><path d="M10.5 7.5v6"/><path d="M7.5 10.5h6"/><path d="M15 15 20 20"/></g></svg>`;
          if(kind === 'zoom-out') return `${wireOpen}${wireGroup}<circle cx="10.5" cy="10.5" r="5"/><path d="M7.5 10.5h6"/><path d="M15 15 20 20"/></g></svg>`;
          if(kind === 'start') return `${wireOpen}${wireGroup}<path d="M6 5v14"/><path d="M18 7.5 11 12l7 4.5"/></g></svg>`;
          if(kind === 'jump-in') return `${wireOpen}${wireGroup}<path d="M7 5v14"/><path d="m17 18-7-6 7-6"/></g></svg>`;
          if(kind === 'play') return `${wireOpen}<path fill="currentColor" d="M8.5 6.2v11.6a.7.7 0 0 0 1.05.6l8.3-5.8a.7.7 0 0 0 0-1.2l-8.3-5.8a.7.7 0 0 0-1.05.6Z"/></svg>`;
          if(kind === 'pause') return `${wireOpen}<rect x="7.3" y="5.4" width="3.8" height="13.2" rx="1" fill="currentColor"/><rect x="12.9" y="5.4" width="3.8" height="13.2" rx="1" fill="currentColor"/></svg>`;
          if(kind === 'jump-out') return `${wireOpen}${wireGroup}<path d="M17 5v14"/><path d="m7 18 7-6-7-6"/></g></svg>`;
          if(kind === 'end') return `${wireOpen}${wireGroup}<path d="M18 5v14"/><path d="M7 7.5 14 12l-7 4.5"/></g></svg>`;
        }
        if(kind === 'magnet') return `${wireOpen}${wireGroup}<path d="M9 3.5h3.5V7H9z"/><path d="M11.5 7v2.5a2.5 2.5 0 1 0 5 0V7"/><path d="M14.5 3.5H18V7h-3.5"/><path d="M8.5 11.5V18a3.5 3.5 0 1 0 7 0v-6.5"/>${wireEnd}`;
        if(kind === 'select') return `${wireOpen}${wireGroup}<path d="M5.5 4.5 10 9l-2.5.5.5 2.5L13 19l2-2-7-7 2.5-.5L5.5 4.5z"/><path d="m13 17 3 3"/>${wireEnd}`;
        if(kind === 'razor') return `${wireOpen}${wireGroup}<path d="M5 19 19 5"/><path d="M8 5h3l2.5 3.5"/><path d="M14 16h3l-2.5-3.5"/><circle cx="7" cy="7" r="1.6"/><circle cx="17" cy="17" r="1.6"/>${wireEnd}`;
        if(kind === 'slip') return `${wireOpen}${wireGroup}<path d="M7 7v10"/><path d="M17 7v10"/><path d="M9.5 9.5h-4"/><path d="M9.5 14.5h-4"/><path d="M14.5 9.5h4"/><path d="M14.5 14.5h4"/>${wireEnd}`;
        if(kind === 'slide') return `${wireOpen}${wireGroup}<path d="M5 9h4v6H5z"/><path d="M15 9h4v6h-4z"/><path d="M9 12h6"/>${wireEnd}`;
        if(kind === 'timeline') return `${wireOpen}${wireGroup}<rect x="4" y="5" width="8" height="3.5" rx="1.75"/><rect x="9" y="10.25" width="11" height="3.5" rx="1.75"/><rect x="4" y="15.5" width="7" height="3.5" rx="1.75"/>${wireEnd}`;
        if(kind === 'monitor') return `${wireOpen}${wireGroup}<rect x="5" y="6" width="14" height="12" rx="2.5"/><path d="M9 21h6"/>${wireEnd}`;
        if(kind === 'in') return `${wireOpen}${wireGroup}<path d="M10 5H7v14h3"/>${wireEnd}`;
        if(kind === 'out') return `${wireOpen}${wireGroup}<path d="M14 5h3v14h-3"/>${wireEnd}`;
        if(kind === 'clear') return `${wireOpen}${wireGroup}<path d="M7 7h10v10H7z" stroke-dasharray="2.5 2"/><path d="m8.5 8.5 7 7"/><path d="m15.5 8.5-7 7"/>${wireEnd}`;
        if(kind === 'zoom-in') return `${wireOpen}${wireGroup}<circle cx="10.5" cy="10.5" r="5.5"/><path d="M15 15 20 20"/><path d="M10.5 8v5"/><path d="M8 10.5h5"/>${wireEnd}`;
        if(kind === 'zoom-out') return `${wireOpen}${wireGroup}<circle cx="10.5" cy="10.5" r="5.5"/><path d="M15 15 20 20"/><path d="M8 10.5h5"/>${wireEnd}`;
        if(kind === 'start') return `${wireOpen}${wireGroup}<path d="M6 5v14"/><path d="M18 6l-8 6 8 6V6Z"/>${wireEnd}`;
        if(kind === 'jump-in') return `${wireOpen}${wireGroup}<path d="M7 5v14"/><path d="m17 18-8-6 8-6"/>${wireEnd}`;
        if(kind === 'play') return `${wireOpen}<path fill="currentColor" d="M8 5.8v12.4a.8.8 0 0 0 1.2.69l9.2-6.2a.8.8 0 0 0 0-1.38l-9.2-6.2A.8.8 0 0 0 8 5.8Z"/></svg>`;
        if(kind === 'pause') return `${wireOpen}<rect x="7" y="5" width="4" height="14" rx="1.2" fill="currentColor"/><rect x="13" y="5" width="4" height="14" rx="1.2" fill="currentColor"/></svg>`;
        if(kind === 'jump-out') return `${wireOpen}${wireGroup}<path d="M17 5v14"/><path d="m7 18 8-6-8-6"/>${wireEnd}`;
        if(kind === 'end') return `${wireOpen}${wireGroup}<path d="M18 5v14"/><path d="M6 6l8 6-8 6V6Z"/>${wireEnd}`;
        return '';
      };

      const makeToolButton = ({ iconKind, label = '', compact = true, active = false, title = '', tooltip = '', dataset = null } = {}) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `editor-tool-btn ${active ? 'is-active' : ''}`.trim();
        if(compact) btn.dataset.compact = 'true';
        if(title || tooltip || label) btn.setAttribute('aria-label', tooltip || title || label);
        if(dataset && typeof dataset === 'object'){
          Object.keys(dataset).forEach((key) => {
            btn.dataset[key] = String(dataset[key]);
          });
        }
        btn.dataset.iconKind = iconKind || '';
        if(label){
          btn.dataset.label = label;
          btn.classList.add('editor-tool-btn--label');
          btn.textContent = label;
        }
        if(tooltip) setTooltip(btn, tooltip);
        return btn;
      };

      const buttons = {
        select: makeToolButton({ iconKind: 'select', active: true, title: 'selection', tooltip: 'selection tool', dataset: { editorTool: 'select' } }),
        range: makeToolButton({ label: 'R', title: 'range select', tooltip: 'range select tool', dataset: { editorTool: 'range' } }),
        razor: makeToolButton({ iconKind: 'razor', title: 'razor', tooltip: 'razor tool', dataset: { editorTool: 'razor' } }),
        slip: makeToolButton({ iconKind: 'slip', title: 'trim or slip', tooltip: 'trim or slip tool', dataset: { editorTool: 'slip' } }),
        slide: makeToolButton({ iconKind: 'slide', title: 'slide', tooltip: 'slide tool', dataset: { editorTool: 'slide' } }),
        magnet: makeToolButton({ iconKind: 'magnet', active: snapEnabled, title: snapEnabled ? 'snapping on' : 'snapping off', tooltip: 'toggle trim snapping' }),
        start: makeToolButton({ iconKind: 'start', title: 'skip to beginning', tooltip: 'jump to beginning' }),
        jumpIn: makeToolButton({ iconKind: 'jump-in', title: 'skip to in point', tooltip: 'jump to in point' }),
        play: makeToolButton({ iconKind: 'play', title: 'play', tooltip: 'play or pause - space' }),
        jumpOut: makeToolButton({ iconKind: 'jump-out', title: 'skip to out point', tooltip: 'jump to out point' }),
        end: makeToolButton({ iconKind: 'end', title: 'skip to end', tooltip: 'jump to end' }),
        setIn: makeToolButton({ iconKind: 'in', title: 'set in point', tooltip: 'set in point - I' }),
        setOut: makeToolButton({ iconKind: 'out', title: 'set out point', tooltip: 'set out point - O' }),
        clear: makeToolButton({ iconKind: 'clear', title: 'clear trim', tooltip: 'clear trim' }),
        zoomOut: makeToolButton({ iconKind: 'zoom-out', title: 'zoom out', tooltip: 'zoom out' }),
        zoomIn: makeToolButton({ iconKind: 'zoom-in', title: 'zoom in', tooltip: 'zoom in' }),
        fit: makeToolButton({ label: 'fit', compact: false, title: 'fit timeline', tooltip: 'fit timeline to view' }),
      };

      const refreshButtonIcons = () => {
        Object.values(buttons).forEach((button) => {
          if(button.dataset.label){
            button.textContent = button.dataset.label;
            return;
          }
          const kind = button.dataset.iconKind || '';
          button.innerHTML = svgIcon(kind, currentVariant ? currentVariant.pack : 'wire');
        });
      };

      const meta = document.createElement('div');
      meta.className = 'editor-meta-grid editor-meta-grid--readout w-full';
      meta.style.minWidth = '0';
      meta.innerHTML = `
        <div class="editor-meta-chip"><div class="text-[11px] uppercase tracking-[0.16em] opacity-60">in</div><div class="text-sm font-mono tabular-nums" data-role="in-label">0:00.00</div></div>
        <div class="editor-meta-chip"><div class="text-[11px] uppercase tracking-[0.16em] opacity-60">out</div><div class="text-sm font-mono tabular-nums" data-role="out-label">0:00.00</div></div>
        <div class="editor-meta-chip"><div class="text-[11px] uppercase tracking-[0.16em] opacity-60">playhead</div><div class="text-sm font-mono tabular-nums" data-role="cursor-label">0:00.00</div></div>
      `;
      const inLabel = meta.querySelector('[data-role="in-label"]');
      const outLabel = meta.querySelector('[data-role="out-label"]');
      const cursorLabel = meta.querySelector('[data-role="cursor-label"]');

      const audio = document.createElement('audio');
      audio.preload = 'metadata';

      const baseTracks = [{ key: 'source', label: 'source', output: null }];
      if(task.id && Array.isArray(task.outputs) && task.outputs.length){
        task.outputs.forEach((output, index) => {
          baseTracks.push({
            key: `output:${index}`,
            label: String(output || `output ${index + 1}`).replace(/\.[^.]+$/, ''),
            output: String(output),
          });
        });
      }

      const tracks = baseTracks.map((baseTrack) => {
        const draft = originalTrackDrafts[baseTrack.key];
        return {
          ...baseTrack,
          clip: cloneClipState(baseTrack.key === 'source' ? sourceOriginalClip : (draft || sourceOriginalClip)),
          durationMs: 0,
          points: [],
          mins: [],
          maxs: [],
          pointCount: 0,
          dom: null,
          waveformLoaded: false,
          previewLoaded: false,
        };
      });

      const timelineTracks = tracks.filter((trackItem) => trackItem.key === 'source');
      const markerState = { nextId: 1, items: [] };
      const clipHistoryPast = [];
      const clipHistoryFuture = [];
      const inspectorState = new Map();
      const trackColorFor = (trackItem, index = 0) => {
        const name = String(trackItem && trackItem.label || '').toLowerCase();
        if(name.includes('voc')) return '#74e0ff';
        if(name.includes('inst') || name.includes('other')) return '#61ccb9';
        if(name.includes('bass')) return '#7dd9c8';
        if(name.includes('drum')) return '#8ab4ff';
        return index === 0 ? '#b5c4d1' : '#8ed8ff';
      };
      const compactTrackLabel = (trackItem) => {
        const name = String(trackItem && trackItem.label || '').toLowerCase();
        if(trackItem && trackItem.key === 'source') return 'SRC';
        if(name.includes('voc')) return 'VOC';
        if(name.includes('inst') || name.includes('other')) return 'INST';
        if(name.includes('bass')) return 'BASS';
        if(name.includes('drum')) return 'DRM';
        return String(trackItem && trackItem.label || 'TRK').slice(0, 4).toUpperCase();
      };
      const displayTrackName = (trackItem) => {
        if(trackItem && trackItem.key === 'source') return 'source';
        return String(trackItem && trackItem.label || 'track').replace(/\.[^.]+$/, '');
      };
      tracks.forEach((trackItem, index) => {
        inspectorState.set(trackItem.key, {
          mute: false,
          solo: false,
          exportEnabled: true,
          levelDb: index === 0 ? 0 : Math.max(-12, -(index * 1.5)),
          color: trackColorFor(trackItem, index),
        });
      });

      const activeTrack = () => tracks.find((trackItem) => trackItem.key === activeTrackKey) || tracks[0];

      const closeContextMenu = () => {
        const menu = contextMenuEl;
        if(menu && menu.parentNode){
          menu.parentNode.removeChild(menu);
        }
        contextMenuEl = null;
        const returnFocus = menu && menu.__returnFocus;
        if(returnFocus && returnFocus.isConnected && typeof returnFocus.focus === 'function'){
          try { returnFocus.focus({ preventScroll: true }); } catch(_) { returnFocus.focus(); }
        }
      };

      const normalizeTrackClip = (trackItem) => {
        if(!trackItem) return;
        const safeDuration = Math.max(0, Math.round(Number(trackItem.durationMs) || 0));
        trackItem.clip.startMs = Math.max(0, Math.min(safeDuration, Math.round(Number(trackItem.clip.startMs) || 0)));
        const fallbackEnd = safeDuration > 0 ? safeDuration : null;
        trackItem.clip.endMs = Number.isFinite(Number(trackItem.clip.endMs)) ? Math.round(Number(trackItem.clip.endMs)) : fallbackEnd;
        if(Number.isFinite(Number(trackItem.clip.endMs))){
          trackItem.clip.endMs = Math.max(trackItem.clip.startMs, Math.min(safeDuration, Number(trackItem.clip.endMs)));
        }
        trackItem.clip.enabled = !!trackItem.clip.enabled && safeDuration > 0 && (
          trackItem.clip.startMs > 0 || (trackItem.clip.endMs !== null && Number(trackItem.clip.endMs) < safeDuration)
        );
        if(!trackItem.clip.enabled){
          trackItem.clip.startMs = 0;
          trackItem.clip.endMs = fallbackEnd;
        }
      };

      const nearestSnapValue = (rawMs, movingTrackKey, movingEdge) => {
        if(!snapEnabled) return rawMs;
        const threshold = snapDistanceMs();
        if(threshold <= 0) return rawMs;
        let best = null;
        tracks.forEach((trackItem) => {
          normalizeTrackClip(trackItem);
          const endValue = Number.isFinite(Number(trackItem.clip.endMs)) ? Number(trackItem.clip.endMs) : trackItem.durationMs;
          [['in', trackItem.clip.startMs], ['out', endValue]].forEach(([edge, value]) => {
            if(trackItem.key === movingTrackKey && edge === movingEdge) return;
            const diff = Math.abs(Number(value) - Number(rawMs));
            if(diff > threshold) return;
            if(!best || diff < best.diff){
              best = { value: Number(value), diff };
            }
          });
        });
        return best ? best.value : rawMs;
      };

      const msFromClientX = (trackItem, clientX) => {
        if(!trackItem || !trackItem.dom || !trackItem.dom.overlay) return 0;
        const rect = trackItem.dom.overlay.getBoundingClientRect();
        if(!rect.width) return 0;
        const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        return Math.round(pct * Math.max(0, trackItem.durationMs || 0));
      };

      const snapshotClips = () => tracks.map((trackItem) => ({
        key: trackItem.key,
        clip: cloneClipState(trackItem.clip),
      }));

      const applyTrackWaveformPayload = (trackItem, payload, { preserveClip = false } = {}) => {
        if(!trackItem || !payload || typeof payload !== 'object') return false;
        const nextPoints = Array.isArray(payload.points) ? payload.points : [];
        const nextMins = Array.isArray(payload.mins) ? payload.mins : [];
        const nextMaxs = Array.isArray(payload.maxs) ? payload.maxs : [];
        const nextPointCount = Math.max(
          nextPoints.length,
          nextMins.length,
          nextMaxs.length,
          Math.round(Number(payload.point_count) || 0),
        );
        const currentPointCount = Math.max(0, Number(trackItem.pointCount) || 0);
        const durationMs = Math.max(0, Math.round(Number(payload.duration_ms) || 0));
        const changed = (
          nextPointCount !== currentPointCount ||
          durationMs !== Math.max(0, Math.round(Number(trackItem.durationMs) || 0)) ||
          nextPoints !== trackItem.points ||
          nextMins !== trackItem.mins ||
          nextMaxs !== trackItem.maxs
        );
        const priorDurationMs = Math.max(0, Math.round(Number(trackItem.durationMs) || 0));
        const shouldBlendDetailUpgrade = changed &&
          currentPointCount > 0 &&
          nextPointCount > currentPointCount &&
          nextMins.length &&
          nextMaxs.length &&
          Array.isArray(trackItem.mins) &&
          trackItem.mins.length &&
          Array.isArray(trackItem.maxs) &&
          trackItem.maxs.length &&
          Math.abs(durationMs - priorDurationMs) <= 2;
        if(shouldBlendDetailUpgrade){
          trackItem.waveformTransition = {
            mins: trackItem.mins.slice(),
            maxs: trackItem.maxs.slice(),
            startedAt: (window.performance && typeof window.performance.now === 'function') ? window.performance.now() : Date.now(),
            durationMs: WAVEFORM_DETAIL_TRANSITION_MS,
          };
        }else{
          trackItem.waveformTransition = null;
        }
        trackItem.points = nextPoints;
        trackItem.mins = nextMins;
        trackItem.maxs = nextMaxs;
        trackItem.pointCount = nextPointCount;
        trackItem.durationMs = durationMs;
        if(!preserveClip && trackItem.key === 'source' && !originalTrackDrafts[trackItem.key]){
          trackItem.clip = {
            startMs: Math.max(0, Math.round(Number(payload.clip_start_ms) || 0)),
            endMs: Number.isFinite(Number(payload.clip_end_ms)) ? Math.round(Number(payload.clip_end_ms)) : null,
            enabled: !!payload.clip_enabled,
          };
        }
        if(changed){
          trackItem.waveformRevision = (trackItem.waveformRevision || 0) + 1;
        }
        return changed;
      };

      const snapshotEditorState = () => ({
        clips: snapshotClips(),
        markers: markerState.items.map((marker) => ({ ...marker })),
        nextMarkerId: markerState.nextId,
        inspector: Array.from(inspectorState.entries()).map(([key, state]) => [key, { ...state }]),
        editorToolMode,
        snapEnabled,
      });

      const applyEditorSnapshot = (snapshot) => {
        const map = new Map((((snapshot && snapshot.clips) || []).map((entry) => [entry.key, cloneClipState(entry.clip)])));
        tracks.forEach((trackItem) => {
          if(map.has(trackItem.key)){
            trackItem.clip = cloneClipState(map.get(trackItem.key));
            normalizeTrackClip(trackItem);
          }
        });
        markerState.items = Array.isArray(snapshot && snapshot.markers)
          ? snapshot.markers.map((marker) => ({ ...marker }))
          : [];
        markerState.nextId = Math.max(1, Number(snapshot && snapshot.nextMarkerId) || (markerState.items.length + 1));
        const inspectorEntries = new Map(Array.isArray(snapshot && snapshot.inspector) ? snapshot.inspector.map(([key, state]) => [key, { ...state }]) : []);
        inspectorState.clear();
        tracks.forEach((trackItem, index) => {
          inspectorState.set(trackItem.key, inspectorEntries.get(trackItem.key) || {
            mute: false,
            solo: false,
            exportEnabled: true,
            levelDb: index === 0 ? 0 : Math.max(-12, -(index * 1.5)),
            color: trackColorFor(trackItem, index),
          });
        });
        editorToolMode = String((snapshot && snapshot.editorToolMode) || editorToolMode || 'select');
        snapEnabled = !!(snapshot && snapshot.snapEnabled);
        editorSetSnapEnabled(snapEnabled);
        syncUi();
      };

      const commitHistory = (beforeSnapshot) => {
        const before = beforeSnapshot && typeof beforeSnapshot === 'object' ? beforeSnapshot : snapshotEditorState();
        const after = snapshotEditorState();
        if(JSON.stringify(before) === JSON.stringify(after)) return;
        clipHistoryPast.push(before);
        if(clipHistoryPast.length > HISTORY_LIMIT){
          clipHistoryPast.shift();
        }
        clipHistoryFuture.length = 0;
      };

      const runUndo = () => {
        if(!clipHistoryPast.length) return;
        const current = snapshotEditorState();
        const previous = clipHistoryPast.pop();
        clipHistoryFuture.push(current);
        applyEditorSnapshot(previous);
      };

      const runRedo = () => {
        if(!clipHistoryFuture.length) return;
        const current = snapshotEditorState();
        const next = clipHistoryFuture.pop();
        clipHistoryPast.push(current);
        applyEditorSnapshot(next);
      };

      const makeCommandButton = ({ label = '', title = '', tooltip = '', tone = 'default' } = {}) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `editor-command-btn ${tone === 'primary' ? 'is-primary' : ''}`.trim();
        btn.textContent = label;
        if(title || tooltip || label) btn.setAttribute('aria-label', tooltip || title || label);
        if(tooltip) setTooltip(btn, tooltip);
        return btn;
      };

      const commandButtons = {
        undo: makeCommandButton({ label: 'undo', title: 'undo trim change', tooltip: 'undo trim change' }),
        redo: makeCommandButton({ label: 'redo', title: 'redo trim change', tooltip: 'redo trim change' }),
        split: makeCommandButton({ label: 'split', title: 'activate razor tool', tooltip: 'activate razor tool' }),
        marker: makeCommandButton({ label: 'marker', title: 'drop timeline marker', tooltip: 'drop marker at playhead' }),
      };
      commandButtons.undo.classList.add('is-quiet');
      commandButtons.redo.classList.add('is-quiet');
      saveBtn.className = 'editor-command-btn is-primary';
      cancelBtn.className = 'editor-command-btn';

      const transportPlayhead = makeElement('div', 'editor-transport-readout');
      const transportSelection = makeElement('div', 'editor-transport-readout');
      const transportZoom = makeElement('div', 'editor-transport-readout');

      const renderInspector = () => {
        if(layoutRefs.inspectorTracks){
          layoutRefs.inspectorTracks.innerHTML = '';
        }
      };

      const applyRulerViewportOffset = () => {
        if(!layoutRefs || !layoutRefs.rulerScale) return;
        const scrollLeft = Math.max(0, Math.round(Number(layoutRefs.timelineScrollLeft) || 0));
        layoutRefs.rulerScale.style.transform = `translate3d(${-scrollLeft}px, 0, 0)`;
      };

      const scheduleRulerRender = () => {
        if(!layoutRefs || layoutRefs.rulerRenderScheduled) return;
        layoutRefs.rulerRenderScheduled = true;
        requestAnimationFrame(() => {
          if(layoutRefs){
            layoutRefs.rulerRenderScheduled = false;
          }
          drawRuler();
        });
      };

      const syncTimelineScroll = (scrollLeft, source = null) => {
        if(layoutRefs.syncingScroll) return;
        layoutRefs.syncingScroll = true;
        const nextScrollLeft = Math.max(0, Math.round(Number(scrollLeft) || 0));
        layoutRefs.timelineScrollLeft = nextScrollLeft;
        applyRulerViewportOffset();
        scheduleRulerRender();
        timelineTracks.forEach((trackItem) => {
          if(trackItem.dom && trackItem.dom.zoomWrap && trackItem.dom.zoomWrap !== source){
            trackItem.dom.zoomWrap.scrollLeft = nextScrollLeft;
            scheduleViewportRender(trackItem);
          }
        });
        layoutRefs.syncingScroll = false;
      };

      const drawRuler = () => {
        if(!layoutRefs.rulerCanvas) return;
        const durationMs = Math.max(1000, ...tracks.map((trackItem) => Math.max(0, Number(trackItem.durationMs) || 0)));
        const canvas = layoutRefs.rulerCanvas;
        const renderWidth = timelineRenderWidth();
        const viewportWidth = Math.max(1, Math.round(layoutRefs.rulerWrap && layoutRefs.rulerWrap.clientWidth ? layoutRefs.rulerWrap.clientWidth : renderWidth));
        const scrollLeft = Math.max(0, Math.round(Number(layoutRefs.timelineScrollLeft) || 0));
        const height = RULER_HEIGHT;
        if(layoutRefs.rulerScale){
          layoutRefs.rulerScale.style.width = `${renderWidth}px`;
        }
        if(layoutRefs.markerScale){
          layoutRefs.markerScale.style.width = `${renderWidth}px`;
        }
        applyRulerViewportOffset();
        const overscanPx = Math.min(Math.max(180, Math.round(viewportWidth * 0.4)), 880);
        const bufferWidth = Math.min(renderWidth, viewportWidth + (overscanPx * 2));
        const maxStart = Math.max(0, renderWidth - bufferWidth);
        const currentStart = Math.max(0, Math.round(Number(layoutRefs.rulerRenderedStart) || 0));
        const currentEnd = Math.max(currentStart, Math.round(Number(layoutRefs.rulerRenderedEnd) || 0));
        const withinRenderedWindow = scrollLeft >= currentStart && (scrollLeft + viewportWidth) <= currentEnd;
        const safeLead = overscanPx * 0.45;
        const safeTrail = overscanPx * 0.45;
        const needsRebuffer = !withinRenderedWindow || scrollLeft < (currentStart + safeLead) || (scrollLeft + viewportWidth) > (currentEnd - safeTrail);
        const desiredStart = Math.max(0, Math.min(maxStart, Math.round(scrollLeft - ((bufferWidth - viewportWidth) / 2))));
        const renderStart = needsRebuffer ? desiredStart : currentStart;
        const renderEnd = Math.min(renderWidth, renderStart + bufferWidth);
        const canvasWidth = Math.max(1, renderEnd - renderStart);
        const drawKey = `${durationMs}:${renderWidth}:${viewportWidth}:${renderStart}:${renderEnd}`;
        canvas.style.left = `${renderStart}px`;
        if(layoutRefs.rulerDrawKey === drawKey) return;
        layoutRefs.rulerDrawKey = drawKey;
        layoutRefs.rulerRenderedStart = renderStart;
        layoutRefs.rulerRenderedEnd = renderEnd;
        const ctx = resizeHiDPICanvas(canvas, canvasWidth, height);
        if(!ctx) return;
        ctx.clearRect(0, 0, canvasWidth, height);
        ctx.fillStyle = 'rgba(11, 18, 23, 0.96)';
        ctx.fillRect(0, 0, canvasWidth, height);
        ctx.strokeStyle = 'rgba(255,255,255,.11)';
        ctx.beginPath();
        ctx.moveTo(0, height - 0.5);
        ctx.lineTo(canvasWidth, height - 0.5);
        ctx.stroke();
        const candidates = [250, 500, 1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000, 300000];
        const pixelsPerMs = renderWidth / Math.max(1, durationMs);
        const majorMs = candidates.find((value) => (value * pixelsPerMs) >= 96) || candidates[candidates.length - 1];
        const minorMs = Math.max(100, Math.round(majorMs / 5));
        ctx.font = '11px "Nunito Sans", sans-serif';
        ctx.textBaseline = 'top';
        const startMs = Math.max(0, Math.floor((renderStart / Math.max(1, renderWidth)) * durationMs));
        const endMs = Math.min(durationMs, Math.ceil((renderEnd / Math.max(1, renderWidth)) * durationMs));
        const firstMinorMs = Math.max(0, Math.floor(startMs / minorMs) * minorMs);
        for(let ms = firstMinorMs; ms <= endMs + minorMs; ms += minorMs){
          const x = ((ms / durationMs) * renderWidth) - renderStart;
          const isMajor = ms % majorMs === 0;
          ctx.strokeStyle = isMajor ? 'rgba(236,247,252,.38)' : 'rgba(255,255,255,.05)';
          ctx.beginPath();
          ctx.moveTo(Math.round(x) + 0.5, height - (isMajor ? 20 : 9));
          ctx.lineTo(Math.round(x) + 0.5, height);
          ctx.stroke();
          if(isMajor && x >= -52 && x <= canvasWidth){
            ctx.fillStyle = 'rgba(228,241,247,.9)';
            ctx.fillText(editorRulerLabel(ms), Math.min(canvasWidth - 52, x + 6), 4);
          }
        }
      };

      const renderMarkerLane = () => {
        if(!layoutRefs.markerScale) return;
        const durationMs = Math.max(1000, ...tracks.map((trackItem) => Math.max(0, Number(trackItem.durationMs) || 0)));
        const active = activeTrack();
        const activeOut = active ? (Number.isFinite(Number(active.clip.endMs)) ? Number(active.clip.endMs) : durationMs) : 0;
        const markerKey = JSON.stringify({
          durationMs,
          activeKey: active ? active.key : '',
          inMs: active ? active.clip.startMs : 0,
          outMs: activeOut,
          markers: markerState.items.map((marker) => [marker.id, marker.timeMs, marker.label]),
        });
        if(layoutRefs.markerScale.dataset.renderKey === markerKey) return;
        layoutRefs.markerScale.dataset.renderKey = markerKey;
        Array.from(layoutRefs.markerScale.querySelectorAll('.editor-timeline-marker')).forEach((node) => node.remove());
        if(active){
          const inCue = makeElement('div', 'editor-timeline-marker is-boundary');
          inCue.dataset.label = 'in';
          inCue.style.left = `${(Math.max(0, active.clip.startMs) / durationMs) * 100}%`;
          layoutRefs.markerScale.appendChild(inCue);
          const outCue = makeElement('div', 'editor-timeline-marker is-boundary');
          outCue.dataset.label = 'out';
          outCue.style.left = `${(Math.max(0, activeOut) / durationMs) * 100}%`;
          layoutRefs.markerScale.appendChild(outCue);
        }
        markerState.items.forEach((marker) => {
          const markerEl = makeElement('div', 'editor-timeline-marker');
          markerEl.dataset.label = marker.label;
          markerEl.style.left = `${(Math.max(0, marker.timeMs) / durationMs) * 100}%`;
          layoutRefs.markerScale.appendChild(markerEl);
        });
      };

      const syncHistoryUi = () => {
        commandButtons.undo.disabled = clipHistoryPast.length === 0;
        commandButtons.redo.disabled = clipHistoryFuture.length === 0;
      };

      const buildLayout = () => {
        const root = makeElement('div', 'editor-workspace');
        const topbar = makeElement('div', 'editor-topbar');
        const titleWrap = makeElement('div', 'editor-titlebar');
        const titlePrimary = makeElement('div', 'editor-titlebar-name', songDisplayTitle(task.name || 'untitled'));
        const titleSecondary = makeElement('div', 'editor-titlebar-meta', 'trim and preview');
        titleWrap.append(titlePrimary, titleSecondary);
        const topbarActions = makeElement('div', 'editor-topbar-actions');
        appendChildren(topbarActions, [
          commandButtons.undo,
          commandButtons.redo,
          commandButtons.split,
          commandButtons.marker,
          saveBtn,
          cancelBtn,
        ]);
        topbar.append(titleWrap, topbarActions);

        const main = makeElement('div', 'editor-workspace-main');
        const center = makeElement('div', 'editor-workspace-center');
        const transport = makeElement('div', 'editor-transportbar');
        const transportGroup = makeElement('div', 'editor-transport-group');
        appendChildren(transportGroup, [
          buttons.jumpIn,
          buttons.play,
          buttons.jumpOut,
        ]);
        const zoomGroup = makeElement('div', 'editor-transport-group');
        appendChildren(zoomGroup, [
          buttons.zoomOut,
          buttons.zoomIn,
          buttons.fit,
        ]);
        const transportReadouts = makeElement('div', 'editor-transport-readouts');
        transportReadouts.append(transportPlayhead, transportSelection, transportZoom);
        transport.append(transportGroup, zoomGroup);

        const timelineShell = makeElement('div', 'editor-timeline-shell');
        const ruler = makeElement('div', 'editor-ruler');
        const rulerLabel = makeElement('div', 'editor-ruler-gutter', 'time');
        const rulerWrap = makeElement('div', 'editor-ruler-scroll');
        const rulerScale = makeElement('div', 'editor-ruler-scale');
        const rulerCanvas = document.createElement('canvas');
        rulerCanvas.className = 'editor-ruler-canvas';
        rulerCanvas.width = 1600;
        rulerCanvas.height = 34;
        const markerScale = makeElement('div', 'editor-ruler-markers');
        const rulerPlayhead = makeElement('div', 'editor-timeline-playhead editor-timeline-playhead--ruler');
        rulerScale.append(rulerCanvas, markerScale, rulerPlayhead);
        rulerWrap.appendChild(rulerScale);
        ruler.append(rulerLabel, rulerWrap);

        const laneHost = makeElement('div', 'editor-lane-host');
        const tracksHost = makeElement('div', 'editor-timeline-tracks');
        laneHost.append(tracksHost);
        timelineShell.append(ruler, laneHost);
        const statusbar = makeElement('div', 'editor-statusbar');
        statusbar.append(meta);
        center.append(transport, timelineShell, statusbar);

        main.append(center);
        root.append(topbar, main);

        rulerWrap.addEventListener('wheel', (event) => {
          if(!event.altKey) return;
          event.preventDefault();
          const step = Math.exp(-event.deltaY * 0.0025);
          const sourceWrap = (timelineTracks[0] && timelineTracks[0].dom && timelineTracks[0].dom.zoomWrap) ? timelineTracks[0].dom.zoomWrap : event.currentTarget;
          updateWaveZoom(waveZoom * step, { anchorClientX: event.clientX, sourceWrap });
        }, { passive: false });

        return {
          root,
          tracksHost,
          inspectorTracks: null,
          statusbar,
          rulerCanvas,
          rulerPlayhead,
          rulerWrap,
          rulerScale,
          markerScale,
          transportPlayhead,
          transportSelection,
          transportZoom,
          timelineScrollLeft: 0,
          syncingScroll: false,
        };
      };

      const renderTrackPlacement = () => {
        if(!layoutRefs || !layoutRefs.tracksHost) return;
        layoutRefs.tracksHost.innerHTML = '';
        timelineTracks.forEach((trackItem) => {
          if(trackItem.dom && trackItem.dom.track.parentNode){
            trackItem.dom.track.remove();
          }
          if(trackItem.dom && trackItem.dom.track){
            layoutRefs.tracksHost.appendChild(trackItem.dom.track);
          }
        });
        renderInspector();
        renderMarkerLane();
        drawRuler();
      };

      const applyVariantTheme = (variant) => {
        card.style.maxWidth = variant.maxWidth || '1560px';
        Object.entries(variant.theme || {}).forEach(([key, value]) => {
          card.style.setProperty(key, value);
          editorShell.style.setProperty(key, value);
        });
        editorShell.dataset.buttonShape = variant.buttonShape || 'rail';
        editorShell.dataset.trackStyle = variant.trackStyle || 'console';
        editorShell.dataset.layoutVariant = String(variant.number || '');
        setTooltip(editorShell, `${variant.name} editor workspace`);
      };

      const syncPlayButton = () => {
        const paused = audio.paused;
        buttons.play.innerHTML = paused ? svgIcon('play') : svgIcon('pause');
        buttons.play.setAttribute('aria-label', paused ? 'play' : 'pause');
        buttons.play.classList.toggle('is-active', !paused);
      };

      const syncPlaybackFrame = () => {
        playbackFrame = 0;
        if(audio.paused) return;
        const nextPlayheadMs = Math.round((Number(audio.currentTime) || 0) * 1000);
        if(nextPlayheadMs !== playheadMs){
          playheadMs = nextPlayheadMs;
          syncTransportUi();
        }
        playbackFrame = requestAnimationFrame(syncPlaybackFrame);
      };

      const startPlaybackLoop = () => {
        if(playbackFrame) return;
        playbackFrame = requestAnimationFrame(syncPlaybackFrame);
      };

      const stopPlaybackLoop = () => {
        if(!playbackFrame) return;
        cancelAnimationFrame(playbackFrame);
        playbackFrame = 0;
      };

      const waveformTransitionNow = () => (
        (window.performance && typeof window.performance.now === 'function') ? window.performance.now() : Date.now()
      );

      const activeWaveformTransition = (trackItem) => {
        if(!trackItem || !trackItem.waveformTransition) return null;
        const transition = trackItem.waveformTransition;
        const durationMs = Math.max(1, Number(transition.durationMs) || WAVEFORM_DETAIL_TRANSITION_MS);
        const progress = clampNumber((waveformTransitionNow() - Number(transition.startedAt || 0)) / durationMs, 0, 1);
        if(progress >= 0.999){
          trackItem.waveformTransition = null;
          return null;
        }
        return {
          ...transition,
          progress,
        };
      };

      const waveformProjectionForRange = ({
        mins,
        maxs,
        renderStart,
        renderEnd,
        renderWidth,
        canvasWidth,
        centerY,
        amplitudeScale,
        densityBias = 1,
      }) => {
        const sampleCount = Math.max(1, Math.min(
          Array.isArray(mins) ? mins.length : 0,
          Array.isArray(maxs) ? maxs.length : 0,
        ));
        const visibleStartIndex = clampNumber(Math.floor((renderStart / Math.max(1, renderWidth)) * sampleCount) - 1, 0, sampleCount - 1);
        const visibleEndIndex = clampNumber(Math.ceil((renderEnd / Math.max(1, renderWidth)) * sampleCount) + 1, visibleStartIndex + 1, sampleCount);
        const visibleCount = Math.max(1, visibleEndIndex - visibleStartIndex);
        const columnTarget = Math.max(1, Math.min(
          visibleCount,
          Math.ceil(canvasWidth * densityBias),
          2400,
        ));
        const projectedUpper = new Array(columnTarget);
        const projectedLower = new Array(columnTarget);
        const projectedX = new Array(columnTarget);
        for(let column = 0; column < columnTarget; column += 1){
          const sliceStart = Math.floor((column / columnTarget) * visibleCount);
          const sliceEnd = Math.max(sliceStart + 1, Math.min(visibleCount, Math.ceil(((column + 1) / columnTarget) * visibleCount)));
          let minValue = 1;
          let maxValue = -1;
          for(let offset = sliceStart; offset < sliceEnd; offset += 1){
            const index = visibleStartIndex + offset;
            const sampleMin = Math.max(-1, Math.min(0, Number(mins[index] ?? 0)));
            const sampleMax = Math.max(0, Math.min(1, Number(maxs[index] ?? 0)));
            if(sampleMin < minValue) minValue = sampleMin;
            if(sampleMax > maxValue) maxValue = sampleMax;
          }
          if(maxValue < 0) maxValue = 0;
          if(minValue > 0) minValue = 0;
          projectedUpper[column] = centerY - (maxValue * amplitudeScale);
          projectedLower[column] = centerY - (minValue * amplitudeScale);
          projectedX[column] = columnTarget <= 1 ? 0 : (column / (columnTarget - 1)) * Math.max(1, canvasWidth - 1);
        }
        return { projectedUpper, projectedLower, projectedX, columnTarget };
      };

      const drawProjectedWaveform = (ctx, projection, { fillColor, strokeColor, alpha = 1 } = {}) => {
        if(!ctx || !projection || !projection.columnTarget) return;
        const safeAlpha = clampNumber(Number(alpha) || 0, 0, 1);
        if(safeAlpha <= 0.001) return;
        const { projectedUpper, projectedLower, projectedX, columnTarget } = projection;
        const priorAlpha = ctx.globalAlpha;
        ctx.globalAlpha = priorAlpha * safeAlpha;
        ctx.beginPath();
        ctx.moveTo(projectedX[0] ?? 0, projectedUpper[0] ?? 0);
        for(let offset = 1; offset < columnTarget; offset += 1){
          ctx.lineTo(projectedX[offset] ?? 0, projectedUpper[offset] ?? 0);
        }
        for(let offset = columnTarget - 1; offset >= 0; offset -= 1){
          ctx.lineTo(projectedX[offset] ?? 0, projectedLower[offset] ?? 0);
        }
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(projectedX[0] ?? 0, projectedUpper[0] ?? 0);
        for(let offset = 1; offset < columnTarget; offset += 1){
          ctx.lineTo(projectedX[offset] ?? 0, projectedUpper[offset] ?? 0);
        }
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(projectedX[0] ?? 0, projectedLower[0] ?? 0);
        for(let offset = 1; offset < columnTarget; offset += 1){
          ctx.lineTo(projectedX[offset] ?? 0, projectedLower[offset] ?? 0);
        }
        ctx.stroke();
        ctx.globalAlpha = priorAlpha;
      };

      const drawTrackWaveform = (trackItem) => {
        if(!trackItem || !trackItem.dom || !trackItem.dom.ctx) return;
        const { canvas, zoomWrap } = trackItem.dom;
        if(!zoomWrap) return;
        const shellHeight = Math.round(trackItem.dom.waveShell && trackItem.dom.waveShell.getBoundingClientRect ? trackItem.dom.waveShell.getBoundingClientRect().height : 0);
        const cssHeight = Math.max(34, Math.round(shellHeight || canvas.getBoundingClientRect().height || (trackItem.key === 'source' ? 528 : 34)));
        const viewportWidth = Math.max(1, Math.round(zoomWrap.clientWidth || 0));
        const height = cssHeight;
        const renderWidth = timelineRenderWidth();
        const scrollLeft = Math.max(0, Math.round(zoomWrap.scrollLeft || 0));
        const overscanPx = Math.min(Math.max(180, Math.round(viewportWidth * 0.35)), 720);
        const bufferWidth = Math.min(renderWidth, viewportWidth + (overscanPx * 2));
        const maxStart = Math.max(0, renderWidth - bufferWidth);
        const currentStart = Math.max(0, Math.round(Number(trackItem.dom.renderedStart) || 0));
        const currentEnd = Math.max(currentStart, Math.round(Number(trackItem.dom.renderedEnd) || 0));
        const withinRenderedWindow = scrollLeft >= currentStart && (scrollLeft + viewportWidth) <= currentEnd;
        const safeLead = overscanPx * 0.45;
        const safeTrail = overscanPx * 0.45;
        const active = trackItem.key === activeTrackKey;
        const needsRebuffer = !withinRenderedWindow || scrollLeft < (currentStart + safeLead) || (scrollLeft + viewportWidth) > (currentEnd - safeTrail);
        const desiredStart = Math.max(0, Math.min(maxStart, Math.round(scrollLeft - ((bufferWidth - viewportWidth) / 2))));
        const renderStart = needsRebuffer ? desiredStart : currentStart;
        const renderEnd = Math.min(renderWidth, renderStart + bufferWidth);
        const canvasWidth = Math.max(1, renderEnd - renderStart);
        const transition = activeWaveformTransition(trackItem);
        const transitionFrame = transition ? Math.round(transition.progress * 12) : 0;
        const drawKey = `${viewportWidth}:${canvasWidth}:${height}:${renderStart}:${renderEnd}:${renderWidth}:${active ? 1 : 0}:${trackItem.waveformRevision || 0}:${transitionFrame}`;
        canvas.style.left = `${renderStart}px`;
        if(trackItem.dom.waveformDrawKey === drawKey){
          if(transition){
            scheduleViewportRender(trackItem);
          }
          return;
        }
        trackItem.dom.waveformDrawKey = drawKey;
        trackItem.dom.renderedStart = renderStart;
        trackItem.dom.renderedEnd = renderEnd;
        const ctx = resizeHiDPICanvas(canvas, canvasWidth, height);
        if(!ctx) return;
        ctx.clearRect(0, 0, canvasWidth, height);
        ctx.fillStyle = currentVariant ? currentVariant.waveBg : 'rgba(12, 24, 31, 0.96)';
        ctx.fillRect(0, 0, canvasWidth, height);
        const values = Array.isArray(trackItem.points) && trackItem.points.length ? trackItem.points : new Array(220).fill(0.12);
        const mins = Array.isArray(trackItem.mins) && trackItem.mins.length ? trackItem.mins : values.map((value) => -(Number(value) || 0));
        const maxs = Array.isArray(trackItem.maxs) && trackItem.maxs.length ? trackItem.maxs : values.map((value) => Number(value) || 0);
        const centerY = height / 2;
        const strokeColor = active
          ? (currentVariant ? currentVariant.waveFill : 'rgba(214, 240, 252, 0.96)')
          : 'rgba(189, 220, 234, 0.82)';
        const fillColor = active
          ? 'rgba(197, 229, 244, 0.96)'
          : 'rgba(173, 209, 224, 0.82)';
        const centerLineColor = active ? 'rgba(255,255,255,0.16)' : 'rgba(255,255,255,0.10)';
        const verticalPadding = trackItem.key === 'source' ? 4 : 8;
        const amplitudeScale = Math.max(12, (height / 2) - verticalPadding);
        const densityBias = trackItem.key === 'source' ? 1.15 : 1;
        const nextProjection = waveformProjectionForRange({
          mins,
          maxs,
          renderStart,
          renderEnd,
          renderWidth,
          canvasWidth,
          centerY,
          amplitudeScale,
          densityBias,
        });
        if(transition && Array.isArray(transition.mins) && Array.isArray(transition.maxs)){
          const priorProjection = waveformProjectionForRange({
            mins: transition.mins,
            maxs: transition.maxs,
            renderStart,
            renderEnd,
            renderWidth,
            canvasWidth,
            centerY,
            amplitudeScale,
            densityBias,
          });
          drawProjectedWaveform(ctx, priorProjection, {
            fillColor,
            strokeColor,
            alpha: 1 - transition.progress,
          });
          drawProjectedWaveform(ctx, nextProjection, {
            fillColor,
            strokeColor,
            alpha: transition.progress,
          });
          scheduleViewportRender(trackItem);
        }else{
          drawProjectedWaveform(ctx, nextProjection, {
            fillColor,
            strokeColor,
            alpha: 1,
          });
        }
        ctx.beginPath();
        ctx.moveTo(0, centerY + 0.5);
        ctx.lineTo(canvasWidth, centerY + 0.5);
        ctx.strokeStyle = centerLineColor;
        ctx.stroke();
      };

      const scheduleViewportRender = (trackItem) => {
        if(!trackItem || !trackItem.dom || trackItem.dom.waveformRenderScheduled) return;
        trackItem.dom.waveformRenderScheduled = true;
        requestAnimationFrame(() => {
          if(trackItem.dom){
            trackItem.dom.waveformRenderScheduled = false;
          }
          syncTrackUi(trackItem);
        });
      };

      const syncTrackUi = (trackItem) => {
        if(!trackItem || !trackItem.dom) return;
        normalizeTrackClip(trackItem);
        drawTrackWaveform(trackItem);
        const safeDuration = Math.max(1, trackItem.durationMs || 1);
        const currentPlayhead = Math.max(0, Math.min(trackItem.durationMs || 0, playheadMs));
        const endValue = Number.isFinite(Number(trackItem.clip.endMs)) ? Number(trackItem.clip.endMs) : safeDuration;
        const startPct = (trackItem.clip.startMs / safeDuration) * 100;
        const endPct = (endValue / safeDuration) * 100;
        const headPct = (currentPlayhead / safeDuration) * 100;
        const trimActive = !!trackItem.clip.enabled;
        trackItem.dom.selection.style.left = `${startPct}%`;
        trackItem.dom.selection.style.width = `${Math.max(0, endPct - startPct)}%`;
        trackItem.dom.selection.classList.toggle('is-hidden', !trimActive);
        if(trackItem.dom.dimBefore){
          trackItem.dom.dimBefore.style.width = trimActive ? `${Math.max(0, startPct)}%` : '0%';
          trackItem.dom.dimBefore.classList.toggle('is-hidden', !trimActive || startPct <= 0.05);
        }
        if(trackItem.dom.dimAfter){
          trackItem.dom.dimAfter.style.width = trimActive ? `${Math.max(0, 100 - endPct)}%` : '0%';
          trackItem.dom.dimAfter.classList.toggle('is-hidden', !trimActive || endPct >= 99.95);
        }
        trackItem.dom.inMarker.style.left = `${startPct}%`;
        trackItem.dom.outMarker.style.left = `${endPct}%`;
        trackItem.dom.playhead.style.left = `${headPct}%`;
        trackItem.dom.track.classList.toggle('is-active', trackItem.key === activeTrackKey);
        if(trackItem.dom.muteBtn || trackItem.dom.soloBtn){
          const state = inspectorState.get(trackItem.key) || { mute: false, solo: false };
          if(trackItem.dom.muteBtn){
            trackItem.dom.muteBtn.classList.toggle('is-active', !!state.mute);
          }
          if(trackItem.dom.soloBtn){
            trackItem.dom.soloBtn.classList.toggle('is-active', !!state.solo);
          }
        }
      };

      const syncTrackTransportUi = (trackItem) => {
        if(!trackItem || !trackItem.dom) return;
        const safeDuration = Math.max(1, trackItem.durationMs || 1);
        const currentPlayhead = Math.max(0, Math.min(trackItem.durationMs || 0, playheadMs));
        const headPct = (currentPlayhead / safeDuration) * 100;
        trackItem.dom.playhead.style.left = `${headPct}%`;
      };

      const syncMetaUi = () => {
        const trackItem = activeTrack();
        normalizeTrackClip(trackItem);
        const safeDuration = Math.max(0, trackItem ? trackItem.durationMs : 0);
        inLabel.textContent = editorTimeLabel(trackItem ? trackItem.clip.startMs : 0);
        outLabel.textContent = editorTimeLabel(trackItem ? (Number.isFinite(Number(trackItem.clip.endMs)) ? Number(trackItem.clip.endMs) : safeDuration) : 0);
        cursorLabel.textContent = editorTimeLabel(playheadMs);
        if(layoutRefs.transportPlayhead){
          layoutRefs.transportPlayhead.textContent = `P ${editorTimeLabel(playheadMs)}`;
        }
        if(layoutRefs.transportSelection){
          const outValue = trackItem ? (Number.isFinite(Number(trackItem.clip.endMs)) ? Number(trackItem.clip.endMs) : safeDuration) : 0;
          const selectionMs = Math.max(0, outValue - (trackItem ? trackItem.clip.startMs : 0));
          layoutRefs.transportSelection.textContent = `SEL ${editorTimeLabel(selectionMs)}`;
        }
        if(layoutRefs.transportZoom){
          layoutRefs.transportZoom.textContent = `${waveZoom.toFixed(2)}x`;
        }
        const maxDuration = Math.max(1, ...tracks.map((entry) => Math.max(0, Number(entry.durationMs) || 0)));
        const renderWidth = timelineRenderWidth();
        const playheadX = (Math.max(0, Math.min(maxDuration, playheadMs)) / maxDuration) * renderWidth;
        if(layoutRefs.rulerPlayhead){
          layoutRefs.rulerPlayhead.style.left = `${playheadX}px`;
        }
      };

      const applyWaveZoom = () => {
        const renderWidth = timelineRenderWidth();
        timelineTracks.forEach((trackItem) => {
          if(!trackItem.dom || !trackItem.dom.waveShell || !trackItem.dom.zoomWrap) return;
          trackItem.dom.waveShell.style.width = `${renderWidth}px`;
          trackItem.dom.zoomWrap.style.overflowX = waveZoom > 1 ? 'auto' : 'hidden';
          if(waveZoom <= 1){
            trackItem.dom.zoomWrap.scrollLeft = 0;
          }
        });
        if(layoutRefs.rulerWrap){
          layoutRefs.rulerWrap.style.overflowX = 'hidden';
        }
        if(waveZoom <= 1){
          layoutRefs.timelineScrollLeft = 0;
        }
        applyRulerViewportOffset();
      };

      const updateWaveZoom = (nextZoom, { anchorClientX = null, sourceWrap = null } = {}) => {
        const clampedZoom = clampNumber(Math.round(Number(nextZoom || 1) * 100) / 100, WAVE_ZOOM_MIN, WAVE_ZOOM_MAX);
        if(Math.abs(clampedZoom - waveZoom) < 0.001) return;
        const wrap = sourceWrap || (timelineTracks[0] && timelineTracks[0].dom ? timelineTracks[0].dom.zoomWrap : null) || layoutRefs.rulerWrap;
        let offsetX = 0;
        let progress = 0;
        if(wrap && Number.isFinite(Number(anchorClientX))){
          const rect = wrap.getBoundingClientRect();
          offsetX = clampNumber((Number(anchorClientX) || 0) - rect.left, 0, rect.width);
          const priorWidth = timelineRenderWidth(waveZoom);
          progress = priorWidth > 0 ? (wrap.scrollLeft + offsetX) / priorWidth : 0;
        }
        waveZoom = clampedZoom;
        const requestedPointTarget = waveformPointTarget(clampedZoom);
        syncUi();
        if(task.id){
          timelineTracks.forEach((trackItem) => {
            const cachedPayload = bestCachedEditorWaveformPayload(task.id, trackItem.output || '', requestedPointTarget);
            if(cachedPayload){
              applyTrackWaveformPayload(trackItem, cachedPayload, { preserveClip: true });
            }
            if((trackItem.pointCount || 0) >= requestedPointTarget) return;
            if(trackItem.detailFetchTimer){
              clearTimeout(trackItem.detailFetchTimer);
            }
            trackItem.detailFetchTimer = setTimeout(() => {
              trackItem.detailFetchTimer = null;
              ensureEditorWaveformPayload(task.id, trackItem.output || '', requestedPointTarget)
                .then((payload) => {
                if(!payload || !overlay.isConnected) return;
                const nextPointCount = Math.max(
                  Array.isArray(payload.points) ? payload.points.length : 0,
                  Array.isArray(payload.mins) ? payload.mins.length : 0,
                  Array.isArray(payload.maxs) ? payload.maxs.length : 0,
                  Math.round(Number(payload.point_count) || 0),
                );
                if(nextPointCount <= (trackItem.pointCount || 0)) return;
                applyTrackWaveformPayload(trackItem, payload, { preserveClip: true });
                const stepIndex = EDITOR_WAVEFORM_POINT_STEPS.indexOf(requestedPointTarget);
                const warmSteps = stepIndex >= 0
                  ? EDITOR_WAVEFORM_POINT_STEPS.slice(stepIndex + 1, stepIndex + 2)
                  : [];
                for(const nextStep of warmSteps){
                  if(nextStep && nextStep <= EDITOR_WAVEFORM_POINTS_MAX){
                    void ensureEditorWaveformPayload(task.id, trackItem.output || '', nextStep);
                  }
                }
                syncUi();
              })
              .catch(() => {});
            }, 140);
          });
        }
        if(wrap && Number.isFinite(Number(anchorClientX))){
          const nextWidth = timelineRenderWidth(waveZoom);
          const maxScroll = Math.max(0, nextWidth - wrap.clientWidth);
          syncTimelineScroll(clampNumber((progress * nextWidth) - offsetX, 0, maxScroll));
        }
      };

      const syncEditorToolUi = () => {
        [buttons.select, buttons.range, buttons.razor, buttons.slip].forEach((button) => {
          const mode = button.dataset.editorTool;
          if(!mode) return;
          button.classList.toggle('is-active', editorToolMode === mode);
        });
        commandButtons.split.classList.toggle('is-active', editorToolMode === 'razor');
      };

      const syncUi = () => {
        applyWaveZoom();
        tracks.forEach(syncTrackUi);
        syncMetaUi();
        syncEditorToolUi();
        buttons.magnet.classList.toggle('is-active', snapEnabled);
        buttons.magnet.setAttribute('aria-label', snapEnabled ? 'snapping on' : 'snapping off');
        renderInspector();
        renderMarkerLane();
        drawRuler();
        syncHistoryUi();
        syncPlayButton();
      };

      const syncTransportUi = () => {
        timelineTracks.forEach(syncTrackTransportUi);
        syncMetaUi();
        syncPlayButton();
      };

      const loadPreviewForTrack = async (trackItem, { preservePlayhead = true, force = false } = {}) => {
        if(!trackItem) return;
        if(!force && loadedPreviewTrackKey === trackItem.key && audio.src){
          return;
        }
        stopPlaybackLoop();
        const resumeAfterLoad = !audio.paused && loadedPreviewTrackKey === trackItem.key;
        audio.pause();
        if(activeObjectUrl){
          URL.revokeObjectURL(activeObjectUrl);
          activeObjectUrl = '';
        }
        if(task.id){
          const previewUrl = trackItem.output
            ? `/api/tasks/${task.id}/preview_audio?output=${encodeURIComponent(trackItem.output)}`
            : `/api/tasks/${task.id}/preview_audio`;
          audio.src = previewUrl;
        }else if(pending && pending.file){
          activeObjectUrl = URL.createObjectURL(pending.file);
          audio.src = activeObjectUrl;
        }
        loadedPreviewTrackKey = trackItem.key;
        audio.load();
        trackItem.previewLoaded = true;
        const targetMs = Math.max(0, Math.min(trackItem.durationMs || 0, playheadMs));
        const seek = async () => {
          try { audio.currentTime = targetMs / 1000; } catch(_) {}
          if(resumeAfterLoad){
            try { await audio.play(); } catch(_) {}
          }
          audio.removeEventListener('loadedmetadata', seek);
        };
        if(preservePlayhead || resumeAfterLoad){
          audio.addEventListener('loadedmetadata', seek);
        }
      };

      const loadWaveformForTrack = async (trackItem) => {
        if(!trackItem || trackItem.waveformLoaded) return;
        const requestedPointTarget = waveformPointTarget();
        if(task.id){
          const cachedPayload = bestCachedEditorWaveformPayload(task.id, trackItem.output || '', requestedPointTarget);
          if(cachedPayload){
            applyTrackWaveformPayload(trackItem, cachedPayload);
          }
        }
        if(task.id){
          try{
            if((trackItem.pointCount || 0) < requestedPointTarget){
              const payload = await ensureEditorWaveformPayload(task.id, trackItem.output || '', requestedPointTarget);
              if(payload){
                applyTrackWaveformPayload(trackItem, payload);
              }
            }
          }catch(_){}
        }
        trackItem.waveformLoaded = true;
        normalizeTrackClip(trackItem);
        syncUi();
      };

      const setActiveTrack = async (trackKey, { preservePlayhead = true } = {}) => {
        const changed = activeTrackKey !== trackKey;
        activeTrackKey = trackKey;
        if(changed){
          renderTrackPlacement();
        }
        syncUi();
        if(changed || !loadedPreviewTrackKey){
          await loadPreviewForTrack(activeTrack(), { preservePlayhead, force: changed });
        }
      };

      const setPlayhead = (nextMs) => {
        playheadMs = Math.max(0, Math.round(Number(nextMs) || 0));
        try { audio.currentTime = playheadMs / 1000; } catch(_) {}
        syncTransportUi();
      };

      const setInPoint = () => {
        const trackItem = activeTrack();
        trackItem.clip.startMs = nearestSnapValue(playheadMs, trackItem.key, 'in');
        trackItem.clip.enabled = true;
        syncUi();
      };

      const setOutPoint = () => {
        const trackItem = activeTrack();
        trackItem.clip.endMs = nearestSnapValue(playheadMs, trackItem.key, 'out');
        trackItem.clip.enabled = true;
        syncUi();
      };

      const togglePlayback = async () => {
        const trackItem = activeTrack();
        await loadPreviewForTrack(trackItem);
        if(audio.paused){
          try { audio.currentTime = playheadMs / 1000; } catch(_) {}
          try { await audio.play(); } catch(_) {}
        }else{
          audio.pause();
          stopPlaybackLoop();
        }
        syncTransportUi();
      };

      const openContextMenuForTrack = (trackItem, clientX, clientY) => {
        closeContextMenu();
        const menu = document.createElement('div');
        menu.className = 'editor-wave-context';
        menu.setAttribute('role', 'menu');
        menu.setAttribute('aria-label', 'waveform trim actions');
        menu.__returnFocus = document.activeElement;
        const copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.textContent = 'copy trim';
        copyBtn.setAttribute('role', 'menuitem');
        const pasteBtn = document.createElement('button');
        pasteBtn.type = 'button';
        pasteBtn.textContent = 'paste trim';
        pasteBtn.setAttribute('role', 'menuitem');
        const clipboard = editorReadClipboard();
        pasteBtn.disabled = !clipboard;
        pasteBtn.style.opacity = clipboard ? '1' : '.4';
        guardClick(copyBtn, (event) => {
          event.preventDefault();
          editorWriteClipboard(trackItem.clip);
          showPopup('copied trim');
          closeContextMenu();
        }, 150);
        guardClick(pasteBtn, (event) => {
          event.preventDefault();
          const nextClip = editorReadClipboard();
          if(!nextClip) return;
          const beforeSnapshot = snapshotEditorState();
          trackItem.clip = cloneClipState(nextClip);
          normalizeTrackClip(trackItem);
          commitHistory(beforeSnapshot);
          syncUi();
          closeContextMenu();
        }, 150);
        menu.append(copyBtn, pasteBtn);
        document.body.appendChild(menu);
        const bounds = menu.getBoundingClientRect();
        menu.style.left = `${Math.max(12, Math.min(window.innerWidth - bounds.width - 12, clientX))}px`;
        menu.style.top = `${Math.max(12, Math.min(window.innerHeight - bounds.height - 12, clientY))}px`;
        menu.addEventListener('keydown', (event) => {
          if(event.key === 'Escape'){
            event.preventDefault();
            closeContextMenu();
            return;
          }
          if(!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const items = [copyBtn, pasteBtn].filter((button) => !button.disabled);
          const current = items.indexOf(document.activeElement);
          let next = 0;
          if(event.key === 'End') next = items.length - 1;
          else if(event.key === 'ArrowUp') next = current <= 0 ? items.length - 1 : current - 1;
          else if(event.key === 'ArrowDown') next = current < 0 || current >= items.length - 1 ? 0 : current + 1;
          items[next].focus();
        });
        contextMenuEl = menu;
        requestAnimationFrame(() => copyBtn.focus({ preventScroll: true }));
      };

      overlay.__dismissGuard = () => !!dragState || Date.now() < dismissLockUntil;

      const createTrackUi = (trackItem) => {
        const trackEl = document.createElement('div');
        trackEl.className = `editor-track ${trackItem.placeholder ? 'editor-track--placeholder' : ''}`.trim();
        if(trackItem.key === 'source'){
          trackEl.classList.add('editor-track--source');
        }
        const gutter = document.createElement('div');
        gutter.className = 'editor-track-gutter';
        gutter.style.cursor = trackItem.placeholder ? 'default' : 'pointer';
        const chip = document.createElement('span');
        chip.className = 'editor-track-chip';
        chip.style.background = trackItem.color || (inspectorState.get(trackItem.key) || {}).color || '#8ed8ff';
        const titleWrap = document.createElement('div');
        titleWrap.className = 'editor-track-gutter-copy';
        const title = document.createElement('div');
        title.className = 'editor-track-title';
        title.textContent = compactTrackLabel(trackItem);
        const subtitle = document.createElement('div');
        subtitle.className = 'editor-track-subtitle';
        subtitle.textContent = displayTrackName(trackItem);
        titleWrap.append(title, subtitle);
        if(trackItem.key === 'source'){
          const state = inspectorState.get(trackItem.key) || { mute: false, solo: false };
          const laneControls = document.createElement('div');
          laneControls.className = 'editor-track-controls';
          const muteBtn = document.createElement('button');
          muteBtn.type = 'button';
          muteBtn.className = `editor-mini-toggle editor-mini-toggle--lane ${state.mute ? 'is-active' : ''}`.trim();
          muteBtn.textContent = 'M';
          const soloBtn = document.createElement('button');
          soloBtn.type = 'button';
          soloBtn.className = `editor-mini-toggle editor-mini-toggle--lane ${state.solo ? 'is-active' : ''}`.trim();
          soloBtn.textContent = 'S';
          setTooltip(muteBtn, 'mute source lane');
          setTooltip(soloBtn, 'solo source lane');
          guardClick(muteBtn, (event) => {
            event.preventDefault();
            event.stopPropagation();
            const beforeSnapshot = snapshotEditorState();
            const nextState = inspectorState.get(trackItem.key) || { mute: false, solo: false, exportEnabled: true, levelDb: 0, color: trackColorFor(trackItem, 0) };
            nextState.mute = !nextState.mute;
            inspectorState.set(trackItem.key, nextState);
            commitHistory(beforeSnapshot);
            syncUi();
          }, 120);
          guardClick(soloBtn, (event) => {
            event.preventDefault();
            event.stopPropagation();
            const beforeSnapshot = snapshotEditorState();
            const nextState = inspectorState.get(trackItem.key) || { mute: false, solo: false, exportEnabled: true, levelDb: 0, color: trackColorFor(trackItem, 0) };
            nextState.solo = !nextState.solo;
            inspectorState.set(trackItem.key, nextState);
            commitHistory(beforeSnapshot);
            syncUi();
          }, 120);
          laneControls.append(muteBtn, soloBtn);
          titleWrap.appendChild(laneControls);
          trackEl.dataset.hasLaneControls = 'true';
          trackItem.dom = trackItem.dom || {};
          trackItem.dom.muteBtn = muteBtn;
          trackItem.dom.soloBtn = soloBtn;
        }
        gutter.append(chip, titleWrap);
        trackEl.appendChild(gutter);

        const body = document.createElement('div');
        body.className = 'editor-track-body';
        const zoomWrap = document.createElement('div');
        zoomWrap.className = 'editor-wave-zoom-wrap';
        body.appendChild(zoomWrap);
        trackEl.appendChild(body);

        if(trackItem.placeholder){
          const placeholderShell = document.createElement('div');
          placeholderShell.className = 'editor-wave-shell editor-wave-shell--placeholder';
          placeholderShell.innerHTML = '<div class="editor-placeholder-grid"></div>';
          zoomWrap.appendChild(placeholderShell);
          trackItem.dom = {
            track: trackEl,
            waveShell: placeholderShell,
            zoomWrap,
          };
          return;
        }

        gutter.addEventListener('click', async (event) => {
          event.preventDefault();
          event.stopPropagation();
          await setActiveTrack(trackItem.key);
        });

        const waveShell = document.createElement('div');
        waveShell.className = 'editor-wave-shell';
        waveShell.innerHTML = `
          <canvas class="editor-wave-canvas" width="1600" height="260"></canvas>
          <div class="editor-wave-overlay">
            <div class="editor-wave-dim editor-wave-dim--before"></div>
            <div class="editor-wave-dim editor-wave-dim--after"></div>
            <div class="editor-wave-selection"></div>
            <div class="editor-marker editor-marker-in">
              <div class="editor-marker-cap"></div>
              <div class="editor-marker-line"></div>
              <div class="editor-marker-handle" data-role="in"></div>
            </div>
            <div class="editor-marker editor-marker-playhead">
              <div class="editor-marker-cap"></div>
              <div class="editor-marker-line"></div>
              <div class="editor-marker-handle" data-role="playhead"></div>
            </div>
            <div class="editor-marker editor-marker-out">
              <div class="editor-marker-cap"></div>
              <div class="editor-marker-line"></div>
              <div class="editor-marker-handle" data-role="out"></div>
            </div>
          </div>
        `;
        zoomWrap.appendChild(waveShell);

        const canvas = waveShell.querySelector('.editor-wave-canvas');
        const ctx = canvas.getContext('2d');
        const overlayEl = waveShell.querySelector('.editor-wave-overlay');
        const dimBefore = waveShell.querySelector('.editor-wave-dim--before');
        const dimAfter = waveShell.querySelector('.editor-wave-dim--after');
        const selectionEl = waveShell.querySelector('.editor-wave-selection');
        const inMarker = waveShell.querySelector('.editor-marker-in');
        const playheadMarker = waveShell.querySelector('.editor-marker-playhead');
        const outMarker = waveShell.querySelector('.editor-marker-out');

        trackItem.dom = {
          track: trackEl,
          canvas,
          ctx,
          overlay: overlayEl,
          dimBefore,
          dimAfter,
          selection: selectionEl,
          inMarker,
          playhead: playheadMarker,
          outMarker,
          waveShell,
          zoomWrap,
          muteBtn: trackItem.dom && trackItem.dom.muteBtn ? trackItem.dom.muteBtn : null,
          soloBtn: trackItem.dom && trackItem.dom.soloBtn ? trackItem.dom.soloBtn : null,
        };

        zoomWrap.addEventListener('scroll', (event) => {
          syncTimelineScroll(event.currentTarget.scrollLeft, event.currentTarget);
          scheduleViewportRender(trackItem);
        });
        zoomWrap.addEventListener('wheel', (event) => {
          if(!event.altKey) return;
          event.preventDefault();
          const step = Math.exp(-event.deltaY * 0.0025);
          updateWaveZoom(waveZoom * step, { anchorClientX: event.clientX, sourceWrap: event.currentTarget });
        }, { passive: false });

        overlayEl.querySelectorAll('.editor-marker-handle').forEach((handle) => {
          handle.addEventListener('pointerdown', async (event) => {
            if(event.button !== 0) return;
            event.preventDefault();
            event.stopPropagation();
            closeContextMenu();
            dismissLockUntil = Date.now() + 320;
            const role = handle.dataset.role || 'playhead';
            dragState = {
              kind: role,
              trackKey: trackItem.key,
              moved: false,
              beforeSnapshot: role === 'playhead' ? null : snapshotEditorState(),
            };
            await setActiveTrack(trackItem.key);
            if(role === 'playhead'){
              playheadMarker.classList.add('is-dragging');
            }
          });
        });

        waveShell.addEventListener('pointerdown', async (event) => {
          if(event.button !== 0) return;
          if(Date.now() < clickLockUntil) return;
          if(event.target.closest('.editor-marker-handle')) return;
          event.preventDefault();
          closeContextMenu();
          dismissLockUntil = Date.now() + 320;
          await setActiveTrack(trackItem.key);
          const nextMs = nearestSnapValue(msFromClientX(trackItem, event.clientX), trackItem.key, 'playhead');
          setPlayhead(nextMs);
          dragState = {
            kind: 'playhead',
            trackKey: trackItem.key,
            moved: false,
            beforeSnapshot: null,
          };
          playheadMarker.classList.add('is-dragging');
        });

        waveShell.addEventListener('contextmenu', async (event) => {
          event.preventDefault();
          event.stopPropagation();
          await setActiveTrack(trackItem.key);
          openContextMenuForTrack(trackItem, event.clientX, event.clientY);
        });
      };

      timelineTracks.forEach(createTrackUi);

      const saveEditorChanges = async (reason) => {
        if(reason === 'cancel'){
          editorRestoreTrackDrafts(task, originalTrackDrafts);
          return true;
        }
        editorPersistTrackDrafts(task, tracks);
        const sourceTrack = tracks.find((trackItem) => trackItem.key === 'source');
        normalizeTrackClip(sourceTrack);
        const payload = cloneClipState(sourceTrack.clip);
        if(task.id){
          const members = taskGroupMembers(task).filter((member) => member && member.id);
          const requestBody = {
            clip_start_ms: payload.startMs,
            clip_end_ms: payload.endMs,
            clip_enabled: payload.enabled,
          };
          const responses = await Promise.all(members.map(async (member) => {
            const res = await fetch(`/api/tasks/${member.id}/edit`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(requestBody),
            });
            const data = await res.json().catch(() => ({}));
            if(!res.ok){
              throw new Error((data && data.detail && data.detail.message) || 'could not save editor changes');
            }
            return { member, data };
          }));
          responses.forEach(({ member, data }) => {
            applyClipState(member, {
              startMs: data.clip_start_ms,
              endMs: data.clip_end_ms,
              enabled: data.clip_enabled,
            });
          });
        }else{
          taskGroupMembers(task).forEach((member) => applyClipState(member, payload));
          syncPendingClipStateToGroup(task, payload);
        }
        saveTasks();
        updateUI();
        if(reason === 'save'){
          showPopup('editor changes saved');
        }
        return true;
      };

      overlay.__cleanupFns = overlay.__cleanupFns || [];
      const handleDocumentPointerMove = (event) => {
        if(!dragState) return;
        const trackItem = tracks.find((entry) => entry.key === dragState.trackKey);
        if(!trackItem) return;
        dragState.moved = true;
        const rawMs = msFromClientX(trackItem, event.clientX);
        if(dragState.kind === 'playhead'){
          playheadMs = nearestSnapValue(rawMs, trackItem.key, 'playhead');
          try { audio.currentTime = playheadMs / 1000; } catch(_) {}
          syncTransportUi();
        }else if(dragState.kind === 'in'){
          const endValue = Number.isFinite(Number(trackItem.clip.endMs)) ? Number(trackItem.clip.endMs) : trackItem.durationMs;
          trackItem.clip.startMs = Math.min(nearestSnapValue(rawMs, trackItem.key, 'in'), endValue);
          trackItem.clip.enabled = true;
          syncUi();
        }else if(dragState.kind === 'out'){
          trackItem.clip.endMs = Math.max(nearestSnapValue(rawMs, trackItem.key, 'out'), trackItem.clip.startMs);
          trackItem.clip.enabled = true;
          syncUi();
        }
      };
      const handleDocumentPointerUp = () => {
        if(!dragState) return;
        const trackItem = tracks.find((entry) => entry.key === dragState.trackKey);
        if(trackItem && trackItem.dom){
          trackItem.dom.playhead.classList.remove('is-dragging');
        }
        dismissLockUntil = Date.now() + 220;
        if(dragState.moved){
          clickLockUntil = Date.now() + 140;
        }
        if(dragState.moved && dragState.beforeSnapshot){
          commitHistory(dragState.beforeSnapshot);
        }
        const endedPlayheadDrag = dragState.kind === 'playhead';
        dragState = null;
        if(endedPlayheadDrag){
          syncTransportUi();
        }else{
          syncUi();
        }
      };
      const handleDocumentPointerDown = (event) => {
        if(contextMenuEl && !contextMenuEl.contains(event.target)){
          closeContextMenu();
        }
      };
      document.addEventListener('pointermove', handleDocumentPointerMove);
      document.addEventListener('pointerup', handleDocumentPointerUp);
      document.addEventListener('pointerdown', handleDocumentPointerDown);

      const handleEditorKeydown = async (event) => {
        if(!overlay.isConnected) return;
        const target = event.target;
        if(target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
        const modifierPressed = event.metaKey || event.ctrlKey;
        if(modifierPressed && event.code === 'KeyZ'){
          event.preventDefault();
          event.stopPropagation();
          if(event.shiftKey){
            runRedo();
          }else{
            runUndo();
          }
          return;
        }
        if(event.repeat && (event.code === 'Space' || event.code === 'KeyI' || event.code === 'KeyO')) return;
        if(event.code === 'Space'){
          event.preventDefault();
          event.stopPropagation();
          await togglePlayback();
          return;
        }
        if(event.code === 'KeyI'){
          event.preventDefault();
          event.stopPropagation();
          const beforeSnapshot = snapshotEditorState();
          setInPoint();
          commitHistory(beforeSnapshot);
          return;
        }
        if(event.code === 'KeyO'){
          event.preventDefault();
          event.stopPropagation();
          const beforeSnapshot = snapshotEditorState();
          setOutPoint();
          commitHistory(beforeSnapshot);
        }
      };
      document.addEventListener('keydown', handleEditorKeydown, true);

      overlay.__cleanupFns.push(() => {
        document.removeEventListener('pointermove', handleDocumentPointerMove);
        document.removeEventListener('pointerup', handleDocumentPointerUp);
        document.removeEventListener('pointerdown', handleDocumentPointerDown);
        document.removeEventListener('keydown', handleEditorKeydown, true);
        closeContextMenu();
        stopPlaybackLoop();
        try { audio.pause(); } catch(_) {}
        if(activeObjectUrl){
          URL.revokeObjectURL(activeObjectUrl);
          activeObjectUrl = '';
        }
      });

      overlay.__requestClose = async (reason = 'dismiss') => {
        if(closingEditor) return;
        closingEditor = true;
        const saveText = saveBtn.textContent;
        if(reason === 'save'){
          saveBtn.dataset.busy = '1';
          saveBtn.textContent = 'saving...';
        }
        try{
          await saveEditorChanges(reason);
          closeOverlay(overlay);
        }catch(err){
          showPopup((err && err.message) || 'could not save editor changes');
          if(reason === 'save'){
            saveBtn.dataset.busy = '';
            saveBtn.textContent = saveText || 'save';
          }
          closingEditor = false;
        }
      };

      audio.addEventListener('loadedmetadata', () => {
        const trackItem = activeTrack();
        if(Number.isFinite(audio.duration) && audio.duration > 0){
          trackItem.durationMs = Math.max(trackItem.durationMs, Math.round(audio.duration * 1000));
        }
        normalizeTrackClip(trackItem);
        if(playheadMs > 0){
          try { audio.currentTime = Math.min(trackItem.durationMs || 0, playheadMs) / 1000; } catch(_) {}
        }
        syncUi();
      });
      audio.addEventListener('timeupdate', () => {
        const nextPlayheadMs = Math.round((Number(audio.currentTime) || 0) * 1000);
        if(nextPlayheadMs !== playheadMs){
          playheadMs = nextPlayheadMs;
          syncTransportUi();
        }
      });
      audio.addEventListener('pause', () => { stopPlaybackLoop(); syncTransportUi(); });
      audio.addEventListener('play', () => { syncTransportUi(); });
      audio.addEventListener('playing', () => { startPlaybackLoop(); syncTransportUi(); });
      audio.addEventListener('ended', () => { stopPlaybackLoop(); syncTransportUi(); });

      [buttons.select, buttons.range, buttons.razor, buttons.slip].forEach((button) => {
        guardClick(button, (event) => {
          event.preventDefault();
          const mode = button.dataset.editorTool;
          if(!mode) return;
          editorToolMode = mode;
          syncUi();
        }, 120);
      });
      guardClick(buttons.zoomIn, (event) => {
        event.preventDefault();
        updateWaveZoom(waveZoom + 0.25);
      }, 120);
      guardClick(buttons.zoomOut, (event) => {
        event.preventDefault();
        updateWaveZoom(waveZoom - 0.25);
      }, 120);
      guardClick(buttons.magnet, (event) => {
        event.preventDefault();
        const beforeSnapshot = snapshotEditorState();
        snapEnabled = !snapEnabled;
        editorSetSnapEnabled(snapEnabled);
        commitHistory(beforeSnapshot);
        syncUi();
      }, 120);
      guardClick(buttons.fit, (event) => {
        event.preventDefault();
        updateWaveZoom(1);
        syncTimelineScroll(0);
      }, 120);
      guardClick(buttons.start, (event) => {
        event.preventDefault();
        audio.pause();
        playheadMs = 0;
        try { audio.currentTime = 0; } catch(_) {}
        syncUi();
      }, 120);
      guardClick(buttons.jumpIn, (event) => {
        event.preventDefault();
        const trackItem = activeTrack();
        setPlayhead(Math.max(0, trackItem.clip.startMs));
      }, 120);
      guardClick(buttons.play, async (event) => {
        event.preventDefault();
        await togglePlayback();
      }, 120);
      guardClick(buttons.jumpOut, (event) => {
        event.preventDefault();
        const trackItem = activeTrack();
        setPlayhead(Number.isFinite(Number(trackItem.clip.endMs)) ? Number(trackItem.clip.endMs) : trackItem.durationMs);
      }, 120);
      guardClick(buttons.end, (event) => {
        event.preventDefault();
        const trackItem = activeTrack();
        audio.pause();
        playheadMs = Math.max(0, trackItem.durationMs || 0);
        try { audio.currentTime = playheadMs / 1000; } catch(_) {}
        syncUi();
      }, 120);
      guardClick(buttons.setIn, (event) => {
        event.preventDefault();
        const beforeSnapshot = snapshotEditorState();
        setInPoint();
        commitHistory(beforeSnapshot);
      }, 120);
      guardClick(buttons.setOut, (event) => {
        event.preventDefault();
        const beforeSnapshot = snapshotEditorState();
        setOutPoint();
        commitHistory(beforeSnapshot);
      }, 120);
      guardClick(buttons.clear, (event) => {
        event.preventDefault();
        const trackItem = activeTrack();
        const beforeSnapshot = snapshotEditorState();
        trackItem.clip.startMs = 0;
        trackItem.clip.endMs = trackItem.durationMs || null;
        trackItem.clip.enabled = false;
        commitHistory(beforeSnapshot);
        syncUi();
      }, 120);
      guardClick(commandButtons.undo, (event) => {
        event.preventDefault();
        runUndo();
      }, 120);
      guardClick(commandButtons.redo, (event) => {
        event.preventDefault();
        runRedo();
      }, 120);
      guardClick(commandButtons.split, (event) => {
        event.preventDefault();
        const beforeSnapshot = snapshotEditorState();
        editorToolMode = 'razor';
        commitHistory(beforeSnapshot);
        syncUi();
      }, 120);
      guardClick(commandButtons.marker, (event) => {
        event.preventDefault();
        const beforeSnapshot = snapshotEditorState();
        markerState.items.push({
          id: markerState.nextId,
          label: `M${markerState.nextId}`,
          timeMs: playheadMs,
        });
        markerState.nextId += 1;
        commitHistory(beforeSnapshot);
        syncUi();
      }, 120);

      guardClick(cancelBtn, async (event) => {
        event.preventDefault();
        await requestOverlayClose(overlay, 'cancel');
      });
      guardClick(saveBtn, async (event) => {
        event.preventDefault();
        if(saveBtn.dataset.busy === '1') return;
        await requestOverlayClose(overlay, 'save');
      });
      setTooltip(cancelBtn, 'close without saving');
      setTooltip(saveBtn, 'save trim changes');

      applyVariantTheme(currentVariant);
      refreshButtonIcons();
      editorShell.innerHTML = '';
      layoutRefs = buildLayout();
      editorShell.appendChild(layoutRefs.root);
      renderTrackPlacement();
      syncHistoryUi();
      syncUi();
      void Promise.all(timelineTracks.map((trackItem) => loadWaveformForTrack(trackItem)));
      await setActiveTrack('source', { preservePlayhead: false });
      playheadMs = sourceOriginalClip.startMs || 0;
      syncUi();
    }

    function formatEta(seconds){
      const value = Number(seconds);
      if(!Number.isFinite(value) || value <= 0) return '';
      const rounded = Math.max(1, Math.round(value));
      if(rounded < 60){
        return `${rounded}s`;
      }
      if(rounded < 3600){
        const minutes = Math.floor(rounded / 60);
        const remainSeconds = Math.round(rounded % 60);
        if(minutes < 5 && remainSeconds){
          return `${minutes}m ${remainSeconds}s`;
        }
        return `${minutes}m`;
      }
      const minutes = Math.round(rounded / 60);
      const hours = Math.floor(minutes / 60);
      const remain = minutes % 60;
      return remain ? `${hours}h ${remain}m` : `${hours}h`;
    }

    function formatProgressChip(pct, etaSeconds, etaState){
      const clamped = Math.round(Math.max(0, Math.min(100, Number(pct) || 0)));
      const eta = formatEta(etaSeconds);
      const state = String(etaState || '').toLowerCase();
      // Keep ETA/state plumbing active, but hide ETA-related copy from the UI.
      void eta;
      void state;
      return `${clamped}%`;
    }

    function shouldUseProgressSummary(stage){
      const normalized = String(stage || '').toLowerCase();
      if(['done', 'error', 'errored', 'stopped'].includes(normalized)) return false;
      return !['ready', 'queued'].includes(normalized);
    }

    async function rehydrateTasks(){
      try{
        const res = await fetch('/api/tasks?limit=100');
        if(!res.ok) return;
        const data = await res.json();
        tasks = data && Array.isArray(data.items) ? data.items.filter((task) => task && task.id) : [];
      }catch(err){
        console.debug('rehydration skipped', err);
      }
    }
    // processing-state helper
    function isProcessing(t){
      if(!t) return false;
      const stage = (t.stage || '').toLowerCase();
      const inactive = ['ready','queued','done','stopped','error'];
      if(inactive.includes(stage)) return false;
      return typeof t.pct === 'number' ? (t.pct >= 0 && t.pct < 100) : true;
    }

    function updateClear(){
      let hasClearable = false;
      if (!startPressed) {
        hasClearable = Array.isArray(tasks) && tasks.length > 0;
      } else {
        hasClearable = tasks && tasks.some(t => t && (t.stage === 'stopped' || (typeof t.pct === 'number' && t.pct >= 100)));
      }
      const itemCount = queueTaskCount();
      const manyItems = itemCount > 6;
      const compact = itemCount <= 2;
      spacer.style.height = hasClearable ? (compact ? '14px' : '22px') : '0px';
      if (bottomPad) {
        if(itemCount === 0){
          bottomPad.style.height = '0px';
        }else if(compact){
          bottomPad.style.height = '10px';
        }else{
          bottomPad.style.height = manyItems ? '34px' : '22px';
        }
      }
      if (hasClearable) {
        clearBtn.hidden = false;
        clearBtn.classList.add('show');
        clearBtn.style.pointerEvents = 'auto';
      } else {
        clearBtn.classList.remove('show');
        clearBtn.style.pointerEvents = 'none';
        clearBtn.hidden = true;
      }
    }

    function updateShellLayout(){
      if(!appShell) return;
      const itemCount = queueTaskCount();
      const empty = itemCount === 0;
      const compact = itemCount > 0 && itemCount <= 2;
      appShell.classList.toggle('empty', empty);
      appShell.classList.toggle('compact', compact);
      appShell.classList.toggle('expanded', !empty && !compact);
    }
    function updateVigs(){
      const maxScrollTop = Math.max(0, appShell.scrollHeight - appShell.clientHeight);
      topVig.style.top = '0px';
      topVig.style.bottom = 'auto';
      topVig.style.transform = 'none';
      topVig.style.opacity = appShell.scrollTop > 0 ? '0.56' : '0';
      botVig.style.opacity = appShell.scrollTop < maxScrollTop - 2 ? '0.56' : '0';
    }

    function enforceCapacity(){
      const activeCount = tasks.filter(t => t && !isFinished(t)).length;
      const full = activeCount >= MAX_TASKS;
      dropzone.classList.toggle('pointer-events-none', full);
      dropzone.classList.toggle('opacity-60', full);
      if(fileInput) fileInput.disabled = full;
      if(full){
        const now = Date.now();
        if(!enforceCapacity._lastNotice || now - enforceCapacity._lastNotice > 8000){
          showPopup('maximum number of songs reached! Please clear all to upload more songs');
          enforceCapacity._lastNotice = now;
        }
      }
      return full;
    }

    function updateUI(){
      regroupQueueDom();
      updateShellLayout();
      updateClear();
      updateVigs();
      const full = enforceCapacity();
      const blocked = full;
      dropzone.classList.toggle('pointer-events-none', blocked);
      dropzone.classList.toggle('opacity-60', blocked);
      if(fileInput) fileInput.disabled = blocked;
      updateStartButton();
    }


    // Helper to apply stem labels
    function applyLabels(container, stemsList){
      if(!container || !stemsList) return;
      container.innerHTML='';
      const pretty = {
        guitar: 'guitar',
        mel_band_karaoke: 'bg vocal',
        all_stems: 'all stems',
        denoise: 'denoise',
        preset_denoise: 'denoise',
        bs_roformer_6s: 'full mix',
        htdemucs_ft_drums: 'drums',
        htdemucs_ft_bass: 'bass',
        htdemucs_ft_other: 'other',
        htdemucs_6s: 'full mix faster',
        drumsep_6s: 'drum split - 6',
        drumsep_4s: 'drum split - 4',
        boost_harmonies: 'boost harmonies',
      };
      stemsList.forEach(name => {
        const base = name.split(' - ').pop() || name;
        const stem = base.replace(/\.wav$/i,'');
        const chip = document.createElement('span');
        chip.className = 'chip chip-dim text-[11px]';
        chip.textContent = pretty[stem] || stem;
        container.appendChild(chip);
      });
    }

    function refreshAllLabelVisibility(){
      queueLeafRows().forEach((row) => {
        const task = getTaskByRow(row);
        if(task){
          applyRowState(row, task);
        }
      });
    }

    function shouldShowStatus(stage){
      const normalized = String(stage || '').toLowerCase();
      if(!normalized || normalized === 'done') return false;
      if(normalized === 'ready') return startPressed;
      if(normalized === 'queued') return queueStarted || startPressed;
      return true;
    }

    function setStatusVisibility(st, stage){
      if(!st) return;
      const visible = stage !== 'done' && shouldShowStatus(stage);
      st.classList.toggle('hidden', !visible);
    }

    function hideStatus(st){
      if(!st) return;
      st.textContent = '';
      st.classList.add('hidden');
      st.style.display = 'none';
    }

    function showStatus(st){
      if(!st) return;
      st.style.display = '';
    }

    function setStopVisibility(stopPad, stage, forceShow = false){
      if(!stopPad) return;
      const normalized = String(stage || '').toLowerCase();
      let inferredActive = false;
      const row = stopPad.closest ? stopPad.closest('.item-row') : null;
      if(row){
        const task = getTaskByRow(row);
        inferredActive = !!task && (isProcessing(task) || (normalized === 'queued' && Number(task.pct || 0) > 0));
      }
      const active = forceShow || inferredActive || (normalized && !['ready','queued','done','stopped','error','errored'].includes(normalized));
      const show = !!active;
      stopPad.style.display = show ? '' : 'none';
      stopPad.classList.toggle('show', show);
      if(!show){
        stopPad.style.pointerEvents = 'none';
        stopPad.style.opacity = '0.6';
      } else {
        stopPad.style.pointerEvents = '';
        stopPad.style.opacity = '';
      }
    }

    function bindFolderButton(btn, taskRef){
      if(!btn || !taskRef) return;
      guardClick(btn, async (e) => {
        e.preventDefault();
        if(btn.dataset.busy === '1') return;
        btn.dataset.busy = '1';
        try{
          const res = await fetch('/reveal/' + taskRef.id, { method: 'POST' });
          if(!res.ok){
            let msg = "file doesn't exist";
            try{
              const data = await res.json();
              msg = (data && data.detail && data.detail.message) || msg;
            }catch(_){}
            showPopup(msg);
          }
        }catch(err){
          showPopup("file doesn't exist");
        }
        setTimeout(() => { btn.dataset.busy = ''; }, 250);
      });
    }

    // --- Helper to set rerun and stop icons ---

    function setRetryIcon(stopPad){
      if(!stopPad) return;
      while (stopPad.firstChild) stopPad.removeChild(stopPad.firstChild);
      stopPad.insertAdjacentHTML('afterbegin', '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" class="w-7 h-7" fill="none" stroke="#0F2027" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>');
    }

    function setStopSquareIcon(stopPad){
      if(!stopPad) return;
      while (stopPad.firstChild) stopPad.removeChild(stopPad.firstChild);
      stopPad.insertAdjacentHTML('afterbegin', '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" class="w-[1.85rem] h-[1.85rem]" aria-hidden="true"><rect x="5.5" y="5.5" width="13" height="13" rx="2.1" ry="2.1" fill="#0F2027"/></svg>');
    }

    function bindStopPad(stopPad, taskRef, ui){
      if(!stopPad || !taskRef || !taskRef.id) return;
      setStopSquareIcon(stopPad);
      stopPad.style.pointerEvents = '';
      stopPad.style.opacity = '';
      stopPad.title = 'stop current task and pause queue';
      guardClick(stopPad, async (e) => {
        if(stopPad.dataset.busy === '1') return;
        stopPad.dataset.busy = '1';
        e.preventDefault();
        console.debug('stemsplat stop click', { taskId: taskRef.id, stage: taskRef.stage, pct: taskRef.pct });
        await requestStop(taskRef.id, taskRef, ui);
        stopPad.dataset.busy = '';
      });
    }

    function setDownloadIcon(btn){
      if(!btn) return;
      while (btn.firstChild) btn.removeChild(btn.firstChild);
      const NS = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(NS,'svg');
      svg.setAttribute('viewBox','0 0 24 24');
      svg.classList.add('w-7','h-7');
      svg.setAttribute('fill','none');
      svg.setAttribute('stroke','#0F2027');
      svg.setAttribute('stroke-width','2.2');
      svg.setAttribute('stroke-linecap','round');
      svg.setAttribute('stroke-linejoin','round');
      const arr = document.createElementNS(NS,'path');
      arr.setAttribute('d','M12 5v10m0 0l-4-4m4 4l4-4');
      const base = document.createElementNS(NS,'path');
      base.setAttribute('d','M5 19h14');
      svg.append(arr, base);
      btn.appendChild(svg);
    }

    function setFolderIcon(btn){
      if(!btn) return;
      while (btn.firstChild) btn.removeChild(btn.firstChild);
      btn.insertAdjacentHTML('afterbegin', '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" class="w-7 h-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>');
    }

    function triggerTaskDownload(taskRef){
      if(!taskRef || !taskRef.id) return;
      const link = document.createElement('a');
      link.href = `/download/${taskRef.id}`;
      link.rel = 'noopener';
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      link.remove();
    }

    function bindDownloadButton(btn, taskRef){
      if(!btn || !taskRef || !taskRef.id) return;
      btn.classList.add('show');
      if(taskRef.delivery === 'browser_download'){
        setDownloadIcon(btn);
        btn.title = 'download';
        guardClick(btn, (e) => {
          e.preventDefault();
          triggerTaskDownload(taskRef);
        });
        if(!taskRef.autoDownloaded){
          taskRef.autoDownloaded = true;
          setTimeout(() => triggerTaskDownload(taskRef), 180);
        }
        return;
      }
      setFolderIcon(btn);
      btn.title = 'open folder';
      bindFolderButton(btn, taskRef);
    }

// --- Helper to rerun a task server-side ---
    async function rerunTask(task, ui, options = {}){
      const {bar, st, dl, stopPad, smooth} = ui;
      if(!task || !task.id) return;
      let ok = false, newId = null, payload = null;
      const body = {};
      const startSettings = {
        ...currentStartSettings(),
        ...(options && typeof options === 'object' ? options : {}),
      };
      if(typeof options.stems === 'string' && options.stems){
        body.stems = options.stems;
      }
      if(typeof startSettings.output_format === 'string' && startSettings.output_format){
        body.output_format = startSettings.output_format;
      }
      if(typeof startSettings.multi_stem_export === 'string' && startSettings.multi_stem_export){
        body.multi_stem_export = startSettings.multi_stem_export;
      }
      if(typeof startSettings.video_handling === 'string' && startSettings.video_handling){
        body.video_handling = startSettings.video_handling;
      }
      if(typeof startSettings.output_root === 'string' && startSettings.output_root){
        body.output_root = startSettings.output_root;
      }
      body.output_same_as_input = !!startSettings.output_same_as_input;
      body.prioritize = options.prioritize !== false;
      try {
        const res = await fetch('/rerun/' + task.id, {
          method: 'POST',
          headers: Object.keys(body).length ? { 'Content-Type': 'application/json' } : undefined,
          body: Object.keys(body).length ? JSON.stringify(body) : undefined,
        });
        if (res.ok) {
          try { payload = await res.json(); } catch(_) { payload = null; }
          newId = payload && (payload.task_id || payload.id || null);
          if (newId) { ok = true; }
        }
      } catch (_) {}
      if (!ok || !newId) {
        // Treat it as a stopped/error card with rerun available
        if (st) { showStatus(st); st.textContent = 'error'; st.classList.remove('hidden'); setStatusVisibility(st, 'error'); }
        const parent = st && st.closest ? st.closest('.card') : null;
        if (parent) parent.classList.add('done');

        if (stopPad) {
          stopPad.title = 'rerun';
          stopPad.innerHTML = '';          // prevent double icon
          setRetryIcon(stopPad);
          stopPad.style.pointerEvents = '';
          stopPad.style.opacity = '';
          guardClick(stopPad, (e) => { e.preventDefault(); rerunTask(task, ui); });
        }
        updateUI();
        showError('rerun failed');
        return;
      }
      // reset visual state
      if(st){ showStatus(st); st.classList.remove('hidden'); st.textContent = 'preparing (0%)'; setStatusVisibility(st, 'preparing'); }
      if(dl){ dl.classList.remove('show'); dl.removeAttribute('href'); }
      const card = stopPad && stopPad.closest('.card');
      if(card){ card.classList.remove('done'); }
      if(smooth){ smooth.setImmediate(0); }
      // show stop square again
      if(stopPad){
        stopPad.title = 'stop current task and pause queue';
        stopPad.innerHTML = '';
        setStopSquareIcon(stopPad);
        stopPad.style.pointerEvents = '';
        stopPad.style.opacity = '';
        guardClick(stopPad, async (e) => {
          if(stopPad.dataset.busy === '1') return;
          stopPad.dataset.busy = '1';
          e.preventDefault();
          await requestStop(task.id, task, { bar, st, dl, stopPad, smooth });
          setTimeout(() => { stopPad.dataset.busy = ''; }, 250);
        });
        setStopVisibility(stopPad, 'preparing');
      }
      task.stage = 'preparing';
      task.pct = 0;
      if(Array.isArray(payload && payload.stems)){
        task.stems = payload.stems;
      }
      task.downloaded = false;
      task.autoDownloaded = false;
      task.out_dir = null;
      saveTasks();
      updateUI();
      // resume tracking
      const oldId = task.id;
      task.id = newId;
      if (dl) { dl.removeAttribute('href'); }
      const row = stopPad && stopPad.closest('.item-row');
      if (row) { row.__taskId = newId; }
      saveTasks();
      trackProgress(newId, bar, st, dl, null, null, task, null, smooth, stopPad);
    }

    function createItem(task){
      primeTaskEditorAssets(task);
      const existing = queueLeafRows().find((row) => row.__taskRowKey === taskRowIdentity(task)) || null;
      if(existing){
        applyLabels(existing.querySelector('.labels'), displayStemsForTask(task));
        applyRowState(existing, task);
        ensureTaskProgressTracking(task);
        updateUI();
        return taskUiForRow(existing);
      }
      const node = template.content.cloneNode(true);
      const li   = node.querySelector('.filename');
      const bar  = node.querySelector('.bar-fill');
      const st   = node.querySelector('.chip.status');
      const dl   = node.querySelector('.dl-pad');
      const card = node.querySelector('.card');
      const stopPad = node.querySelector('.stop-pad');
      applyFilename(li, task.name);
      const labels = node.querySelector('.labels');
      const smooth = makeProgressSmoother(bar);
      const initOverall = Math.max(0, Math.min(100, task.pct || 0));
      smooth.setImmediate(initOverall);
      if(task.stage === 'done'){
        hideStatus(st);
        dl.classList.add('show');
        dl.title = 'open folder';
        bindFolderButton(dl, task);
        card.classList.add('done');
        if(stopPad) stopPad.remove();
      }
      else if(task.stage === 'stopped'){
        // Treat as finished visually
        showStatus(st);
        st.classList.add('hidden');
        card.classList.add('done');
        if (stopPad){
          stopPad.style.display = '';
          stopPad.classList.add('show');
          stopPad.innerHTML = '';
          setRetryIcon(stopPad);
          stopPad.title = 'rerun';
        guardClick(stopPad, (e) => { e.preventDefault(); rerunTask(task, { bar, st, dl, stopPad, smooth }); });
      }
      }
      else if (task.stage === 'error' || task.pct < 0) {
        // Treat like stopped; keep label visible for context
        showStatus(st);
        st.textContent = 'error';
        const parent = st.closest('.card'); if (parent) parent.classList.add('done');
        if (stopPad){
          stopPad.style.display = '';
          stopPad.classList.add('show');
          stopPad.title = 'rerun';
          stopPad.innerHTML = '';
          setRetryIcon(stopPad);
        guardClick(stopPad, (e) => { e.preventDefault(); rerunTask(task, { bar, st, dl, stopPad, smooth }); });
      }
      }
      else {
        const sp = Math.round(Math.max(0, Math.min(100, task.pct)));
        showStatus(st);
        const normalizedStage = String(task.stage || '').toLowerCase();
        if(!shouldUseProgressSummary(normalizedStage)){
          st.textContent = displayStage(task.stage) || 'queued';
          smooth.setImmediate(0);
        }else{
          st.textContent = formatProgressChip(sp, task.eta_seconds, task.eta_state);
          if(normalizedStage === 'ready' || normalizedStage === 'queued'){
            smooth.setImmediate(0);
          }else{
            smooth.setImmediate(sp);
          }
        }
        setStopVisibility(stopPad, task.stage);
        if(stopPad && task.id && isProcessing(task)){
          bindStopPad(stopPad, task, { bar, st, dl, stopPad, smooth });
        }
      }
      if(dl && (task.stage === 'stopped' || task.stage === 'error' || task.pct < 0)){
        dl.classList.add('show');
        dl.title = 'retry';
        dl.innerHTML = '';
        setRetryIcon(dl);
        guardClick(dl, (e) => { e.preventDefault(); rerunTask(task, { bar, st, dl, stopPad, smooth }); });
      }
      setStatusVisibility(st, task.stage);
      const inserted = node.children[0];
      inserted.__taskId = task.id;
      inserted.__tempKey = task.tempKey || '';
      inserted.__batchKey = task.batchKey || '';
      inserted.__taskRowKey = taskRowIdentity(task);
      inserted.__stems = Array.isArray(task.stems) ? task.stems.slice() : [];
      queue.insertBefore(inserted, queue.firstChild);
      stageQueueEntry(inserted, `task:${inserted.__taskRowKey}`, Math.min(80, queue.childElementCount * 18));
      // Apply any stem labels if present
      applyLabels(inserted.querySelector('.labels'), displayStemsForTask(task));
      applyRowState(inserted, task);
      ensureTaskProgressTracking(task);
      updateUI();
      return {bar, st, dl, stopPad, smooth};
    }

    function createGroupParentRow(kind, groupKey){
      const frag = template.content.cloneNode(true);
      const row = frag.firstElementChild;
      row.className = 'queue-group-parent-row';
      row.dataset.groupKind = kind;
      row.dataset.groupKey = groupKey;
      row.dataset.forceHideEditor = kind === 'batch' ? '1' : '0';
      const stopPad = row.querySelector('.stop-pad');
      if(stopPad){
        stopPad.style.display = '';
        stopPad.classList.add('show');
        stopPad.title = 'expand';
        stopPad.innerHTML = '<span class="queue-parent-caret" aria-hidden="true"><svg viewBox="0 0 24 24" class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"></path></svg></span>';
      }
      const card = row.querySelector('.card');
      if(card){
        const stack = document.createElement('div');
        stack.className = 'queue-card-stack';
        stack.setAttribute('aria-hidden', 'true');
        stack.innerHTML = `
          <span class="queue-card-stack-layer"></span>
          <span class="queue-card-stack-layer"></span>
          <span class="queue-card-stack-layer"></span>
        `;
        card.appendChild(stack);
      }
      return row;
    }

    function aggregateGroupTaskState(groupTasks){
      const tasksList = Array.isArray(groupTasks) ? groupTasks.filter(Boolean) : [];
      if(!tasksList.length){
        return { stage: 'ready', pct: 0 };
      }
      const normalizedStages = tasksList.map((task) => String(task.stage || '').toLowerCase());
      if(normalizedStages.some((stage) => stage === 'error')){
        return { stage: 'error', pct: -1 };
      }
      const allDone = normalizedStages.every((stage) => stage === 'done');
      if(allDone){
        return { stage: 'done', pct: 100 };
      }
      const allStopped = normalizedStages.every((stage) => stage === 'stopped');
      if(allStopped){
        return { stage: 'stopped', pct: 0 };
      }
      const activeTasks = tasksList.filter((task) => isProcessing(task));
      if(activeTasks.length){
        const pct = Math.round(activeTasks.reduce((sum, task) => sum + Math.max(0, Math.min(100, Number(task.pct) || 0)), 0) / activeTasks.length);
        const stageTask = activeTasks.find((task) => !['queued', 'ready'].includes(String(task.stage || '').toLowerCase())) || activeTasks[0];
        return { stage: String(stageTask.stage || 'queued').toLowerCase(), pct };
      }
      const queued = normalizedStages.some((stage) => stage === 'queued');
      if(queued){
        return { stage: 'queued', pct: 0 };
      }
      const ready = normalizedStages.some((stage) => stage === 'ready');
      return { stage: ready ? 'ready' : normalizedStages[0] || 'ready', pct: 0 };
    }

    function queueArtworkGridCell(task){
      const taskId = task && task.id ? String(task.id) : '';
      if(taskId){
        const image = document.createElement('img');
        image.className = 'queue-artwork-grid-cell';
        image.src = `${artworkUrl(taskId)}?v=${encodeURIComponent(taskId)}`;
        image.alt = '';
        return image;
      }
      const label = songDisplayTitle(task && task.name).slice(0, 2).toUpperCase();
      const fallback = document.createElement('div');
      fallback.className = 'queue-artwork-grid-cell is-fallback';
      fallback.textContent = label || '♪';
      return fallback;
    }

    function syncParentRow(row, parentTask, childTasks, { kind = 'song', title = '', artworkGridTasks = [] } = {}){
      if(!row) return;
      const filename = row.querySelector('.filename');
      const dl = row.querySelector('.dl-pad');
      const st = row.querySelector('.chip.status');
      const stopPad = row.querySelector('.stop-pad');
      const bar = row.querySelector('.bar-fill');
      const labels = row.querySelector('.labels');
      const editorBtn = row.querySelector('.task-editor-btn');
      const presetBtn = row.querySelector('.task-preset-btn');
      const shell = row.querySelector('.artwork-shell');
      const fallback = row.querySelector('.artwork-fallback');
      const sourceCount = kind === 'batch'
        ? Math.max(0, artworkGridTasks.length)
        : Math.max(0, Array.isArray(childTasks) ? childTasks.length : 0);
      const stackCount = Math.min(3, Math.max(0, sourceCount - 1));
      row.dataset.stackCount = String(stackCount);
      row.style.setProperty('--queue-stack-space', stackCount > 0 ? `${stackCount * 10 + 8}px` : '0px');
      let grid = row.querySelector('.queue-artwork-grid');
      if(!grid && shell){
        grid = document.createElement('div');
        grid.className = 'queue-artwork-grid';
        shell.appendChild(grid);
      }
      applyFilename(filename, title || parentTask.name || '');
      applyLabels(labels, displayStemsForTask(parentTask));
      if(bar){
        const pct = Math.max(0, Math.min(100, Number(parentTask.pct) || 0));
        bar.style.width = `${pct}%`;
      }
      if(st){
        if(parentTask.stage === 'done'){
          hideStatus(st);
        }else if(parentTask.stage === 'error'){
          showStatus(st);
          st.textContent = 'error';
        }else if(shouldUseProgressSummary(parentTask.stage)){
          showStatus(st);
          st.textContent = formatProgressChip(parentTask.pct || 0, null, null);
        }else{
          showStatus(st);
          st.textContent = displayStage(parentTask.stage) || 'queued';
        }
        setStatusVisibility(st, parentTask.stage);
      }
      if(editorBtn){
        editorBtn.hidden = kind === 'batch';
        editorBtn.disabled = kind === 'batch';
      }
      if(presetBtn){
        presetBtn.hidden = true;
      }
      if(kind === 'batch'){
        row.dataset.forceHideEditor = '1';
        if(grid){
          grid.hidden = false;
          grid.replaceChildren(...artworkGridTasks.slice(0, 4).map(queueArtworkGridCell));
        }
        if(shell){
          shell.classList.add('has-image');
        }
        if(fallback){
          fallback.hidden = true;
        }
        const img = row.querySelector('.artwork-img');
        if(img){
          img.hidden = true;
          img.removeAttribute('src');
        }
        if(dl){
          dl.classList.remove('show');
        }
      }else{
        row.dataset.forceHideEditor = '0';
        if(grid){
          grid.hidden = true;
          grid.innerHTML = '';
        }
        applyArtwork(row, childTasks.find((task) => task && task.id) || childTasks[0] || parentTask);
        if(dl){
          const doneRef = childTasks.find((task) => task && task.stage === 'done' && task.out_dir);
          if(doneRef){
            dl.classList.add('show');
            dl.title = 'open folder';
            bindFolderButton(dl, doneRef);
          }else{
            dl.classList.remove('show');
          }
        }
      }
      if(stopPad){
        stopPad.style.display = '';
      }
      applyRowState(row, parentTask);
    }

    function applyGroupShellState(shell, expanded){
      if(!shell) return;
      shell.classList.toggle('is-expanded', !!expanded);
      shell.classList.toggle('is-collapsed', !expanded);
      const children = shell.querySelector(':scope > .queue-group-children');
      if(children){
        children.style.maxHeight = expanded ? 'none' : '0px';
      }
    }

    function regroupQueueDom(){
      const leafRows = queueLeafRows();
      if(!leafRows.length) return;
      const songOrder = [];
      const songs = new Map();
      leafRows.forEach((row) => {
        const task = getTaskByRow(row);
        const songKey = queueSongGroupKey(task || { tempKey: row.__tempKey, id: row.__taskId });
        const batchKey = (task && task.batchKey) || row.__batchKey || '';
        if(!songs.has(songKey)){
          songs.set(songKey, { key: songKey, batchKey, rows: [], tasks: [] });
          songOrder.push(songKey);
        }
        const group = songs.get(songKey);
        group.rows.push(row);
        if(task) group.tasks.push(task);
      });
      const batchSongKeys = new Map();
      songOrder.forEach((songKey) => {
        const group = songs.get(songKey);
        if(group && group.batchKey){
          const list = batchSongKeys.get(group.batchKey) || [];
          list.push(songKey);
          batchSongKeys.set(group.batchKey, list);
        }
      });

      const buildSongEntry = (songGroup, { nested = false } = {}) => {
        const songTasks = songGroup.tasks;
        const representative = songTasks[0] || null;
        const multiStem = songGroup.rows.length > 1;
        songGroup.rows.forEach((row) => {
          row.dataset.forceHideEditor = multiStem ? '1' : '0';
          const task = getTaskByRow(row);
          if(task){
            applyRowState(row, task);
          }
        });
        if(!multiStem){
          return songGroup.rows[0];
        }
        const shell = document.createElement('div');
        shell.className = 'queue-group-shell queue-song-shell';
        const groupKey = songGroup.key;
        const expanded = isGroupExpanded('song', groupKey);
        const parentRow = createGroupParentRow('song', groupKey);
        const children = document.createElement('div');
        children.className = 'queue-group-children';
        songGroup.rows.forEach((row) => children.appendChild(row));
        const unionStems = representative && Array.isArray(representative.groupStems) && representative.groupStems.length
          ? representative.groupStems.slice()
          : Array.from(new Set(songTasks.flatMap((task) => Array.isArray(task.stems) ? task.stems : [])));
        const aggregate = aggregateGroupTaskState(songTasks);
        const parentTask = representative ? {
          ...representative,
          name: representative.name,
          stage: aggregate.stage,
          pct: aggregate.pct,
          groupStems: unionStems,
          stems: unionStems,
        } : { name: 'song', stage: 'ready', pct: 0, stems: unionStems, groupStems: unionStems };
        syncParentRow(parentRow, parentTask, songTasks, { kind: 'song', title: songDisplayTitle(parentTask.name) });
        stageQueueEntry(parentRow, `song-group:${groupKey}`);
        const toggle = (event) => {
          if(event.target.closest('button')) return;
          setGroupExpanded('song', groupKey, !isGroupExpanded('song', groupKey));
          regroupQueueDom();
          updateUI();
        };
        parentRow.addEventListener('click', toggle);
        const caretBtn = parentRow.querySelector('.stop-pad');
        if(caretBtn){
          guardClick(caretBtn, async (event) => {
            event.preventDefault();
            event.stopPropagation();
            setGroupExpanded('song', groupKey, !isGroupExpanded('song', groupKey));
            regroupQueueDom();
            updateUI();
          }, 120);
        }
        shell.append(parentRow, children);
        applyGroupShellState(shell, expanded);
        return shell;
      };

      const fragment = document.createDocumentFragment();
      const handledBatches = new Set();
      songOrder.forEach((songKey) => {
        const songGroup = songs.get(songKey);
        if(!songGroup) return;
        const songBatchKey = songGroup.batchKey;
        const batchSongs = songBatchKey ? (batchSongKeys.get(songBatchKey) || []) : [];
        if(songBatchKey && batchSongs.length > 1){
          if(handledBatches.has(songBatchKey)) return;
          handledBatches.add(songBatchKey);
          const batchShell = document.createElement('div');
          batchShell.className = 'queue-group-shell queue-batch-shell';
          const parentRow = createGroupParentRow('batch', songBatchKey);
          const children = document.createElement('div');
          children.className = 'queue-group-children';
          const songGroups = batchSongs.map((key) => songs.get(key)).filter(Boolean);
          songGroups.forEach((childSongGroup) => {
            children.appendChild(buildSongEntry(childSongGroup, { nested: true }));
          });
          const batchTasks = songGroups.flatMap((group) => group.tasks);
          const aggregate = aggregateGroupTaskState(batchTasks);
          const unionStems = Array.from(new Set(batchTasks.flatMap((task) => Array.isArray(task.groupStems) ? task.groupStems : (Array.isArray(task.stems) ? task.stems : []))));
          const parentTask = {
            name: megaCardTitle(songGroups.flatMap((group) => group.tasks.slice(0, 1))),
            stage: aggregate.stage,
            pct: aggregate.pct,
            stems: unionStems,
            groupStems: unionStems,
            disableEditor: true,
          };
          syncParentRow(parentRow, parentTask, batchTasks, {
            kind: 'batch',
            title: megaCardTitle(songGroups.flatMap((group) => group.tasks.slice(0, 1))),
            artworkGridTasks: songGroups.flatMap((group) => group.tasks.slice(0, 1)),
          });
          stageQueueEntry(parentRow, `batch-group:${songBatchKey}`);
          parentRow.addEventListener('click', (event) => {
            if(event.target.closest('button')) return;
            setGroupExpanded('batch', songBatchKey, !isGroupExpanded('batch', songBatchKey));
            regroupQueueDom();
            updateUI();
          });
          const caretBtn = parentRow.querySelector('.stop-pad');
          if(caretBtn){
            guardClick(caretBtn, async (event) => {
              event.preventDefault();
              event.stopPropagation();
              setGroupExpanded('batch', songBatchKey, !isGroupExpanded('batch', songBatchKey));
              regroupQueueDom();
              updateUI();
            }, 120);
          }
          batchShell.append(parentRow, children);
          applyGroupShellState(batchShell, isGroupExpanded('batch', songBatchKey));
          fragment.appendChild(batchShell);
          return;
        }
        fragment.appendChild(buildSongEntry(songGroup));
      });

      const existingChildren = Array.from(queue.childNodes);
      existingChildren.forEach((node) => queue.removeChild(node));
      queue.appendChild(fragment);
    }

    function withPageFade(doWork){
      const mask = document.createElement('div');
      mask.className = 'page-mask';
      document.body.appendChild(mask);
      requestAnimationFrame(() => {
        mask.classList.add('show');
        setTimeout(() => {
          try { doWork(); } finally {
            mask.classList.remove('show');
            setTimeout(() => mask.remove(), 300);
          }
        }, 280);
      });
    }

    function wait(ms){
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function nextFrame(){
      return new Promise((resolve) => requestAnimationFrame(() => resolve()));
    }

    async function runClearAllTransition(doWork){
      const body = document.body;
      if(!body){
        await doWork();
        return;
      }
      const prefersReducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
      let overlay = document.getElementById('clear-all-overlay');
      const createdOverlay = !overlay;
      if(!overlay){
        overlay = document.createElement('div');
        overlay.id = 'clear-all-overlay';
        overlay.className = 'clear-all-overlay';
        body.appendChild(overlay);
      }
      body.classList.remove('clear-all-exit', 'clear-all-enter-start');
      body.classList.add('clear-all-anim');
      try{
        await nextFrame();
        if(!prefersReducedMotion){
          body.classList.add('clear-all-exit');
          await wait(360);
        }
        await doWork();
        if(!prefersReducedMotion){
          body.classList.remove('clear-all-exit');
          body.classList.add('clear-all-enter-start');
          await nextFrame();
          await nextFrame();
          body.classList.remove('clear-all-enter-start');
          await wait(420);
        }
      }finally{
        body.classList.remove('clear-all-exit', 'clear-all-enter-start', 'clear-all-anim');
        if(createdOverlay && overlay && overlay.parentElement){
          overlay.remove();
        }
      }
    }

    function closeActiveStreams(){
      activeStreams.forEach((stream) => {
        try{ stream.close(); } catch(_){}
      });
      activeStreams.clear();
    }

    function abortPendingUploads(){
      pendingItems.forEach((pending) => {
        if(pending && pending.xhr){
          try{ pending.xhr.abort(); } catch(_){}
        }
      });
    }

    async function clearAllTasks({ showToast = false } = {}){
      if(isClearing) return;
      isClearing = true;
      closeActiveStreams();
      abortPendingUploads();
      if(Array.isArray(tasks)){
        tasks.forEach(t => {
          if(t && t.id){
            try { fetch('/stop/' + t.id, {method:'POST'}); } catch(_){}
          }
        });
      }
      await runClearAllTransition(async () => {
        for (const el of Array.from(queue.children)) { el.remove(); }
        tasks = [];
        pendingItems.length = 0;
        startLock = false;
        startPressed = false;
        queueStarted = false;
        saveTasks();
        updateUI();
      });
      try { await fetch('/clear_all_uploads', {method:'POST'}); } catch(_){ }
      isClearing = false;
      if(showToast) showPopup('all tasks cleared');
    }

    guardClick(clearBtn, async () => {
      const hasRunning = Array.isArray(tasks) && tasks.some(t => isProcessing(t));
      if(hasRunning){
        const ok = await showConfirm('a process is still running. clear all anyway?');
        if(!ok) return;
      }
      await clearAllTasks();
    });

    rehydrateTasks().finally(() => {
      // restore previous tasks
      queueStarted = tasks.some(t => t && ((t.stage && t.stage !== 'ready') || (typeof t.pct === 'number' && t.pct > 0) || t.out_dir));
      tasks.forEach(t => {
        const {bar, st, dl, stopPad, smooth} = createItem(t);
        const last = queue.lastElementChild;
        if(last){
          requestAnimationFrame(() => {
            last.classList.add('enter-active');
          });
        }
        ensureTaskProgressTracking(t);
      });
      refreshAllLabelVisibility();
      updateUI();
      appShell.addEventListener('scroll', updateVigs, { passive: true });
      if(detachAppShellSoftScroll){
        detachAppShellSoftScroll();
      }
      detachAppShellSoftScroll = attachSoftWheelScroll(appShell, updateVigs);
    });
    // settings button → open overlay
    const settingsBtn = document.getElementById('settings-btn');
    const bottomLeftControls = document.querySelector('.bottom-left-controls');
    if(settingsBtn){ settingsBtn.addEventListener('click', openSettings); }
    if(historyBtn){ historyBtn.addEventListener('click', openPreviousFilesCard); }

    // Register this synchronously so keyboard dismissal is available even while
    // the asynchronous settings/bootstrap requests are still completing.
    document.addEventListener('keydown', (event) => {
      if(event.key === 'Escape'){
        if(event.defaultPrevented) return;
        closeOutputFormatMenu();
        closeMultiStemExportMenu();
        closeLanTtlMenu();
        closePreviousFilesRetentionMenu();
        const presetOverlay = presetSettingsOverlay || document.getElementById('preset-settings-overlay');
        const mainOverlay = settingsOverlay || document.getElementById('settings-overlay');
        if(presetOverlay && !presetOverlay.classList.contains('hidden')){
          closePresetSettings();
        }else if(mainOverlay && !mainOverlay.classList.contains('hidden')){
          closeSettings();
        }
        return;
      }
      if(shouldHandleStartShortcut(event)){
        event.preventDefault();
        void startSequentialProcessing();
      }
    });

    // initialize overlay controls on load
    window.addEventListener('load', async () => {
      applyDesktopShellState();
      bindChoiceTooltips();
      settingsOverlay = document.getElementById('settings-overlay');
      presetSettingsOverlay = document.getElementById('preset-settings-overlay');
      presetSettingsBtn = document.getElementById('preset-settings-btn');
      if(isLanClient){
        if(bottomLeftControls){
          bottomLeftControls.hidden = true;
          bottomLeftControls.style.display = 'none';
        }
        if(presetSettingsBtn){
          presetSettingsBtn.hidden = true;
          presetSettingsBtn.style.display = 'none';
        }
      }
      outputFormatSelect = document.getElementById('output-format');
      multiStemExportSelect = document.getElementById('multi-stem-export');
      previousFilesRetentionSelect = document.getElementById('previous-files-retention');
      previousFilesRetentionButton = document.getElementById('previous-files-retention-button');
      previousFilesRetentionMenu = document.getElementById('previous-files-retention-menu');
      previousFilesRetentionLabel = document.getElementById('previous-files-retention-label');
	      previousFilesLimitInput = document.getElementById('previous-files-limit-gb');
	      previousFilesWarnInput = document.getElementById('previous-files-warn-gb');
	      editorSnapDistanceInput = document.getElementById('editor-snap-distance-ms');
	      outputFormatButton = document.getElementById('output-format-button');
      outputFormatMenu = document.getElementById('output-format-menu');
      outputFormatLabel = document.getElementById('output-format-label');
      multiStemExportButton = document.getElementById('multi-stem-export-button');
      multiStemExportMenu = document.getElementById('multi-stem-export-menu');
      multiStemExportLabel = document.getElementById('multi-stem-export-label');
      outputFolderInput = document.getElementById('output-folder-path');
      outputSameAsInput = document.getElementById('output-same-as-input');
      outputFolderChoose = document.getElementById('output-folder-choose');
      outputFolderOpen = document.getElementById('output-folder-open');
      settingsScrollIndicator = document.getElementById('settings-scroll-indicator');
      settingsScrollBody = document.getElementById('settings-card-scroll');
      settingsScrollThumb = document.getElementById('settings-scroll-thumb');
      videoAudioOnly = document.getElementById('video-audio-only');
      nerdStuffToggle = document.getElementById('nerd-stuff-toggle');
      nerdStuffWrap = document.getElementById('nerd-stuff-wrap');
      lanAccessHeading = document.getElementById('lan-access-heading');
      lanLocalText = document.getElementById('lan-local-text');
      lanIpText = document.getElementById('lan-ip-text');
      lanCopyLocalBtn = document.getElementById('lan-copy-local-btn');
      lanCopyIpBtn = document.getElementById('lan-copy-ip-btn');
      lanPasscodeEnabled = document.getElementById('lan-passcode-enabled');
      lanPasscodeInput = document.getElementById('lan-passcode-input');
      lanPasscodeIndicator = document.getElementById('lan-passcode-indicator');
      lanPasscodeVisibilityBtn = document.getElementById('lan-passcode-visibility');
      lanPasscodeTtl = document.getElementById('lan-passcode-ttl');
      lanPasscodeTtlButton = document.getElementById('lan-passcode-ttl-button');
      lanPasscodeTtlMenu = document.getElementById('lan-passcode-ttl-menu');
      lanPasscodeTtlLabel = document.getElementById('lan-passcode-ttl-label');
      lanPasscodeWrap = document.getElementById('lan-passcode-wrap');
      lanPasscodeTtlWrap = document.getElementById('lan-passcode-ttl-wrap');
      portStatusText = document.getElementById('port-status-text');
      modelsList = document.getElementById('models-list');
      modelsNote = document.getElementById('models-note');
      modelsDownloadBtn = document.getElementById('models-download');
      modelsCtaWrap = document.getElementById('models-cta-wrap');
      modelsFolderBtn = document.getElementById('models-folder-open');
      modelsProgressWrap = document.getElementById('models-progress-wrap');
      modelsTotal = document.getElementById('models-total');
      modelsProgressBar = document.getElementById('models-progress-bar');
      modelsProgressMeta = document.getElementById('models-progress-meta');
      boostHarmoniesBackgroundSlider = document.getElementById('boost-harmonies-background-slider');
      boostHarmoniesBaseSlider = document.getElementById('boost-harmonies-base-slider');
      boostHarmoniesBackgroundValue = document.getElementById('boost-harmonies-background-value');
      boostHarmoniesBaseValue = document.getElementById('boost-harmonies-base-value');
      const closeBtn = document.getElementById('settings-close');
      const presetCloseBtn = document.getElementById('preset-settings-close');
      const forceBtn = document.getElementById('force-clear');
      if(settingsOverlay) settingsOverlay.addEventListener('keydown', trapDialogFocus);
      if(presetSettingsOverlay) presetSettingsOverlay.addEventListener('keydown', trapDialogFocus);
      installAccessibleListbox(outputFormatButton, outputFormatMenu);
      installAccessibleListbox(multiStemExportButton, multiStemExportMenu);
      installAccessibleListbox(previousFilesRetentionButton, previousFilesRetentionMenu);
      installAccessibleListbox(lanPasscodeTtlButton, lanPasscodeTtlMenu);
      if(nerdStuffToggle){
        nerdStuffToggle.addEventListener('click', () => {
          setNerdStuffExpanded(!nerdStuffExpanded);
        });
      }
      if(nerdStuffWrap){
        nerdStuffWrap.addEventListener('transitionend', (event) => {
          if(event.propertyName !== 'max-height') return;
          if(nerdStuffExpanded){
            nerdStuffWrap.style.maxHeight = `${nerdStuffWrap.scrollHeight}px`;
          }
          updateSettingsScrollIndicator();
        });
        setNerdStuffExpanded(false, { immediate: true });
      }
      if(closeBtn) closeBtn.addEventListener('click', closeSettings);
      if(presetSettingsBtn) presetSettingsBtn.addEventListener('click', openPresetSettings);
      if(presetCloseBtn) presetCloseBtn.addEventListener('click', closePresetSettings);
      if(settingsOverlay) settingsOverlay.addEventListener('click', (e)=>{ if(e.target === settingsOverlay) closeSettings(); });
      if(presetSettingsOverlay) presetSettingsOverlay.addEventListener('click', (e)=>{ if(e.target === presetSettingsOverlay) closePresetSettings(); });
      if(settingsOverlay){
        const settingsCardEl = document.getElementById('settings-card');
        if(settingsCardEl && settingsScrollBody){
          settingsScrollBody.addEventListener('scroll', updateSettingsScrollIndicator, { passive: true });
          attachSoftWheelScroll(settingsScrollBody, updateSettingsScrollIndicator);
        }
      }
      if(forceBtn) forceBtn.addEventListener('click', async ()=>{
        try{
          await clearAllTasks();
          closeSettings();
          window.location.reload();
        }catch(err){ showError('force clear failed'); }
      });
      [
        [boostHarmoniesBackgroundSlider, 'boost_harmonies_background_vocals_gain_db'],
        [boostHarmoniesBaseSlider, 'boost_harmonies_base_song_gain_db'],
      ].forEach(([slider, key]) => {
        if(!slider) return;
        slider.addEventListener('input', () => {
          schedulePresetSettingsPersist({ [key]: Number(slider.value || 0) });
        });
      });
      checkStorage();
      checkMemory();
      setInterval(checkStorage, 15000);
      setInterval(checkMemory, 12000);
      await loadSettings();
      updateSettingsScrollIndicator();
      await refreshRuntimeStatus({ showPortNotice: true });
      startLanDisconnectMonitor();
      await refreshModelStatus();
      await checkForReleaseUpdate();
      setInterval(checkForReleaseUpdate, 12 * 60 * 60 * 1000);
      if(outputFormatSelect){
        outputFormatSelect.addEventListener('change', () => {
          persistSettings({ output_format: outputFormatSelect.value }, { showMissingPopup:false });
        });
      }
      if(multiStemExportSelect){
        multiStemExportSelect.addEventListener('change', () => {
          persistSettings({ multi_stem_export: multiStemExportSelect.value }, { showMissingPopup:false });
        });
      }
      if(outputFormatButton){
        outputFormatButton.addEventListener('click', () => {
          toggleOutputFormatMenu();
        });
      }
      if(multiStemExportButton){
        multiStemExportButton.addEventListener('click', () => {
          toggleMultiStemExportMenu();
        });
      }
      if(previousFilesRetentionButton){
        previousFilesRetentionButton.addEventListener('click', () => {
          togglePreviousFilesRetentionMenu();
        });
      }
      if(outputFormatMenu){
        outputFormatMenu.querySelectorAll('.fancy-select-option').forEach((option) => {
          option.addEventListener('click', () => {
            const nextValue = option.dataset.value || 'same_as_input';
            if(outputFormatSelect){
              outputFormatSelect.value = nextValue;
            }
            settingsState = { ...settingsState, output_format: nextValue };
            applySettingsUI();
            closeOutputFormatMenu();
            persistSettings({ output_format: nextValue }, { showMissingPopup:false });
          });
        });
      }
      if(multiStemExportMenu){
        multiStemExportMenu.querySelectorAll('.fancy-select-option').forEach((option) => {
          option.addEventListener('click', () => {
            const nextValue = option.dataset.value || 'zip';
            if(multiStemExportSelect){
              multiStemExportSelect.value = nextValue;
            }
            settingsState = { ...settingsState, multi_stem_export: nextValue };
            applySettingsUI();
            closeMultiStemExportMenu();
            persistSettings({ multi_stem_export: nextValue }, { showMissingPopup:false });
          });
        });
      }
      if(previousFilesRetentionMenu){
        previousFilesRetentionMenu.querySelectorAll('.fancy-select-option').forEach((option) => {
          option.addEventListener('click', () => {
            const nextValue = option.dataset.value || '1w';
            if(previousFilesRetentionSelect){
              previousFilesRetentionSelect.value = nextValue;
            }
            settingsState = { ...settingsState, previous_files_retention: nextValue };
            applySettingsUI();
            closePreviousFilesRetentionMenu();
            persistSettings({ previous_files_retention: nextValue }, { showMissingPopup:false });
          });
        });
      }
      const persistPreviousFilesStorageSettings = () => {
        const limitGb = coerceStorageSettingValue(previousFilesLimitInput ? previousFilesLimitInput.value : settingsState.previous_files_limit_gb, settingsState.previous_files_limit_gb ?? 10, 0.5);
        let warnGb = coerceStorageSettingValue(previousFilesWarnInput ? previousFilesWarnInput.value : settingsState.previous_files_warn_gb, settingsState.previous_files_warn_gb ?? 8, 0.1);
        warnGb = Math.min(limitGb, warnGb);
        settingsState = { ...settingsState, previous_files_limit_gb: limitGb, previous_files_warn_gb: warnGb };
        applySettingsUI();
        persistSettings({ previous_files_limit_gb: limitGb, previous_files_warn_gb: warnGb }, { showMissingPopup:false });
      };
	      [previousFilesLimitInput, previousFilesWarnInput].forEach((input) => {
	        if(!input) return;
	        input.addEventListener('change', persistPreviousFilesStorageSettings);
	        input.addEventListener('blur', persistPreviousFilesStorageSettings);
	      });
	      if(editorSnapDistanceInput){
	        const persistEditorSnapDistance = () => {
	          const parsed = Math.max(0, Math.min(2000, Math.round(Number(editorSnapDistanceInput.value) || 0)));
	          settingsState = { ...settingsState, editor_snap_distance_ms: parsed };
	          applySettingsUI();
	          persistSettings({ editor_snap_distance_ms: parsed }, { showMissingPopup:false });
	        };
	        editorSnapDistanceInput.addEventListener('change', persistEditorSnapDistance);
	        editorSnapDistanceInput.addEventListener('blur', persistEditorSnapDistance);
	      }
      document.addEventListener('click', (event) => {
        if(outputFormatButton && outputFormatMenu){
          if(!(outputFormatButton.contains(event.target) || outputFormatMenu.contains(event.target))){
            closeOutputFormatMenu();
          }
        }
        if(multiStemExportButton && multiStemExportMenu){
          if(!(multiStemExportButton.contains(event.target) || multiStemExportMenu.contains(event.target))){
            closeMultiStemExportMenu();
          }
        }
        if(lanPasscodeTtlButton && lanPasscodeTtlMenu){
          if(!(lanPasscodeTtlButton.contains(event.target) || lanPasscodeTtlMenu.contains(event.target))){
            closeLanTtlMenu();
          }
        }
        if(previousFilesRetentionButton && previousFilesRetentionMenu){
          if(!(previousFilesRetentionButton.contains(event.target) || previousFilesRetentionMenu.contains(event.target))){
            closePreviousFilesRetentionMenu();
          }
        }
      });
      if(outputSameAsInput){
        outputSameAsInput.addEventListener('change', () => {
          persistSettings({ output_same_as_input: outputSameAsInput.checked }, { showMissingPopup:false });
        });
      }
      if(videoAudioOnly){
        videoAudioOnly.addEventListener('change', () => {
          if(videoAudioOnly.checked){
            persistSettings({ video_handling: 'audio_only' }, { showMissingPopup:false });
          }
        });
      }
      if(outputFolderChoose){
        outputFolderChoose.addEventListener('click', async () => {
          if(settingsState.output_same_as_input) return;
          try{
            if(window.pywebview && window.pywebview.api && typeof window.pywebview.api.pick_output_folder === 'function'){
              const chosen = await window.pywebview.api.pick_output_folder();
              if(chosen){
                await persistSettings({ output_root: chosen }, { showMissingPopup:false });
              }
            }else{
              const res = await fetch('/api/settings/output_root/pick', { method: 'POST' });
              if(!res.ok) return;
              const data = await res.json();
              if(!data.cancelled && data.output_root){
                settingsState = { ...settingsState, output_root: data.output_root };
                applySettingsUI();
              }
            }
          }catch(err){
            showPopup('could not choose folder');
          }
        });
      }
      if(outputFolderOpen){
        outputFolderOpen.addEventListener('click', async () => {
          if(settingsState.output_same_as_input) return;
          try{
            if(window.pywebview && window.pywebview.api && typeof window.pywebview.api.open_path === 'function'){
              const ok = await window.pywebview.api.open_path(settingsState.output_root || '');
              if(!ok){
                showPopup('could not open folder');
              }
            }else{
              await fetch('/api/settings/output_root/open', { method: 'POST' });
            }
          }catch(err){
            showPopup('could not open folder');
          }
        });
      }
      if(lanCopyLocalBtn){
        lanCopyLocalBtn.innerHTML = createCopyIconMarkup();
        lanCopyLocalBtn.dataset.copied = '0';
        lanCopyLocalBtn.addEventListener('click', async () => {
          const runtime = settingsState.runtime || {};
          const target = runtime.lan_local_display || '';
          if(!target) return;
          const ok = await copyText(target);
          if(ok){
            pulseCopiedState(lanCopyLocalBtn);
          }
        });
      }
      if(lanCopyIpBtn){
        lanCopyIpBtn.innerHTML = createCopyIconMarkup();
        lanCopyIpBtn.dataset.copied = '0';
        lanCopyIpBtn.addEventListener('click', async () => {
          const runtime = settingsState.runtime || {};
          const target = runtime.lan_display || '';
          if(!target) return;
          const ok = await copyText(target);
          if(ok){
            pulseCopiedState(lanCopyIpBtn);
          }
        });
      }
      if(lanPasscodeEnabled){
        lanPasscodeEnabled.addEventListener('change', async () => {
          lanPasscodeDirty = false;
          lanPasscodeVisible = false;
          lanPasscodeDraft = '';
          if(!lanPasscodeEnabled.checked){
            try{ await persistLanConfig({ enabled: false }); }
            catch(error){ showPopup(error.message || 'could not disable LAN access'); await loadSettings(); }
            return;
          }
          if(settingsState.lan_passcode_configured){
            try{ await persistLanConfig({ enabled: true }); }
            catch(error){ showPopup(error.message || 'could not enable LAN access'); await loadSettings(); }
            return;
          }
          settingsState = { ...settingsState, lan_passcode_enabled: true, lan_access_enabled: true };
          applySettingsUI();
          if(lanPasscodeInput){ lanPasscodeInput.focus(); }
          showPopup('enter and confirm a passcode to enable LAN access');
        });
      }
      if(lanPasscodeInput){
        lanPasscodeInput.maxLength = 24;
        lanPasscodeInput.addEventListener('input', () => {
          if(lanPasscodeInput.value.length > 24){
            lanPasscodeInput.value = lanPasscodeInput.value.slice(0, 24);
          }
          lanPasscodeDraft = String(lanPasscodeInput.value || '');
          lanPasscodeDirty = lanPasscodeDraft !== String(settingsState.lan_passcode || '');
          applySettingsUI();
        });
        lanPasscodeInput.addEventListener('keydown', (event) => {
          if(event.key === 'Enter' && lanPasscodeIndicator && !lanPasscodeIndicator.disabled){
            event.preventDefault();
            lanPasscodeIndicator.click();
          }
        });
      }
      if(lanPasscodeIndicator){
        lanPasscodeIndicator.addEventListener('click', async () => {
          if(lanPasscodeIndicator.disabled || !lanPasscodeInput || !settingsState.lan_passcode_enabled){
            return;
          }
          lanPasscodeDraft = String(lanPasscodeInput.value || '');
          try{
            await persistLanConfig({ enabled: true, passcode: lanPasscodeDraft });
            lanPasscodeDirty = false;
            lanPasscodeDraft = '';
            applySettingsUI();
          }catch(error){
            lanPasscodeDirty = true;
            showPopup(error.message || 'could not enable LAN access');
          }
        });
      }
      if(lanPasscodeVisibilityBtn){
        lanPasscodeVisibilityBtn.addEventListener('click', () => {
          if(lanPasscodeVisibilityBtn.disabled){
            return;
          }
          lanPasscodeVisible = !lanPasscodeVisible;
          applySettingsUI();
        });
      }
      if(lanPasscodeTtl){
        lanPasscodeTtl.addEventListener('change', () => {
          settingsState = { ...settingsState, lan_passcode_ttl: lanPasscodeTtl.value };
          if(settingsState.lan_access_enabled){
            persistLanConfig({ enabled: true }).catch((error) => showPopup(error.message || 'could not update LAN access'));
          }else{
            persistSettings({ lan_passcode_ttl: lanPasscodeTtl.value }, { showMissingPopup:false });
          }
        });
      }
      if(lanPasscodeTtlButton){
        lanPasscodeTtlButton.addEventListener('click', () => {
          toggleLanTtlMenu();
        });
      }
      if(lanPasscodeTtlMenu){
        lanPasscodeTtlMenu.querySelectorAll('.fancy-select-option').forEach((option) => {
          option.addEventListener('click', () => {
            const nextValue = option.dataset.value || '1d';
            if(lanPasscodeTtl){
              lanPasscodeTtl.value = nextValue;
            }
            settingsState = { ...settingsState, lan_passcode_ttl: nextValue };
            applySettingsUI();
            closeLanTtlMenu();
            if(settingsState.lan_access_enabled){
              persistLanConfig({ enabled: true }).catch((error) => showPopup(error.message || 'could not update LAN access'));
            }else{
              persistSettings({ lan_passcode_ttl: nextValue }, { showMissingPopup:false });
            }
          });
        });
      }
      if(modelsDownloadBtn){
        modelsDownloadBtn.addEventListener('click', async () => {
          if(modelPreviewActive){
            simulateModelPreviewDownload();
            return;
          }
          const missing = Array.isArray(lastModelStatus && lastModelStatus.missing) ? lastModelStatus.missing : [];
          await beginModelDownload(missing);
        });
      }
      if(modelsFolderBtn){
        modelsFolderBtn.addEventListener('click', () => openModelsFolderAction());
      }
      if(structureCheck){
        structureCheck.addEventListener('change', () => {
          if(structureCheck.checked){
            persistSettings({ structure_mode: 'structured' });
          }
        });
      }
      if(structurelessCheck){
        structurelessCheck.addEventListener('change', () => {
          if(structurelessCheck.checked){
            persistSettings({ structure_mode: 'flat' });
          }
        });
      }
      window.addEventListener('resize', () => {
        updateSettingsScrollIndicator();
        settleFullscreenTransition();
        positionModeSwitcherPill();
      });
    });
    window.addEventListener('pywebviewready', applyDesktopShellState);
    if(windowCloseBtn){
      windowCloseBtn.addEventListener('click', async () => {
        const handled = await callDesktopAction('close_window');
        if(!handled) window.close();
      });
    }
    if(windowMinimizeBtn){
      windowMinimizeBtn.addEventListener('click', () => {
        callDesktopAction('minimize_window');
      });
    }
    if(windowFullscreenBtn){
      windowFullscreenBtn.addEventListener('click', () => {
        beginFullscreenTransition();
        callDesktopAction('toggle_fullscreen_window');
      });
    }

    // drag-and-drop highlights
    ['dragenter','dragover'].forEach(evt =>
      dropzone.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.add('ring-2','ring-[var(--accent)]');
      }));
    ['dragleave','drop'].forEach(evt =>
      dropzone.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.remove('ring-2','ring-[var(--accent)]');
      }));

    // open file picker on click/keyboard
    const triggerPicker = async () => {
      if(window.pywebview && window.pywebview.api && window.pywebview.api.pick_media_files){
        const picked = await window.pywebview.api.pick_media_files();
        await handleDesktopPickedPaths(Array.isArray(picked) ? picked : []);
        return;
      }
      try {
        if (fileInput.showPicker) { fileInput.showPicker(); return; }
      } catch(_) {}
      fileInput.click();
    };
    dropzone.addEventListener('click', (e) => {
      e.preventDefault();
      triggerPicker();
    });
    dropzone.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        void triggerPicker();
      }
    });

    // handle both dropped and picked files
    dropzone.addEventListener('drop', e => {
      captureDroppedSourceDirs(e);
      handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', e => handleFiles(e.target.files));
    if(modeSwitcher){
      modeSwitcher.addEventListener('click', (event) => {
        const button = event.target.closest('.mode-switch-btn');
        if(!button) return;
        const nextTab = button.dataset.tab || 'single';
        if(nextTab === activeModeTab) return;
        activeModeTab = nextTab;
        ensureSelectionForActiveTab();
        renderModeChoices(true);
        Promise.resolve(syncQueuedStemSelection()).catch(() => {});
      });
    }
    if(modeChoices){
      modeChoices.addEventListener('click', (event) => {
        const button = event.target.closest('.stem-choice, .preset-choice');
        if(!button) return;
        if(button.dataset.preset){
          const nextPreset = button.dataset.preset || '';
          if(selectedPresetMode === nextPreset){
            ensureSelectionForActiveTab();
          }else{
            selectedPresetMode = nextPreset || null;
            selectedStemModes = new Set();
          }
          updateModeChoiceUI();
          Promise.resolve(syncQueuedStemSelection()).catch(() => {});
          return;
        }
        const nextMode = button.dataset.mode || 'vocals';
        if(event.shiftKey){
          selectStemModeRange(nextMode);
        }else{
          toggleStemModeSelection(nextMode, !!(event.metaKey || event.ctrlKey));
        }
        updateModeChoiceUI();
        Promise.resolve(syncQueuedStemSelection()).catch(() => {});
      });
    }
    if(queue){
      queue.addEventListener('contextmenu', (event) => {
        const target = event.target instanceof Element ? event.target : null;
        const row = target ? target.closest('.item-row') : null;
        if(!row) return;
        openQueueContextMenu(event, row);
      });
    }
    ensureSelectionForActiveTab();
    renderModeChoices();

    function showError(msg){
      const message = String(msg || 'unknown error');
      const { overlay, card } = createOverlayCard('upload failed');
      const copy = document.createElement('p');
      copy.className = 'text-sm leading-relaxed text-[var(--txt-main)]/85';
      copy.textContent = message;
      const actions = document.createElement('div');
      actions.className = 'flex justify-end gap-2';
      const okBtn = makeActionButton('ok', 'bg-white text-black');
      actions.append(okBtn);
      card.append(copy, actions);
      let settled = false;
      const finish = () => {
        if(settled) return;
        settled = true;
        closeOverlay(overlay);
      };
      okBtn.addEventListener('click', (event) => {
        event.preventDefault();
        finish();
      });
      overlay.addEventListener('click', (event) => {
        if(event.target === overlay || (event.target && event.target.classList && event.target.classList.contains('overlay-bg'))){
          finish();
        }
      }, true);
      overlay.addEventListener('keydown', (event) => {
        if(event.key === 'Escape' || event.key === 'Enter'){
          event.preventDefault();
          event.stopPropagation();
          finish();
        }
      }, true);
    }

    function ensureToastWrap(){
      let wrap = document.getElementById('toast-wrap');
      if(!wrap){
        wrap = document.createElement('div');
        wrap.id = 'toast-wrap';
        document.body.appendChild(wrap);
      }
      return wrap;
    }

    function showPopup(msg){
      const wrap = ensureToastWrap();
      const card = document.createElement('div');
      card.textContent = msg;
      card.className = 'toast-card glass text-[var(--txt-main)]';
      wrap.appendChild(card);
      const isErrorLike = /errorcode\.|failed|error/i.test(String(msg || ''));
      setTimeout(() => dismissToastCard(card), isErrorLike ? 8000 : 3000);
    }
    const reopenParam = new URLSearchParams(window.location.search).get('reopen');
    if(reopenParam === '1'){
      showPopup("sorry, it appears stemsplat wasn't shut down. please restart the app.");
      window.history.replaceState({}, '', window.location.pathname);
    }

    function showConfirm(msg){
      return new Promise((resolve) => {
        const { overlay, card } = createOverlayCard('confirm action');
        const copy = document.createElement('p');
        copy.className = 'text-sm leading-relaxed text-[var(--txt-main)]/85';
        copy.textContent = String(msg || '');
        const actions = document.createElement('div');
        actions.className = 'flex justify-end gap-2';
        const cancelBtn = makeActionButton('cancel', 'bg-white/10 text-white');
        const okBtn = makeActionButton('yes', 'bg-white text-black');
        actions.append(cancelBtn, okBtn);
        card.append(copy, actions);
        let settled = false;
        const finish = (result) => {
          if(settled) return;
          settled = true;
          closeOverlay(overlay);
          resolve(result);
        };
        cancelBtn.addEventListener('click', (event) => {
          event.preventDefault();
          finish(false);
        });
        okBtn.addEventListener('click', (event) => {
          event.preventDefault();
          finish(true);
        });
        overlay.addEventListener('click', (event) => {
          if(event.target === overlay || (event.target && event.target.classList && event.target.classList.contains('overlay-bg'))){
            finish(false);
          }
        }, true);
        overlay.addEventListener('keydown', (event) => {
          if(event.key === 'Escape'){
            event.preventDefault();
            event.stopPropagation();
            finish(false);
          }else if(event.key === 'Enter'){
            event.preventDefault();
            event.stopPropagation();
            finish(true);
          }
        }, true);
      });
    }

    function dismissToastCard(card, { immediate = false } = {}){
      if(!card) return;
      if(immediate){
        if(card.parentNode) card.parentNode.removeChild(card);
        return;
      }
      if(card.dataset.toastClosing === '1') return;
      card.dataset.toastClosing = '1';
      card.classList.add('toast-card-closing');
      setTimeout(() => {
        if(card.parentNode) card.parentNode.removeChild(card);
      }, 190);
    }

    function closeOverlay(overlay){
      if(!overlay || !overlay.parentNode) return;
      if(Array.isArray(overlay.__cleanupFns)){
        overlay.__cleanupFns.forEach((cleanup) => {
          try { cleanup(); } catch(_) {}
        });
        overlay.__cleanupFns = [];
      }
      if(overlay === activeModelOverlay){
        activeModelOverlay = null;
        activeModelOverlayUi = null;
      }
      if(overlay === activeHostClosedOverlay){
        activeHostClosedOverlay = null;
      }
      const bg = overlay.querySelector('.overlay-bg');
      const card = overlay.querySelector('.overlay-card');
      if(bg){
        bg.classList.remove('anim');
        bg.classList.add('anim-out');
      }
      if(card){
        card.classList.remove('settings-card-in');
        card.classList.add('settings-card-out');
      }
      setTimeout(() => {
        if(overlay.parentNode){
          overlay.parentNode.removeChild(overlay);
        }
      }, 190);
    }

    async function copyText(text){
      try{
        if(navigator.clipboard && navigator.clipboard.writeText){
          await navigator.clipboard.writeText(text);
          return true;
        }
      }catch(_){}
      try{
        const input = document.createElement('textarea');
        input.value = text;
        input.setAttribute('readonly', '');
        input.style.position = 'absolute';
        input.style.left = '-9999px';
        document.body.appendChild(input);
        input.select();
        const ok = document.execCommand('copy');
        input.remove();
        return ok;
      }catch(_){
        return false;
      }
    }

    function createCopyIconMarkup(){
      return `
        <span class="icon-wrap" aria-hidden="true">
          <svg class="icon-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="10" height="10" rx="2"></rect>
            <path d="M5 15V7a2 2 0 0 1 2-2h8"></path>
          </svg>
          <svg class="icon-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M5 12l4.2 4.2L19 6.5"></path>
          </svg>
        </span>`;
    }

    function pulseCopiedState(btn){
      if(!btn) return;
      const existing = copyResetTimers.get(btn);
      if(existing){
        clearTimeout(existing);
      }
      btn.dataset.copied = '1';
      const timer = setTimeout(() => {
        btn.dataset.copied = '0';
        copyResetTimers.delete(btn);
      }, 5000);
      copyResetTimers.set(btn, timer);
    }

    function requestOverlayClose(overlay, reason = 'dismiss'){
      if(!overlay) return;
      if(typeof overlay.__requestClose === 'function'){
        return overlay.__requestClose(reason);
      }
      closeOverlay(overlay);
    }

    function createOverlayCard(titleText){
      const overlay = document.createElement('div');
      overlay.className = 'overlay-root fixed inset-0 z-[80] grid place-items-center px-4';
      overlay.tabIndex = -1;
      const bg = document.createElement('div');
      bg.className = 'overlay-bg anim';
      const card = document.createElement('div');
      card.className = 'overlay-card glass settings-card-in w-full max-w-md rounded-2xl p-6 text-[var(--txt-main)] flex flex-col gap-4 relative';
      const titleEl = document.createElement('h3');
      titleEl.className = 'text-xl font-light';
      titleEl.textContent = titleText;
      card.appendChild(titleEl);
      overlay.append(bg, card);
      document.body.appendChild(overlay);
      requestAnimationFrame(() => {
        try { overlay.focus({ preventScroll: true }); } catch(_) {}
      });
      overlay.addEventListener('click', (e) => {
        if(e.target === overlay || e.target === bg){
          if(typeof overlay.__dismissGuard === 'function' && overlay.__dismissGuard()){
            return;
          }
          requestOverlayClose(overlay, 'dismiss');
          if(activeReleaseOverlay === overlay){
            dismissReleaseUntilNextOpen(activeReleaseVersion);
            activeReleaseOverlay = null;
            activeReleaseOverlayUi = null;
            activeReleaseVersion = '';
            stopReleaseStatusPolling();
          }
          if(activePortOverlay === overlay) activePortOverlay = null;
          if(activeModelOverlay === overlay) activeModelOverlay = null;
          if(activeHostClosedOverlay === overlay) activeHostClosedOverlay = null;
        }
      });
      overlay.addEventListener('keydown', (e) => {
        if(e.key === 'Escape'){
          requestOverlayClose(overlay, 'dismiss');
          if(activeReleaseOverlay === overlay){
            dismissReleaseUntilNextOpen(activeReleaseVersion);
            activeReleaseOverlay = null;
            activeReleaseOverlayUi = null;
            activeReleaseVersion = '';
            stopReleaseStatusPolling();
          }
          if(activePortOverlay === overlay) activePortOverlay = null;
          if(activeModelOverlay === overlay) activeModelOverlay = null;
          if(activeHostClosedOverlay === overlay) activeHostClosedOverlay = null;
        }
      });
      overlay.tabIndex = 0;
      overlay.focus();
      return { overlay, card };
    }

    function makeActionButton(label, extraClass = ''){
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `btn px-3 py-2 text-xs ${extraClass}`.trim();
      btn.textContent = label;
      return btn;
    }

    function parseReleaseHighlights(notes){
      const text = String(notes || '').trim();
      if(!text) return [];
      const lines = text
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .filter((line) => !/^summary:/i.test(line))
        .filter((line) => !/^highlights:/i.test(line));
      const bulletLines = lines
        .filter((line) => /^[-*•]/.test(line))
        .map((line) => line.replace(/^[-*•]\s*/, '').trim())
        .filter(Boolean);
      if(bulletLines.length){
        return bulletLines.slice(0, 5);
      }
      const plain = lines
        .map((line) => line.replace(/^[-*•]\s*/, '').trim())
        .filter(Boolean)
        .join(' ');
      if(!plain) return [];
      return plain
        .split(/\s*[;,]\s*/)
        .map((part) => part.trim())
        .filter(Boolean)
        .slice(0, 5);
    }

    function applyRuntimeUI(runtime){
      const data = runtime && typeof runtime === 'object' ? runtime : {};
      const appVersion = document.getElementById('app-version');
      if(appVersion){ appVersion.textContent = data.app_version ? `v${data.app_version}` : 'version unavailable'; }
      const lanDisplay = data.lan_display || '';
      const lanLocalDisplay = data.lan_local_display || '';
      const networkName = (data.network_name || '').trim();
      if(lanAccessHeading){
        lanAccessHeading.textContent = networkName
          ? `lan access - connect from other devices on the '${networkName}' network`
          : 'lan access - connect from other devices on the local network';
      }
      if(lanLocalText){
        lanLocalText.textContent = lanLocalDisplay || 'unavailable';
      }
      if(lanIpText){
        lanIpText.textContent = lanDisplay ? `${lanDisplay} (not recommended)` : 'unavailable';
      }
      if(lanCopyLocalBtn){
        lanCopyLocalBtn.disabled = !lanLocalDisplay;
        lanCopyLocalBtn.classList.toggle('dimmed-control', !lanLocalDisplay);
      }
      if(lanCopyIpBtn){
        lanCopyIpBtn.disabled = !lanDisplay;
        lanCopyIpBtn.classList.toggle('dimmed-control', !lanDisplay);
      }
      if(portStatusText){
        if(data.port_conflict){
          portStatusText.hidden = false;
          portStatusText.textContent = `port ${data.preferred_port || 9876} is in use. currently running on ${data.current_port || data.preferred_port || 9876}.`;
        }else{
          portStatusText.hidden = true;
          portStatusText.textContent = '';
        }
      }
      if(data.ffmpeg_available === false && !ffmpegNoticeShown){
        ffmpegNoticeShown = true;
        showPopup(data.ffmpeg_message || 'ffmpeg is missing. install ffmpeg or use the bundled fallback build.');
      }
      if(nerdStuffExpanded && nerdStuffWrap){
        nerdStuffWrap.style.maxHeight = `${nerdStuffWrap.scrollHeight}px`;
      }
    }

    function renderModelChecklist(status){
      if(!modelsList) return;
      const missing = new Set(Array.isArray(status && status.missing) ? status.missing : []);
      const modelMap = new Map(
        (Array.isArray(status && status.models) ? status.models : []).map((item) => [item.key, item])
      );
      let installedBytes = 0;
      modelsList.innerHTML = '';
      MODEL_CHECKLIST_ORDER.forEach((key) => {
        const detail = modelMap.get(key) || null;
        const row = document.createElement('div');
        const ready = !missing.has(key);
        row.className = `model-item${ready ? ' ready' : ''}`;
        const main = document.createElement('div');
        main.className = 'model-item-main';
        const box = document.createElement('span');
        box.className = 'model-state';
        box.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4L9 15"></path><path d="M21 19L3 19"></path><path d="M9 15L4 10"></path></svg>`;
        const copy = document.createElement('span');
        copy.className = 'model-item-copy';
        const label = document.createElement('span');
        label.className = 'model-item-label';
        label.textContent = MODEL_LABELS[key] || key;
        copy.appendChild(label);
        const displaySize = ready ? Number(detail && detail.size_bytes) : Number(detail && detail.expected_size_bytes);
        if(ready && Number(detail && detail.size_bytes) > 0){
          installedBytes += Number(detail.size_bytes) || 0;
        }
        if(displaySize > 0){
          const size = document.createElement('span');
          size.className = 'model-item-size';
          size.textContent = formatBytes(displaySize);
          copy.appendChild(size);
        }
        main.append(box, copy);
        row.appendChild(main);
        const repoUrl = MODEL_SOURCE_URLS[key];
        if(repoUrl){
          const actions = document.createElement('div');
          actions.className = 'model-item-actions';
          const repoLink = document.createElement('a');
          repoLink.className = 'model-action-btn';
          repoLink.href = repoUrl;
          repoLink.target = '_blank';
          repoLink.rel = 'noopener noreferrer';
          repoLink.title = `open ${MODEL_LABELS[key] || key} page`;
          repoLink.setAttribute('aria-label', `open ${MODEL_LABELS[key] || key} page`);
          repoLink.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3h7v7"></path><path d="M10 14 21 3"></path><path d="M21 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path></svg>`;
          actions.appendChild(repoLink);
          row.appendChild(actions);
        }
        modelsList.appendChild(row);
      });
      if(modelsTotal){
        const expectedTotalBytes = Number(status && status.expected_total_bytes) || 0;
        const totalBytes = expectedTotalBytes > 0 ? expectedTotalBytes : installedBytes;
        if(totalBytes > 0 && installedBytes > 0 && expectedTotalBytes > installedBytes){
          modelsTotal.textContent = `${formatBytes(installedBytes)} installed · ${formatBytes(totalBytes)} total`;
        }else{
          modelsTotal.textContent = totalBytes > 0 ? `total ${formatBytes(totalBytes)}` : '';
        }
        modelsTotal.hidden = totalBytes <= 0;
      }
    }

    async function refreshRuntimeStatus({ showPortNotice = false } = {}){
      try{
        const res = await fetch('/api/runtime_status');
        if(res.status === 401){
          if(isLanClient){
            window.location.reload();
          }
          return null;
        }
        if(!res.ok) return null;
        const data = await res.json();
        lanRuntimeFailures = 0;
        settingsState.runtime = data;
        applyRuntimeUI(data);
        if(showPortNotice && data.show_port_notice){
          showPortConflictOverlay(data);
        }
        return data;
      }catch(err){
        console.warn('runtime status failed', err);
        return null;
      }
    }

    function showHostClosedOverlay(){
      if(activeHostClosedOverlay) return;
      closeOverlay(activePortOverlay);
      closeOverlay(activeReleaseOverlay);
      closeOverlay(activeModelOverlay);
      activePortOverlay = null;
      activeReleaseOverlay = null;
      activeReleaseOverlayUi = null;
      activeReleaseVersion = '';
      stopReleaseStatusPolling();
      activeModelOverlay = null;

      const { overlay, card } = createOverlayCard('program was closed on host');
      activeHostClosedOverlay = overlay;

      const info = document.createElement('div');
      info.className = 'text-sm opacity-90 whitespace-pre-line';
      info.textContent = 'the stemsplat app was closed on the host machine. reopen it on the host to continue using lan access.';
      card.appendChild(info);

      const row = document.createElement('div');
      row.className = 'flex gap-3 justify-end flex-wrap';
      const reloadBtn = makeActionButton('reload', 'bg-white text-black');
      reloadBtn.onclick = () => window.location.reload();
      row.appendChild(reloadBtn);
      card.appendChild(row);
    }

    function startLanDisconnectMonitor(){
      if(!isLanClient || lanRuntimeHeartbeat) return;
      const tick = async () => {
        if(activeHostClosedOverlay) return;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 2500);
        try{
          const res = await fetch(`/api/runtime_status?_=${Date.now()}`, {
            cache: 'no-store',
            signal: controller.signal,
          });
          clearTimeout(timer);
          if(res.status === 401){
            window.location.reload();
            return;
          }
          if(!res.ok){
            throw new Error(`runtime status ${res.status}`);
          }
          lanRuntimeFailures = 0;
        }catch(err){
          clearTimeout(timer);
          lanRuntimeFailures += 1;
          if(lanRuntimeFailures >= 2){
            showHostClosedOverlay();
          }
        }
      };
      tick();
      lanRuntimeHeartbeat = setInterval(tick, 3000);
    }

    function showPortConflictOverlay(status){
      closeOverlay(activePortOverlay);
      const preferredPort = status.preferred_port || 9876;
      const currentPort = status.current_port || preferredPort;
      const { overlay, card } = createOverlayCard(`port ${preferredPort} is busy`);
      activePortOverlay = overlay;

      const info = document.createElement('div');
      info.className = 'text-sm opacity-90 whitespace-pre-line';
      info.textContent = `stemsplat started on port ${currentPort} because port ${preferredPort} is already in use. you can stay on this port or free ${preferredPort} and retry.`;
      card.appendChild(info);

      const commandWrap = document.createElement('div');
      commandWrap.className = 'flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2';
      const commandCode = document.createElement('code');
      commandCode.className = 'flex-1 text-xs break-all opacity-80';
      commandCode.textContent = status.kill_command || `kill -9 $(lsof -ti tcp:${preferredPort})`;
      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.className = 'copy-button';
      copyBtn.title = 'copy command';
      copyBtn.setAttribute('aria-label', 'copy command');
      copyBtn.dataset.copied = '0';
      copyBtn.innerHTML = createCopyIconMarkup();
      copyBtn.onclick = async () => {
        const ok = await copyText(commandCode.textContent || '');
        if(ok){
          pulseCopiedState(copyBtn);
        }
      };
      commandWrap.append(commandCode, copyBtn);
      card.appendChild(commandWrap);

      const row = document.createElement('div');
      row.className = 'grid grid-cols-1 gap-3 sm:grid-cols-3';
      const stayBtn = makeActionButton('i just want to use the app', 'w-full min-w-0 whitespace-normal text-center leading-tight bg-white/10 text-white');
      const terminalBtn = makeActionButton('open terminal', 'w-full min-w-0 whitespace-normal text-center leading-tight bg-white/10 text-white');
      const retryBtn = makeActionButton(`retry port ${preferredPort}`, 'w-full min-w-0 whitespace-normal text-center leading-tight bg-white text-black');

      terminalBtn.onclick = async () => {
        try{
          const res = await fetch('/api/open_terminal', { method: 'POST' });
          if(!res.ok){
            showPopup(await responseErrorMessage(res, 'could not open terminal'));
          }
        }catch(err){
          showPopup('could not open terminal');
        }
      };

      stayBtn.onclick = async () => {
        let next = null;
        try{
          if(window.pywebview && window.pywebview.api && window.pywebview.api.acknowledge_port_conflict){
            next = await window.pywebview.api.acknowledge_port_conflict();
          }
        }catch(err){
          console.warn('port conflict acknowledge failed', err);
        }
        if(!next){
          const baseRuntime = settingsState.runtime || {};
          const lanBase = String(baseRuntime.lan_display || '').split(':')[0];
          const lanLocalBase = String(baseRuntime.lan_local_display || '').split(':')[0];
          next = {
            ...baseRuntime,
            preferred_port: preferredPort,
            current_port: currentPort,
            port_conflict: false,
            show_port_notice: false,
            lan_display: lanBase ? `${lanBase}:${currentPort}` : (status.lan_display || ''),
            lan_local_display: lanLocalBase ? `${lanLocalBase}:${currentPort}` : (status.lan_local_display || ''),
          };
        }
        settingsState.runtime = next;
        applyRuntimeUI(next);
        closeOverlay(activePortOverlay);
        activePortOverlay = null;
      };

      retryBtn.onclick = async () => {
        retryBtn.disabled = true;
        stayBtn.disabled = true;
        try{
          if(!(window.pywebview && window.pywebview.api && window.pywebview.api.retry_preferred_port)){
            showPopup('retry is only available inside the packaged app');
            return;
          }
          const next = await window.pywebview.api.retry_preferred_port();
          if(next){
            settingsState.runtime = next;
            applyRuntimeUI(next);
          }
          if(next && next.switched && next.client_url){
            closeOverlay(activePortOverlay);
            activePortOverlay = null;
            window.location.replace(next.client_url);
            return;
          }
          showPopup((next && next.error) || `port ${preferredPort} is still in use`);
        }catch(err){
          console.warn('port retry failed', err);
          showPopup(`could not switch to port ${preferredPort}`);
        }finally{
          retryBtn.disabled = false;
          stayBtn.disabled = false;
          terminalBtn.disabled = false;
        }
      };

      row.append(stayBtn, terminalBtn, retryBtn);
      card.appendChild(row);
    }

    function formatBytes(bytes){
      const value = Number(bytes || 0);
      if(!value || value < 1024) return '0 b';
      const units = ['kb', 'mb', 'gb', 'tb'];
      let size = value / 1024;
      let unit = units[0];
      for(let i = 1; i < units.length && size >= 1024; i += 1){
        size /= 1024;
        unit = units[i];
      }
      return `${size.toFixed(size >= 100 ? 0 : 1)} ${unit}`;
    }

    function allMissingModels(status = lastModelStatus){
      const missing = new Set(Array.isArray(status && status.missing) ? status.missing : []);
      return (Array.isArray(status && status.models) ? status.models : [])
        .filter((model) => model && missing.has(model.key) && model.auto_download && model.release_eligible)
        .map((model) => model.key);
    }

    function expectedMissingModelBytes(status = lastModelStatus){
      const value = Number(status && status.expected_missing_total_bytes);
      return Number.isFinite(value) && value > 0 ? value : 0;
    }

    function formatDuration(seconds){
      const value = Number(seconds);
      if(!Number.isFinite(value) || value < 0) return '';
      if(value < 60) return `${Math.round(value)}s left`;
      const mins = Math.floor(value / 60);
      const secs = Math.round(value % 60);
      if(mins < 60) return `${mins}m ${secs}s left`;
      const hours = Math.floor(mins / 60);
      const remMins = mins % 60;
      return `${hours}h ${remMins}m left`;
    }

    function stopModelDownloadAttention(...buttons){
      buttons.forEach((button) => {
        if(!button) return;
        button.classList.remove('attention-sequence', 'attention-ring');
      });
      if(modelsDownloadBtn){
        modelsDownloadBtn.classList.remove('attention-sequence', 'attention-ring');
      }
    }

    function isModelDownloadBusyStatus(status){
      const value = String(status || '').toLowerCase();
      return value === 'downloading' || value === 'retrying';
    }

    function currentModelBlockMessage(){
      const status = String(lastModelStatus && lastModelStatus.status || '').toLowerCase();
      if(isModelDownloadBusyStatus(status)){
        return 'models still downloading';
      }
      const missing = relevantMissingModels();
      if(missing.length === 0){
        return 'download the models before splitting';
      }
      const labels = missing.map((key) => MODEL_LABELS[key] || key);
      return `download ${labels.join(', ')} before splitting`;
    }

    function clearModelToastTimer(){
      if(modelToastTimer){
        clearTimeout(modelToastTimer);
        modelToastTimer = null;
      }
    }

    function removeModelDownloadToast({ immediate = false } = {}){
      clearModelToastTimer();
      if(!activeModelToast) return;
      const card = activeModelToast.card;
      activeModelToast = null;
      if(!card) return;
      dismissToastCard(card, { immediate });
    }

    function queueModelToastRemoval(delayMs){
      clearModelToastTimer();
      modelToastTimer = setTimeout(() => {
        modelToastTimer = null;
        removeModelDownloadToast();
      }, delayMs);
    }

    function ensureModelDownloadToast(){
      if(activeModelToast && activeModelToast.card && activeModelToast.card.isConnected){
        return activeModelToast;
      }
      const wrap = ensureToastWrap();
      const card = document.createElement('div');
      card.className = 'toast-card glass';
      const title = document.createElement('div');
      title.className = 'toast-title';
      const body = document.createElement('div');
      body.className = 'toast-body';
      const progress = document.createElement('div');
      progress.className = 'toast-progress';
      const bar = document.createElement('div');
      bar.className = 'bar';
      const fill = document.createElement('div');
      fill.className = 'bar-fill';
      bar.appendChild(fill);
      const meta = document.createElement('div');
      meta.className = 'text-xs opacity-70';
      progress.append(bar, meta);
      card.append(title, body, progress);
      wrap.appendChild(card);
      activeModelToast = { card, title, body, fill, meta, progress };
      return activeModelToast;
    }

    function syncModelDownloadToast(status){
      const missing = Array.isArray(status && status.missing) ? status.missing : [];
      const statusText = String(status && status.status || '').toLowerCase();
      const isDownloading = statusText === 'downloading';
      const isRetrying = statusText === 'retrying';
      const isError = !!(status && status.status === 'error');
      const isDone = !!(status && status.status === 'done' && missing.length === 0);
      if(isDownloading || isRetrying){
        modelDownloadObservedActive = true;
      }
      const shouldRender = modelDownloadIntentActive || modelDownloadObservedActive || isDownloading || isRetrying;
      if(!shouldRender && !isError && !isDone){
        removeModelDownloadToast({ immediate: true });
        return;
      }
      if(isDone && !shouldRender){
        removeModelDownloadToast({ immediate: true });
        return;
      }
      if(!isDownloading && !isRetrying && !isError && !isDone){
        removeModelDownloadToast({ immediate: true });
        return;
      }

      const toast = ensureModelDownloadToast();
      const pct = Math.max(0, Math.min(100, Number(status && status.pct) || 0));
      toast.fill.style.width = `${isDone ? 100 : pct}%`;

      if(isDownloading){
        clearModelToastTimer();
        const label = status.current_model || 'models';
        toast.title.textContent = status.total_bytes ? `downloading ${label}` : 'starting model download...';
        const bits = [];
        if(status.total_bytes){
          bits.push(`${pct}%`);
          bits.push(`${formatBytes(status.downloaded_bytes)} / ${formatBytes(status.total_bytes)}`);
        }
        const rate = formatDownloadRate(status.download_rate_bytes_per_sec);
        if(rate) bits.push(rate);
        toast.body.textContent = bits.join(' · ');
        toast.meta.textContent = '';
        toast.progress.hidden = false;
        return;
      }

      if(isRetrying){
        clearModelToastTimer();
        toast.title.textContent = 'model download paused';
        toast.body.textContent = status.error || `network dropped, retrying in 5s... (${status.retry_label || '1'})`;
        toast.meta.textContent = status.current_model ? `last file: ${status.current_model}` : '';
        toast.progress.hidden = false;
        return;
      }

      if(isError){
        toast.title.textContent = 'model download failed';
        toast.body.textContent = status.error || 'could not download models';
        toast.meta.textContent = '';
        toast.progress.hidden = false;
        modelDownloadIntentActive = false;
        modelDownloadObservedActive = false;
        queueModelToastRemoval(8000);
        return;
      }

      toast.title.textContent = 'models downloaded';
      toast.body.textContent = 'all required models are ready';
      toast.meta.textContent = '';
      toast.progress.hidden = false;
      modelDownloadIntentActive = false;
      modelDownloadObservedActive = false;
      queueModelToastRemoval(2600);
    }

    function removeModelReminderToast(){
      if(!activeModelReminderToast) return;
      const card = activeModelReminderToast.card;
      activeModelReminderToast = null;
      if(card) dismissToastCard(card, { immediate: true });
    }

    function ensureModelReminderToast(missing){
      if(activeModelReminderToast && activeModelReminderToast.card && activeModelReminderToast.card.isConnected){
        return activeModelReminderToast;
      }
      const wrap = ensureToastWrap();
      const card = document.createElement('div');
      card.className = 'toast-card glass';
      const title = document.createElement('div');
      title.className = 'toast-title';
      const body = document.createElement('div');
      body.className = 'toast-body';
      const row = document.createElement('div');
      row.className = 'flex gap-2 mt-3';
      const downloadBtn = document.createElement('button');
      downloadBtn.type = 'button';
      downloadBtn.className = 'btn bg-white text-black px-3 py-2 text-xs';
      downloadBtn.textContent = 'download';
      const folderBtn = document.createElement('button');
      folderBtn.type = 'button';
      folderBtn.className = 'btn bg-white/10 text-white px-3 py-2 text-xs';
      folderBtn.textContent = 'open folder';
      row.append(downloadBtn, folderBtn);
      card.append(title, body, row);
      wrap.appendChild(card);
      activeModelReminderToast = { card, title, body, downloadBtn, folderBtn };
      downloadBtn.addEventListener('click', async () => {
        const currentMissing = allMissingModels();
        const selection = currentMissing.length ? currentMissing : missing;
        await beginModelDownload(selection);
      });
      folderBtn.addEventListener('click', () => openModelsFolderAction());
      return activeModelReminderToast;
    }

    function syncModelReminderToast(status){
      const missing = allMissingModels(status);
      const isDownloading = isModelDownloadBusyStatus(status && status.status);
      if(!missing.length || isDownloading || modelDownloadIntentActive){
        removeModelReminderToast();
        return;
      }
      const toast = ensureModelReminderToast(missing);
      toast.title.textContent = `${missing.length} model${missing.length === 1 ? '' : 's'} missing`;
      const totalBytes = expectedMissingModelBytes(status);
      toast.body.textContent = totalBytes > 0
        ? `download all missing models (${formatBytes(totalBytes)}) before splitting songs.`
        : 'download all missing models before splitting songs.';
    }

    async function beginModelDownload(missing){
      const list = Array.isArray(missing) && missing.length ? missing : (Array.isArray(lastModelStatus && lastModelStatus.missing) ? lastModelStatus.missing : []);
      modelDownloadIntentActive = true;
      stopModelDownloadAttention(modelsDownloadBtn);
      removeModelReminderToast();
      await dismissModelPrompt();
      syncModelDownloadToast({
        status: 'downloading',
        pct: 0,
        current_model: '',
        downloaded_bytes: 0,
        total_bytes: 0,
        eta_seconds: null,
        eta_state: null,
        missing: list,
      });
      await startModelDownload(list.length ? list : null);
    }

    function flashModelDownloadCTA(){
      if(!modelsDownloadBtn || modelsDownloadBtn.hidden) return;
      openSettings();
      setNerdStuffExpanded(true);
      modelsDownloadBtn.classList.add('attention-ring');
      setTimeout(() => {
        if(modelsDownloadBtn) modelsDownloadBtn.classList.remove('attention-ring');
      }, 2200);
    }

    function stopModelStatusPolling(){
      if(modelStatusPoll){
        clearInterval(modelStatusPoll);
        modelStatusPoll = null;
      }
    }

    function clearModelPreviewTimer(){
      if(modelPreviewTimer){
        clearTimeout(modelPreviewTimer);
        modelPreviewTimer = null;
      }
    }

    function restoreModelPreview({ immediate = false } = {}){
      clearModelPreviewTimer();
      if(!modelPreviewActive && !immediate) return;
      modelPreviewActive = false;
      const doRestore = () => refreshModelStatus({ allowPrompt: false });
      if(immediate){
        doRestore();
        return;
      }
      modelPreviewTimer = setTimeout(() => {
        modelPreviewTimer = null;
        doRestore();
      }, 300);
    }

    function beginModelPreview(){
      clearModelPreviewTimer();
      modelPreviewActive = true;
      updateModelSettings({
        status: 'idle',
        pct: 0,
        current_model: '',
        downloaded_bytes: 0,
        total_bytes: 0,
        eta_seconds: null,
        eta_state: null,
        error: '',
        missing: MODEL_CHECKLIST_ORDER,
      });
      showModelMissingOverlay({
        missing: MODEL_CHECKLIST_ORDER,
      }, { preview: true });
    }

    function simulateModelPreviewDownload(){
      clearModelPreviewTimer();
      modelPreviewActive = true;
      const totalBytes = 2.1 * 1024 * 1024 * 1024;
      let pct = 0;
      const tick = () => {
        pct = Math.min(100, pct + 8);
        updateModelSettings({
          status: pct >= 100 ? 'done' : 'downloading',
          pct,
          current_model: pct < 34 ? 'vocals' : (pct < 68 ? 'instrumental' : 'full mix'),
          downloaded_bytes: Math.round(totalBytes * (pct / 100)),
          total_bytes: Math.round(totalBytes),
          eta_seconds: pct >= 100 ? 0 : Math.max(0, Math.round((100 - pct) * 0.55)),
          error: '',
          missing: pct >= 100 ? [] : MODEL_CHECKLIST_ORDER,
        });
        if(pct >= 100){
          modelPreviewTimer = setTimeout(() => {
            modelPreviewTimer = null;
            restoreModelPreview({ immediate: true });
          }, 900);
          return;
        }
        modelPreviewTimer = setTimeout(tick, 170);
      };
      tick();
    }

    function showModelMissingOverlay(status, { preview = false } = {}){
      closeOverlay(activeModelOverlay);
      activeModelOverlayUi = null;
      const missing = Array.isArray(status && status.missing) ? status.missing : MODEL_CHECKLIST_ORDER;
      const { overlay, card } = createOverlayCard('models not downloaded');
      activeModelOverlay = overlay;

      const body = document.createElement('div');
      body.className = 'text-sm opacity-90 leading-relaxed';
      const label = document.createElement('span');
      label.textContent = 'models: ';
      body.appendChild(label);
      missing.forEach((key, index) => {
        const anchor = document.createElement('a');
        anchor.href = MODEL_SOURCE_URLS[key] || '#';
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.className = 'underline';
        anchor.textContent = key === 'deux' ? 'deux' : (MODEL_LABELS[key] || key);
        body.appendChild(anchor);
        if(index < missing.length - 1){
          body.appendChild(document.createTextNode(', '));
        }
      });
      card.appendChild(body);

      const totalMissingBytes = expectedMissingModelBytes(status);
      if(totalMissingBytes > 0){
        const sizeNote = document.createElement('div');
        sizeNote.className = 'text-xs opacity-70';
        sizeNote.textContent = `${missing.length} model${missing.length === 1 ? '' : 's'} · ${formatBytes(totalMissingBytes)} total`;
        card.appendChild(sizeNote);
      }

      const row = document.createElement('div');
      row.className = 'flex gap-3 justify-end flex-wrap';
      const folderBtn = makeActionButton('open models folder', 'bg-white/10 text-white');
      const downloadBtn = makeActionButton('download automatically', 'bg-white text-black attention-sequence');
      downloadBtn.addEventListener('pointerdown', () => {
        stopModelDownloadAttention(downloadBtn);
      }, { passive: true });

      folderBtn.onclick = async () => {
        if(!preview){
          await dismissModelPrompt();
        }
        await openModelsFolderAction();
        closeOverlay(activeModelOverlay);
        activeModelOverlay = null;
        activeModelOverlayUi = null;
      };

      downloadBtn.onclick = async () => {
        stopModelDownloadAttention(downloadBtn);
        closeOverlay(overlay);
        if(preview){
          simulateModelPreviewDownload();
          return;
        }
        await beginModelDownload(missing);
      };

      row.append(folderBtn, downloadBtn);
      card.appendChild(row);

      const progressWrap = document.createElement('div');
      progressWrap.className = 'model-overlay-progress space-y-2';
      const progressNote = document.createElement('div');
      progressNote.className = 'model-overlay-note';
      const progressBar = document.createElement('div');
      progressBar.className = 'bar';
      const progressFill = document.createElement('div');
      progressFill.className = 'bar-fill';
      progressBar.appendChild(progressFill);
      const progressMeta = document.createElement('div');
      progressMeta.className = 'text-xs opacity-70';
      progressWrap.append(progressNote, progressBar, progressMeta);
      card.appendChild(progressWrap);

      activeModelOverlayUi = {
        overlay,
        folderBtn,
        downloadBtn,
        progressWrap,
        progressFill,
        progressMeta,
        progressNote,
      };
      updateModelSettings(status || { missing });
    }

    function startModelStatusPolling(){
      if(modelStatusPoll) return;
      modelStatusPoll = setInterval(() => {
        refreshModelStatus({ allowPrompt: false });
      }, 1200);
    }

    function updateModelSettings(status){
      lastModelStatus = status;
      const missing = Array.isArray(status && status.missing) ? status.missing : [];
      const relevantMissing = relevantMissingModels(status);
      const statusText = String(status && status.status || '').toLowerCase();
      const isDownloading = isModelDownloadBusyStatus(statusText);
      const isRetrying = statusText === 'retrying';
      const isError = !!(status && status.status === 'error');
      const downloadedTotalBytes = Number(status && status.downloaded_total_bytes) || 0;
      modelsBlocked = relevantMissing.length > 0 || isDownloading;
      renderModelChecklist(status);
      syncModelDownloadToast(status);
      syncModelReminderToast(status);
      if(modelsNote){
        let noteMessage = '';
        if(isError){
          noteMessage = status.error || 'model download failed';
        }else if(isRetrying){
          noteMessage = status.error || `network dropped, retrying in 5s... (${status.retry_label || '1'})`;
        }else if(isDownloading){
          noteMessage = `downloading ${status.current_model || 'models'}`;
        }else if(missing.length && allMissingModels(status).length === 0){
          noteMessage = 'automatic model setup is blocked pending immutable hashes and license review';
        }else if(missing.length && downloadedTotalBytes <= 0){
          noteMessage = `${missing.length} model${missing.length === 1 ? '' : 's'} missing`;
        }else if(missing.length){
          noteMessage = relevantMissing.length
            ? `${relevantMissing.length} required model${relevantMissing.length === 1 ? '' : 's'} missing`
            : `${missing.length} optional model${missing.length === 1 ? '' : 's'} missing`;
        }
        modelsNote.textContent = noteMessage;
        modelsNote.hidden = !noteMessage;
      }
      if(modelsProgressWrap){
        modelsProgressWrap.hidden = !isDownloading && !isError;
      }
      if(modelsProgressBar){
        modelsProgressBar.style.width = `${Math.max(0, status && status.pct ? status.pct : 0)}%`;
      }
      if(modelsProgressMeta){
        if(isDownloading){
          if(isRetrying){
            modelsProgressMeta.textContent = status.error || `network dropped, retrying in 5s... (${status.retry_label || '1'})`;
          }else{
            const bits = [
              `${Math.max(0, status.pct || 0)}%`,
              `${formatBytes(status.downloaded_bytes)} / ${formatBytes(status.total_bytes)}`,
            ];
            const rate = formatDownloadRate(status.download_rate_bytes_per_sec);
            if(rate) bits.push(rate);
            modelsProgressMeta.textContent = bits.join(' · ');
          }
        }else if(isError){
          modelsProgressMeta.textContent = status.error || 'download failed';
        }else{
          modelsProgressMeta.textContent = '';
        }
      }
      if(modelsDownloadBtn){
        const downloadableMissing = allMissingModels(status);
        const showDownloadButton = downloadableMissing.length > 0 && !isDownloading;
        modelsDownloadBtn.hidden = !showDownloadButton;
        modelsDownloadBtn.disabled = !showDownloadButton;
        modelsDownloadBtn.classList.toggle('attention-sequence', showDownloadButton);
        const totalMissingBytes = expectedMissingModelBytes(status);
        modelsDownloadBtn.textContent = isError
          ? 'retry download'
          : (totalMissingBytes > 0 ? `download models (${formatBytes(totalMissingBytes)})` : 'download models');
      }
      if(modelsFolderBtn){
        modelsFolderBtn.hidden = isLanClient;
      }
      if(modelsCtaWrap){
        const allChildrenHidden = Array.from(modelsCtaWrap.children).every((child) => child.hidden);
        modelsCtaWrap.hidden = allChildrenHidden;
      }
      if((isDownloading || missing.length > 0) && nerdStuffWrap){
        setNerdStuffExpanded(true);
      }else if(nerdStuffExpanded && nerdStuffWrap){
        nerdStuffWrap.style.maxHeight = `${nerdStuffWrap.scrollHeight}px`;
      }
      updateUI();
    }

    async function openModelsFolderAction(){
      try{
        await fetch('/api/open_models_folder', { method: 'POST' });
      }catch(err){
        console.warn('open models folder failed', err);
      }
    }

    async function dismissModelPrompt(){
      try{
        const res = await fetch('/api/model_download_prompt/dismiss', { method: 'POST' });
        if(res.ok){
          const data = await res.json();
          updateModelSettings(data);
        }
      }catch(err){
        console.warn('dismiss model prompt failed', err);
      }
    }

    async function startModelDownload(models = null){
      try{
        const res = await fetch('/api/model_downloads/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(models ? { models } : {}),
        });
        if(!res.ok){
          modelDownloadIntentActive = false;
          modelDownloadObservedActive = false;
          removeModelDownloadToast({ immediate: true });
          syncModelReminderToast(lastModelStatus || { missing: models || [] });
          showPopup('could not start model download');
          return;
        }
        const data = await res.json();
        updateModelSettings(data);
        startModelStatusPolling();
      }catch(err){
        modelDownloadIntentActive = false;
        modelDownloadObservedActive = false;
        removeModelDownloadToast({ immediate: true });
        syncModelReminderToast(lastModelStatus || { missing: models || [] });
        console.warn('start model download failed', err);
        showPopup('could not start model download');
      }
    }

    async function refreshModelStatus({ allowPrompt = true } = {}){
      try{
        const res = await fetch('/api/model_download_status');
        if(!res.ok) return null;
        const data = await res.json();
        updateModelSettings(data);
        if(isModelDownloadBusyStatus(data.status)){
          startModelStatusPolling();
        }else if(data.status === 'error'){
          stopModelStatusPolling();
        }else{
          stopModelStatusPolling();
          if(data.status === 'done' && (!data.missing || data.missing.length === 0)){
            if(modelsProgressWrap) modelsProgressWrap.hidden = true;
          }else if(allowPrompt && data.missing && data.missing.length && data.prompt_state === 'pending'){
            syncModelReminderToast(data);
          }
        }
        return data;
      }catch(err){
        console.warn('model status failed', err);
        return null;
      }
    }

    async function acknowledgeRelease(){
      try{
        await fetch('/api/release_status/ack', { method: 'POST' });
      }catch(err){
        console.warn('release ack failed', err);
      }
    }

    function stopReleaseStatusPolling(){
      if(releaseStatusPoll){
        clearInterval(releaseStatusPoll);
        releaseStatusPoll = null;
      }
    }

    function clearReleaseToastTimer(){
      if(releaseToastTimer){
        clearTimeout(releaseToastTimer);
        releaseToastTimer = null;
      }
    }

    function removeReleaseDownloadToast({ immediate = false } = {}){
      clearReleaseToastTimer();
      if(!activeReleaseToast) return;
      const card = activeReleaseToast.card;
      activeReleaseToast = null;
      dismissToastCard(card, { immediate });
    }

    function queueReleaseToastRemoval(delayMs){
      clearReleaseToastTimer();
      releaseToastTimer = setTimeout(() => {
        releaseToastTimer = null;
        removeReleaseDownloadToast();
      }, delayMs);
    }

    function ensureReleaseDownloadToast(){
      if(activeReleaseToast && activeReleaseToast.card && activeReleaseToast.card.isConnected){
        return activeReleaseToast;
      }
      const wrap = ensureToastWrap();
      const card = document.createElement('div');
      card.className = 'toast-card glass';
      const title = document.createElement('div');
      title.className = 'toast-title';
      const body = document.createElement('div');
      body.className = 'toast-body';
      const progress = document.createElement('div');
      progress.className = 'toast-progress';
      const bar = document.createElement('div');
      bar.className = 'bar';
      const fill = document.createElement('div');
      fill.className = 'bar-fill';
      bar.appendChild(fill);
      const meta = document.createElement('div');
      meta.className = 'text-xs opacity-70';
      progress.append(bar, meta);
      card.append(title, body, progress);
      wrap.appendChild(card);
      activeReleaseToast = { card, title, body, fill, meta, progress };
      return activeReleaseToast;
    }

    function syncReleaseDownloadToast(status){
      const statusText = String(status && status.status || '').toLowerCase();
      const isStarting = statusText === 'starting';
      const isDownloading = statusText === 'downloading';
      const isBusy = isStarting || isDownloading;
      const isError = statusText === 'error';
      const isDone = statusText === 'done';
      if(!isBusy && !isError && !isDone){
        removeReleaseDownloadToast({ immediate: true });
        return;
      }

      const toast = ensureReleaseDownloadToast();
      const pct = Math.max(0, Math.min(100, Number(status && status.pct) || 0));
      toast.fill.style.width = `${isDone ? 100 : pct}%`;

      if(isBusy){
        clearReleaseToastTimer();
        const label = status.current_asset || status.filename || 'update';
        const bits = [];
        if(Number(status.total_bytes || 0) > 0){
          bits.push(`${Math.max(0, status.pct || 0)}%`);
          bits.push(`${formatBytes(status.downloaded_bytes)} / ${formatBytes(status.total_bytes)}`);
        }
        const rate = formatDownloadRate(status.download_rate_bytes_per_sec);
        if(rate) bits.push(rate);
        const eta = formatEta(status.eta_seconds);
        toast.title.textContent = isStarting ? 'starting update download...' : `downloading ${label}`;
        toast.body.textContent = bits.join(' · ');
        toast.meta.textContent = eta ? `${eta} remaining` : '';
        toast.progress.hidden = false;
        return;
      }

      if(isError){
        toast.title.textContent = 'update download failed';
        toast.body.textContent = status.error || 'could not download update';
        toast.meta.textContent = '';
        toast.progress.hidden = false;
        stopReleaseStatusPolling();
        queueReleaseToastRemoval(8000);
        return;
      }

      toast.title.textContent = 'update downloaded';
      toast.body.textContent = `${status.filename || 'update'} · ${formatBytes(status.downloaded_bytes)} saved`;
      toast.meta.textContent = 'Downloads';
      toast.progress.hidden = false;
      stopReleaseStatusPolling();
      queueReleaseToastRemoval(3200);
    }

    async function refreshReleaseDownloadStatus(){
      try{
        const res = await fetch('/api/release_download_status');
        if(!res.ok){
          return null;
        }
        const data = await res.json();
        syncReleaseDownloadToast(data);
        return data;
      }catch(err){
        console.warn('release download status failed', err);
        return null;
      }
    }

    function startReleaseStatusPolling(){
      if(releaseStatusPoll) return;
      releaseStatusPoll = setInterval(() => {
        refreshReleaseDownloadStatus();
      }, 800);
    }

    async function downloadReleaseUpdate(){
      stopReleaseStatusPolling();
      closeOverlay(activeReleaseOverlay);
      activeReleaseOverlay = null;
      activeReleaseOverlayUi = null;
      activeReleaseVersion = '';
      try{
        const res = await fetch('/api/release_download', { method: 'POST' });
        if(!res.ok){
          throw new Error(await responseErrorMessage(res, 'could not download update'));
        }
        const data = await res.json();
        syncReleaseDownloadToast(data);
        const statusText = String(data && data.status || '').toLowerCase();
        if(statusText === 'starting' || statusText === 'downloading'){
          startReleaseStatusPolling();
        }
      }catch(err){
        removeReleaseDownloadToast({ immediate: true });
        showPopup((err && err.message) || 'could not download update');
      }
    }

    function clearLegacyReleaseSnooze(){
      try{
        localStorage.removeItem(RELEASE_SNOOZE_STORAGE_KEY);
      }catch(err){
        console.warn('release snooze clear failed', err);
      }
    }

    function readReleaseSessionDismiss(){
      try{
        clearLegacyReleaseSnooze();
        return String(sessionStorage.getItem(RELEASE_SESSION_DISMISS_KEY) || '');
      }catch(err){
        console.warn('release dismiss read failed', err);
        return '';
      }
    }

    function dismissReleaseUntilNextOpen(version){
      try{
        clearLegacyReleaseSnooze();
        if(!version){
          sessionStorage.removeItem(RELEASE_SESSION_DISMISS_KEY);
          return;
        }
        sessionStorage.setItem(RELEASE_SESSION_DISMISS_KEY, version);
      }catch(err){
        console.warn('release dismiss write failed', err);
      }
    }

    function showReleaseOverlay(status){
      stopReleaseStatusPolling();
      closeOverlay(activeReleaseOverlay);
      activeReleaseOverlayUi = null;
      activeReleaseVersion = '';
      const { overlay, card } = createOverlayCard('new release available');
      activeReleaseOverlay = overlay;
      activeReleaseVersion = status.latest_version || '';
      const headline = status.latest_name || status.latest_version || 'new release';
      const info = document.createElement('div');
      info.className = 'text-sm opacity-90';
      info.textContent = `${headline} is ready.`;
      card.appendChild(info);
      const bullets = parseReleaseHighlights(status.notes);
      if(bullets.length){
        const list = document.createElement('ul');
        list.className = 'list-disc pl-5 text-sm opacity-90 space-y-1';
        bullets.forEach((item) => {
          const li = document.createElement('li');
          li.textContent = item;
          list.appendChild(li);
        });
        card.appendChild(list);
      }
      const row = document.createElement('div');
      row.className = 'flex gap-3 justify-end flex-wrap';
      const laterBtn = makeActionButton('later', 'bg-white/10 text-white');
      const downloadBtn = makeActionButton('download update', 'bg-white text-black');
      laterBtn.onclick = async () => {
        dismissReleaseUntilNextOpen(status.latest_version || '');
        stopReleaseStatusPolling();
        closeOverlay(activeReleaseOverlay);
        activeReleaseOverlay = null;
        activeReleaseOverlayUi = null;
        activeReleaseVersion = '';
      };
      downloadBtn.onclick = async () => {
        await downloadReleaseUpdate();
      };
      row.append(laterBtn, downloadBtn);
      card.appendChild(row);
      activeReleaseOverlayUi = { overlay, laterBtn, downloadBtn };
    }

    async function checkForReleaseUpdate(){
      try{
        const res = await fetch('/api/release_status?refresh=1');
        if(!res.ok) return;
        const data = await res.json();
        const latest = data.latest_version || '';
        if(!data.update_available || !latest) return;
        const dismissedVersion = readReleaseSessionDismiss();
        if(dismissedVersion === latest) return;
        if(dismissedVersion && dismissedVersion !== latest){
          dismissReleaseUntilNextOpen('');
        }
        if(latest === (data.skipped_version || '')) return;
        showReleaseOverlay(data);
      }catch(err){
        console.warn('release check failed', err);
      }
    }

    /* === upload & progress logic === */
    // queue of files added by user but not yet uploaded
    const pendingItems = [];
    const activeStreams = new Map();
    let isClearing = false;
    function closeTaskStream(taskId){
      if(!taskId) return;
      const stream = activeStreams.get(taskId);
      if(stream){
        try{ stream.close(); }catch(_){ }
        activeStreams.delete(taskId);
      }
    }

    function markTaskStopped(taskRef, ui){
      if(!taskRef || !ui) return;
      const {st, dl, stopPad, smooth, bar} = ui;
      taskRef.stopRequested = false;
      taskRef.stage = 'stopped';
      taskRef.pct = 0;
      saveTasks();
      if(smooth){ smooth.setImmediate(0); }
      if(st){ showStatus(st); st.textContent = 'stopped'; st.classList.remove('hidden'); }
      const parent = st && st.closest ? st.closest('.card') : null;
      if(parent){ parent.classList.add('done'); }
      if(dl){ dl.classList.remove('show'); }
      if(stopPad){
        stopPad.style.pointerEvents = '';
        stopPad.style.opacity = '';
        stopPad.title = 'rerun';
        setRetryIcon(stopPad);
        guardClick(stopPad, (ev) => { ev.preventDefault(); rerunTask(taskRef, { bar, st, dl, stopPad, smooth }); });
      }
      if(ui.item){
        applyRowState(ui.item, taskRef);
      }
      updateUI();
    }

    function markTaskStopping(taskRef, ui){
      if(!taskRef || !ui) return;
      const {st, dl, stopPad} = ui;
      taskRef.stopRequested = true;
      taskRef.stage = 'stopping';
      saveTasks();
      if(st){
        showStatus(st);
        st.textContent = 'stopping';
        st.classList.remove('hidden');
      }
      if(dl){
        dl.classList.remove('show');
      }
      if(stopPad){
        setStopSquareIcon(stopPad);
        stopPad.style.pointerEvents = 'none';
        stopPad.style.opacity = '0.65';
        stopPad.title = 'stopping';
      }
      if(ui.item){
        applyRowState(ui.item, taskRef);
      }
      updateUI();
    }

    async function requestStop(taskId, taskRef, ui){
      if(!taskId || !taskRef || !ui) return;
      markTaskStopping(taskRef, ui);
      try{
        const res = await fetch('/stop/' + taskId, { method:'POST' });
        if(res.ok){
          const data = await res.json().catch(() => null);
          setQueuePausedUi(true);
          showPopup('current task stopped; queue paused');
          if(data && data.status === 'stopped'){
            markTaskStopped(taskRef, ui);
          }
        }else{
          taskRef.stopRequested = false;
          saveTasks();
          if(ui.stopPad){
            setStopSquareIcon(ui.stopPad);
            ui.stopPad.style.pointerEvents = '';
            ui.stopPad.style.opacity = '';
            ui.stopPad.title = 'stop current task and pause queue';
          }
          showPopup(await responseErrorMessage(res, 'could not stop song'));
        }
      }catch(_){
        taskRef.stopRequested = false;
        saveTasks();
        if(ui.stopPad){
          setStopSquareIcon(ui.stopPad);
          ui.stopPad.style.pointerEvents = '';
          ui.stopPad.style.opacity = '';
          ui.stopPad.title = 'stop current task and pause queue';
        }
        showPopup('could not stop song');
      }
    }
    function normalizeStemList(stems){
      return Array.isArray(stems) ? stems.map((item) => String(item || '').toLowerCase()).filter(Boolean) : [];
    }
    function displayStemsForTask(task){
      if(task && Array.isArray(task.groupStems) && task.groupStems.length){
        return task.groupStems;
      }
      return Array.isArray(task && task.stems) ? task.stems : [];
    }
    function stemsMatch(a, b){
      const left = normalizeStemList(a);
      const right = normalizeStemList(b);
      if(left.length !== right.length) return false;
      return left.every((item, index) => item === right[index]);
    }
    function findRowForTask(task){
      if(!task) return null;
      const rowKey = taskRowIdentity(task);
      return queueLeafRows().find((row) => {
        if(rowKey && row.__taskRowKey === rowKey) return true;
        if(task.id && row.__taskId === task.id) return true;
        if(task.tempKey && row.__tempKey === task.tempKey && stemsIdentityKey(task.stems || []) === stemsIdentityKey(row.__stems || [])) return true;
        return false;
      }) || null;
    }
    function taskUiForRow(row){
      if(!row) return null;
      const item = row;
      const bar = row.querySelector('.bar-fill');
      const st = row.querySelector('.chip.status');
      const dl = row.querySelector('.dl-pad');
      const card = row.querySelector('.card');
      const stopPad = row.querySelector('.stop-pad');
      const smooth = makeProgressSmoother(bar);
      return { item, bar, st, dl, card, stopPad, smooth };
    }
    function ensureTaskProgressTracking(task){
      if(!task || !task.id) return;
      const stage = String(task.stage || '').toLowerCase();
      if(['ready', 'done', 'error', 'stopped'].includes(stage)) return;
      if(activeStreams.has(task.id)) return;
      const row = findRowForTask(task);
      const ui = taskUiForRow(row);
      if(!ui) return;
      trackProgress(task.id, ui.bar, ui.st, ui.dl, null, null, task, null, ui.smooth, ui.stopPad);
    }
    async function restartTaskWithSelection(task, stems){
      if(!task || !task.id) return false;
      const row = findRowForTask(task);
      const ui = taskUiForRow(row);
      if(!ui) return false;
      await rerunTask(task, ui, {
        stems: stems.join(','),
        ...currentStartSettings(),
        prioritize: true,
      });
      return true;
    }
    function dropPendingById(id){
      if(!id) return;
      const idx = pendingItems.findIndex(p => p && p.id === id);
      if(idx >= 0) pendingItems.splice(idx,1);
    }
    function dropPendingByTempKey(tempKey){
      if(!tempKey) return;
      for(let index = pendingItems.length - 1; index >= 0; index -= 1){
        const pending = pendingItems[index];
        if(pending && pending.tempKey === tempKey){
          pendingItems.splice(index, 1);
        }
      }
    }
    function getTaskById(id){
      return tasks.find(t => t && t.id === id);
    }

    function getTaskByTempKey(tempKey){
      if(!tempKey) return null;
      return tasks.find(t => t && t.tempKey === tempKey && !isFinished(t) && String(t.stage || '').toLowerCase() !== 'error')
        || tasks.find(t => t && t.tempKey === tempKey)
        || null;
    }

    function getTaskByRow(row){
      if(!row) return null;
      const rowKey = row.__taskRowKey || '';
      if(rowKey){
        const exact = tasks.find((task) => task && taskRowIdentity(task) === rowKey);
        if(exact) return exact;
      }
      const rowTaskId = row.__taskId || null;
      if(rowTaskId) return getTaskById(rowTaskId);
      return getTaskByTempKey(row.__tempKey || null);
    }

    function groupHasRemainingTasks(task, excludeTaskId = null){
      if(!task || !task.tempKey) return false;
      return tasks.some((candidate) => {
        if(!candidate || candidate.tempKey !== task.tempKey) return false;
        if(excludeTaskId && candidate.id === excludeTaskId) return false;
        const stage = String(candidate.stage || '').toLowerCase();
        return !(candidate.pct >= 100 || stage === 'stopped' || stage === 'error');
      });
    }

    function removeQueueRowAnimated(row, onDone){
      if(!row || !row.parentElement){
        if(typeof onDone === 'function') onDone();
        return;
      }
      const computed = window.getComputedStyle(row);
      const startHeight = row.offsetHeight;
      const startMarginTop = parseFloat(computed.marginTop || '0') || 0;
      const startMarginBottom = parseFloat(computed.marginBottom || '0') || 0;

      row.style.height = `${startHeight}px`;
      row.style.marginTop = `${startMarginTop}px`;
      row.style.marginBottom = `${startMarginBottom}px`;
      row.style.overflow = 'hidden';
      row.style.willChange = 'height, margin, opacity, transform';
      row.classList.add('leave', 'leave-active');

      requestAnimationFrame(() => {
        row.style.transition = 'height .24s cubic-bezier(.22,.61,.36,1), margin .24s cubic-bezier(.22,.61,.36,1), opacity .18s ease, transform .18s ease';
        row.classList.add('leave-to');
        row.style.height = '0px';
        row.style.marginTop = '0px';
        row.style.marginBottom = '0px';
      });

      let settled = false;
      const finalize = () => {
        if(settled) return;
        settled = true;
        if(row.parentElement){
          row.remove();
        }
        if(typeof onDone === 'function') onDone();
      };
      setTimeout(finalize, 270);
      row.addEventListener('transitionend', (event) => {
        if(event.propertyName === 'height'){
          finalize();
        }
      }, { once: true });
    }

    function failPendingUpload(pending, message){
      if(!pending) return;
      const pendingRowKey = `temp:${pending.tempKey || ''}:${stemsIdentityKey(pending.stems || [])}`;
      for(let index = pendingItems.length - 1; index >= 0; index -= 1){
        const candidate = pendingItems[index];
        if(!candidate) continue;
        if(`temp:${candidate.tempKey || ''}:${stemsIdentityKey(candidate.stems || [])}` === pendingRowKey){
          pendingItems.splice(index, 1);
        }
      }
      const taskIndex = tasks.findIndex((task) => task && taskRowIdentity(task) === pendingRowKey && !task.id);
      if(taskIndex >= 0){
        tasks.splice(taskIndex, 1);
      }
      if(pending.ui && pending.ui.smooth && typeof pending.ui.smooth.stop === 'function'){
        pending.ui.smooth.stop();
      }
      const row = pending.ui && pending.ui.item;
      if(row && row.parentElement){
        removeQueueRowAnimated(row, () => {
          saveTasks();
          updateUI();
        });
      }else{
        saveTasks();
        updateUI();
      }
      if(message && !isClearing){
        showPopup(message);
      }
    }

    function revealStatusesAfterStart(){
      queueStarted = true;
      document.querySelectorAll('#queue .chip').forEach(chip => chip.classList.remove('hidden'));
      refreshAllLabelVisibility();
      document.querySelectorAll('#queue .stop-pad').forEach(pad => {
        const row = pad.closest('.item-row');
        const task = getTaskByRow(row);
        const stage = task ? task.stage : null;
        setStopVisibility(pad, stage);
      });
    }

    async function updateReadyTaskSelection(taskId, stems){
      if(!taskId || !Array.isArray(stems) || stems.length === 0) return null;
      try{
        const res = await fetch(`/api/tasks/${taskId}/selection`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stems: stems.join(',') }),
        });
        if(!res.ok) return null;
        return await res.json();
      }catch(_){
        return null;
      }
    }

    async function syncQueuedStemSelection(){
      const groups = selectedStemGroups();
      if(groups.length !== 1) return;
      const stems = groups[0].slice();
      pendingItems.forEach((pending) => {
        if(!pending || pending.frozen) return;
        const linkedTask = pending.id
          ? getTaskById(pending.id)
          : (tasks.find((task) => task && taskRowIdentity(task) === `temp:${pending.tempKey || ''}:${stemsIdentityKey(pending.stems || [])}`) || getTaskByTempKey(pending.tempKey));
        if(linkedTask && (linkedTask.stage !== 'ready' || Number(linkedTask.pct || 0) !== 0 || linkedTask.frozen)) return;
        pending.stems = stems.slice();
        if(pending.ui && pending.ui.item){
          applyLabels(pending.ui.item.querySelector('.labels'), pending.stems);
        }
      });
      const updates = [];
      tasks.forEach((task) => {
        if(!task || task.frozen || task.stage !== 'ready' || task.pct !== 0) return;
        task.stems = stems.slice();
        const row = findRowForTask(task);
        if(row){
          const labels = row.querySelector('.labels');
          applyLabels(labels, task.stems);
        }
        if(task.id){
          updates.push(updateReadyTaskSelection(task.id, stems).then((data) => {
            if(data && Array.isArray(data.stems)){
              task.stems = data.stems;
              if(row){
                const labels = row.querySelector('.labels');
                applyLabels(labels, task.stems);
              }
            }
          }));
        }
      });
      saveTasks();
      if(updates.length){
        await Promise.all(updates);
        saveTasks();
      }
    }

    async function updateTaskSelectionWithResult(taskId, stems){
      if(!taskId || !Array.isArray(stems) || stems.length === 0){
        return { ok: false, message: 'invalid stem selection', data: null };
      }
      try{
        const res = await fetch(`/api/tasks/${taskId}/selection`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stems: stems.join(',') }),
        });
        if(!res.ok){
          return { ok: false, message: await responseErrorMessage(res, 'could not change stem'), data: null };
        }
        const data = await res.json();
        return { ok: true, message: '', data };
      }catch(_){
        return { ok: false, message: 'could not change stem', data: null };
      }
    }

    function queueRowTaskCandidates(row){
      if(!row) return [];
      const exact = getTaskByRow(row);
      return exact ? [exact] : [];
    }

    function queueModeKeyForTask(task){
      if(!task) return null;
      const rawMode = String(task.mode || '').replace(/^preset_/, '');
      if(QUEUE_STEM_OPTION_BY_MODE.has(rawMode)) return rawMode;
      const stems = normalizeStemList(task.stems);
      if(stemsMatch(stems, ['vocals', 'instrumental'])) return 'both_separate';
      for(const option of QUEUE_STEM_OPTION_BY_MODE.values()){
        if(stemsMatch(stems, option.stems)) return option.modeKey;
      }
      return null;
    }

    function seedQueueRowStemSelection(row){
      const candidates = queueRowTaskCandidates(row);
      const seed = new Set();
      candidates.forEach((task) => {
        const modeKey = queueModeKeyForTask(task);
        if(modeKey === 'both_separate'){
          seed.add('vocals');
          seed.add('instrumental');
          return;
        }
        if(modeKey && QUEUE_STEM_OPTION_BY_MODE.has(modeKey)){
          seed.add(modeKey);
        }
      });
      if(!seed.size){
        seed.add('vocals');
      }
      row.__queueStemSelection = Array.from(seed);
      row.__queueStemAnchor = row.__queueStemSelection[row.__queueStemSelection.length - 1] || 'vocals';
    }

    function applyQueueStemModifierSelection(row, modeKey, triggerEvent){
      const selection = new Set(Array.isArray(row.__queueStemSelection) ? row.__queueStemSelection : ['vocals']);
      let anchor = typeof row.__queueStemAnchor === 'string' ? row.__queueStemAnchor : (selection.values().next().value || 'vocals');
      if(triggerEvent && triggerEvent.shiftKey){
        const targetIndex = QUEUE_STEM_MODE_ORDER.indexOf(modeKey);
        const anchorIndex = QUEUE_STEM_MODE_ORDER.indexOf(anchor);
        if(targetIndex >= 0 && anchorIndex >= 0){
          const start = Math.min(anchorIndex, targetIndex);
          const end = Math.max(anchorIndex, targetIndex);
          selection.clear();
          QUEUE_STEM_MODE_ORDER.slice(start, end + 1).forEach((key) => selection.add(key));
        }else{
          selection.clear();
          selection.add(modeKey);
        }
      }else if(triggerEvent && (triggerEvent.metaKey || triggerEvent.ctrlKey)){
        if(selection.has(modeKey)){
          if(selection.size > 1){
            selection.delete(modeKey);
          }
        }else{
          selection.add(modeKey);
        }
      }else{
        selection.clear();
        selection.add(modeKey);
      }
      anchor = modeKey;
      row.__queueStemSelection = Array.from(selection);
      row.__queueStemAnchor = anchor;
      return selection;
    }

    function queueStemsFromSelection(modeSelection){
      const selected = Array.from(modeSelection || []);
      if(selected.length === 1){
        const option = QUEUE_STEM_OPTION_BY_MODE.get(selected[0]);
        return option ? option.stems.slice() : null;
      }
      if(selected.length === 2 && selected.includes('vocals') && selected.includes('instrumental')){
        return ['vocals', 'instrumental'];
      }
      return null;
    }

    async function removeQueueRowFromQueue(row){
      const candidates = queueRowTaskCandidates(row);
      if(!candidates.length) return false;
      const tempKey = row.__tempKey || null;
      const rowKey = row.__taskRowKey || '';
      const ids = candidates.map((task) => task && task.id).filter(Boolean);
      for(const taskId of ids){
        closeTaskStream(taskId);
      }
      if(tempKey){
        pendingItems.forEach((pending) => {
          if(!pending || pending.tempKey !== tempKey) return;
          if(rowKey && `temp:${pending.tempKey || ''}:${stemsIdentityKey(pending.stems || [])}` !== rowKey) return;
          if(pending.xhr && typeof pending.xhr.abort === 'function'){
            try{ pending.xhr.abort(); }catch(_){}
          }
        });
      }
      for(const taskId of ids){
        try{
          const res = await fetch(`/remove/${taskId}`, { method: 'POST' });
          if(!res.ok){
            showPopup(await responseErrorMessage(res, 'could not remove from queue'));
            return false;
          }
        }catch(_){
          showPopup('could not remove from queue');
          return false;
        }
      }
      if(tempKey){
        if(rowKey){
          for(let index = pendingItems.length - 1; index >= 0; index -= 1){
            const pending = pendingItems[index];
            if(!pending) continue;
            if(`temp:${pending.tempKey || ''}:${stemsIdentityKey(pending.stems || [])}` === rowKey){
              pendingItems.splice(index, 1);
            }
          }
        }else{
          dropPendingByTempKey(tempKey);
        }
      }
      const idSet = new Set(ids);
      tasks = tasks.filter((task) => {
        if(!task) return false;
        if(idSet.has(task.id)) return false;
        if(tempKey && task.tempKey === tempKey) return false;
        return true;
      });
      if(row && row.parentElement){
        removeQueueRowAnimated(row, () => {
          saveTasks();
          updateUI();
        });
      }else{
        saveTasks();
        updateUI();
      }
      showPopup('removed from queue');
      return true;
    }

    async function applyQueueRowStemSelection(row, modeKey, triggerEvent){
      const candidates = queueRowTaskCandidates(row);
      if(candidates.length !== 1){
        showPopup('change stem works when this song has one queued split');
        return false;
      }
      const task = candidates[0];
      if(!task){
        showPopup('song is unavailable');
        return false;
      }
      if(isProcessing(task)){
        showPopup('you can only change stems before processing starts');
        return false;
      }
      const stage = String(task.stage || '').toLowerCase();
      if(['done', 'stopped', 'error'].includes(stage)){
        showPopup('you can only change stems for queued songs');
        return false;
      }
      const selection = applyQueueStemModifierSelection(row, modeKey, triggerEvent);
      const nextStems = queueStemsFromSelection(selection);
      if(!nextStems){
        showPopup('that multi-selection is not supported for one queued song');
        return false;
      }
      if(task.id){
        const result = await updateTaskSelectionWithResult(task.id, nextStems);
        if(!result.ok){
          showPopup(result.message || 'could not change stem');
          return false;
        }
        const data = result.data || null;
        if(data && typeof data.mode === 'string'){
          task.mode = data.mode;
        }
        task.stems = data && Array.isArray(data.stems) ? data.stems.slice() : nextStems.slice();
      }else{
        task.stems = nextStems.slice();
      }
      task.groupStems = task.stems.slice();
      if(task.tempKey){
        pendingItems.forEach((pending) => {
          if(!pending || pending.tempKey !== task.tempKey) return;
          pending.stems = task.stems.slice();
          if(pending.ui && pending.ui.item){
            applyLabels(pending.ui.item.querySelector('.labels'), task.stems);
          }
        });
      }
      applyLabels(row ? row.querySelector('.labels') : null, task.stems);
      if(row){
        applyRowState(row, task);
      }
      saveTasks();
      updateUI();
      showPopup('stem updated');
      return true;
    }

    function closeQueueContextMenu(){
      if(!queueContextMenuEl) return;
      const menu = queueContextMenuEl;
      if(typeof menu.__outsidePointerDown === 'function'){
        document.removeEventListener('pointerdown', menu.__outsidePointerDown, true);
      }
      if(typeof menu.__escapeHandler === 'function'){
        document.removeEventListener('keydown', menu.__escapeHandler, true);
      }
      queueContextMenuEl = null;
      menu.remove();
      const returnFocus = menu.__returnFocus;
      if(returnFocus && returnFocus.isConnected && typeof returnFocus.focus === 'function'){
        try { returnFocus.focus({ preventScroll: true }); } catch(_) { returnFocus.focus(); }
      }
    }

    function measureQueueNestedRequiredHeight(menu){
      if(!menu) return 0;
      const groupMenu = menu.querySelector('.queue-context-groups');
      const changeWrap = menu.querySelector('.queue-context-change-wrap');
      if(!groupMenu || !changeWrap) return 0;
      const nestedTopOffset = changeWrap.offsetTop || 0;
      const restore = [];
      const setMeasureState = (el) => {
        restore.push({
          el,
          display: el.style.display,
          visibility: el.style.visibility,
          pointerEvents: el.style.pointerEvents,
        });
        el.style.display = 'grid';
        el.style.visibility = 'hidden';
        el.style.pointerEvents = 'none';
      };
      setMeasureState(groupMenu);
      let nestedHeight = groupMenu.getBoundingClientRect().height || 0;
      restore.reverse().forEach((entry) => {
        entry.el.style.display = entry.display;
        entry.el.style.visibility = entry.visibility;
        entry.el.style.pointerEvents = entry.pointerEvents;
      });
      return nestedTopOffset + nestedHeight;
    }

    function openQueueContextMenu(event, row){
      if(!row) return;
      event.preventDefault();
      closeQueueContextMenu();
      seedQueueRowStemSelection(row);

      const menu = document.createElement('div');
      menu.className = 'queue-context-menu';
      menu.setAttribute('role', 'menu');
      menu.setAttribute('aria-label', 'queue item actions');
      menu.__returnFocus = document.activeElement;

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'queue-context-action';
      removeBtn.setAttribute('role', 'menuitem');
      removeBtn.textContent = 'remove from queue';
      removeBtn.addEventListener('click', async (clickEvent) => {
        clickEvent.preventDefault();
        clickEvent.stopPropagation();
        const removed = await removeQueueRowFromQueue(row);
        if(removed){
          closeQueueContextMenu();
        }
      });
      menu.appendChild(removeBtn);

      const changeWrap = document.createElement('div');
      changeWrap.className = 'queue-context-change-wrap';
      const changeBtn = document.createElement('button');
      changeBtn.type = 'button';
      changeBtn.className = 'queue-context-action';
      changeBtn.setAttribute('role', 'menuitem');
      const changeLabel = document.createElement('span');
      changeLabel.textContent = 'change stem';
      const changeCaret = document.createElement('span');
      changeCaret.className = 'queue-context-caret';
      changeCaret.textContent = '▸';
      changeBtn.append(changeLabel, changeCaret);
      changeWrap.appendChild(changeBtn);

      const groupMenu = document.createElement('div');
      groupMenu.className = 'queue-context-groups';
      QUEUE_STEM_MENU_GROUPS.forEach((group) => {
        const groupWrap = document.createElement('div');
        groupWrap.className = 'queue-context-group';
        const groupBtn = document.createElement('button');
        groupBtn.type = 'button';
        groupBtn.className = 'queue-context-action';
        groupBtn.setAttribute('role', 'menuitem');
        const groupLabel = document.createElement('span');
        groupLabel.textContent = group.label;
        const groupCaret = document.createElement('span');
        groupCaret.className = 'queue-context-caret';
        groupCaret.textContent = '▸';
        groupBtn.append(groupLabel, groupCaret);
        groupWrap.appendChild(groupBtn);

        const optionMenu = document.createElement('div');
        optionMenu.className = 'queue-context-options';
        group.options.forEach((option) => {
          const optionBtn = document.createElement('button');
          optionBtn.type = 'button';
          optionBtn.className = 'queue-context-action';
          optionBtn.setAttribute('role', 'menuitem');
          optionBtn.textContent = option.label;
          optionBtn.addEventListener('click', async (clickEvent) => {
            clickEvent.preventDefault();
            clickEvent.stopPropagation();
            const changed = await applyQueueRowStemSelection(row, option.modeKey, clickEvent);
            if(changed){
              closeQueueContextMenu();
            }
          });
          optionMenu.appendChild(optionBtn);
        });
        groupWrap.appendChild(optionMenu);
        groupMenu.appendChild(groupWrap);
      });
      changeWrap.appendChild(groupMenu);
      menu.appendChild(changeWrap);

      appShell.appendChild(menu);
      const nestedAllowance = 420;
      const shellRect = appShell.getBoundingClientRect();
      const localClientX = event.clientX - shellRect.left;
      if(localClientX > (appShell.clientWidth - nestedAllowance)){
        menu.classList.add('open-left');
      }
      const rect = menu.getBoundingClientRect();
      const nestedRequiredHeight = measureQueueNestedRequiredHeight(menu);
      const shellViewX = event.clientX - shellRect.left;
      const shellViewY = event.clientY - shellRect.top;
      const baseX = appShell.scrollLeft + shellViewX;
      const baseY = appShell.scrollTop + shellViewY;
      const minLeft = appShell.scrollLeft + 8;
      const maxLeft = appShell.scrollLeft + appShell.clientWidth - rect.width - 8;
      const minTop = appShell.scrollTop + 8;
      const requiredHeight = Math.max(rect.height, nestedRequiredHeight || 0);
      const maxTop = appShell.scrollTop + appShell.clientHeight - requiredHeight - 8;
      const safeMaxLeft = Math.max(minLeft, maxLeft);
      const safeMaxTop = Math.max(minTop, maxTop);
      const left = Math.max(minLeft, Math.min(baseX, safeMaxLeft));
      const top = Math.max(minTop, Math.min(baseY, safeMaxTop));
      menu.style.left = `${left}px`;
      menu.style.top = `${top}px`;

      menu.__outsidePointerDown = (pointerEvent) => {
        if(menu.contains(pointerEvent.target)) return;
        closeQueueContextMenu();
      };
      menu.__escapeHandler = (keyEvent) => {
        if(keyEvent.key === 'Escape'){
          keyEvent.preventDefault();
          closeQueueContextMenu();
          return;
        }
        if(['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(keyEvent.key)){
          const items = Array.from(menu.querySelectorAll('[role="menuitem"]'))
            .filter((item) => item.getClientRects().length > 0 && getComputedStyle(item).visibility !== 'hidden');
          if(!items.length) return;
          keyEvent.preventDefault();
          const current = items.indexOf(document.activeElement);
          let next = 0;
          if(keyEvent.key === 'End') next = items.length - 1;
          else if(keyEvent.key === 'ArrowUp') next = current <= 0 ? items.length - 1 : current - 1;
          else if(keyEvent.key === 'ArrowDown') next = current < 0 || current >= items.length - 1 ? 0 : current + 1;
          items[next].focus();
        }
      };
      document.addEventListener('pointerdown', menu.__outsidePointerDown, true);
      document.addEventListener('keydown', menu.__escapeHandler, true);
      queueContextMenuEl = menu;
      requestAnimationFrame(() => removeBtn.focus({ preventScroll: true }));
    }

    function selectedStemGroups(){
      if(selectedPresetMode === 'all_stems'){
        return [['all_stems']];
      }
      if(selectedPresetMode === 'denoise'){
        return [['preset_denoise']];
      }
      if(selectedPresetMode === 'mel_band_karaoke'){
        return [['mel_band_karaoke']];
      }
      if(selectedPresetMode === 'boost_harmonies'){
        return [['boost_harmonies']];
      }
      const groups = [];
      const pushGroup = (modeKey, stems) => {
        if(selectedStemModes.has(modeKey)){
          groups.push(stems.slice());
        }
      };
      pushGroup('vocals', ['vocals']);
      pushGroup('instrumental', ['instrumental']);
      pushGroup('guitar', ['guitar']);
      pushGroup('mel_band_karaoke', ['mel_band_karaoke']);
      pushGroup('bs_roformer_6s', ['bs_roformer_6s']);
      pushGroup('htdemucs_ft_drums', ['htdemucs_ft_drums']);
      pushGroup('htdemucs_ft_bass', ['htdemucs_ft_bass']);
      pushGroup('htdemucs_ft_other', ['htdemucs_ft_other']);
      pushGroup('htdemucs_6s', ['htdemucs_6s']);
      pushGroup('drumsep_6s', ['drumsep_6s']);
      pushGroup('drumsep_4s', ['drumsep_4s']);
      return groups;
    }

    function selectedStemUnion(){
      const seen = new Set();
      const stems = [];
      selectedStemGroups().forEach((group) => {
        group.forEach((stem) => {
          if(seen.has(stem)) return;
          seen.add(stem);
          stems.push(stem);
        });
      });
      return stems;
    }

    function hasQueuedUploadReadyToStart(){
      if(pendingItems.some((pending) => !!pending)) return true;
      return tasks.some((task) => {
        if(!task || isFinished(task)) return false;
        return String(task.stage || '').toLowerCase() !== 'error';
      });
    }

    function hasOpenBlockingOverlay(){
      if(settingsOverlay && !settingsOverlay.classList.contains('hidden')) return true;
      if(presetSettingsOverlay && !presetSettingsOverlay.classList.contains('hidden')) return true;
      return !!document.getElementById('cfm-ok');
    }

    function shouldHandleStartShortcut(event){
      if(event.key !== 'Enter' || event.defaultPrevented || event.isComposing || event.repeat) return false;
      if(event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return false;
      if(hasOpenBlockingOverlay()) return false;
      const target = event.target;
      if(target instanceof Element && target.closest('input, textarea, select, button, a, [contenteditable=""], [contenteditable="true"], [role="button"]')){
        return false;
      }
      return hasQueuedUploadReadyToStart();
    }

    function selectedStems(){
      const groups = selectedStemGroups();
      return groups[0] ? groups[0].slice() : [];
    }

    function currentStartSettings(){
      return {
        output_format: settingsState.output_format || 'same_as_input',
        multi_stem_export: settingsState.multi_stem_export || 'zip',
        video_handling: settingsState.video_handling || 'audio_only',
        output_root: settingsState.output_root || '',
        output_same_as_input: !!settingsState.output_same_as_input,
      };
    }

    async function responseErrorMessage(res, fallback){
      try{
        const payload = await res.json();
        return payload?.message || payload?.detail?.message || payload?.detail || fallback;
      }catch(_){
        return fallback;
      }
    }

    async function requestTaskStart(taskId, startSettings){
      const res = await fetch('/start/' + taskId, {
        method:'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(startSettings),
      });
      if(!res.ok){
        throw new Error(await responseErrorMessage(res, 'could not start song'));
      }
      return await res.json();
    }

    const droppedSourceDirByFile = new WeakMap();

    function normalizeSourceDirPath(pathText){
      if(typeof pathText !== 'string' || !pathText) return '';
      const normalized = pathText.replace(/\\/g, '/');
      const index = normalized.lastIndexOf('/');
      return index > 0 ? normalized.slice(0, index) : '';
    }

    function normalizePickedPath(pathText){
      if(typeof pathText !== 'string') return '';
      let normalized = pathText.trim();
      if(!normalized) return '';
      if(
        normalized.length >= 2 &&
        ((normalized.startsWith('"') && normalized.endsWith('"')) ||
          (normalized.startsWith("'") && normalized.endsWith("'")))
      ){
        normalized = normalized.slice(1, -1).trim();
      }
      if(normalized.startsWith('file://')){
        try{
          const url = new URL(normalized);
          let resolved = decodeURIComponent(url.pathname || '');
          if(/^\/[A-Za-z]:\//.test(resolved)){
            resolved = resolved.slice(1);
          }
          if(url.hostname && url.hostname !== 'localhost'){
            resolved = `//${url.hostname}${resolved}`;
          }
          if(resolved){
            normalized = resolved;
          }
        }catch(_){
          // keep original text if parsing fails
        }
      }
      return normalized;
    }

    function captureDroppedSourceDirs(event){
      const dt = event && event.dataTransfer;
      if(!dt) return;
      const files = Array.from(dt.files || []);
      if(!files.length) return;
      const uriPayload = String(dt.getData('text/uri-list') || '').trim();
      const plainPayload = String(dt.getData('text/plain') || '').trim();
      const rawLines = (uriPayload || plainPayload)
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith('#'));
      if(!rawLines.length) return;

      const droppedPaths = [];
      rawLines.forEach((line) => {
        if(line.startsWith('file://')){
          try{
            const url = new URL(line);
            let pathText = decodeURIComponent(url.pathname || '');
            if(/^\/[A-Za-z]:\//.test(pathText)){
              pathText = pathText.slice(1);
            }
            if(pathText){
              droppedPaths.push(pathText);
            }
            return;
          }catch(_){
            // fall through to plain path parsing
          }
        }
        if(line.startsWith('/')){
          droppedPaths.push(line);
        }
      });

      if(!droppedPaths.length) return;
      const byName = new Map();
      droppedPaths.forEach((pathText) => {
        const normalized = pathText.replace(/\\/g, '/');
        const name = normalized.slice(normalized.lastIndexOf('/') + 1);
        if(!name) return;
        const bucket = byName.get(name) || [];
        bucket.push(pathText);
        byName.set(name, bucket);
      });

      files.forEach((file, index) => {
        if(!file || (typeof file.path === 'string' && file.path)) return;
        const name = String(file.name || '');
        const namedBucket = byName.get(name);
        const matchedPath = namedBucket && namedBucket.length
          ? namedBucket.shift()
          : (droppedPaths[index] || '');
        const sourceDir = normalizeSourceDirPath(matchedPath);
        if(sourceDir){
          droppedSourceDirByFile.set(file, sourceDir);
        }
      });
    }

    function sourceDirForFile(file){
      if(!file) return '';
      if(typeof file.path === 'string' && file.path){
        return normalizeSourceDirPath(file.path);
      }
      const inferred = droppedSourceDirByFile.get(file);
      return typeof inferred === 'string' ? inferred : '';
    }

    function createPendingQueueRow(name, stems, tempKey, options = {}){
      const frag = template.content.cloneNode(true);
      const item = frag.firstElementChild;
      const li   = item.querySelector('.filename');
      const bar  = item.querySelector('.bar-fill');
      const st   = item.querySelector('.chip.status');
      const dl   = item.querySelector('.dl-pad');
      const card = item.querySelector('.card');
      const stopPad = item.querySelector('.stop-pad');
      applyFilename(li, name);
      applyLabels(item.querySelector('.labels'), stems);
      st.textContent = 'checking';
      st.classList.add('chip-dim');
      st.classList.add('hidden');
      if(startPressed){ st.classList.remove('hidden'); }
      dl.classList.remove('show');
      if (stopPad) stopPad.style.display = 'none';
      item.classList.add('enter-pre');
      item.__tempKey = tempKey;
      item.__batchKey = options.batchKey || '';
      item.__taskRowKey = `temp:${tempKey || ''}:${stemsIdentityKey(stems || [])}`;
      item.__stems = Array.isArray(stems) ? stems.slice() : [];
      queue.insertBefore(item, queue.firstChild);
      requestAnimationFrame(() => {
        item.classList.add('enter-active');
      });
      updateVigs();
      const smooth = makeProgressSmoother(bar);
      smooth.setImmediate(0);
      applyRowState(item, { stage: 'ready', pct: 0, id: null, stems, groupStems: stems });
      return {item, li, bar, st, dl, card, stopPad, smooth};
    }

    async function handleFiles(fileList){
      if(modelsBlocked){
        showPopup(currentModelBlockMessage());
        await refreshModelStatus({ allowPrompt: true });
        flashModelDownloadCTA();
        if (fileInput) fileInput.value = '';
        return;
      }
      const storageOk = await checkStorage();
      const memoryOk = checkMemory();
      if(!storageOk || !memoryOk){
        showPopup('cannot add songs until resources are available');
        return;
      }
      const existingCount = tasks.filter(Boolean).length;
      if(existingCount >= MAX_TASKS){
        showPopup('you already have 50 songs queued; clear some to upload more');
        return;
      }
      const makeTempKey = () => (crypto && crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));
      const stemGroups = selectedStemGroups();
      const uploadStemGroups = stemGroups.length ? stemGroups : [[]];
      const displayStems = selectedStemUnion();
      const newPending = [];
      const files = [...fileList];
      const audioFiles = files.filter((file) => {
        return (file.type && (file.type.startsWith('audio/') || file.type.startsWith('video/'))) || /\.(wav|wave|mp3|m4a|aac|flac|ogg|oga|aif|aiff|alac|opus|mp4|m4v|mov|webm|mkv|avi)$/i.test(file.name);
      });
      const taskCountNeeded = audioFiles.length * uploadStemGroups.length;
      if(existingCount + taskCountNeeded > MAX_TASKS){
        const slots = Math.max(0, MAX_TASKS - existingCount);
        const maxSongs = Math.floor(slots / uploadStemGroups.length);
        if(maxSongs <= 0){
          showPopup(`you need ${uploadStemGroups.length} free slots for the selected models on one song`);
        }else{
          const slotLabel = maxSongs === 1 ? 'song' : 'songs';
          showPopup(`with ${uploadStemGroups.length} selected models, you can only add ${maxSongs} more ${slotLabel}`);
        }
        return;
      }
      const uploadBatchKey = audioFiles.length > 1 ? makeUiGroupKey('batch') : '';
      let availableSlots = Math.max(0, MAX_TASKS - existingCount);
      for(const file of files){
        if(availableSlots <= 0) break;
        // accept common audio even when type is empty
        const looksMedia = (file.type && (file.type.startsWith('audio/') || file.type.startsWith('video/'))) || /\.(wav|wave|mp3|m4a|aac|flac|ogg|oga|aif|aiff|alac|opus|mp4|m4v|mov|webm|mkv|avi)$/i.test(file.name);
        if(!looksMedia) continue;
        const tempKey = makeTempKey();
        for(const stems of uploadStemGroups){
          if(availableSlots <= 0) break;
          availableSlots -= 1;
          const rowUi = createPendingQueueRow(file.name, stems, tempKey, { batchKey: uploadBatchKey });
          // record pending entry
          const pending = { file, ui:rowUi, started:false, id:null, stems:stems.slice(), groupStems: displayStems.slice(), tempKey, batchKey: uploadBatchKey, autoStart:startPressed, startedProcessing:false, frozen:false, clip_start_ms:0, clip_end_ms:null, clip_enabled:false, outputs:[] };
          pending.validationPromise = (async () => {
            const duration = await getAudioDuration(file);
            if(duration > 0){
              if(duration > FIVE_HOURS_SEC){
                failPendingUpload(pending, 'single song limit is 5 hours');
                return false;
              }
              if(duration > LONG_TRACK_SEC && !hasMemoryForDuration(duration)){
                failPendingUpload(pending);
                return false;
              }
            }
            return true;
          })();
          pendingItems.push(pending);
          newPending.push(pending);
          // persist placeholder task (no id yet)
          tasks.push({ id:null, name:file.name, pct:0, stage:'ready', stems:stems.slice(), groupStems: displayStems.slice(), file:null, tempKey, batchKey: uploadBatchKey, out_dir:null, clip_start_ms:0, clip_end_ms:null, clip_enabled:false, outputs:[], downloaded:false, delivery:'folder', autoDownloaded:false, frozen:false });
          saveTasks();
          updateUI();
        }
      }
      if (fileInput) fileInput.value = '';
      await Promise.all(newPending.map(async (p) => {
        const valid = await p.validationPromise;
        if(!valid) return;
        return p.started ? p.uploadPromise : uploadWithStems(p, p.stems);
      }));
      await syncQueuedStemSelection();
      updateStartButton();
    }

    function uploadWithStems(pending, stems){
      pending.started = true;
      pending.uploadPromise = new Promise((resolve) => {
        const {file, ui} = pending;
        const {st, dl, card, stopPad, bar, smooth} = ui;
        const data = new FormData();
        data.append('file', file);
        data.append('stems', stems.join(','));
        data.append('output_format', settingsState.output_format || 'same_as_input');
        data.append('multi_stem_export', settingsState.multi_stem_export || 'zip');
        data.append('video_handling', settingsState.video_handling || 'audio_only');
        data.append('clip_start_ms', String(Number(pending.clip_start_ms) || 0));
        if(Number.isFinite(Number(pending.clip_end_ms))){
          data.append('clip_end_ms', String(Math.max(0, Number(pending.clip_end_ms))));
        }
        data.append('clip_enabled', pending.clip_enabled ? 'true' : 'false');
        const sourceDir = sourceDirForFile(file);
        if(sourceDir){
          data.append('source_dir', sourceDir);
        }
        const xhr = new XMLHttpRequest();
        pending.xhr = xhr;
        xhr.open('POST', '/upload');
        xhr.upload.onprogress = (evt) => {
          if(evt.lengthComputable){
            const pct = Math.round(evt.loaded / evt.total * 100);
            st.textContent = `uploading (${pct}%)`;
            st.classList.remove('chip-dim');
          }
        };
        xhr.onload = () => {
          if(xhr.status >= 400){
            let message = 'could not upload file';
            try{
              const payload = JSON.parse(xhr.responseText || '{}');
              message = payload?.detail?.message || payload?.message || message;
            }catch(_){}
            failPendingUpload(pending, message);
            return resolve();
          }
          st.classList.remove('chip-dim');
          st.textContent = 'ready';
          smooth.setImmediate(0);
          let res = {};
          try {
            res = JSON.parse(xhr.responseText || '{}');
            const msg = res.detail && res.detail.message;
            if(msg){
              showStatus(st);
              st.textContent = 'error';
              st.classList.add('chip-dim');
              if (stopPad) stopPad.style.display = 'none';
              else showError(msg);
            }
          }catch(_){ }
          if(res.detail && res.detail.message){
            failPendingUpload(pending, res.detail.message);
            return resolve();
          }
          const parsed = res;
          pending.id = parsed.task_id;
          pending.stems = parsed.stems;
          ui.item.__taskId = parsed.task_id;
          // replace placeholder task
          const idx = tasks.findIndex(t => t.tempKey === pending.tempKey);
          const t = { id:parsed.task_id, name:file.name, mode: parsed.mode || null, pct:0, stage:'ready', stems:parsed.stems, groupStems: pending.groupStems || parsed.stems, tempKey: pending.tempKey, batchKey: pending.batchKey || '', out_dir: null, preset_settings: parsed.preset_settings || null, can_adjust_preset: !!parsed.can_adjust_preset, clip_start_ms: Number(parsed.clip_start_ms) || Number(pending.clip_start_ms) || 0, clip_end_ms: Number.isFinite(Number(parsed.clip_end_ms)) ? Number(parsed.clip_end_ms) : (Number.isFinite(Number(pending.clip_end_ms)) ? Number(pending.clip_end_ms) : null), clip_enabled: typeof parsed.clip_enabled === 'boolean' ? !!parsed.clip_enabled : !!pending.clip_enabled, outputs: Array.isArray(parsed.outputs) ? parsed.outputs.slice() : [], downloaded:false, delivery: parsed.delivery || 'folder', autoDownloaded:false, frozen: !!pending.frozen };
          if(idx>=0) tasks[idx]=t; else tasks.push(t);
          if(startPressed || pending.autoStart){
            t.stage = 'queued';
            t.pct = 0;
            t.frozen = true;
            pending.frozen = true;
            if(st){
              showStatus(st);
              st.textContent = 'queued';
              st.classList.remove('hidden');
            }
          }
          applyRowState(ui.item, t);
          saveTasks(); updateUI();
          // enable stop now that we have an id
          if (stopPad){
            bindStopPad(stopPad, t, { bar, st, dl, stopPad, smooth });
            setStopVisibility(stopPad, (startPressed || pending.autoStart) ? 'queued' : 'ready');
          }
          if(!queueStarted && !startPressed){
            Promise.resolve(syncQueuedStemSelection()).catch(() => {});
          }
          if(pending.autoStart){
            startTaskWhenReady(pending);
          }
          resolve();
        };
        xhr.onerror = () => {
          if(!isClearing){
            failPendingUpload(pending, 'could not upload file');
          }
          resolve();
        };
        xhr.send(data);
      });
      return pending.uploadPromise;
    }

    function startTaskWhenReady(pending, startSettings = null){
      if(!pending || pending.startedProcessing) return;
      if(!pending.uploadPromise){ return; }
      pending.startedProcessing = true;
      pending.uploadPromise.then(async () => {
        if(!pending.id) return;
        try{
          const body = startSettings || currentStartSettings();
          const startedTask = await requestTaskStart(pending.id, body);
          if(startedTask && typeof startedTask.message === 'string' && startedTask.message.trim() && !isClearing){
            showPopup(startedTask.message);
          }
          const task = getTaskById(pending.id);
          if(task){
            task.stage = startedTask.stage || startedTask.status || 'queued';
            task.pct = typeof startedTask.pct === 'number' ? startedTask.pct : 0;
            task.frozen = true;
            saveTasks();
            ensureTaskProgressTracking(task);
            const row = findRowForTask(task);
            if(row) applyRowState(row, task);
          }
        }catch(err){
          pending.startedProcessing = false;
          const task = getTaskById(pending.id);
          if(task){
            task.stage = 'ready';
            task.pct = 0;
            task.frozen = false;
            saveTasks();
            const row = findRowForTask(task);
            if(row){
              applyRowState(row, task);
              const st = row.querySelector('.chip.status');
              if(st){
                showStatus(st);
                st.textContent = 'ready';
              }
              const stopPad = row.querySelector('.stop-pad');
              setStopVisibility(stopPad, 'ready');
            }
          }
          if(!isClearing){
            showPopup(err && err.message ? err.message : 'could not start song');
          }
        }
      });
    }

    async function startSequentialProcessing(){
      if(startLock || (startBtn && startBtn.disabled)) return;
      if(modelsBlocked){
        showPopup(currentModelBlockMessage());
        await refreshModelStatus({ allowPrompt: true });
        flashModelDownloadCTA();
        return;
      }
      const storageOk = await checkStorage();
      const memoryOk = checkMemory();
      if(!storageOk || !memoryOk){
        showPopup('cannot start until storage and memory are sufficient');
        return;
      }
      startLock = true;
      updateStartButton();
      let markedStarted = false;
      const freezeReadyBatch = () => {
        let hasStartable = false;
        pendingItems.forEach((pending) => {
          if(!pending || pending.frozen) return;
          const task = pending.id ? getTaskById(pending.id) : getTaskByTempKey(pending.tempKey);
          if(task && (task.stage !== 'ready' || Number(task.pct || 0) !== 0)) return;
          pending.frozen = true;
          pending.autoStart = true;
          if(task){
            task.frozen = true;
          }
          hasStartable = true;
        });
        tasks.forEach((task) => {
          if(!task || !task.id || task.frozen || task.stage !== 'ready' || Number(task.pct || 0) !== 0) return;
          task.frozen = true;
          hasStartable = true;
        });
        if(hasStartable){
          saveTasks();
        }
        return hasStartable;
      };
      const markReadyQueuedUI = () => {
        const rows = queueLeafRows();
        let changed = false;
        rows.forEach(row => {
          const task = getTaskByRow(row);
          const chip = row.querySelector('.chip.status');
          if(task && (!task.stage || task.stage === 'ready' || task.stage === 'queued')){
            if(task.stage !== 'queued' || task.pct !== 0){
              task.stage = 'queued';
              task.pct = 0;
              changed = true;
            }
            if(chip){
              showStatus(chip);
              chip.textContent = 'queued';
              chip.classList.remove('hidden');
            }
            applyRowState(row, task);
            const stopPad = row.querySelector('.stop-pad');
            setStopVisibility(stopPad, 'queued');
            return;
          }
          if(chip && (chip.textContent || '').toLowerCase() === 'ready'){
            showStatus(chip);
            chip.textContent = 'queued';
            chip.classList.remove('hidden');
          }
          const fallbackTask = task || { stage: 'queued', pct: 0, id: id || null };
          applyRowState(row, fallbackTask);
          const stopPad = row.querySelector('.stop-pad');
          setStopVisibility(stopPad, fallbackTask.stage, !!task && isProcessing(task));
        });
        if(changed){
          saveTasks();
        }
      };
      const startReadyTasks = async (startSettings) => {
        const existing = Array.isArray(tasks) ? tasks.filter(Boolean) : [];
        let startError = '';
        let queueNotice = '';
        for(const t of existing){
          if(!t || !t.id || isProcessing(t)) continue;
          const normalizedStage = String(t.stage || '').toLowerCase();
          const isStartable = t.frozen && Number(t.pct || 0) === 0 && (normalizedStage === 'ready' || normalizedStage === 'queued');
          if(isStartable){
            try{
              const startedTask = await requestTaskStart(t.id, startSettings);
              if(!queueNotice && startedTask && typeof startedTask.message === 'string' && startedTask.message.trim()){
                queueNotice = startedTask.message;
              }
              t.stage = startedTask.stage || startedTask.status || 'queued';
              t.pct = typeof startedTask.pct === 'number' ? startedTask.pct : 0;
              t.downloaded = false;
              t.autoDownloaded = false;
              t.out_dir = null;
              t.frozen = true;
              ensureTaskProgressTracking(t);
            }catch(err){
              t.stage = 'ready';
              t.pct = 0;
              t.frozen = false;
              if(!startError){
                startError = err && err.message ? err.message : 'could not start song';
              }
            }
          }
        }
        saveTasks();
        markReadyQueuedUI();
        existing.forEach((task) => {
          if(!task || task.frozen || task.stage !== 'ready' || Number(task.pct || 0) !== 0) return;
          const row = findRowForTask(task);
          if(!row) return;
          applyRowState(row, task);
          const st = row.querySelector('.chip.status');
          if(st){
            showStatus(st);
            st.textContent = 'ready';
          }
          const stopPad = row.querySelector('.stop-pad');
          setStopVisibility(stopPad, 'ready');
        });
        if(startError && !isClearing){
          showPopup(startError);
        }else if(queueNotice && !isClearing){
          showPopup(queueNotice);
        }
      };
      try{
        if(!freezeReadyBatch()){
          showPopup('there are no songs to start');
          return;
        }
        const batchStartSettings = currentStartSettings();
        startGeneration += 1;
        startPressed = true;
        markedStarted = true;
        markReadyQueuedUI();
        revealStatusesAfterStart();
        const pendingBatch = pendingItems.slice();
        for(const p of pendingBatch){
          if(!p || !p.frozen) continue;
          if(!p.uploadPromise){
            uploadWithStems(p, p.stems);
          }
          if(p.uploadPromise){
            startTaskWhenReady(p, batchStartSettings);
          }
        }
        await startReadyTasks(batchStartSettings);
        startPressed = false;
        markedStarted = false;
        pendingItems.forEach(p => { if(p){ p.autoStart = false; } });
        updateUI();
      } finally {
        if(markedStarted){
          startPressed = false;
        }
        startLock = false;
        updateStartButton();
      }
    }

    async function handleDesktopPickedPaths(paths){
      if(!Array.isArray(paths) || paths.length === 0){
        return;
      }
      if(modelsBlocked){
        showPopup(currentModelBlockMessage());
        await refreshModelStatus({ allowPrompt: true });
        flashModelDownloadCTA();
        return;
      }
      const storageOk = await checkStorage();
      const memoryOk = checkMemory();
      if(!storageOk || !memoryOk){
        showPopup('cannot add songs until resources are available');
        return;
      }
      const existingCount = tasks.filter(Boolean).length;
      if(existingCount >= MAX_TASKS){
        showPopup('you already have 50 songs queued; clear some to upload more');
        return;
      }
      const stemGroups = selectedStemGroups();
      const uploadStemGroups = stemGroups.length ? stemGroups : [[]];
      const availableSlots = Math.max(0, MAX_TASKS - existingCount);
      const maxSongs = Math.floor(availableSlots / uploadStemGroups.length);
      if(maxSongs <= 0){
        showPopup(`you need ${uploadStemGroups.length} free slots for the selected models on one song`);
        return;
      }
      const normalizedPaths = paths.map(normalizePickedPath).filter((pathText) => !!pathText);
      const trimmed = normalizedPaths.slice(0, maxSongs);
      if(!trimmed.length){
        return;
      }
      try{
        const sourceDirs = trimmed.map((pathText) => normalizeSourceDirPath(pathText));
        const importedTasks = [];
        const importBatchKey = trimmed.length > 1 ? makeUiGroupKey('batch') : '';
        const songKeys = trimmed.map(() => makeUiGroupKey('song'));
        for(const stems of uploadStemGroups){
          const res = await fetch('/api/import_paths', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              paths: trimmed,
              source_dirs: sourceDirs,
              stems: stems.join(','),
              output_format: settingsState.output_format || 'same_as_input',
              output_same_as_input: !!settingsState.output_same_as_input,
              multi_stem_export: settingsState.multi_stem_export || 'zip',
              video_handling: settingsState.video_handling || 'audio_only',
            }),
          });
          const payload = await res.json().catch(() => ({}));
          if(!res.ok){
            const msg = payload && payload.message ? payload.message : 'could not import files';
            showPopup(msg);
            return;
          }
          (Array.isArray(payload.tasks) ? payload.tasks : []).forEach((task, index) => {
            importedTasks.push({
              ...task,
              __tempKey: songKeys[index] || makeUiGroupKey('song'),
              __batchKey: importBatchKey,
            });
          });
        }
        importedTasks.forEach((task) => {
          const item = {
            id: task.task_id || task.id || null,
            name: task.name,
            mode: task.mode || null,
            pct: typeof task.pct === 'number' ? task.pct : 0,
            stage: task.stage || 'ready',
            stems: Array.isArray(task.stems) ? task.stems : [],
            out_dir: task.out_dir || null,
            tempKey: task.__tempKey || '',
            batchKey: task.__batchKey || '',
            preset_settings: task.preset_settings || null,
            can_adjust_preset: !!task.can_adjust_preset,
            clip_start_ms: Number(task.clip_start_ms) || 0,
            clip_end_ms: Number.isFinite(Number(task.clip_end_ms)) ? Number(task.clip_end_ms) : null,
            clip_enabled: !!task.clip_enabled,
            outputs: Array.isArray(task.outputs) ? task.outputs.slice() : [],
            downloaded: false,
            delivery: task.delivery || 'folder',
            autoDownloaded: false,
            frozen: false,
          };
          tasks.push(item);
          createItem(item);
        });
        saveTasks();
        updateUI();
      }catch(err){
        console.warn('desktop import failed', err);
        showPopup('could not import files');
      }
    }
    if(startBtn) guardClick(startBtn, startSequentialProcessing);

    // --- Progress tracking for uploads (SSE) ---
    function displayStage(stage){
      if(!stage) return 'queued';
      const lower = stage.toLowerCase();
      if(lower.includes('deux')) return 'voc/inst';
      if(lower.includes('harmon')) return 'harmonies';
      if(lower.includes('vocals')) return 'vocals';
      if(lower.includes('instrumental')) return 'instrumental';
      if(lower.includes('background vocal') || lower.includes('bg vocal') || lower.includes('karaoke')) return 'bg vocal';
      if(lower.includes('denoise')) return 'denoise';
      if(lower.includes('drums')) return 'drums';
      if(lower.includes('bass')) return 'bass';
      if(lower.includes('other')) return 'other';
      if(lower.includes('guitar')) return 'guitar';
      if(lower.includes('piano')) return 'piano';
      if(lower.includes('6s')) return 'full mix';
      if(lower.includes('error')) return 'error';
      if(lower.startsWith('prepare')) return 'preparing';
      if(lower.startsWith('load_audio')) return 'loading';
      if(lower.startsWith('write')) return 'finishing';
      const parts = lower.split('.');
      return parts[parts.length-1] || lower;
    }

    async function recoverTaskFromServer(taskId, taskRef, ui){
      try{
        const res = await fetch(`/api/tasks/${taskId}?_=${Date.now()}`, { cache: 'no-store' });
        if(!res.ok) return false;
        const data = await res.json();
        const {bar, st, dl, stopPad, smooth} = ui;
        const stage = data.status === 'error' ? 'error' : (data.stage || data.status || 'queued');
        if(taskRef){
          taskRef.stage = stage;
          taskRef.pct = typeof data.pct === 'number' ? data.pct : taskRef.pct;
          taskRef.eta_seconds = data.eta_seconds;
          taskRef.eta_state = data.eta_state ?? taskRef.eta_state ?? null;
          taskRef.mode = data.mode || taskRef.mode;
          if(typeof data.clip_start_ms !== 'undefined') taskRef.clip_start_ms = Number(data.clip_start_ms) || 0;
          if(typeof data.clip_end_ms !== 'undefined') taskRef.clip_end_ms = Number.isFinite(Number(data.clip_end_ms)) ? Number(data.clip_end_ms) : null;
          if(typeof data.clip_enabled === 'boolean') taskRef.clip_enabled = !!data.clip_enabled;
          taskRef.out_dir = data.out_dir || null;
          taskRef.delivery = data.delivery || taskRef.delivery;
          taskRef.error = data.error || null;
          if(Array.isArray(data.stems)) taskRef.stems = data.stems;
          if(Array.isArray(data.outputs)) taskRef.outputs = data.outputs.slice();
          if(data.preset_settings) taskRef.preset_settings = data.preset_settings;
          if(typeof data.can_adjust_preset === 'boolean') taskRef.can_adjust_preset = data.can_adjust_preset;
          saveTasks();
        }
        const currentRow = st && st.closest ? st.closest('.item-row') : null;
        if(currentRow){
          applyRowState(currentRow, taskRef || { id: taskId, stage, pct: data.pct || 0 });
        }
        if(data.status === 'done'){
          if(smooth) smooth.setTarget(100);
          hideStatus(st);
          if(dl){
            const ref = taskRef || { id: taskId, out_dir: data.out_dir, delivery: data.delivery || 'folder', downloaded: false, autoDownloaded: false };
            if(ref.delivery !== 'browser_download' && ref.downloaded){
              setFolderIcon(dl);
              dl.classList.add('show');
              dl.title = 'open folder';
              bindFolderButton(dl, ref);
            }else{
              bindDownloadButton(dl, ref);
            }
          }
          if(stopPad) stopPad.remove();
          const parent = st && st.closest ? st.closest('.card') : null;
          if(parent) parent.classList.add('done');
          updateUI();
          return true;
        }
        if(data.status === 'stopped'){
          markTaskStopped(taskRef, { ...ui, item: currentRow });
          return true;
        }
        if(data.status === 'error'){
          if(st){
            showStatus(st);
            st.textContent = 'error';
            st.classList.remove('hidden');
          }
          setStatusVisibility(st, 'error');
          const parent = st && st.closest ? st.closest('.card') : null;
          if(parent) parent.classList.add('done');
          if(stopPad){
            stopPad.style.display = '';
            stopPad.classList.add('show');
            stopPad.title = 'rerun';
            stopPad.innerHTML = '';
            setRetryIcon(stopPad);
            guardClick(stopPad, (e) => {
              e.preventDefault();
              if(taskRef) rerunTask(taskRef, { ...ui, item: currentRow });
            });
          }
          if(data.error) showPopup(data.error);
          updateUI();
          return true;
        }
        if(st){
          showStatus(st);
          if(shouldUseProgressSummary(stage)){
            st.textContent = formatProgressChip(data.pct || 0, data.eta_seconds, data.eta_state);
          }else{
            st.textContent = displayStage(stage) || 'queued';
          }
        }
        setStatusVisibility(st, stage);
        setStopVisibility(stopPad, stage, !!taskRef && isProcessing(taskRef));
        if(stopPad && taskRef && taskRef.id && isProcessing(taskRef)){
          bindStopPad(stopPad, taskRef, { bar, st, dl, stopPad, smooth });
        }
        setTimeout(() => {
          trackProgress(taskId, bar, st, dl, null, null, taskRef, null, smooth, stopPad);
        }, 450);
        return true;
      }catch(_){
        return false;
      }
    }

    function trackProgress(taskId, bar, st, dl, _a, _b, taskRef, _c, smooth, stopPad){
      try{
        const es = new EventSource('/progress/' + taskId);
        activeStreams.set(taskId, es);
        es.addEventListener('message', (ev) => {
          if(isClearing) return;
          let data = null;
          try { data = JSON.parse(ev.data); } catch(_) {}
          if(!data) return;
          const rawStage = data.stage;
          const stage = rawStage === 'errored' ? 'error' : rawStage;
          const rawPct = data.pct;
          const previousPct = taskRef && typeof taskRef.pct === 'number' && taskRef.pct >= 0 ? taskRef.pct : 0;
          const pct = (typeof rawPct === 'number' && rawPct >= 0 && !['done', 'error', 'stopped'].includes(String(stage || '').toLowerCase()))
            ? Math.max(rawPct, previousPct)
            : rawPct;
          const stopPending = !!(taskRef && taskRef.stopRequested);
          const stopTerminal = stage === 'stopped' || stage === 'error' || stage === 'done';
          if(stopPending && !stopTerminal){
            return;
          }
          if(taskRef){
            taskRef.stage = stage;
            taskRef.pct = pct;
            if(typeof data.eta_seconds === 'number' || data.eta_seconds === null){
              taskRef.eta_seconds = data.eta_seconds;
            }
            taskRef.eta_state = data.eta_state ?? taskRef.eta_state ?? null;
            if(data.mode) taskRef.mode = data.mode;
            if(typeof data.clip_start_ms !== 'undefined') taskRef.clip_start_ms = Number(data.clip_start_ms) || 0;
            if(typeof data.clip_end_ms !== 'undefined') taskRef.clip_end_ms = Number.isFinite(Number(data.clip_end_ms)) ? Number(data.clip_end_ms) : null;
            if(typeof data.clip_enabled === 'boolean') taskRef.clip_enabled = !!data.clip_enabled;
            if(data.stems) taskRef.stems = data.stems;
            if(Array.isArray(data.outputs)) taskRef.outputs = data.outputs.slice();
            if(data.out_dir) taskRef.out_dir = data.out_dir;
            if(data.zip) taskRef.zip = data.zip;
            if(data.delivery) taskRef.delivery = data.delivery;
            if(data.preset_settings) taskRef.preset_settings = data.preset_settings;
            if(typeof data.can_adjust_preset === 'boolean') taskRef.can_adjust_preset = data.can_adjust_preset;
            if(typeof taskRef.downloaded === 'undefined'){ taskRef.downloaded = false; }
            if(typeof taskRef.autoDownloaded === 'undefined'){ taskRef.autoDownloaded = false; }
            saveTasks();
          }
          const currentRow = st && st.closest ? st.closest('.item-row') : null;
          if(currentRow){
            applyRowState(currentRow, taskRef || { id: taskId, stage, pct });
          }
          if(data.stems && st){
            const parentRow = st.closest('.item-row');
            applyLabels(parentRow ? parentRow.querySelector('.labels') : null, displayStemsForTask(taskRef || { stems: data.stems }));
          }
          if(st){ showStatus(st); }
          if(stage === 'error'){
            if(taskRef){
              taskRef.stopRequested = false;
              taskRef.stage = 'error';
              taskRef.pct = -1;
              saveTasks();
            }
            st.textContent = 'error';
            st.classList.remove('hidden');
            if(dl){
              dl.classList.remove('show');
            }
            setStatusVisibility(st, 'error');
            const parent = st.closest('.card'); if(parent) parent.classList.add('done');
            if (stopPad) {
              stopPad.style.display = '';
              stopPad.classList.add('show');
              stopPad.title = 'rerun';
              stopPad.innerHTML = '';
              setRetryIcon(stopPad);
              guardClick(stopPad, (e) => { e.preventDefault(); rerunTask(taskRef, { bar, st, dl, stopPad, smooth }); });
            }
            dropPendingById(taskId);
            es.close();
            activeStreams.delete(taskId);
            if(data.error){
              showPopup(data.error);
            }
            if(currentRow){
              applyRowState(currentRow, taskRef || { id: taskId, stage: 'error', pct: -1 });
            }
            updateUI(); return;
          }
          if(stage === 'done'){
            const hasRemainingGroupTasks = !!(taskRef && groupHasRemainingTasks(taskRef, taskId));
            if(taskRef){
              taskRef.stopRequested = false;
            }
            smooth.setTarget(100);
            dropPendingById(taskId);
            es.close();
            activeStreams.delete(taskId);
            if(hasRemainingGroupTasks){
              showStatus(st);
              st.textContent = 'queued';
              st.classList.remove('hidden');
              smooth.setImmediate(0);
              if(dl) dl.classList.remove('show');
              if(stopPad){
                stopPad.style.display = 'none';
                stopPad.classList.remove('show');
              }
              const parent = st.closest('.card'); if(parent) parent.classList.remove('done');
              if(currentRow){
                applyLabels(currentRow.querySelector('.labels'), displayStemsForTask(taskRef));
                applyRowState(currentRow, getTaskByTempKey(taskRef.tempKey) || taskRef);
              }
              updateUI(); return;
            }
            hideStatus(st);
            if(dl){
              const ref = taskRef || { id: taskId, out_dir: data.out_dir, zip: data.zip, downloaded:false, delivery: data.delivery || 'folder', autoDownloaded:false };
              if(ref.delivery !== 'browser_download' && ref.downloaded){
                setFolderIcon(dl);
                dl.classList.add('show');
                dl.title = 'open folder';
                bindFolderButton(dl, ref);
              }else{
                bindDownloadButton(dl, ref);
              }
            }
            if(stopPad) stopPad.remove();
            const parent = st.closest('.card'); if(parent) parent.classList.add('done');
            if(currentRow){
              applyRowState(currentRow, taskRef || { id: taskId, stage: 'done', pct: 100 });
            }
            updateUI(); return;
          }
          if (stage === 'stopped') {
            markTaskStopped(taskRef, { bar, st, dl, stopPad, smooth, item: currentRow });
            dropPendingById(taskId);
            es.close();
            activeStreams.delete(taskId);
            if(currentRow){
              applyRowState(currentRow, taskRef || { id: taskId, stage: 'stopped', pct: 0 });
            }
            updateUI(); return;
          }
          if (typeof pct === 'number' && pct >= 0) {
            const sp = Math.round(Math.max(0, Math.min(100, pct)));
            const normalizedStage = String(stage || '').toLowerCase();
            if (!shouldUseProgressSummary(normalizedStage)) {
              const friendlyStage = displayStage(stage);
              st.textContent = friendlyStage || 'queued';
              smooth.setImmediate(0);
            } else {
              const etaValue = typeof data.eta_seconds === 'number' ? data.eta_seconds : (taskRef ? taskRef.eta_seconds : null);
              st.textContent = formatProgressChip(sp, etaValue, data.eta_state || (taskRef ? taskRef.eta_state : null));
              if(normalizedStage === 'ready' || normalizedStage === 'queued'){
                smooth.setImmediate(0);
              }else{
                smooth.setTarget(sp);
              }
            }
            if(stage !== 'done' && dl){
              dl.classList.remove('show');
            }
          } else {
            st.textContent = formatStageLabel(effectiveStage, 0);
          }
          setStatusVisibility(st, stage);
          setStopVisibility(stopPad, stage);
          if(stopPad && taskRef && taskRef.id && isProcessing(taskRef)){
            bindStopPad(stopPad, taskRef, { bar, st, dl, stopPad, smooth });
          }
        });
        es.addEventListener('error', (ev) => {
          if (taskRef && taskRef.stage === 'stopped') {
            es.close();
            activeStreams.delete(taskId);
            updateUI();
            return;
          }
          let errPayload = null;
          let errCode = null;
          let errMsg = null;
          try{
            if(ev && ev.data){ errPayload = JSON.parse(ev.data); }
          }catch(_){}
          if(errPayload){
            errCode = errPayload.code || (errPayload.detail && errPayload.detail.code);
            errMsg = errPayload.message || (errPayload.detail && errPayload.detail.message);
          }
          es.close();
          activeStreams.delete(taskId);
          if(!errMsg){
            recoverTaskFromServer(taskId, taskRef, { bar, st, dl, stopPad, smooth, item: st && st.closest ? st.closest('.item-row') : null }).then((recovered) => {
              if(recovered) return;
              if (taskRef) {
                taskRef.stage = 'error';
                taskRef.pct = -1;
                saveTasks();
              }
              const currentRow = st && st.closest ? st.closest('.item-row') : null;
              if (st) {
                showStatus(st);
                st.textContent = 'error';
                st.classList.remove('hidden');
              }
              setStatusVisibility(st, 'error');
              const parent = st && st.closest ? st.closest('.card') : null;
              if (parent) parent.classList.add('done');

              if (stopPad) {
                stopPad.style.display = '';
                stopPad.classList.add('show');
                stopPad.title = 'rerun';
                stopPad.innerHTML = '';
                setRetryIcon(stopPad);
                guardClick(stopPad, (e) => {
                  e.preventDefault();
                  if (taskRef) rerunTask(taskRef, { bar, st, dl, stopPad, smooth });
                });
              }
              dropPendingById(taskId);
              if(currentRow){
                applyRowState(currentRow, taskRef || { id: taskId, stage: 'error', pct: -1 });
              }
              updateUI();
            });
            return;
          }
          if (taskRef) {
            taskRef.stage = 'error';
            taskRef.pct = -1;
            saveTasks();
          }
          const currentRow = st && st.closest ? st.closest('.item-row') : null;
          if (st) {
            showStatus(st);
            st.textContent = 'error';
            st.classList.remove('hidden');
          }
          setStatusVisibility(st, 'error');
          const parent = st && st.closest ? st.closest('.card') : null;
          if (parent) parent.classList.add('done');

          if (stopPad) {
            stopPad.style.display = '';
            stopPad.classList.add('show');
            stopPad.title = 'rerun';
            stopPad.innerHTML = '';
            setRetryIcon(stopPad);
            guardClick(stopPad, (e) => {
              e.preventDefault();
              if (taskRef) rerunTask(taskRef, { bar, st, dl, stopPad, smooth });
            });
          }
          dropPendingById(taskId);
          if(currentRow){
            applyRowState(currentRow, taskRef || { id: taskId, stage: 'error', pct: -1 });
          }
          if(errMsg){
            showPopup(errMsg);
          }
          updateUI();
        });
              }catch(e){
        showStatus(st);
        st.textContent = 'error';
        if(taskRef){ taskRef.stage = 'error'; taskRef.pct = -1; saveTasks(); }
        updateUI();
      }
    }
