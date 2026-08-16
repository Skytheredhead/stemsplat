# Legacy skeleton audit (not the v0.4.2 product)

> **Scope warning:** This preserved report describes the abandoned, unrelated
> Rust/Python skeleton from the old checkout. It does **not** describe the
> working v0.4.2 Apple Silicon application at upstream commit `63ab006` and
> must not be used as evidence that v0.4.2 lacks inference or implements
> playback. Applicable defensive behaviors are tracked in
> [`reconciliation-matrix.md`](reconciliation-matrix.md) and are ported
> manually; the histories have no common ancestor and the skeleton is not
> merged into the release branch.

---

# Stemsplat pre-release QA, reliability, and regression audit

Audit date: 2026-08-10 CDT (final runtime evidence crossed into 2026-08-11 UTC)

## Executive verdict

The repository, both server implementations, browser application, Rust
workspace, SQLite persistence, import/file boundaries, queue/playback state,
packaging, and deployment definitions were broadly inspected and exercised.
The application is substantially harder to corrupt or misrepresent than at the
start of the sweep: unsafe uploads are scoped and bounded, placeholder audio is
no longer reported as separated stems, state mutations are serialized, failed
artifacts are cleaned, browser state survives refresh/restart, and error paths
are explicit.

This build is **not release-ready as a working stem separator**. There are no P0
findings, but three P1 release boundaries remain:

1. Real model inference adapters and model/config assets do not exist. Imports
   intentionally fail closed.
2. Playback is a persisted state machine, not an audio-output engine. The UI now
   says so explicitly.
3. There is no authentication or authorization. The default loopback boundary
   is acceptable for local use only; any LAN or internet exposure requires an
   authenticated reverse proxy.

The work is in the **reference_safe** lane. Default chunk sizes, overlap,
precision, device selection policy, and model behavior were not replaced with
lower-quality alternatives. False placeholder output was invalidated rather
than silently preserved. No whole-pipeline quality-safety claim is possible
because this repository has no `quality.py`, model adapters, checkpoints, or
drift harness.

## 1. System map

### Runtime and entry points

- Python 3.10+ package managed by Poetry metadata plus `requirements.txt`.
- `python -m stemrunner` / `stemrunner.cli:main`: Click CLI.
- `stemrunner.server:app`: retained headless legacy FastAPI API.
- `stemrunner.ml_worker`: JSON-progress Python subprocess used by Rust.
- Rust 2021 Axum/Tokio primary server: `rust/crates/stemsplat-server`.
- Rust workspace lockfile contains 216 dependencies.
- Dependency-free, single-file browser application: `web/index.html`.

### Rust architecture

- `stemsplat-core`: DTOs, settings, constants, capability types.
- `stemsplat-db`: SQLite schema/migrations, WAL pool, settings, import, track,
  waveform, stem, and queue persistence.
- `stemsplat-import`: upload finalization, concurrency admission, metadata,
  analysis, waveform, Python worker lifecycle, stem validation, progress.
- `stemsplat-metadata`: currently WAV metadata plus format identification.
- `stemsplat-audio-analysis`: WAV key/BPM analysis.
- `stemsplat-waveform`: WAV waveform reduction.
- `stemsplat-queue`: queue operations over transactional DB methods.
- `stemsplat-playback`: persisted/in-memory playback-state transitions only.
- Axum routes cover imports/SSE, tracks, waveforms, stems/downloads, queue,
  playback state, settings, and system capabilities.

### Storage and persistence

- SQLite at `STEMSPLAT_DATA_DIR/stemsplat.sqlite`.
- Imported files and generated stems under `STEMSPLAT_UPLOAD_DIR`.
- Foreign keys, WAL, an eight-connection pool, a five-second busy timeout,
  transactional queue rewrites, and indexes for import/queue access.
- Recent import progress persists and is replayed after refresh/restart.
- The legacy Python API keeps its task registry only in process memory.

### Authentication, external services, and configuration

- No accounts, sessions, roles, or resource ownership exist.
- Default Rust binding is `127.0.0.1:8000`; Host and Origin guards reduce DNS
  rebinding/cross-origin browser attacks but are not authentication.
- External runtime dependencies are PyTorch, Torchaudio, TorchCodec, FFmpeg,
  and future local model/config files. There are no cloud APIs or services.
- Relevant environment variables: `STEMSPLAT_BIND_ADDR`,
  `STEMSPLAT_REPO_ROOT`, `STEMSPLAT_DATA_DIR`, `STEMSPLAT_UPLOAD_DIR`,
  `STEMSPLAT_MAX_UPLOAD_BYTES`, `STEMSPLAT_ALLOWED_HOSTS`,
  `STEMSPLAT_WORKER_TIMEOUT_SECONDS`, and `RUST_LOG`.
- CPU and CUDA Dockerfiles use a Rust build stage, non-root runtime user,
  healthcheck, pinned direct dependencies, FFmpeg, and persistent mount points.

### Major user flows traced

1. Browser selects/drops file -> multipart upload -> random staging file ->
   signature/size validation -> atomic track+import row creation -> background
   import -> SSE progress -> persisted recent-import card.
2. Import -> metadata -> key/BPM -> waveform -> Python worker -> seven validated
   24-bit WAV stems -> optional metadata flag -> completed track. The worker
   currently stops at the unimplemented model boundary.
3. Completed track -> library -> waveform/stem fetches -> queue add/reorder ->
   playback-state transitions -> download.
4. Settings/capabilities -> partial settings PATCH -> SQLite -> immutable
   per-import execution snapshot.
5. Legacy FastAPI upload -> task-scoped file -> serialized Python processing ->
   polling/SSE and downloads. Its root now identifies this as a headless legacy
   surface instead of serving an incompatible Rust UI.

## 2. Baseline and static bug hunt

The tracked baseline had one pipeline test. That test accepted empty files and
the model implementation cloned the full input into every named stem, so a
green test actively certified false output. The tracked browser depended on
runtime Google Fonts/Tailwind CDNs, called the legacy Python endpoints, parsed
SSE data as a bare integer even though the API emits JSON, and had no library,
settings, persistence, or resilient error states.

The worktree also contained an untracked Rust application, which was treated as
in scope and preserved rather than discarded. It already had broad architecture
but lacked reliable transaction/concurrency boundaries, durable progress,
complete validation, production disclosure, and regression depth.

Static searches covered TODO/FIXME/HACK/XXX markers, ignored/disabled tests,
broad catches, unsafe casts, promise/error handling, DOM sinks, file paths,
subprocesses, timers/listeners, database queries/indexes, settings mutations,
hard-coded environment assumptions, secrets, and dependency advisories. No
disabled Rust/Python tests or committed secrets were found. One FFmpeg-dependent
parameterized test skips only when FFmpeg is unavailable; FFmpeg 8.1 was present,
so all nine cases ran.

## 3. Testing performed

### Automated layers

- 49 Python unit/integration tests: CLI validation, model fail-closed behavior,
  deterministic PCM24 encoding, WAV fallbacks, nine compressed formats,
  streamed uploads, path/content/size validation, cleanup, task limits,
  concurrency serialization, downloads, errors, and security headers.
- 47 Rust unit/integration tests: audio analysis, DB rollback/recovery,
  placeholder invalidation, import slot races, immutable settings snapshots,
  hung-worker termination, failed-artifact cleanup, stem validation, metadata,
  playback/queue concurrency, multipart truncation, capability validation,
  path/error disclosure, and API status contracts.
- Static JS compilation and HTML validation.
- Strict Ruff, Ruff formatting, mypy including untyped function bodies, Rustfmt,
  Clippy with warnings denied, Python compileall, package build, pip check,
  Dockerfile lint, `git diff --check`, Bandit, detect-secrets, pip-audit, and
  cargo-audit.
- Optimized Rust workspace build from the final source.

### Input, boundary, state, and failure matrix

- Empty, zero-byte, wrong-extension, wrong-signature, path traversal, control
  characters, overlong names, Unicode names, duplicate names, malformed UUIDs,
  unknown JSON fields, malformed JSON, missing rows, negative/zero/out-of-range
  settings, unavailable GPU IDs, and unavailable metadata embedding.
- Upload byte limits, file-count limits, truncated multipart bodies, one/many
  queue items, queue boundary reorders, seek clamping, unknown duration, empty
  libraries/queues, and terminal/nonterminal imports.
- Repeated/rapid actions, concurrent queue additions, concurrent play calls,
  partial settings updates from separate clients, active-item deletion, restart
  during playback, restart with interrupted imports, refresh during/after import,
  and stale frontend responses.
- HTTP 400, 403, 404, 409, 413, 422, generic worker failure,
  malformed/closed SSE handling, hostile Host, hostile Origin, and network/UI
  retry states. No endpoint intentionally emits 401/429 because there is no
  authentication or rate limiter; that absence is a known risk.
- AAC, AIFF, FLAC, M4A, MP3, MP4, OGA, OGG, Opus, 16/24-bit WAV, stereo WAV,
  wrong content, and empty content. `.aif`/`.wave` share the tested AIFF/WAV
  container signatures.

### Browser and manual application testing

The final release binary was launched against a temporary persistent SQLite and
upload directory and exercised through the collaborative browser plus direct
HTTP clients.

- Initial/empty/error/loaded states; settings, capabilities, library, queue,
  playback state, recent imports, waveform/stem fallbacks, refresh, and restart.
- Valid WAV and MP3 uploads reached the worker and failed with the expected
  generic error because inference is unavailable; no server path leaked.
- Unicode `Final Ω.wav` and `Cleanup Probe.wav` persisted correctly.
- A stored filename containing `"><img ... onerror=...>` rendered literally;
  no image node appeared and no script executed.
- Recent import cards survived reload/restart, did not reconnect terminal SSE
  streams, and remained newest-first after an ordering regression was repaired.
- A post-fix failed upload left its DB error record but no copied upload or
  partial stem file. Historic failed artifacts were cleaned on restart.
- Rapid settings changes converged to the last value. Concurrent independent
  PATCHes preserved both fields. Settings and queue state survived restart.
- Thirty-two concurrent queue inserts retained unique contiguous positions;
  reorder/delete/play/pause/seek/stop, concurrent play, active delete, and stale
  restart state were exercised.
- A seeded completed track exercised waveform/stem/download and queue happy
  paths. It was not represented as evidence of real separation.
- Security headers, generic errors, Host/Origin rejection, invalid routes, and
  missing resources were checked live.
- At 375x667 and 768x1024 the page had no horizontal overflow. A prior 320px
  probe exposed a long-filename overflow; after repair the constrained app/card
  widths matched their scroll widths. A later preview resize to 320px timed out,
  so the final unusually-narrow check was DOM-constrained rather than a second
  native viewport capture.
- All interactive elements are native buttons/inputs/selects/links with names;
  the final DOM had zero unnamed controls, real `<progress>` elements, live
  regions, a labeled dropzone, and visible-focus CSS. The hidden preview did not
  advance real Tab focus reliably, so no assistive-technology certification is
  claimed.
- Browser snapshot and dedicated performance-trace capture were unavailable.
  DOM/resource/performance entries were inspected instead.

Single-route navigation means route-parameter and multi-page back/forward cases
are not applicable. Refresh/reload/restart persistence was tested directly.

## 4-8. Bugs, severity, causes, fixes, and regression coverage

| Severity | Bug | Root Cause | Fix | Regression Test | Status |
|---|---|---|---|---|---|
| P1 | Seven fake stems could be reported as successful | Placeholder model methods cloned the input | Model fails closed; prior placeholder completions/queue rows are invalidated | Python refusal; Rust one-time invalidation | Fixed; inference open |
| P1 | Upload traversal/overwrite and memory exhaustion | User filenames/storage paths were trusted and bodies were not safely bounded | Streamed random task/staging paths, signature checks, caps, safe cleanup | Python/Rust hostile filename, oversized, truncated tests | Fixed |
| P1 | Stored DOM XSS risk in server-controlled names/data | Dynamic API data could reach HTML-style rendering paths | DOM APIs/`textContent`; encoded URLs | Malicious persisted filename browser probe; sink scan | Fixed |
| P1 | GPU serialization could be bypassed mid-import | Admission and launch reread different settings | Immutable settings snapshot controls admission/execution | Admission-snapshot test | Fixed |
| P1 | Hung worker could block every GPU import forever | Unbounded child/stdout wait held the single slot | Configurable six-hour watchdog, child termination, bounded stderr | Hung-worker and timeout parser tests | Fixed |
| P1 | Queue/playback races created duplicate positions or ghost state | Multi-step DB/in-memory mutation lacked one boundary | Locks, transactions, unique positions, atomic delete/restart recovery | Concurrent add/play/delete/restart tests | Fixed |
| P1 | Multi-tab settings updates lost unrelated fields | Stale full objects plus racy read/modify/write | Dirty-field PATCHes, frontend serialization, server mutex/versioning | Concurrent partial settings test/browser probe | Fixed |
| P2 | Failed imports permanently consumed copied/partial files | Error/restart paths only changed DB status | Scoped cleanup after error and on startup; error remains visible | Cleanup scope test and live disk check | Fixed |
| P2 | Progress/cards disappeared after refresh/restart | Progress existed only in channel/UI memory | Persist/recover, recent-import API, replay/reconnect UI | DB recovery/SSE tests and browser restart | Fixed |
| P2 | Partial track/import/staging artifacts survived failures | Writes and multipart ownership were split | Atomic track+import transaction; pre-chunk cleanup ownership | DB rollback/truncated multipart tests | Fixed for one-file flow |
| P2 | Stereo duration was halved and title became a UUID | Frames divided by channels twice; storage basename treated as metadata | Preserve original title; use Hound duration | Metadata regression tests | Fixed |
| P2 | “24-bit” stems were silently 16-bit | TorchCodec ignored encoding request | Deterministic vectorized PCM24 WAV writer | Sample-width/roundtrip/property tests/benchmark | Fixed |
| P2 | Compressed worker decoding failed after install | TorchCodec/FFmpeg prerequisites absent | Pin TorchCodec, install FFmpeg, align extensions | Nine real decode tests | Fixed for worker |
| P2 | Duplicate uploads/CLI inputs collided | Shared filename paths/no normalized preflight | UUID directories and NFC/casefold collision rejection | Duplicate upload/CLI tests | Fixed |
| P2 | Paths/errors/download headers leaked or were unsafe | Direct DB/error serialization and title in header | Public DTOs, generic errors, allowlist, sanitized attachment, streaming | Leak/error/header/download tests | Fixed |
| P2 | CPU-only systems retained GPU mode and used one thread | GPU-on default was only masked by UI | Startup reconciles persisted settings to capabilities | Capability reconciliation test | Fixed |
| P2 | Legacy API served a frontend for different routes | Shared HTML targeted only Rust `/api/*` | Explicit headless JSON landing and `/docs` pointer | Root contract test/live Uvicorn probe | Fixed |
| P3 | UI had stale races, duplicate actions, phantom loading, weak errors | Missing versions/guards/settled refresh; brittle SSE | Versions, locks, retries, validation, watchdog/reconnect, explicit states | Browser failure/rapid/reload tests | Fixed |
| P3 | Long filenames overflowed narrow layouts | Flex/grid intrinsic minimum widths | Bounded items, `min-width: 0`, nonshrinking badge | 320-constrained and 375/768 probes | Fixed |
| P3 | Upload/progress controls lacked robust semantics | Clickable container/div progress/weak status semantics | Native button/progress/labels/live regions/focus styles | HTML validation/DOM audit | Fixed; AT pass open |
| P3 | Persisted imports reversed order after reload | Newest-first results were prepended forward | Reverse iteration while prepending | Browser reload newest-first | Fixed |
| P3 | Recent import refresh scanned/sorted full history | CASE order defeated timestamp index | Matching expression index | Query plan uses priority index | Fixed |
| P3 | Production UI used CDNs/container served incomplete surface | Runtime Tailwind/fonts and legacy entrypoint | Self-contained page; multistage non-root Rust images | Resource/build/lint checks | Fixed statically |
| P1 | Real model inference is absent | No adapters/configs/checkpoints; methods intentionally raise | None; fail closed and release warning | Fail-closed tests | **Open release blocker** |
| P1 | Playback emits no audio | Crate only mutates state; no sink/decoder/audio element | UI now explicitly says state-only | State tests/disclosure check | **Open if playback ships** |
| P1 conditional | No auth/authorization | Local skeleton has no identity/ownership | Loopback default, Host/Origin guard, warning | Security boundary tests | **Open; block exposure** |
| P2 | Compressed tracks lack Rust metadata/key/BPM/waveform | Those readers are WAV-only; waveform error is nonfatal | Limitation documented | Worker decode tests do not cover Rust analysis | **Open format blocker** |
| P2 | Three advisories remain in two Python packages | Compatibility blocks fixed Torch/setuptools versions | Scope documented; vulnerable APIs unused now | `pip-audit` | Open/upstream constrained |
| P2 | Legacy tasks are restart-volatile | In-memory Python task dictionary | Live cap/TTL/cleanup only | Task lifecycle tests | Open |
| P2 | Multi-file Rust import is non-atomic/body cap conflicts | Sequential finalize/spawn; one-file-sized router limit | UI posts one file/request; remaining staging cleans | Multipart tests | Open for direct clients |
| P2/P3 | Large library causes unbounded list plus `2N` requests | No pagination/lazy loading | Recent imports only are capped/indexed | Load/resource inspection | Open scalability risk |
| P3 | No cancel endpoint; timeout is coarse | Watchdog exists but no user cancellation | Six-hour bound | Watchdog test | Open UX gap |
| P3 | No successful-track deletion/retention | Skeleton only deletes queue rows | Failed files now auto-clean | Cleanup tests | Open lifecycle gap |
| P3 | No frontend E2E, CI, or quality harness | Repository began as skeleton | Backend suites/manual browser expanded | 96 tests/browser evidence | Open process risk |

## 9. Performance and reliability sweep

### Measured results

- Final release binary: about 6.8 MiB; self-contained HTML: 40 KiB.
- Concurrent release-binary smoke load:
  - `/`: 1,000 requests, concurrency 20, 0 failed, 4,354.45 req/s,
    4.593 ms mean request time.
  - `/api/tracks`: 1,000 requests, concurrency 20, 0 failed, 4,434.39
    req/s, 4.510 ms mean request time.
  - These ran concurrently and are not a controlled comparative benchmark.
- Recent-import query before its expression index: full scan plus temporary
  B-tree sort. After migration: `SCAN i USING INDEX
  idx_imports_priority_started`, with no temporary sort.
- Deterministic 1,000,000-sample PCM24 packing microbenchmark:
  - scalar reference: 0.529144 seconds
  - vectorized implementation: 0.002085 seconds
  - encoder-only speedup: 253.8x
  - 3,000,000 bytes, byte-for-byte identical

The speed claim applies only to PCM byte packing, not separation. The property
test uses a deterministic random seed and checks scalar/vector identity.

### Remaining performance risks

- Full track and queue listings are unpaginated.
- The browser loads waveform plus stems for every completed track (`2N` API
  calls) and retains all corresponding DOM/data.
- Imports have no explicit admission queue cap or user cancellation.
- The watchdog bounds hangs, but six hours can still retain a GPU slot by design.
- No bundle profiler is applicable to the 40 KiB dependency-free page. A Core
  Web Vitals trace was unavailable in the collaborative browser.

## 10. Production-specific checks

- Optimized Rust workspace build passes with the locked graph.
- Host/Origin checks, CSP, no-store, nosniff, frame denial, referrer and
  permissions policies cover success and error responses.
- CSP needs `'unsafe-inline'` because CSS/JS are in one HTML file. No dynamic
  HTML/eval sinks remain; externalizing assets would permit a stricter policy.
- HSTS/TLS, identity, rate limiting, quotas, and proxy headers are not built in.
- Dockerfiles are statically valid, use the Rust server, add FFmpeg, run
  non-root, and name persistent paths. Docker was not running, so images were
  not built or started.
- The host Homebrew Rust toolchain was broken by an external LLVM/Z3 dylib
  mismatch. Validation used independent rustup stable 1.94.1. Docker specifies
  Rust 1.95, but the container build could not run.
- The branch is 8 commits ahead and 177 behind `origin/main`. Upstream
  integration risk was not mutated during QA.

## 11. Commands executed

Representative exact commands follow. HTTP calls were repeated with valid,
malformed, missing, hostile, concurrent, and restart-state payloads.

### Python/dependencies/package

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
uvx ruff check stemrunner
uvx ruff format --check stemrunner
uvx mypy --python-executable .venv/bin/python --check-untyped-defs stemrunner
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m compileall -q stemrunner
.venv/bin/python -m pip wheel --no-deps --wheel-dir <temporary-directory> .
uv pip compile requirements.txt --python-version 3.10 --python-platform linux
.venv/bin/pip-audit -r requirements.txt
.venv/bin/pip-audit -r requirements.txt --format json
uvx bandit -q -r stemrunner -x stemrunner/tests
uvx detect-secrets scan stemrunner pyproject.toml requirements.txt Dockerfile.cpu Dockerfile.gpu web/index.html
.venv/bin/python -m pytest -q -p no:cacheprovider stemrunner/tests/test_pipeline.py::test_runtime_decodes_supported_compressed_audio
```

### Rust

```bash
PATH=/Users/skylarenns/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin cargo +stable fmt --all -- --check
PATH=/Users/skylarenns/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin cargo +stable test --workspace --locked
PATH=/Users/skylarenns/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin cargo +stable clippy --workspace --all-targets --locked -- -D warnings
PATH=/Users/skylarenns/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin cargo +stable build --workspace --release --locked
PATH=/Users/skylarenns/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin cargo audit --file Cargo.lock
```

### Frontend, deployment, database, and live runtime

```bash
node -e '<extract inline script and compile with vm.Script>'
npx --yes html-validate@10.11.1 web/index.html
npx --yes dockerfilelint Dockerfile.cpu Dockerfile.gpu
git diff --check
docker info
ffmpeg -version
sqlite3 <db> '.indexes'
sqlite3 <db> 'EXPLAIN QUERY PLAN SELECT ... recent imports ...'
curl -F 'file=@...;filename=Final Ω.wav;type=audio/wav' http://127.0.0.1:8000/api/import
curl -N --max-time 10 http://127.0.0.1:8000/api/imports/<id>/events
curl -H 'Host: attacker.example' http://127.0.0.1:8000/api/tracks
curl -X POST -H 'Origin: https://attacker.example' http://127.0.0.1:8000/api/playback/stop
ab -n 1000 -c 20 http://127.0.0.1:8000/
ab -n 1000 -c 20 http://127.0.0.1:8000/api/tracks
PYTHONPATH=. .venv/bin/uvicorn stemrunner.server:app --host 127.0.0.1 --port 8001
```

The release server was repeatedly restarted against the same temporary database
to validate migrations, recovery, persistence, stale playback reset, import
restoration, and cleanup.

## 12. Files changed

The worktree was already dirty and the complete `rust/` tree was untracked at
audit start. This list describes the current audited/modified surface, not an
ownership claim over unrelated pre-existing changes. `.DS_Store` and
`agents.md` were left untouched.

- Root/build/docs: `.gitignore`, `.dockerignore`, `Dockerfile.cpu`,
  `Dockerfile.gpu`, `README.md`, `pyproject.toml`, `requirements.txt`, and this
  `QA_AUDIT_REPORT.md`.
- Python: `stemrunner/__main__.py`, `cli.py`, `models.py`, `pipeline.py`,
  `server.py`, `ml_worker.py`, and all three files in `stemrunner/tests/`.
- Frontend: `web/index.html`.
- Rust: `rust/Cargo.toml`, `rust/Cargo.lock`, and source/Cargo manifests for
  `stemsplat-core`, `stemsplat-db`, `stemsplat-import`, `stemsplat-metadata`,
  `stemsplat-audio-analysis`, `stemsplat-waveform`, `stemsplat-queue`,
  `stemsplat-playback`, and `stemsplat-server`.

## 13. Remaining known issues and future investigation

### Release blockers

1. Implement real, versioned model adapters and obtain legal/verified model and
   config assets. Do not remove fail-closed behavior until quality gates pass.
2. Add actual audio output/transport, or remove playback from release scope.
3. Add authentication, authorization, per-user ownership, upload quotas/rate
   limiting, and proxy-aware secure deployment before exposure.
4. Decode compressed inputs for Rust metadata/key/BPM/waveform, or restrict the
   primary server to its fully supported formats.
5. Create the required reference/turbo quality harness with representative
   audio, deterministic metrics, drift thresholds, and promotion tests.

### High-value follow-up

- Resolve PyTorch/Torchaudio against a release containing the
  `PYSEC-2025-194` fix, and setuptools against 83+ once Torch allows it. Current
  code does not call `torch.jit.script` or load user-controlled model objects;
  the setuptools advisory concerns Unicode normalization in sdist exclusions.
- Persist or remove the legacy Python task surface; its in-memory registry loses
  status on restart.
- Define atomic semantics for direct multi-file clients and align whole-request
  and per-file count/size limits.
- Add track/queue pagination, lazy waveform/stem loading, successful-track
  deletion, import retry/cancel, storage quotas, and retention policies.
- Add frontend component/E2E tests, real keyboard/screen-reader/mobile passes,
  Docker runtime tests, CI, migration fixtures, and controlled Web Vitals/load
  benchmarks.
- Reconcile the branch with 177 upstream commits in a clean worktree and repeat
  the full suite after integration.

## Areas that could not be fully tested

- Real stem quality, model loading, CUDA inference, long separation, and drift:
  adapters/checkpoints/configs/harness are absent.
- Actual audio playback: no playback engine exists.
- Docker image build/container health: Docker daemon unavailable.
- NVIDIA CUDA: this host exposed Apple MPS, not CUDA.
- Real multi-user/auth/session/permission cases: those systems do not exist.
- Physical devices, assistive technology, and full keyboard traversal: browser
  snapshot/focus/resize automation was partially unavailable. Semantic and
  responsive DOM checks were completed.
- Offline/slow/throttled traces/Core Web Vitals: no trace-capable attachment.
  Explicit API/SSE failures and resource counts were used instead.
- Upstream integration/deployment: no rebase, commit, push, publish, or deploy
  was authorized or performed.

## Final validation results

| Validation | Final result |
|---|---|
| Python tests | **PASS — 49 passed** |
| Rust tests | **PASS — 47 passed** |
| Ruff lint/format | **PASS** |
| mypy including untyped bodies | **PASS — 10 source files** |
| Rustfmt | **PASS** |
| Clippy warnings denied | **PASS** |
| Python compileall/pip check | **PASS** |
| Python wheel build/contents | **PASS — package and web asset included** |
| Linux/Python 3.10 dependency resolution | **PASS** |
| Frontend JS syntax | **PASS** |
| HTML validation | **PASS** |
| Dockerfile lint | **PASS** |
| Rust release build | **PASS** |
| cargo-audit | **PASS — no known Rust advisories** |
| Bandit | **PASS** |
| detect-secrets | **PASS — no findings** |
| pip-audit | **FAIL — 3 records in 2 packages (Torch/setuptools)** |
| Docker image build/runtime | **NOT RUN — daemon unavailable** |
| Quality/drift harness | **NOT RUN — harness/model absent** |
| Release readiness | **BLOCKED — inference, playback scope, access control** |

All final code/build/test checks are green except for the explicitly recorded
dependency advisories and unavailable environment/feature gates. Green tooling
does not override the release blockers above.
