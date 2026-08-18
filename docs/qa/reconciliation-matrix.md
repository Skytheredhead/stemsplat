# v0.4.3 audit reconciliation matrix

Base: `origin/main` at `63ab006181af19d8fedb76a19608675e96647930`
Lane: `reference_safe`
Owner: Stemsplat maintainers unless a different owner is named below.

The old audit was produced from a repository with no Git ancestor in common
with the released Mac app. No Rust or Python implementation from that checkout
is cherry-picked. This matrix maps the report's substantive findings to the
v0.4.3 product.

| Legacy finding | Disposition | v0.4.3 implementation/evidence | Owner/status |
|---|---|---|---|
| Placeholder inference and missing adapters | Already implemented | v0.4.2 has real Demucs, MDX, BS-Roformer, and Mel-Roformer adapters. Real-model drift is governed by `quality.py`; fake-inference tests are not quality evidence. | Release owner; quality run required |
| Playback is state-only | Not applicable | v0.4.2 does not ship or advertise a playback engine. | Closed |
| No authentication/authorization | Port | LAN is disabled by default, isolated on its own HTTPS app/listener, passcode sessions are server-side, and local-only routes are excluded from the LAN router. | Security owner; automated tests included |
| Upload traversal, overwrite, memory exhaustion | Port | `stemsplat/uploads.py` enforces normalized display names, UUID paths, byte/staging/concurrency/task caps, signatures, and media probing. | Backend owner; automated tests included |
| Stored DOM XSS in names/API data | Port | Browser storage is decoded defensively; new shared UI helpers use DOM/text APIs. Remaining legacy sinks are tracked by the frontend lint gate. | Frontend owner; E2E/manual pass required |
| Mutable execution settings / GPU serialization race | Already implemented | v0.4.2 task records already snapshot execution settings and serialize the worker. SQLite persists the immutable snapshot. | Backend owner; regression tests |
| Hung worker blocks the queue | Already implemented | v0.4.2 has watchdog, cancellation, timeout, and subprocess cleanup logic. | Closed; existing tests |
| Queue/task state races and restart loss | Port | `stemsplat/state.py` uses SQLite WAL, foreign keys, migrations, transactions, and interrupted-state recovery. | Backend owner; recovery tests included |
| Multi-client settings overwrite | Port | Versioned partial `PATCH /api/settings`; legacy `/settings` delegates to the same settings service. | Backend/frontend owner |
| Failed artifact cleanup | Already implemented + port | v0.4.2 scoped work cleanup remains; upload admission now cleans every rejection/cancellation and startup cleanup is scope-checked. | Backend owner |
| Refresh/restart loses task cards | Port | Server task/history APIs are authoritative and paginated; browser storage is presentation-only. | Backend/frontend owner; E2E required |
| Partial multi-file admission | Port | Local batches are validated and staged before any task is registered. | Backend owner; batch rollback tests |
| Audio metadata/duration/title bugs | Already implemented | v0.4.2 probes real media, retains original display names, and its expanded battery covers mono/stereo/sample-rate/duration behavior. | Closed; existing tests |
| 24-bit output claim was false | Not applicable | v0.4.2 intentionally defaults to its existing 16-bit output semantics. v0.4.3 does not relabel or change them. | Closed |
| Compressed decode prerequisites | Already implemented | v0.4.2 uses packaged FFmpeg/ffprobe and covers WAV, VBR MP3, and compressed decode paths. | Packaging owner; clean-install test required |
| Duplicate upload collision | Port | UUID-only storage plus NFC/casefold duplicate detection for batches. | Backend owner |
| Path/error/header disclosure | Port | New DTO and deletion paths expose opaque IDs/display names, never absolute storage paths or internal exception text. | Security owner |
| CPU/GPU setting reconciliation | Not applicable | Supported v0.4.3 target is Apple Silicon MPS only; unsupported Docker release definitions are removed. | Closed |
| Legacy API served mismatched frontend | Not applicable | v0.4.2's FastAPI app and packaged desktop/mobile UI are one product. Missing UI assets now fail startup instead of serving an embedded fallback. | Closed |
| UI races, weak SSE recovery, phantom loading | Port | Shared storage/API/SSE helpers and authoritative server state are introduced without a framework rewrite. | Frontend owner; Playwright required |
| Narrow-layout filename overflow | Already implemented | Preserve v0.4.2 rendered behavior; capture/recheck 320, 375, 768, and 1440 px. | Frontend owner; screenshot evidence required |
| Upload/progress accessibility | Port | Keyboard/focus behavior remains a release checklist and E2E requirement for custom controls. | Frontend owner; manual AT signoff open |
| Import ordering and unbounded library queries | Port | Cursor-based task/history APIs return stable newest-first pages. | Backend owner; pagination tests |
| Runtime CDN dependencies | Port | Remote fonts and runtime Tailwind are removed; all executable and style resources are packaged same-origin. | Frontend/packaging owner |
| Dependency advisories | Port | Reproducible ARM64 lock inputs, `pip-audit`, secret scan, and SBOM jobs are release gates. Exceptions require advisory, scope, owner, and expiry. | Release owner |
| Model supply-chain trust | New v0.4.3 blocker | Manifest revisions, lengths, hashes, configs, loader types, SPDX identifiers, and attribution are mandatory before automatic download/load. | Model owner; real hashes/licenses required |
| Quality/drift harness absent | Port | Root `quality.py` records provenance, structural/audio drift, public metrics, MPS benchmarks, and promotion reports. | Quality owner; corpora/model run required |
| CI/frontend E2E absent | Port | GitHub Actions and Playwright smoke projects cover unit/static/resource/offline/security flows. | Release owner |
| Ad-hoc packaging | Port + external prerequisite | Version source and verification scripts are unified. Developer ID signing/notarization/stapling remain hard credentialed release gates. | Release owner; credentials required |
| Rust server/compiler experiments | Deferred experimental | May be retained outside the release branch only under `experimental/compiler_lab`; never a normal dependency or default path. | Research owner |

## Promotion rule

A row is complete only when its implementation and named test/evidence exist.
Documentation alone is not completion. Model license/hash, real MPS quality,
private listening, and Apple notarization rows are hard release prerequisites
and cannot be waived by an ad-hoc development build.
