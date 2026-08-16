#!/usr/bin/env python3
"""Reference-safe quality and performance evidence for Stemsplat releases.

Audio and licensed corpora remain outside Git. Reports contain paths relative to
the supplied roots, content hashes, aggregate metrics, and provenance only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf

from stemsplat.atomic import atomic_write_json
from stemsplat.version import __version__

SUPPORTED_AUDIO = {".wav", ".wave", ".flac", ".aif", ".aiff", ".mp3", ".m4a", ".ogg", ".opus"}
REFERENCE_SAFE_GATES = {
    "median_si_sdr_db": 50.0,
    "minimum_si_sdr_db": 40.0,
    "maximum_loudness_delta_db": 0.1,
    "maximum_peak_delta_db": 0.1,
    "maximum_public_median_sdr_degradation_db": 0.10,
    "maximum_public_fifth_percentile_degradation_db": 0.25,
    "maximum_runtime_regression_percent": 5.0,
    "maximum_memory_regression_percent": 10.0,
}
REQUIRED_PUBLIC_TARGETS = {"vocals", "accompaniment", "drums", "bass", "other", "guitar", "piano"}
REQUIRED_PUBLIC_CORPORA = {"musdb18-hq", "slakh2100-tiny"}
REQUIRED_PRIVATE_TAGS = {
    "vocals",
    "instrumental",
    "dense_mix",
    "acoustic",
    "bass",
    "drums",
    "guitar",
    "piano",
    "noisy",
    "denoise",
}


class QualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioMetric:
    relative_path: str
    sha256: str
    decoded_waveform_sha256: str
    frames: int
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    integrated_loudness_lufs: float
    finite: bool
    silent: bool
    clipped: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _ffmpeg_executable() -> str:
    direct = shutil.which("ffmpeg")
    if direct:
        return direct
    try:
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise QualityError("FFmpeg is required for BS.1770 integrated loudness measurements") from exc
    if not candidate:
        raise QualityError("FFmpeg is required for BS.1770 integrated loudness measurements")
    return candidate


def _integrated_loudness_lufs(path: Path) -> float:
    result = subprocess.run(
        [
            _ffmpeg_executable(),
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-filter_complex",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise QualityError(f"FFmpeg loudness measurement failed for {path.name}")
    matches = re.findall(r"^\s*I:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+LUFS\s*$", result.stderr, flags=re.MULTILINE)
    if not matches:
        raise QualityError(f"FFmpeg did not return integrated loudness for {path.name}")
    value = float(matches[-1])
    if not math.isfinite(value):
        return -120.0
    return value


def _audio_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO and not path.is_symlink()
    )


def measure_audio(path: Path, root: Path) -> AudioMetric:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    finite = bool(np.isfinite(data).all())
    absolute = np.abs(data) if data.size else np.asarray([], dtype=np.float32)
    peak = float(np.max(absolute)) if absolute.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)))) if data.size else 0.0
    waveform_digest = hashlib.sha256()
    waveform_digest.update(int(sample_rate).to_bytes(8, "little", signed=False))
    waveform_digest.update(int(data.shape[1]).to_bytes(4, "little", signed=False))
    waveform_digest.update(np.ascontiguousarray(data, dtype="<f4").tobytes())
    return AudioMetric(
        relative_path=path.relative_to(root).as_posix(),
        sha256=_sha256(path),
        decoded_waveform_sha256=waveform_digest.hexdigest(),
        frames=int(data.shape[0]),
        sample_rate=int(sample_rate),
        channels=int(data.shape[1]),
        peak_dbfs=_db(peak),
        rms_dbfs=_db(rms),
        integrated_loudness_lufs=_integrated_loudness_lufs(path),
        finite=finite,
        silent=not data.size or peak < 1e-8,
        clipped=bool(peak >= 1.0 - 1e-7),
    )


def inventory(root: Path) -> list[AudioMetric]:
    if not root.is_dir():
        raise QualityError(f"audio root does not exist: {root}")
    files = _audio_files(root)
    if not files:
        raise QualityError(f"no supported audio files found under {root}")
    return [measure_audio(path, root) for path in files]


def _provenance() -> dict[str, Any]:
    return {
        "stemsplat_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "lane": "reference_safe",
        "created_at": int(time.time()),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _validated_hashes(values: Sequence[str] | None, label: str) -> list[str]:
    hashes = sorted({str(value).strip().lower() for value in values or []})
    if not hashes:
        raise QualityError(f"at least one {label} SHA-256 is required")
    for digest in hashes:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise QualityError(f"invalid {label} SHA-256: {digest}")
    return hashes


def command_prepare(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise QualityError(
            f"corpus manifest missing: {manifest_path}. Copy quality/corpus.example.json and use external paths."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("corpora") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise QualityError("corpus manifest must contain a non-empty 'corpora' list")
    checked: list[dict[str, Any]] = []
    public_corpora: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise QualityError("corpus entries must be objects")
        name = str(entry.get("name") or "").strip()
        root = Path(str(entry.get("path") or "")).expanduser().resolve()
        license_name = str(entry.get("license") or "").strip()
        if not name or not license_name:
            raise QualityError("every corpus requires a name and license/provenance label")
        files = _audio_files(root) if root.is_dir() else []
        if not files:
            raise QualityError(f"corpus '{name}' has no readable audio under {root}")
        checked_entry: dict[str, Any] = {"name": name, "license": license_name, "files": len(files)}
        if bool(entry.get("private")):
            excerpts = entry.get("excerpts")
            if not isinstance(excerpts, list) or len(excerpts) < 12:
                raise QualityError("the private corpus requires at least 12 excerpt hash/tag records")
            available_hashes = {_sha256(path) for path in files}
            declared_hashes: set[str] = set()
            declared_tags: set[str] = set()
            sanitized_excerpts: list[dict[str, Any]] = []
            for excerpt in excerpts:
                if not isinstance(excerpt, dict):
                    raise QualityError("private excerpt records must be objects")
                digest = str(excerpt.get("sha256") or "").strip().lower()
                tags = sorted({str(tag).strip() for tag in excerpt.get("tags", []) if str(tag).strip()})
                if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest) or not tags:
                    raise QualityError("private excerpts require a SHA-256 and at least one tag")
                declared_hashes.add(digest)
                declared_tags.update(tags)
                sanitized_excerpts.append({"sha256": digest, "tags": tags})
            if not declared_hashes.issubset(available_hashes):
                raise QualityError("private excerpt hashes do not match the external corpus")
            missing_tags = sorted(REQUIRED_PRIVATE_TAGS - declared_tags)
            if missing_tags:
                raise QualityError("private corpus is missing required tags: " + ", ".join(missing_tags))
            checked_entry["excerpts"] = sanitized_excerpts
        else:
            public_corpora.add(re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"))
        checked.append(checked_entry)
    missing_public = sorted(REQUIRED_PUBLIC_CORPORA - public_corpora)
    if missing_public:
        raise QualityError("corpus manifest is missing required public corpora: " + ", ".join(missing_public))
    report = {"command": "prepare", "provenance": _provenance(), "corpora": checked, "ready": True}
    if args.output:
        _write_report(Path(args.output), report)
    print(json.dumps(report, indent=2))
    return 0


def command_baseline(args: argparse.Namespace) -> int:
    root = Path(args.reference_dir).expanduser().resolve()
    checkpoint_hashes = _validated_hashes(args.checkpoint_sha256, "checkpoint")
    config_hashes = _validated_hashes(args.config_sha256, "config")
    device_id = str(args.device_id or "").strip()
    if not device_id:
        raise QualityError("an exact device identifier is required")
    metrics = inventory(root)
    failures = [
        f"invalid reference signal: {item.relative_path}"
        for item in metrics
        if not item.finite or item.silent or item.clipped or item.frames <= 0
    ]
    report = {
        "schema_version": 1,
        "command": "baseline",
        "provenance": _provenance(),
        "checkpoint_hashes": checkpoint_hashes,
        "config_hashes": config_hashes,
        "device_id": device_id,
        "passed": not failures,
        "failures": failures,
        "audio": [asdict(item) for item in metrics],
        "audio_count": len(metrics),
        "audio_root_hash": hashlib.sha256(
            "\n".join(f"{item.relative_path}:{item.sha256}" for item in metrics).encode("utf-8")
        ).hexdigest(),
    }
    _write_report(Path(args.output), report)
    print(f"recorded {len(metrics)} reference outputs in {args.output}")
    return 0 if report["passed"] else 2


def _si_sdr(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference64 = reference.astype(np.float64, copy=False).reshape(-1)
    candidate64 = candidate.astype(np.float64, copy=False).reshape(-1)
    reference64 -= np.mean(reference64)
    candidate64 -= np.mean(candidate64)
    reference_energy = float(np.dot(reference64, reference64))
    if reference_energy <= 1e-20:
        return float("inf") if np.max(np.abs(candidate64)) <= 1e-10 else float("-inf")
    scale = float(np.dot(candidate64, reference64)) / reference_energy
    target = scale * reference64
    noise = candidate64 - target
    return 10.0 * math.log10(max(float(np.dot(target, target)), 1e-20) / max(float(np.dot(noise, noise)), 1e-20))


def _fifth_percentile(values: Sequence[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 5))


def command_compare(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline).expanduser().resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reference_root = Path(args.reference_dir).expanduser().resolve()
    candidate_root = Path(args.candidate_dir).expanduser().resolve()
    expected = {item["relative_path"]: item for item in baseline.get("audio", [])}
    candidate_paths = {path.relative_to(candidate_root).as_posix(): path for path in _audio_files(candidate_root)}
    failures: list[str] = []
    checkpoint_hashes = _validated_hashes(args.checkpoint_sha256, "checkpoint")
    config_hashes = _validated_hashes(args.config_sha256, "config")
    device_id = str(args.device_id or "").strip()
    if not device_id:
        raise QualityError("an exact device identifier is required")
    if checkpoint_hashes != baseline.get("checkpoint_hashes"):
        failures.append("checkpoint hashes differ from baseline")
    if config_hashes != baseline.get("config_hashes"):
        failures.append("config hashes differ from baseline")
    if device_id != str(baseline.get("device_id") or ""):
        failures.append("device identifier differs from baseline")
    rows: list[dict[str, Any]] = []
    if set(expected) != set(candidate_paths):
        failures.append("stem paths/count differ from baseline")
    for relative_path, reference_metric in expected.items():
        candidate_path = candidate_paths.get(relative_path)
        reference_path = reference_root / relative_path
        if candidate_path is None or not reference_path.is_file():
            failures.append(f"missing output: {relative_path}")
            continue
        reference, reference_rate = sf.read(reference_path, always_2d=True, dtype="float32")
        candidate, candidate_rate = sf.read(candidate_path, always_2d=True, dtype="float32")
        candidate_metric = measure_audio(candidate_path, candidate_root)
        structural = {
            "frames_match": int(reference.shape[0]) == int(candidate.shape[0]),
            "sample_rate_match": int(reference_rate) == int(candidate_rate),
            "channels_match": int(reference.shape[1]) == int(candidate.shape[1]),
        }
        if not all(structural.values()):
            failures.append(f"audio structure differs: {relative_path}")
            si_sdr = float("-inf")
        else:
            si_sdr = _si_sdr(reference, candidate)
        peak_delta = abs(float(candidate_metric.peak_dbfs) - float(reference_metric["peak_dbfs"]))
        reference_loudness = reference_metric.get("integrated_loudness_lufs")
        if reference_loudness is None:
            failures.append(f"baseline integrated loudness is missing: {relative_path}")
            loudness_delta = float("inf")
        else:
            loudness_delta = abs(float(candidate_metric.integrated_loudness_lufs) - float(reference_loudness))
        if not candidate_metric.finite or candidate_metric.silent or candidate_metric.clipped:
            failures.append(f"invalid signal: {relative_path}")
        rows.append(
            {
                "relative_path": relative_path,
                **structural,
                "si_sdr_db": si_sdr,
                "peak_delta_db": peak_delta,
                "loudness_delta_db": loudness_delta,
                "finite": candidate_metric.finite,
                "silent": candidate_metric.silent,
                "clipped": candidate_metric.clipped,
            }
        )
    si_sdr_values = [float(row["si_sdr_db"]) for row in rows]
    median_si_sdr = statistics.median(si_sdr_values) if si_sdr_values else float("-inf")
    min_si_sdr = min(si_sdr_values, default=float("-inf"))
    max_peak_delta = max((float(row["peak_delta_db"]) for row in rows), default=float("inf"))
    max_loudness_delta = max((float(row["loudness_delta_db"]) for row in rows), default=float("inf"))
    gate_results = {
        "median_si_sdr": median_si_sdr >= REFERENCE_SAFE_GATES["median_si_sdr_db"],
        "minimum_si_sdr": min_si_sdr >= REFERENCE_SAFE_GATES["minimum_si_sdr_db"],
        "peak_delta": max_peak_delta <= REFERENCE_SAFE_GATES["maximum_peak_delta_db"],
        "loudness_delta": max_loudness_delta <= REFERENCE_SAFE_GATES["maximum_loudness_delta_db"],
        "zero_failures": not failures,
    }
    report = {
        "schema_version": 1,
        "command": "compare",
        "provenance": _provenance(),
        "checkpoint_hashes": checkpoint_hashes,
        "config_hashes": config_hashes,
        "device_id": device_id,
        "gates": REFERENCE_SAFE_GATES,
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
        "summary": {
            "outputs": len(rows),
            "median_si_sdr_db": median_si_sdr,
            "minimum_si_sdr_db": min_si_sdr,
            "maximum_peak_delta_db": max_peak_delta,
            "maximum_loudness_delta_db": max_loudness_delta,
        },
        "failures": sorted(set(failures)),
        "audio": rows,
    }
    _write_report(Path(args.output), report)
    print(json.dumps({"passed": report["passed"], **report["summary"], "failures": report["failures"]}, indent=2))
    return 0 if report["passed"] else 2


def _peak_rss_bytes(process: subprocess.Popen[bytes]) -> int:
    peak = 0
    try:
        import psutil

        try:
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                with monitored.oneshot():
                    peak = max(peak, int(monitored.memory_info().rss))
                    for child in monitored.children(recursive=True):
                        try:
                            peak = max(peak, int(child.memory_info().rss))
                        except psutil.Error:
                            continue
                time.sleep(0.05)
        except psutil.Error:
            pass
    except ImportError:
        pass
    process.wait()
    return peak


def _benchmark_trial(command: Sequence[str], environment: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, env=environment)
        peak_rss = _peak_rss_bytes(process)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    elapsed = time.perf_counter() - started
    return {
        "returncode": process.returncode,
        "wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "peak_mps_bytes": None,
        "stdout_tail": stdout.decode("utf-8", "replace")[-2000:],
        "stderr_tail": stderr.decode("utf-8", "replace")[-2000:],
    }


def command_benchmark(args: argparse.Namespace) -> int:
    commands = {
        "cold": shlex.split(args.cold_command),
        "warm": shlex.split(args.warm_command),
    }
    if not all(commands.values()):
        raise QualityError("cold and warm benchmark commands are required")
    environment = dict(os.environ)
    environment["STEMSPLAT_QUALITY_LANE"] = "reference_safe"
    trials: list[dict[str, Any]] = []
    for kind, count in (("cold", args.cold), ("warm", args.warm)):
        for index in range(count):
            trial = _benchmark_trial(commands[kind], environment)
            trial.update({"kind": kind, "index": index + 1})
            trials.append(trial)
            if trial["returncode"] != 0:
                break
    passed = all(trial["returncode"] == 0 for trial in trials)
    summary: dict[str, Any] = {}
    for kind in ("cold", "warm"):
        matching = [trial for trial in trials if trial["kind"] == kind and trial["returncode"] == 0]
        summary[kind] = {
            "trials": len(matching),
            "median_wall_seconds": statistics.median([trial["wall_seconds"] for trial in matching]) if matching else None,
            "median_peak_rss_bytes": statistics.median([trial["peak_rss_bytes"] for trial in matching]) if matching else None,
            "peak_mps_bytes": int(getattr(args, f"{kind}_peak_mps_bytes")),
            "output_audio_seconds": float(args.audio_seconds) if args.audio_seconds else None,
            "median_realtime_factor": (
                float(args.audio_seconds) / statistics.median([trial["wall_seconds"] for trial in matching])
                if matching and args.audio_seconds
                else None
            ),
        }
    report = {
        "schema_version": 1,
        "command": "benchmark",
        "provenance": _provenance(),
        "role": args.role,
        "mode": args.mode,
        "device_id": args.device_id,
        "invocations": commands,
        "passed": passed,
        "summary": summary,
        "trials": trials,
        "note": "Cold and warm peak MPS values are mandatory and must come from the measured worker's torch.mps allocation counter.",
    }
    _write_report(Path(args.output), report)
    print(json.dumps({"passed": passed, "summary": summary}, indent=2))
    return 0 if passed else 2


def _validate_public_metrics(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("metric") or "").lower() not in {"bsseval_v4", "museval_bsseval_v4"}:
        raise QualityError("public metrics must identify museval BSSEval v4")
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise QualityError("public metrics require a targets object")
    missing = sorted(REQUIRED_PUBLIC_TARGETS - set(targets))
    failures: list[str] = []
    tool_name = str(payload.get("source_tool") or "").strip().lower()
    tool_version = str(payload.get("museval_version") or "").strip()
    corpora = {str(value).strip().lower() for value in payload.get("corpora", []) if str(value).strip()}
    if tool_name != "museval" or not tool_version:
        failures.append("public metrics require the exact museval version")
    missing_corpora = sorted(REQUIRED_PUBLIC_CORPORA - corpora)
    if missing_corpora:
        failures.append("public corpora missing: " + ", ".join(missing_corpora))
    if missing:
        failures.append("public targets missing: " + ", ".join(missing))
    if payload.get("failed_tracks"):
        failures.append("public corpus contains failed tracks")
    rows: dict[str, Any] = {}
    for target, values in sorted(targets.items()):
        if not isinstance(values, dict):
            failures.append(f"invalid public metrics for {target}")
            continue
        try:
            baseline = [float(value) for value in values.get("baseline_sdr_db", [])]
            candidate = [float(value) for value in values.get("candidate_sdr_db", [])]
        except (TypeError, ValueError) as exc:
            raise QualityError(f"public metrics for {target} contain a non-numeric value") from exc
        if not baseline or len(baseline) != len(candidate):
            failures.append(f"public metrics for {target} have mismatched tracks")
            continue
        median_degradation = statistics.median(baseline) - statistics.median(candidate)
        fifth_degradation = _fifth_percentile(baseline) - _fifth_percentile(candidate)
        passed = (
            median_degradation <= REFERENCE_SAFE_GATES["maximum_public_median_sdr_degradation_db"]
            and fifth_degradation <= REFERENCE_SAFE_GATES["maximum_public_fifth_percentile_degradation_db"]
        )
        if not passed:
            failures.append(f"public SDR gate failed for {target}")
        rows[target] = {
            "tracks": len(baseline),
            "median_sdr_degradation_db": median_degradation,
            "fifth_percentile_sdr_degradation_db": fifth_degradation,
            "passed": passed,
        }
    return {
        "source": path.name,
        "source_tool": "museval",
        "museval_version": tool_version,
        "corpora": sorted(corpora),
        "metric": "museval BSSEval v4",
        "targets": rows,
        "passed": not failures,
    }, failures


def _validate_listening_signoff(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    excerpts = payload.get("excerpts") if isinstance(payload, dict) else None
    failures: list[str] = []
    if not str(payload.get("reviewer") or "").strip() or not str(payload.get("signed_at") or "").strip():
        failures.append("manual listening reviewer/date missing")
    if not isinstance(excerpts, list) or len(excerpts) < 5:
        failures.append("manual blind listening requires at least five excerpts")
        excerpts = []
    sanitized: list[dict[str, Any]] = []
    for excerpt in excerpts:
        digest = str(excerpt.get("sha256") or "").strip().lower() if isinstance(excerpt, dict) else ""
        result = str(excerpt.get("result") or "").strip().lower() if isinstance(excerpt, dict) else ""
        if len(digest) != 64 or result != "pass":
            failures.append("manual listening excerpt is missing a passing hash-only result")
            continue
        sanitized.append({"sha256": digest, "result": result})
    return {
        "reviewer": str(payload.get("reviewer") or ""),
        "signed_at": str(payload.get("signed_at") or ""),
        "excerpts": sanitized,
        "passed": not failures,
    }, failures


def _benchmark_regressions(reports: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    benchmarks = [report for report in reports if report.get("command") == "benchmark"]
    if len(benchmarks) < 2:
        return {}, ["baseline and candidate benchmark reports are required"]
    failures: list[str] = []
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for benchmark in benchmarks:
        role = str(benchmark.get("role") or "")
        mode = str(benchmark.get("mode") or "")
        device_id = str(benchmark.get("device_id") or "")
        if role not in {"baseline", "candidate"} or not mode or not device_id:
            failures.append("benchmark report is missing role, mode, or device identity")
            continue
        bucket = grouped.setdefault((mode, device_id), {})
        if role in bucket:
            failures.append(f"duplicate {role} benchmark for {mode} on {device_id}")
        bucket[role] = benchmark

    mode_reports: dict[str, Any] = {}
    for (mode, device_id), pair in sorted(grouped.items()):
        if set(pair) != {"baseline", "candidate"}:
            failures.append(f"baseline and candidate benchmarks are required for {mode} on {device_id}")
            continue
        baseline, candidate = pair["baseline"], pair["candidate"]
        rows: dict[str, Any] = {}
        for kind in ("cold", "warm"):
            base = baseline.get("summary", {}).get(kind, {})
            cand = candidate.get("summary", {}).get(kind, {})
            try:
                base_trials = int(base["trials"])
                cand_trials = int(cand["trials"])
                base_wall = float(base["median_wall_seconds"])
                cand_wall = float(cand["median_wall_seconds"])
                base_rss = float(base["median_peak_rss_bytes"])
                cand_rss = float(cand["median_peak_rss_bytes"])
                base_mps = float(base["peak_mps_bytes"])
                cand_mps = float(cand["peak_mps_bytes"])
            except (KeyError, TypeError, ValueError):
                failures.append(f"{mode} {kind} benchmark is missing wall/RSS/MPS evidence")
                continue
            if base_trials < 3 or cand_trials < 3:
                failures.append(f"{mode} {kind} benchmark requires three baseline and candidate trials")
            runtime_regression = ((cand_wall / base_wall) - 1.0) * 100.0 if base_wall > 0 else float("inf")
            rss_regression = ((cand_rss / base_rss) - 1.0) * 100.0 if base_rss > 0 else float("inf")
            mps_regression = ((cand_mps / base_mps) - 1.0) * 100.0 if base_mps > 0 else float("inf")
            passed = (
                runtime_regression <= REFERENCE_SAFE_GATES["maximum_runtime_regression_percent"]
                and rss_regression <= REFERENCE_SAFE_GATES["maximum_memory_regression_percent"]
                and mps_regression <= REFERENCE_SAFE_GATES["maximum_memory_regression_percent"]
            )
            if not passed:
                failures.append(f"{mode} {kind} performance regression gate failed")
            rows[kind] = {
                "runtime_regression_percent": runtime_regression,
                "peak_rss_regression_percent": rss_regression,
                "peak_mps_regression_percent": mps_regression,
                "passed": passed,
            }
        mode_reports[f"{mode}@{device_id}"] = {"device_id": device_id, "trials": rows}
    return {"modes": mode_reports}, failures


def command_report(args: argparse.Namespace) -> int:
    inputs = [Path(path).expanduser().resolve() for path in args.inputs]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    blockers: list[str] = []
    for path, report in zip(inputs, reports):
        if report.get("passed") is False or report.get("ready") is False:
            blockers.append(path.name)
    commands = {str(report.get("command") or "") for report in reports}
    for required_command in ("prepare", "baseline", "compare"):
        if required_command not in commands:
            blockers.append(f"{required_command} report missing")
    public_metrics, public_failures = _validate_public_metrics(Path(args.public_metrics).expanduser().resolve())
    blockers.extend(public_failures)
    listening, listening_failures = _validate_listening_signoff(Path(args.listening_signoff).expanduser().resolve())
    blockers.extend(listening_failures)
    performance, performance_failures = _benchmark_regressions(reports)
    blockers.extend(performance_failures)
    combined = {
        "schema_version": 1,
        "command": "report",
        "provenance": _provenance(),
        "passed": not blockers,
        "manual_blind_listening_signoff": listening,
        "public_quality": public_metrics,
        "performance": performance,
        "inputs": [path.name for path in inputs],
        "blockers": blockers,
        "reports": reports,
    }
    _write_report(Path(args.output), combined)
    print(json.dumps({"passed": combined["passed"], "blockers": blockers}, indent=2))
    return 0 if combined["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stemsplat reference-safe quality and benchmark gates")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", default="quality/private-corpus.json")
    prepare.add_argument("--output", default="reports/quality/prepare.json")
    prepare.set_defaults(handler=command_prepare)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--reference-dir", required=True)
    baseline.add_argument("--output", default="reports/quality/v0.4.2-baseline.json")
    baseline.add_argument("--checkpoint-sha256", action="append")
    baseline.add_argument("--config-sha256", action="append")
    baseline.add_argument("--device-id", required=True)
    baseline.set_defaults(handler=command_baseline)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--reference-dir", required=True)
    compare.add_argument("--candidate-dir", required=True)
    compare.add_argument("--output", default="reports/quality/compare.json")
    compare.add_argument("--checkpoint-sha256", action="append", required=True)
    compare.add_argument("--config-sha256", action="append", required=True)
    compare.add_argument("--device-id", required=True)
    compare.set_defaults(handler=command_compare)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--cold-command", required=True)
    benchmark.add_argument("--warm-command", required=True)
    benchmark.add_argument("--role", choices=("baseline", "candidate"), required=True)
    benchmark.add_argument("--mode", required=True)
    benchmark.add_argument("--device-id", required=True)
    benchmark.add_argument("--cold", type=int, default=3)
    benchmark.add_argument("--warm", type=int, default=3)
    benchmark.add_argument("--output", default="reports/quality/benchmark.json")
    benchmark.add_argument("--audio-seconds", type=float)
    benchmark.add_argument("--cold-peak-mps-bytes", type=int, required=True)
    benchmark.add_argument("--warm-peak-mps-bytes", type=int, required=True)
    benchmark.set_defaults(handler=command_benchmark)

    report = subparsers.add_parser("report")
    report.add_argument("inputs", nargs="+")
    report.add_argument("--public-metrics", required=True)
    report.add_argument("--listening-signoff", required=True)
    report.add_argument("--output", default="reports/quality/release-report.json")
    report.set_defaults(handler=command_report)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if getattr(args, "cold", 1) < 1 or getattr(args, "warm", 1) < 1:
            raise QualityError("benchmark trial counts must be positive")
        if getattr(args, "audio_seconds", None) is not None and args.audio_seconds <= 0:
            raise QualityError("benchmark audio duration must be positive")
        for name in ("cold_peak_mps_bytes", "warm_peak_mps_bytes"):
            if getattr(args, name, 1) <= 0:
                raise QualityError("peak MPS allocation must be positive")
        return int(args.handler(args))
    except (OSError, ValueError, json.JSONDecodeError, QualityError) as exc:
        print(f"quality error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
