from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StepResult:
    name: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _run_step(name: str, command: list[str], *, env: dict[str, str] | None = None) -> StepResult:
    print(f"\n== {name} ==")
    result = subprocess.run(command, cwd=ROOT, env=env)
    return StepResult(name=name, returncode=result.returncode)


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Run the expanded Stemsplat test battery.")
    parser.add_argument(
        "--include-build",
        action="store_true",
        help="Run build_app.sh as the final packaging smoke step.",
    )
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("STEMSPLAT_DISABLE_BACKGROUND_THREADS", "1")

    python = sys.executable
    baseline_modules = [
        "tests.test_downloader_checksums",
        "tests.test_model_download_retry_policy",
        "tests.test_model_registry",
        "tests.test_output_layout",
        "tests.test_runtime_estimator",
    ]

    steps = [
        (
            "Syntax Smoke",
            [
                python,
                "-m",
                "py_compile",
                "main.py",
                "downloader.py",
                "launcher.py",
                "install.py",
                "app_paths.py",
                "tests/test_expanded_battery.py",
            ],
        ),
        (
            "Import Smoke",
            [
                python,
                "-c",
                "import main, downloader, launcher; print('import smoke ok')",
            ],
        ),
        (
            "Baseline Unit Suite",
            [python, "-m", "unittest", *baseline_modules, "-v"],
        ),
        (
            "Expanded Battery",
            [python, "-m", "unittest", "tests.test_expanded_battery", "-v"],
        ),
    ]

    results = [_run_step(name, command, env=env) for name, command in steps]

    if args.include_build:
        build_env = dict(env)
        build_env["PYTHON_BIN"] = python
        results.append(_run_step("Build Smoke", [str(ROOT / "build_app.sh")], env=build_env))

    print("\n== Summary ==")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"{status:4}  {result.name}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
