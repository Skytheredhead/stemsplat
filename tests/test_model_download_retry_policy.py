from __future__ import annotations

import ast
from pathlib import Path
import unittest

import downloader


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"


def _load_retry_helpers() -> dict[str, object]:
    module = ast.parse(MAIN_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PATH))
    wanted_assignments = {"MODEL_DOWNLOAD_MAX_RETRIES"}
    wanted_functions = {
        "_model_download_error_message",
        "_model_download_retry_delay_seconds",
        "_should_retry_model_download",
    }
    selected: list[ast.AST] = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_assignments:
                    selected.append(node)
                    break
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected.append(node)

    helper_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(helper_module)
    namespace = {"ModelDownloadError": downloader.ModelDownloadError}
    exec(compile(helper_module, str(MAIN_PATH), "exec"), namespace)
    return namespace


class ModelDownloadRetryPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        namespace = _load_retry_helpers()
        cls.max_retries = int(namespace["MODEL_DOWNLOAD_MAX_RETRIES"])
        cls.error_message = staticmethod(namespace["_model_download_error_message"])
        cls.retry_delay = staticmethod(namespace["_model_download_retry_delay_seconds"])
        cls.should_retry = staticmethod(namespace["_should_retry_model_download"])

    def test_retryable_download_errors_retry_until_cap(self) -> None:
        exc = downloader.ModelDownloadError("network-error", "temporary problem", retryable=True)
        self.assertTrue(self.should_retry(exc, 1))
        self.assertFalse(self.should_retry(exc, self.max_retries))

    def test_non_retryable_download_errors_stop_immediately(self) -> None:
        exc = downloader.ModelDownloadError("permission-denied", "no write access", retryable=False)
        self.assertFalse(self.should_retry(exc, 1))

    def test_generic_exceptions_retry_until_cap(self) -> None:
        self.assertTrue(self.should_retry(RuntimeError("boom"), 1))
        self.assertFalse(self.should_retry(RuntimeError("boom"), self.max_retries))

    def test_retry_delay_uses_backoff_with_cap(self) -> None:
        self.assertEqual(self.retry_delay(1), 5)
        self.assertEqual(self.retry_delay(3), 15)
        self.assertEqual(self.retry_delay(9), 30)

    def test_error_message_prefers_structured_download_error(self) -> None:
        self.assertEqual(self.error_message(downloader.ModelDownloadError("checksum-mismatch", "bad hash", retryable=True)), "bad hash")
        self.assertEqual(self.error_message(RuntimeError("boom")), "download failed unexpectedly")


if __name__ == "__main__":
    unittest.main()
