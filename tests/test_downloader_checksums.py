from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

import downloader


class _FakeDownloadResponse:
    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._buffer = io.BytesIO(payload)
        self.headers = headers or {}

    def __enter__(self) -> "_FakeDownloadResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


class DownloaderChecksumTests(unittest.TestCase):
    def _single_file_entry(self, *, url: str = "https://example.test/verified.ckpt") -> list[dict[str, str]]:
        return [{"url": url, "subdir": "models", "filename": "verified.ckpt", "tag": "verified"}]

    def test_cached_file_is_skipped_only_when_sha256_matches(self) -> None:
        payload = b"verified-model-contents"
        expected_sha256 = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            dest = base_dir / "models" / "verified.ckpt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(size=len(payload), sha256=expected_sha256),
            ), mock.patch.object(downloader, "urlopen") as mocked_urlopen:
                downloader.download_to(base_dir)

        mocked_urlopen.assert_not_called()

    def test_cached_file_is_redownloaded_when_sha256_mismatch(self) -> None:
        stale_payload = b"stale-model-contents"
        fresh_payload = b"fresh-model-contents"
        self.assertEqual(len(stale_payload), len(fresh_payload))
        expected_sha256 = hashlib.sha256(fresh_payload).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            dest = base_dir / "models" / "verified.ckpt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(stale_payload)

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(size=len(fresh_payload), sha256=expected_sha256),
            ), mock.patch.object(
                downloader,
                "urlopen",
                return_value=_FakeDownloadResponse(fresh_payload),
            ) as mocked_urlopen:
                downloader.download_to(base_dir)

            self.assertEqual(dest.read_bytes(), fresh_payload)
            self.assertFalse((base_dir / "models" / "verified.ckpt.part").exists())

        mocked_urlopen.assert_called_once()

    def test_download_fails_when_payload_sha256_does_not_match(self) -> None:
        expected_payload = b"expected-model-contents"
        actual_payload = b"unexpected-model-data!"
        expected_sha256 = hashlib.sha256(expected_payload).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            dest = base_dir / "models" / "verified.ckpt"

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(size=len(actual_payload), sha256=expected_sha256),
            ), mock.patch.object(
                downloader,
                "urlopen",
                return_value=_FakeDownloadResponse(actual_payload),
            ):
                with self.assertRaises(downloader.ModelDownloadError) as ctx:
                    downloader.download_to(base_dir)

            self.assertEqual(ctx.exception.code, "checksum-mismatch")
            self.assertTrue(ctx.exception.retryable)
            self.assertFalse(dest.exists())
            self.assertFalse((base_dir / "models" / "verified.ckpt.part").exists())

    def test_download_fails_when_payload_size_does_not_match(self) -> None:
        payload = b"short-payload"

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(size=len(payload) + 5, sha256=None),
            ), mock.patch.object(
                downloader,
                "urlopen",
                return_value=_FakeDownloadResponse(payload),
            ):
                with self.assertRaises(downloader.ModelDownloadError) as ctx:
                    downloader.download_to(base_dir)

            self.assertEqual(ctx.exception.code, "size-mismatch")
            self.assertTrue(ctx.exception.retryable)

    def test_download_fails_when_payload_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(),
            ), mock.patch.object(
                downloader,
                "urlopen",
                return_value=_FakeDownloadResponse(b""),
            ):
                with self.assertRaises(downloader.ModelDownloadError) as ctx:
                    downloader.download_to(base_dir)

            self.assertEqual(ctx.exception.code, "empty-download")
            self.assertTrue(ctx.exception.retryable)

    def test_download_uses_get_headers_when_head_metadata_is_missing(self) -> None:
        payload = b"verified-model-contents"
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        headers = {"Content-Length": str(len(payload)), "ETag": f'"{expected_sha256}"'}

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            dest = base_dir / "models" / "verified.ckpt"

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(),
            ), mock.patch.object(
                downloader,
                "urlopen",
                return_value=_FakeDownloadResponse(payload, headers=headers),
            ):
                downloader.download_to(base_dir)

            self.assertEqual(dest.read_bytes(), payload)

    def test_download_rejects_directory_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            dest = base_dir / "models" / "verified.ckpt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.mkdir()

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(),
            ):
                with self.assertRaises(downloader.ModelDownloadError) as ctx:
                    downloader.download_to(base_dir)

            self.assertEqual(ctx.exception.code, "invalid-path")
            self.assertFalse(ctx.exception.retryable)

    def test_download_rejects_insecure_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(url="http://example.test/verified.ckpt"),
            ):
                with self.assertRaises(downloader.ModelDownloadError) as ctx:
                    downloader.download_to(base_dir)

            self.assertEqual(ctx.exception.code, "insecure-url")
            self.assertFalse(ctx.exception.retryable)

    def test_download_fails_when_disk_space_is_insufficient(self) -> None:
        payload = b"verified-model-contents"

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(size=len(payload), sha256=None),
            ), mock.patch.object(
                downloader.shutil,
                "disk_usage",
                return_value=shutil._ntuple_diskusage(10, 9, 1),
            ):
                with self.assertRaises(downloader.ModelDownloadError) as ctx:
                    downloader.download_to(base_dir)

            self.assertEqual(ctx.exception.code, "disk-full")
            self.assertFalse(ctx.exception.retryable)

    def test_download_classifies_missing_remote_as_permanent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(),
            ), mock.patch.object(
                downloader,
                "urlopen",
                side_effect=HTTPError(
                    url="https://example.test/verified.ckpt",
                    code=404,
                    msg="Not Found",
                    hdrs=None,
                    fp=None,
                ),
            ):
                with self.assertRaises(downloader.ModelDownloadError) as ctx:
                    downloader.download_to(base_dir)

            self.assertEqual(ctx.exception.code, "remote-missing")
            self.assertFalse(ctx.exception.retryable)

    def test_download_deduplicates_selected_tags(self) -> None:
        payload = b"verified-model-contents"
        expected_sha256 = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            with mock.patch.object(
                downloader,
                "FILES",
                self._single_file_entry(),
            ), mock.patch.object(
                downloader,
                "get_remote_file_metadata",
                return_value=downloader.RemoteFileMetadata(size=len(payload), sha256=expected_sha256),
            ), mock.patch.object(
                downloader,
                "urlopen",
                return_value=_FakeDownloadResponse(payload),
            ) as mocked_urlopen:
                downloader.download_to(base_dir, selected=["verified", "verified", "missing"])

        mocked_urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
