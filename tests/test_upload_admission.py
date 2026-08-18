from __future__ import annotations

import asyncio
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from stemsplat.uploads import (
    UploadAdmissionController,
    UploadAdmissionError,
    UploadPolicy,
    normalize_display_name,
    stage_local_batch,
    stage_upload,
    validate_signature,
)


class FakeUpload:
    def __init__(self, payload: bytes, filename: str, content_type: str = "audio/wav") -> None:
        self.payload = payload
        self.filename = filename
        self.content_type = content_type
        self.offset = 0
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.payload):
            return b""
        if size < 0:
            size = len(self.payload)
        result = self.payload[self.offset : self.offset + size]
        self.offset += len(result)
        return result

    async def close(self) -> None:
        self.closed = True


def wav_bytes(frames: int = 32, channels: int = 1) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        with wave.open(handle.name, "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(44100)
            output.writeframes(struct.pack("<h", 128) * frames * channels)
        return Path(handle.name).read_bytes()


class UploadAdmissionTests(unittest.TestCase):
    def test_filename_normalization_and_controls(self) -> None:
        self.assertEqual(normalize_display_name("e\u0301.wav"), "é.wav")
        with self.assertRaises(UploadAdmissionError):
            normalize_display_name("bad\nname.wav")
        with self.assertRaises(UploadAdmissionError):
            normalize_display_name("../escape.wav")
        with self.assertRaises(UploadAdmissionError):
            normalize_display_name(r"folder\escape.wav")
        with self.assertRaises(UploadAdmissionError):
            normalize_display_name("x" * 181 + ".wav")

    def test_supported_audio_and_video_container_signatures(self) -> None:
        fixtures = {
            "sample.wav": b"RIFF\x00\x00\x00\x00WAVEfmt ",
            "sample.aiff": b"FORM\x00\x00\x00\x00AIFFCOMM",
            "sample.flac": b"fLaC\x00\x00\x00\x22",
            "sample.mp3": b"ID3\x04\x00\x00\x00\x00\x00\x00",
            "sample.aac": b"\xff\xf1\x50\x80\x00\x1f\xfc",
            "sample.ogg": b"OggS\x00\x02\x00\x00",
            "sample.opus": b"OggS\x00\x02OpusHead",
            "sample.m4a": b"\x00\x00\x00\x18ftypM4A ",
            "sample.mp4": b"\x00\x00\x00\x18ftypisom",
            "sample.webm": b"\x1aE\xdf\xa3\x9fB\x86\x81",
            "sample.mkv": b"\x1aE\xdf\xa3\x9fB\x86\x81",
            "sample.avi": b"RIFF\x00\x00\x00\x00AVI LIST",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename, content in fixtures.items():
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_bytes(content)
                    self.assertTrue(validate_signature(path, filename))

    def test_signature_mismatch_and_truncation_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "wrong.mp3"
            wrong.write_bytes(wav_bytes())
            with self.assertRaises(UploadAdmissionError) as mismatch:
                validate_signature(wrong, wrong.name)
            self.assertEqual(mismatch.exception.status_code, 415)
            tiny = root / "tiny.wav"
            tiny.write_bytes(b"RI")
            with self.assertRaises(UploadAdmissionError):
                validate_signature(tiny, tiny.name)

    def test_oversize_upload_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = UploadPolicy(max_file_bytes=32, max_staged_bytes=1024, chunk_bytes=16)
            controller = UploadAdmissionController(root, policy=policy)
            upload = FakeUpload(b"x" * 64, "large.wav")
            with self.assertRaises(UploadAdmissionError) as caught:
                asyncio.run(
                    stage_upload(
                        upload,
                        root,
                        controller=controller,
                        session_id="client",
                        outstanding_tasks=0,
                        ffprobe="/usr/bin/false",
                    )
                )
            self.assertEqual(caught.exception.status_code, 413)
            self.assertEqual(list(root.iterdir()), [])
            self.assertTrue(upload.closed)

    def test_admission_limits_return_429(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = UploadAdmissionController(
                Path(temporary),
                policy=UploadPolicy(max_global_uploads=1, max_session_uploads=1, max_session_tasks=2),
            )

            async def exercise() -> None:
                async with controller.slot("a", outstanding_tasks=0):
                    with self.assertRaises(UploadAdmissionError) as concurrent:
                        async with controller.slot("b", outstanding_tasks=0):
                            pass
                    self.assertEqual(concurrent.exception.status_code, 429)
                with self.assertRaises(UploadAdmissionError) as tasks:
                    async with controller.slot("a", outstanding_tasks=2):
                        pass
                self.assertEqual(tasks.exception.status_code, 429)

            asyncio.run(exercise())

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe is required")
    def test_local_batch_is_atomic_and_rejects_duplicate_normalized_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.wav"
            second = root / "second.wav"
            first.write_bytes(wav_bytes())
            second.write_bytes(b"not audio")
            destination = root / "stage"
            with self.assertRaises(UploadAdmissionError):
                stage_local_batch(
                    [first, second],
                    destination,
                    policy=UploadPolicy(),
                    ffprobe=shutil.which("ffprobe") or "ffprobe",
                )
            self.assertEqual([path for path in destination.iterdir() if not path.name.startswith(".batch-")], [])


if __name__ == "__main__":
    unittest.main()
