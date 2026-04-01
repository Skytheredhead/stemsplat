# Model Download Hardening

This project now defends against these 25 model download failure scenarios:

1. Download entry is missing a required field.
2. Download URL is not HTTPS.
3. Filename contains invalid path components.
4. Subdirectory contains invalid path components.
5. Destination path escapes the app directory.
6. Configured `sha256` value is malformed.
7. Selected model tags are duplicated.
8. Selected model tags include unknown values.
9. Destination parent path exists but is not a directory.
10. Destination file path is a directory.
11. Destination file path is a symlink.
12. Partial download path is a directory.
13. Partial download path is a symlink.
14. Stale partial download file is left behind from an earlier crash.
15. App cannot create the destination directory.
16. App cannot inspect free disk space.
17. Device does not have enough free disk space.
18. Cached file has the right size but the wrong checksum.
19. HEAD metadata is unavailable but GET headers still provide integrity metadata.
20. Server returns an empty payload.
21. Server returns fewer or more bytes than promised.
22. Downloaded payload hash does not match the expected checksum.
23. Filesystem write fails during download.
24. Final file replace fails after download.
25. Remote fetch fails permanently or transiently and needs different retry behavior.

Implemented safeguards live in [downloader.py](/Users/skylarenns/Desktop/stemsplat/downloader.py) and [main.py](/Users/skylarenns/Desktop/stemsplat/main.py):

- Invalid config and unsafe paths are rejected before any network or filesystem writes start.
- Remote metadata is gathered from HEAD when possible and from GET response headers as a fallback.
- Cached files are re-verified before being trusted.
- Partial downloads are cleaned up before retries and after failures.
- Disk-space and filesystem problems are surfaced with specific permanent errors.
- Network, timeout, and server-side failures are marked retryable.
- Missing files, permission errors, and invalid config stop immediately instead of retrying forever.
- Retry attempts use bounded backoff and stop after a maximum retry count.
