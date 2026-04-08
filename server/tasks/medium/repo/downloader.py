# downloader.py — File download utilities with streaming and parallel downloads
# Uses `requests` streaming API and threading for parallel downloads.
# Must be migrated to `httpx` streaming and async.
#
# Migration challenges:
#   - requests.get(stream=True) → httpx.stream("GET", url) or httpx.Client.stream
#   - response.iter_content(chunk_size) → response.iter_bytes(chunk_size)
#   - allow_redirects=True → follow_redirects=True
#   - download_parallel: threading → async with httpx.AsyncClient
#   - The parallel function must become an async def

import os
import requests
import threading
from config import DEFAULT_TIMEOUT


def download_file(url: str, dest_path: str, timeout: int = DEFAULT_TIMEOUT,
                  on_progress=None) -> int:
    """Download a file using streaming.

    Args:
        url: URL to download from
        dest_path: Local file path to save to
        timeout: Request timeout in seconds
        on_progress: Optional callback(bytes_written, total) called per chunk
    Returns:
        Total number of bytes written
    """
    bytes_written = 0
    response = requests.get(
        url, stream=True, allow_redirects=True, timeout=timeout
    )
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0))

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bytes_written += len(chunk)
            if on_progress:
                on_progress(bytes_written, total)

    return bytes_written


def download_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Download text content from a URL using streaming.

    Returns the full decoded text.
    """
    parts = []
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
        parts.append(chunk)
    return "".join(parts)


def get_content_length(url: str, timeout: int = DEFAULT_TIMEOUT) -> int | None:
    """Get content length via HEAD request without downloading.

    Returns content-length as int, or None if not provided.
    """
    response = requests.head(url, allow_redirects=True, timeout=timeout)
    length = response.headers.get("Content-Length")
    return int(length) if length else None


def download_parallel(urls: list[str], dest_dir: str,
                      max_workers: int = 4,
                      timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Download multiple files in parallel using threads.

    Must be migrated to async with httpx.AsyncClient.

    Args:
        urls: List of URLs to download
        dest_dir: Directory to save downloaded files
        max_workers: Maximum concurrent downloads (unused after async migration)
        timeout: Per-download timeout
    Returns:
        dict with 'completed' (url→bytes) and 'errors' (url→message)
    """
    completed = {}
    errors = {}
    lock = threading.Lock()

    def _download_one(url):
        filename = url.rsplit("/", 1)[-1] or "download"
        path = os.path.join(dest_dir, filename)
        try:
            nbytes = download_file(url, path, timeout=timeout)
            with lock:
                completed[url] = nbytes
        except Exception as e:
            with lock:
                errors[url] = str(e)

    threads = []
    for url in urls:
        t = threading.Thread(target=_download_one, args=(url,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return {"completed": completed, "errors": errors}
