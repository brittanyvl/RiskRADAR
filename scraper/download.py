"""
scraper/download.py
-------------------
Helpers to detect, wait for, and finalize file downloads.

- Watches the configured download directory for new files.
- Waits until a file is "stable" (size stops changing).
- Moves and renames downloads atomically into target directories.

Use this to guarantee clean, complete files before downstream processing.
"""
from __future__ import annotations
from pathlib import Path
import time
import shutil
from typing import Callable
from .config import BrowserConfig

def _has_partial_suffix(p: Path, suffixes: tuple[str, ...]) -> bool:
    return any(str(p).endswith(s) for s in suffixes)

def _newest_file(dirpath: Path) -> Path | None:
    files = [p for p in dirpath.iterdir() if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def wait_for_new_download(cfg: BrowserConfig, predicate: Callable[[], bool] | None = None) -> Path:
    if predicate:
        assert predicate() is True

    deadline = time.time() + cfg.download_timeout_sec
    start_snapshot = {p: p.stat().st_size for p in cfg.downloads_dir.glob("*")}

    candidate: Path | None = None
    last_size: int | None = None

    while time.time() < deadline:
        newest = _newest_file(cfg.downloads_dir)
        if newest and newest not in start_snapshot and not _has_partial_suffix(newest, cfg.partial_suffixes):
            size = newest.stat().st_size
            if candidate is None:
                candidate, last_size = newest, size
            else:
                if newest == candidate and size == last_size:
                    return newest
                candidate, last_size = newest, size
        time.sleep(cfg.stable_check_interval_sec)
    raise TimeoutError("Timed out waiting for download to complete.")

def move_and_rename(src: Path, target_dir: Path, new_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / new_name
    tmp = final_path.with_suffix(final_path.suffix + ".tmpmove")
    shutil.copy2(src, tmp)
    tmp.replace(final_path)
    try:
        src.unlink()
    except Exception:
        pass
    return final_path
