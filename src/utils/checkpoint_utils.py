from __future__ import annotations

from pathlib import Path


LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        return first_line == LFS_POINTER_PREFIX
    except (UnicodeDecodeError, OSError):
        return False


def checkpoint_issue(path: Path) -> str | None:
    if not path.exists():
        return f"Checkpoint not found: {path}"
    if is_git_lfs_pointer(path):
        return (
            f"Checkpoint exists but is still a Git LFS pointer: {path}. "
            "Run `git lfs install` once, then `git lfs pull` in the repo root."
        )
    return None
