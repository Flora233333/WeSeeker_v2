from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from config.settings import KBConfig


@dataclass(frozen=True)
class FileRecord:
    path: Path
    size: int
    mtime: float

    @property
    def path_str(self) -> str:
        return self.path.as_posix()


@dataclass(frozen=True)
class SkippedFile:
    path: str
    reason: str


@dataclass(frozen=True)
class ScanResult:
    files: list[FileRecord]
    added: list[FileRecord]
    modified: list[FileRecord]
    deleted: list[str]
    unchanged: list[FileRecord]
    skipped: list[SkippedFile]


_PDF_DUPLICATE_PRIORITY = {
    ".md": 3,
    ".docx": 2,
    ".txt": 1,
    ".pdf": 0,
}


def scan_kb(
    kb_config: KBConfig,
    *,
    manifest_files: dict[str, dict[str, object]] | None,
    max_file_size_mb: int,
) -> ScanResult:
    root = Path(kb_config.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"知识库根目录不存在: {root}")

    include_ext = {suffix.lower() for suffix in kb_config.include_ext}
    skipped: list[SkippedFile] = []
    candidate_files: list[FileRecord] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        path_str = path.as_posix()
        if _is_excluded(path_str, kb_config.exclude_glob):
            skipped.append(SkippedFile(path=path_str, reason="excluded_by_glob"))
            continue

        suffix = path.suffix.lower()
        if suffix not in include_ext:
            if suffix == ".pptx":
                skipped.append(SkippedFile(path=path_str, reason="pptx_not_supported_in_step1"))
            else:
                skipped.append(SkippedFile(path=path_str, reason="unsupported_extension"))
            continue

        stat_result = path.stat()
        if stat_result.st_size > max_file_size_mb * 1024 * 1024:
            skipped.append(SkippedFile(path=path_str, reason="size_exceeds_limit"))
            continue

        candidate_files.append(
            FileRecord(path=path, size=stat_result.st_size, mtime=stat_result.st_mtime)
        )

    files, duplicate_skipped = _apply_duplicate_pdf_suppression(root, candidate_files)
    skipped.extend(duplicate_skipped)

    previous = manifest_files or {}
    previous_keys = set(previous)
    current_keys = {file.path_str for file in files}

    added: list[FileRecord] = []
    modified: list[FileRecord] = []
    unchanged: list[FileRecord] = []

    for file in files:
        previous_record = previous.get(file.path_str)
        if previous_record is None:
            added.append(file)
            continue

        previous_mtime = float(previous_record.get("mtime") or 0)
        previous_size = int(previous_record.get("size") or 0)
        if previous_mtime != file.mtime or previous_size != file.size:
            modified.append(file)
        else:
            unchanged.append(file)

    deleted = sorted(previous_keys - current_keys)
    return ScanResult(
        files=files,
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
        skipped=skipped,
    )


def _is_excluded(path_str: str, patterns: list[str]) -> bool:
    normalized = path_str.replace("\\", "/")
    return any(fnmatch(normalized, pattern) for pattern in patterns)


def _apply_duplicate_pdf_suppression(
    root: Path,
    files: list[FileRecord],
) -> tuple[list[FileRecord], list[SkippedFile]]:
    grouped: dict[str, list[FileRecord]] = {}
    for file in files:
        # 只在同一相对路径 stem 下比较，避免不同目录下的同名文件互相误伤。
        relative_key = file.path.relative_to(root).with_suffix("").as_posix()
        grouped.setdefault(relative_key, []).append(file)

    skipped: list[SkippedFile] = []
    suppressed_paths: set[str] = set()

    for group in grouped.values():
        if len(group) <= 1:
            continue

        highest = max(_priority_of(record.path.suffix.lower()) for record in group)
        higher_sources = [
            record.path.suffix.lower()
            for record in group
            if _priority_of(record.path.suffix.lower()) == highest and highest > 0
        ]
        for record in group:
            suffix = record.path.suffix.lower()
            if suffix != ".pdf":
                continue
            if highest <= _priority_of(suffix):
                continue

            preferred = sorted(set(higher_sources))[0]
            skipped.append(
                SkippedFile(
                    path=record.path.as_posix(),
                    reason=f"duplicate_higher_priority_source:{preferred}",
                )
            )
            suppressed_paths.add(record.path.as_posix())

    # 保持稳定顺序，方便 report 和 manifest 对比。
    ordered_kept = [file for file in files if file.path.as_posix() not in suppressed_paths]
    return ordered_kept, skipped


def _priority_of(suffix: str) -> int:
    return _PDF_DUPLICATE_PRIORITY.get(suffix, 0)
