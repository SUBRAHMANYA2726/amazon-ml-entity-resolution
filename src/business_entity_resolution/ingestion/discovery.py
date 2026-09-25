"""Reproducible discovery of the supplied challenge files.

Nothing here assumes a filename. Files are found under the configured dataset
root, filtered by extension, classified from their own path/stem using
configurable keywords, and labelled with the split and role they actually
declare. OS metadata (AppleDouble files, ``.DS_Store``) is excluded by
configuration so it can never be mistaken for a challenge file.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from business_entity_resolution.config.settings import DatasetSettings
from business_entity_resolution.exceptions import ConfigurationError
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

SOURCE_ROLE = "source"
GROUND_TRUTH_ROLE = "ground_truth"
UNKNOWN_ROLE = "unknown"


class DatasetDiscoveryError(ConfigurationError):
    """Raised when the configured dataset root cannot supply a usable split."""


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """One real dataset file and the role its own path declares."""

    path: Path
    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    split: str | None
    role: str
    source_index: int | None
    is_readable: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the file record."""

        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "filename": self.filename,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "size_mib": round(self.size_bytes / (1024 * 1024), 2),
            "split": self.split,
            "role": self.role,
            "source_index": self.source_index,
            "is_readable": self.is_readable,
        }


@dataclass(frozen=True, slots=True)
class DiscoveredDataset:
    """Every discovered file, split by the role and split it declares."""

    root: Path
    files: tuple[DiscoveredFile, ...]
    ignored: tuple[str, ...]

    @property
    def train_sources(self) -> tuple[DiscoveredFile, ...]:
        """Return training source files ordered by their declared source index."""

        return tuple(
            sorted(
                (
                    item
                    for item in self.files
                    if item.split == "train" and item.role == SOURCE_ROLE
                ),
                key=lambda item: (item.source_index is None, item.source_index),
            )
        )

    @property
    def train_ground_truth(self) -> DiscoveredFile | None:
        """Return the single training ground-truth file, if one exists."""

        matches = [
            item
            for item in self.files
            if item.split == "train" and item.role == GROUND_TRUTH_ROLE
        ]
        return matches[0] if len(matches) == 1 else None

    @property
    def test_sources(self) -> tuple[DiscoveredFile, ...]:
        """Return test source files without ever reading their contents."""

        return tuple(
            sorted(
                (
                    item
                    for item in self.files
                    if item.split == "test" and item.role == SOURCE_ROLE
                ),
                key=lambda item: (item.source_index is None, item.source_index),
            )
        )

    def source_file(self, split: str, index: int) -> DiscoveredFile:
        """Return the file a split declares as ``source{index}``."""

        for item in self.files:
            if item.split == split and item.role == SOURCE_ROLE and item.source_index == index:
                return item
        raise DatasetDiscoveryError(
            f"No file declares split '{split}' source index {index} under {self.root}"
        )

    def read_scope(self) -> tuple[DiscoveredFile, ...]:
        """Return the only files Phase 1 and Phase 2 are allowed to read."""

        return tuple(item for item in self.files if item.split == "train")


def discover_dataset(settings: DatasetSettings, project_root: Path) -> DiscoveredDataset:
    """Walk the configured dataset root and classify every real dataset file."""

    root = (project_root / settings.root).resolve()
    if not root.is_dir():
        raise DatasetDiscoveryError(f"Configured dataset root does not exist: {root}")

    found: list[DiscoveredFile] = []
    ignored: list[str] = []
    for path in _iter_candidate_files(root, settings):
        relative = path.relative_to(root)
        if _is_ignored(relative, settings.ignore_globs):
            ignored.append(relative.as_posix())
            continue
        found.append(_classify(path, relative, settings))

    found.sort(key=lambda item: item.relative_path)
    ignored.sort()
    dataset = DiscoveredDataset(root=root, files=tuple(found), ignored=tuple(ignored))
    _log_inventory(dataset)
    return dataset


def _iter_candidate_files(root: Path, settings: DatasetSettings) -> Iterator[Path]:
    pattern = "**/*" if settings.recursive else "*"
    for path in sorted(root.glob(pattern)):
        if path.is_dir():
            continue
        if path.suffix.lower() not in settings.extensions:
            continue
        yield path


def _is_ignored(relative: Path, ignore_globs: tuple[str, ...]) -> bool:
    posix = relative.as_posix()
    name = relative.name
    for pattern in ignore_globs:
        if fnmatch.fnmatch(posix, pattern) or fnmatch.fnmatch(name, pattern):
            return True
        if pattern.endswith("/**") and posix.startswith(pattern[:-3].rstrip("*") + "/"):
            return True
    return False


def _classify(path: Path, relative: Path, settings: DatasetSettings) -> DiscoveredFile:
    """Derive split, role, and source index from the file's own location."""

    stem = path.stem.lower()
    parts = [part.lower() for part in relative.parts]
    lowered = path.name.lower()

    split = _match_split(stem, parts, settings.split_keywords)
    role = _match_role(stem, lowered, settings)
    source_index = _match_source_index(stem, settings.source_keywords) if role == SOURCE_ROLE else None

    try:
        size = path.stat().st_size
        readable = path.is_file()
    except OSError:
        size = 0
        readable = False

    return DiscoveredFile(
        path=path,
        relative_path=relative.as_posix(),
        filename=path.name,
        extension=path.suffix.lower(),
        size_bytes=size,
        split=split,
        role=role,
        source_index=source_index,
        is_readable=readable,
    )


def _match_split(
    stem: str,
    parts: list[str],
    split_keywords: object,
) -> str | None:
    mapping = {key: value for key, value in dict(split_keywords).items()}  # type: ignore[arg-type]
    candidates = [*parts, stem]
    for name, keyword in mapping.items():
        if any(candidate == keyword or keyword in candidate.split("_") for candidate in candidates):
            return name
    for name, keyword in mapping.items():
        if any(keyword in candidate for candidate in candidates):
            return name
    return None


def _match_role(stem: str, lowered_name: str, settings: DatasetSettings) -> str:
    for keyword in settings.ground_truth_keywords:
        if keyword in stem or keyword in lowered_name:
            return GROUND_TRUTH_ROLE
    if any(keyword in stem for keyword in settings.source_keywords):
        return SOURCE_ROLE
    return UNKNOWN_ROLE


def _match_source_index(stem: str, source_keywords: tuple[str, ...]) -> int | None:
    digits = ""
    for keyword in source_keywords:
        index = stem.find(keyword)
        if index < 0:
            continue
        tail = stem[index + len(keyword) :]
        for character in tail:
            if character.isdigit():
                digits += character
            elif digits:
                break
        if digits:
            break
    return int(digits) if digits else None


def _log_inventory(dataset: DiscoveredDataset) -> None:
    LOGGER.info(
        "Discovered %d dataset file(s) under %s (%d path(s) ignored as OS metadata).",
        len(dataset.files),
        dataset.root,
        len(dataset.ignored),
    )
    for item in dataset.files:
        LOGGER.info(
            "  %s | split=%s role=%s source_index=%s size=%.2f MiB",
            item.relative_path,
            item.split,
            item.role,
            item.source_index,
            item.size_bytes / (1024 * 1024),
        )
