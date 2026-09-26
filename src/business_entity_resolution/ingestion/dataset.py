"""Train-scoped, memory-bounded loaders for the discovered challenge files.

Nulls are read strictly: only a byte-empty field is treated as missing, so
country labels such as ``NA`` are never silently converted to null by a
default NA list. Raw text is always read as ``str``; type inference happens in
the profiler, never in the loader.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from business_entity_resolution.config.settings import DatasetSettings
from business_entity_resolution.ingestion.discovery import DiscoveredFile
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedFile:
    """A discovered file plus the schema its header actually declares."""

    file: DiscoveredFile
    columns: tuple[str, ...]
    delimiter: str
    encoding: str

    @property
    def key(self) -> str:
        """Return a stable label for reporting, derived from the file's own path."""

        return self.file.relative_path

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable schema view."""

        return {
            "path": str(self.file.path),
            "relative_path": self.file.relative_path,
            "filename": self.file.filename,
            "extension": self.file.extension,
            "size_bytes": self.file.size_bytes,
            "size_mib": round(self.file.size_bytes / (1024 * 1024), 2),
            "delimiter": repr(self.delimiter),
            "encoding": self.encoding,
            "split": self.file.split,
            "role": self.file.role,
            "source_index": self.file.source_index,
            "columns": list(self.columns),
            "column_count": len(self.columns),
        }


def read_header(item: DiscoveredFile, settings: DatasetSettings) -> LoadedFile:
    """Read only the header row to obtain the file's real column names."""

    frame = pd.read_csv(
        item.path,
        sep=settings.delimiter,
        encoding=settings.encoding,
        nrows=0,
        dtype=str,
    )
    return LoadedFile(
        file=item,
        columns=tuple(str(column) for column in frame.columns),
        delimiter=settings.delimiter,
        encoding=settings.encoding,
    )


def iter_chunks(
    loaded: LoadedFile,
    settings: DatasetSettings,
    *,
    columns: list[str] | None = None,
    chunksize: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield a file in bounded chunks without materializing the whole table."""

    size = chunksize or settings.chunksize
    reader = pd.read_csv(
        loaded.file.path,
        sep=settings.delimiter,
        encoding=settings.encoding,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        encoding_errors="replace",
        usecols=columns,
        chunksize=size,
    )
    try:
        for chunk in reader:
            chunk.index = pd.RangeIndex(len(chunk))
            yield chunk
    finally:
        reader.close()


def read_sample(
    loaded: LoadedFile,
    settings: DatasetSettings,
    *,
    rows: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return the leading rows of a file as a deterministic exploration sample."""

    limit = rows if rows is not None else settings.sample_rows
    frame = pd.read_csv(
        loaded.file.path,
        sep=settings.delimiter,
        encoding=settings.encoding,
        nrows=limit,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        encoding_errors="replace",
        usecols=columns,
    )
    return frame.reset_index(drop=True)


def read_filtered(
    loaded: LoadedFile,
    settings: DatasetSettings,
    *,
    key_column: str,
    wanted: set[str],
) -> dict[str, dict[str, str | None]]:
    """Stream a file and return only the rows whose key column is in ``wanted``.

    The full file is read in bounded chunks, so the resident set stays
    proportional to the number of requested identifiers rather than the file.
    """

    if not wanted:
        return {}
    collected: dict[str, dict[str, str | None]] = {}
    for chunk in iter_chunks(loaded, settings):
        subset = chunk[chunk[key_column].isin(wanted)]
        if subset.empty:
            continue
        for values in subset.to_dict(orient="records"):
            key = values.get(key_column)
            if isinstance(key, str):
                collected[key] = values
    LOGGER.info(
        "Streamed %d requested record(s) out of %s", len(collected), loaded.key
    )
    return collected


def resolve_path(project_root: Path, relative: str) -> Path:
    """Resolve a configured project-relative path without hard-coding a machine."""

    return (project_root / relative).resolve()
