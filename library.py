"""
library.py — A folder of reference shoe/tread photos, indexed and cached.

ShoeLibrary points at a folder of images. build()/refresh() extracts
features for every image in it (via engine.extract_features) and caches
them to a pickle file inside that folder, keyed by filename with mtime+size
so unchanged images aren't ever re-processed on a later run — only new or
modified files, and files that were deleted are dropped from the cache.

query() takes a single feature dict (e.g. from a freshly uploaded photo)
and scores it against every cached entry, returning matches sorted best
first.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from engine import (
    IMAGE_EXTS,
    compute_similarity,
    extract_features,
    label_for_score,
)

CACHE_VERSION = 2  # bump if engine's feature extraction changes shape/meaning
CACHE_FILENAME = ".shoe_index.pkl"


@dataclass
class MatchResult:
    score: float
    sub: dict
    label: str
    name: str
    path: Optional[Path]
    features: dict = field(repr=False)


class ShoeLibrary:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._cache: dict[str, dict] = {}  # filename -> {mtime, size, features}

    # -- cache I/O ----------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.folder / CACHE_FILENAME

    def load_cache(self) -> None:
        self._cache = {}
        if not self.index_path.exists():
            return
        try:
            with open(self.index_path, "rb") as fh:
                data = pickle.load(fh)
            if data.get("version") == CACHE_VERSION:
                self._cache = data.get("entries", {})
        except (pickle.PickleError, EOFError, OSError, AttributeError, KeyError):
            self._cache = {}

    def save_cache(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"version": CACHE_VERSION, "entries": self._cache}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.index_path)

    # -- building -------------------------------------------------------

    def list_images(self):
        if not self.folder.exists():
            return []
        return sorted(
            p for p in self.folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    def build(self, progress_callback: Optional[Callable[[int, int, str], None]] = None,
              force: bool = False) -> tuple[int, int, int]:
        """
        Scan self.folder, extract features for new/changed images, drop
        entries for files that no longer exist, save the cache.

        progress_callback(done, total, current_filename) is called after
        each image is processed (or skipped because it's unchanged).

        Returns (added, skipped_unchanged, removed).
        """
        self.load_cache()
        images = self.list_images()
        total = len(images)
        seen_names = set()
        added = 0
        skipped = 0

        for i, path in enumerate(images, start=1):
            name = path.name
            seen_names.add(name)
            try:
                stat = path.stat()
            except OSError:
                continue
            cached = self._cache.get(name)
            unchanged = (
                not force and cached is not None
                and cached.get("mtime") == stat.st_mtime
                and cached.get("size") == stat.st_size
                and cached.get("features") is not None
            )
            if unchanged:
                skipped += 1
            else:
                feats = extract_features(path)
                self._cache[name] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "features": feats,
                }
                added += 1
            if progress_callback:
                progress_callback(i, total, name)

        removed_names = set(self._cache.keys()) - seen_names
        for name in removed_names:
            del self._cache[name]

        self.save_cache()
        return added, skipped, len(removed_names)

    # -- querying -------------------------------------------------------

    def entries(self):
        """Yield (name, features) for every successfully-processed cached image."""
        for name, rec in self._cache.items():
            feats = rec.get("features")
            if feats is not None:
                yield name, feats

    def __len__(self):
        return sum(1 for _ in self.entries())

    def query(self, query_features: dict, top_k: int = 15,
              progress_callback: Optional[Callable[[int, int, str], None]] = None
              ) -> list[MatchResult]:
        all_entries = list(self.entries())
        total = len(all_entries)
        results = []
        for i, (name, feats) in enumerate(all_entries, start=1):
            score, sub = compute_similarity(query_features, feats)
            results.append(MatchResult(
                score=score, sub=sub, label=label_for_score(score),
                name=name, path=feats.get("path"), features=feats,
            ))
            if progress_callback:
                progress_callback(i, total, name)
        results.sort(key=lambda r: -r.score)
        return results[:top_k] if top_k else results
