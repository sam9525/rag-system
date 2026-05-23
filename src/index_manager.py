"""Index persistence and dirty tracking for RAG system."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IndexManifest:
    """Manifest tracking which files were indexed and their checksums."""

    index_dir: Path
    files: Dict[str, Dict] = None

    def __post_init__(self):
        if self.files is None:
            self.files = {}

    def add_file(self, filename: str, checksum: str, chunk_count: int):
        """Add or update a file entry in the manifest."""
        self.files[filename] = {
            "checksum": checksum,
            "chunk_count": chunk_count,
            "indexed_at": str(Path(self.index_dir) / f"{filename}.index"),
        }

    def remove_file(self, filename: str):
        """Remove a file entry from the manifest."""
        if filename in self.files:
            del self.files[filename]

    def to_dict(self) -> Dict:
        """Serialize manifest to dict."""
        return {"index_dir": str(self.index_dir), "files": self.files}

    @staticmethod
    def from_dict(data: Dict, index_dir: Path) -> "IndexManifest":
        """Deserialize manifest from dict."""
        manifest = IndexManifest(index_dir=index_dir)
        manifest.files = data.get("files", {})
        return manifest


class IndexManager:
    """Manages index persistence and dirty file detection."""

    MANIFEST_FILENAME = ".rag_manifest.json"

    def __init__(self, index_dir: Path):
        """Initialize with index directory.

        Args:
            index_dir: Directory to store index files and manifest
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def manifest_path(self) -> Path:
        """Get path to the manifest file."""
        return self.index_dir / self.MANIFEST_FILENAME

    def save_manifest(self, manifest: IndexManifest):
        """Save manifest to disk."""
        with open(self.manifest_path(), "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def load_manifest(self) -> Optional[IndexManifest]:
        """Load manifest from disk, or None if not found."""
        if not self.manifest_path().exists():
            return None

        with open(self.manifest_path()) as f:
            data = json.load(f)
        return IndexManifest.from_dict(data, self.index_dir)

    def file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def detect_dirty_files(
        self, source_dir: Path, current_files: List[str]
    ) -> Dict[str, List[str]]:
        """Detect which files are new, modified, or deleted since last index.

        Args:
            source_dir: Directory containing source PDFs
            current_files: List of filenames currently in source_dir

        Returns:
            Dict with 'new', 'modified', 'deleted' lists
        """
        source_dir = Path(source_dir)  # Ensure Path type
        manifest = self.load_manifest()
        if manifest is None:
            return {"new": current_files, "modified": [], "deleted": []}

        current_set = set(current_files)
        manifest_set = set(manifest.files.keys())

        new_files = list(current_set - manifest_set)
        deleted_files = list(manifest_set - current_set)

        modified_files = []
        for filename in current_set & manifest_set:
            file_path = source_dir / filename
            if file_path.exists():
                current_checksum = self.file_checksum(file_path)
                if current_checksum != manifest.files[filename]["checksum"]:
                    modified_files.append(filename)

        return {"new": new_files, "modified": modified_files, "deleted": deleted_files}

    def is_index_valid(self, source_dir: Path) -> bool:
        """Check if the index is valid (exists and matches manifest)."""
        manifest = self.load_manifest()
        if manifest is None:
            return False

        # Get actual current files from source directory (not manifest files)
        current_files = [f.name for f in Path(source_dir).glob("*.pdf")]
        dirty = self.detect_dirty_files(source_dir, current_files)
        return (
            len(dirty["new"]) == 0
            and len(dirty["modified"]) == 0
            and len(dirty["deleted"]) == 0
        )
