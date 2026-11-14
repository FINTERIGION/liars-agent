"""Validate portable data against a manifest committed with this repository.

The extracted run contains only a release identifier and manifest digest. File
hashes are always read from datasets/<release_id>/manifest.json in the code
checkout, never from an untrusted manifest supplied inside the downloaded run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

from liars_agent.contracts import require_contract

RELEASE_SCHEMA_VERSION = "liars_dataset_release_v1"
RELEASE_MANIFEST_ROOT = Path(__file__).resolve().parents[3] / "datasets"
SUPPORTED_RELEASE_IDS = frozenset({"selfplay-v3-20261114"})
RELEASE_MARKER = "dataset_release.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative_path(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"Invalid release-relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in path.parts[0]:
        raise ValueError(f"Release paths must stay inside the run: {value!r}")
    return path


def _validate_manifest(manifest: dict, release_id: str) -> None:
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != RELEASE_SCHEMA_VERSION
            or manifest.get("release_id") != release_id):
        raise ValueError("Missing or incompatible dataset release schema/identity.")
    require_contract(manifest, "dataset release manifest")
    run_name = _relative_path(manifest.get("run_directory"))
    if len(run_name.parts) != 1:
        raise ValueError("run_directory must be the original top-level directory name.")
    identity = manifest.get("source_identity")
    if not isinstance(identity, dict):
        raise ValueError("Dataset release requires its source game identity.")
    for key, item_type in (("game_ids", str), ("game_seeds", int)):
        values = identity.get(key)
        if (not isinstance(values, list) or any(type(item) is not item_type for item in values)
                or len(set(values)) != len(values)
                or (key == "game_ids" and any(not item for item in values))):
            raise ValueError(f"Invalid dataset release source_identity.{key}.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Dataset release manifest has no files.")
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Invalid dataset release file entry.")
        path = _relative_path(item.get("path"))
        if str(path) in seen or str(path) == RELEASE_MARKER:
            raise ValueError(f"Duplicate or self-referential release path: {path}")
        seen.add(str(path))
        if (type(item.get("size_bytes")) is not int or item["size_bytes"] < 0
                or type(item.get("immutable")) is not bool or not _digest(item.get("sha256"))):
            raise ValueError(f"Invalid release file size/hash/immutability: {path}")


def validate_dataset_release(run_dir: str | Path, required: bool = False,
                             verify_all: bool = False) -> dict | None:
    """Verify pinned release inputs, or all original files immediately after download.

    The directory may be renamed or moved. Mutable outputs (costs, evaluations,
    run status) can change after the initial verify_all check. Existing runs
    without a release marker retain their original validation workflow.
    """
    run = Path(run_dir).resolve()
    marker_path = run / RELEASE_MARKER
    if not marker_path.exists() and not marker_path.is_symlink():
        if required:
            raise ValueError(f"Dataset release marker is missing: {marker_path}")
        return None
    if not marker_path.resolve().is_relative_to(run):
        raise ValueError("Dataset release marker escapes the run directory.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (not isinstance(marker, dict) or set(marker) != {"release_id", "manifest_sha256"}
            or not isinstance(marker.get("release_id"), str)
            or marker["release_id"] not in SUPPORTED_RELEASE_IDS
            or not _digest(marker.get("manifest_sha256"))):
        raise ValueError("Unknown or invalid dataset release marker.")
    release_id = marker["release_id"]
    manifest_path = RELEASE_MANIFEST_ROOT / release_id / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"The repository has no pinned manifest for release {release_id}.")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != marker["manifest_sha256"]:
        raise ValueError("Dataset marker SHA256 differs from the repository-pinned manifest.")
    manifest = json.loads(raw)
    _validate_manifest(manifest, release_id)
    for item in manifest["files"]:
        path = run.joinpath(*PurePosixPath(item["path"]).parts)
        if not path.resolve().is_relative_to(run):
            raise ValueError(f"Release file escapes the run directory: {item['path']}")
        if not (verify_all or item["immutable"]):
            continue
        if not path.is_file():
            raise ValueError(f"Dataset release file is missing: {item['path']}")
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Dataset release file size differs: {item['path']}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"Dataset release file SHA256 differs: {item['path']}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check every released file using the repository-pinned manifest; no model calls.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = validate_dataset_release(args.run_dir, required=True, verify_all=True)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Dataset release verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "release_id": manifest["release_id"],
                      "verified_files": len(manifest["files"]), "verify_all": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
