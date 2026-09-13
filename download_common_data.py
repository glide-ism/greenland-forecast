#!/usr/bin/env python3
"""
Download model input bundles from a manifest.

Example:
    python scripts/download_inputs.py \
        --manifest https://data.glide-ism.org/data/latest.json \
        --out data/raw

Optional extraction:
    python scripts/download_inputs.py \
        --manifest https://data.glide-ism.org/data/latest.json \
        --out data/raw \
        --extract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


def human_size(n: int | None) -> str:
    if n is None:
        return "unknown size"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for unit in units:
        if x < 1024 or unit == units[-1]:
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{n} B"


def load_manifest(url_or_path: str) -> dict[str, Any]:
    if url_or_path.startswith(("http://", "https://")):
        response = requests.get(url_or_path, timeout=30)
        response.raise_for_status()
        return response.json()

    with open(url_or_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, expected_sha256: str | None, expected_size: int | None) -> bool:
    if not path.exists():
        return False

    if expected_size is not None and path.stat().st_size != expected_size:
        return False

    if expected_sha256:
        actual = sha256_file(path)
        return actual.lower() == expected_sha256.lower()

    return True


def download_with_resume(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    retries: int = 5,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if verify_file(dest, expected_sha256, expected_size):
        print(f"Already present and verified: {dest}")
        return

    for attempt in range(1, retries + 1):
        existing = part.stat().st_size if part.exists() else 0
        headers = {}

        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            print(f"Resuming {dest.name} from {human_size(existing)}")
        else:
            print(f"Downloading {dest.name}")

        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
                # If the server ignores Range and returns 200, restart cleanly.
                if existing > 0 and r.status_code == 200:
                    print("Server did not honor Range request; restarting partial download.")
                    part.unlink(missing_ok=True)
                    existing = 0

                r.raise_for_status()

                mode = "ab" if existing > 0 and r.status_code == 206 else "wb"
                bytes_written = existing
                last_report = time.time()

                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue

                        f.write(chunk)
                        bytes_written += len(chunk)

                        now = time.time()
                        if now - last_report >= 5:
                            if expected_size:
                                pct = 100.0 * bytes_written / expected_size
                                print(
                                    f"  {human_size(bytes_written)} / "
                                    f"{human_size(expected_size)} "
                                    f"({pct:.1f}%)"
                                )
                            else:
                                print(f"  {human_size(bytes_written)}")
                            last_report = now

            if expected_size is not None and part.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Downloaded size mismatch for {part}: "
                    f"got {part.stat().st_size}, expected {expected_size}"
                )

            if expected_sha256:
                print(f"Verifying SHA-256 for {dest.name}")
                actual = sha256_file(part)
                if actual.lower() != expected_sha256.lower():
                    raise RuntimeError(
                        f"SHA-256 mismatch for {dest.name}\n"
                        f"  actual:   {actual}\n"
                        f"  expected: {expected_sha256}"
                    )

            part.replace(dest)
            print(f"Finished: {dest}")
            return

        except Exception as e:
            print(f"Attempt {attempt}/{retries} failed: {e}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 60))


def strip_archive_suffix(filename: str) -> str:
    for suffix in [".tar.gz", ".tgz", ".tar.zst", ".tar"]:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def extract_archive(archive: Path, extract_to: Path) -> None:
    name = archive.name

    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        tar_args = ["tar", "-xzf", str(archive)]

    elif name.endswith(".tar.zst"):
        tar_args = ["tar", "--zstd", "-xf", str(archive)]

    elif name.endswith(".tar"):
        tar_args = ["tar", "-xf", str(archive)]

    else:
        print(f"Skipping extraction for unsupported archive type: {archive}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive.name} -> {extract_to}")
    subprocess.run(
        tar_args + ["-C", str(extract_to)],
        check=True,
    )
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="https://data.glide-ism.org/data/latest.json",
        help="URL or local path to manifest JSON, e.g. https://.../latest.json",
    )
    parser.add_argument(
        "--out",
        default="./",
        help="Output directory. Default: data/raw",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract files after download.",
    )
    parser.add_argument(
        "--extract-to",
        default=".",
        help="Directory where archives should be extracted. Default: current working directory.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination file already verifies.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    manifest = load_manifest(args.manifest)

    version = manifest.get("version", "unknown")
    print(f"Input bundle version: {version}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest must contain a non-empty 'files' list.")

    downloaded: list[Path] = []

    for item in files:
        filename = item["filename"]
        url = item.get("url")

        # Allows either absolute file URLs or a manifest-level base_url.
        if not url:
            base_url = manifest.get("base_url")
            if not base_url:
                raise ValueError(f"No URL or base_url specified for {filename}")
            url = urljoin(base_url.rstrip("/") + "/", filename)

        expected_sha256 = item.get("sha256")
        expected_size = item.get("size_bytes")

        dest = out_dir / filename

        if args.force:
            dest.unlink(missing_ok=True)
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)

        download_with_resume(
            url=url,
            dest=dest,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )

        downloaded.append(dest)

    # Save the manifest beside the data for provenance.
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_downloaded.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest copy: {manifest_path}")

    if args.extract:
        if shutil.which("tar") is None:
            raise RuntimeError("Cannot extract: 'tar' not found on PATH.")

        extract_to = Path(args.extract_to)

        for path in downloaded:
            extract_archive(path, extract_to)

if __name__ == "__main__":
    main()
