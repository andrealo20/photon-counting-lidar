"""Download one sample of SP-TransientBench, outside the repository, and
check every file against the digest the server publishes.

The data is licensed CC BY-NC-SA 4.0 and must never be committed; the default
destination is ignored by git. Each file is compared with the hash listed by
the Hugging Face tree endpoint: the sha256 for files stored with LFS, the git
blob id for the others. A file already on disk that passes is not fetched
again. An entry that lists neither digest stops the run, because a download
that cannot be checked is not used.

Usage::

    python tools/fetch_stb.py --sample 1 [--dest DIR]

The destination defaults to $PLIDAR_STB_DIR, then <repo>/data/stb.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

REPO = "Shao132/SP-TransientBench"
ROOT = Path(__file__).resolve().parent.parent
AGENT = "photon-counting-lidar"
CONFIG = ["IRF_global.csv", "config.yaml", "final_extrinsic_sp_to_livox.txt",
          "offset.txt"]


def sample_files(sample: int) -> list[str]:
    """Paths, relative to the dataset root, needed for one sample."""
    return (["README.md"] + ["config/" + f for f in CONFIG]
            + [f"DepthEstimate/Histogram/{sample}.txt",
               f"DepthEstimate/gt/{sample}.csv"])


def git_blob_sha1(data: bytes) -> str:
    """The git blob id of a file: sha1 over a size header and the content."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def expected_digest(entry: dict) -> tuple[str, str]:
    """The digest the server publishes for a tree entry. LFS files carry a
    sha256 of the content; the others carry the git blob id."""
    if entry.get("lfs", {}).get("oid"):
        return "sha256", entry["lfs"]["oid"]
    if entry.get("oid"):
        return "git-sha1", entry["oid"]
    raise SystemExit(f"{entry.get('path', '?')}: the server lists no digest, "
                     "refusing to use an unverifiable file")


def verify(path: Path, entry: dict) -> bool:
    """True when the file has the listed size and the listed digest."""
    kind, want = expected_digest(entry)
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if kind == "git-sha1":  # not LFS, so small enough to read at once
        return git_blob_sha1(path.read_bytes()) == want
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest() == want


def _open(url: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=60)


def _tree(folder: str) -> dict[str, dict]:
    # shortcut: no pagination, these folders hold a few dozen entries at most
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
    with _open(url + ("/" + folder if folder else "")) as r:
        return {e["path"]: e for e in json.load(r) if e.get("type") == "file"}


def _download(path: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with _open(f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}") as r, \
            part.open("wb") as f:
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
    part.replace(out)


def fetch(sample: int, dest: Path) -> Path:
    """Make sure every file of the sample is in dest and verified."""
    wanted = sample_files(sample)
    entries: dict[str, dict] = {}
    for folder in sorted({str(Path(p).parent).replace("\\", "/") for p in wanted}):
        entries.update(_tree("" if folder == "." else folder))
    for p in wanted:
        if p not in entries:
            raise SystemExit(f"{p}: not listed by the server")
        out = dest / p
        if verify(out, entries[p]):
            print(f"verified    {p}")
            continue
        print(f"downloading {p} ({entries[p]['size']} bytes)")
        _download(p, out)
        if not verify(out, entries[p]):
            out.unlink()
            raise SystemExit(f"{p}: digest mismatch after download, removed")
        print(f"verified    {p}")
    return dest


def _citation(readme: Path) -> str:
    text = readme.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"@\w+\{.*?\n\}", text, re.S)
    return m.group(0) if m else "Dong et al., 2026, arXiv 2606.18952"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--dest", type=Path, default=Path(
        os.environ.get("PLIDAR_STB_DIR") or ROOT / "data" / "stb"))
    a = ap.parse_args()
    dest = fetch(a.sample, a.dest)
    print("\nLicence: CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution)")
    print("Cite as:\n" + _citation(dest / "README.md"))
    print("\nDo not commit these files: the licence forbids redistributing "
          "them from this repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
