"""Offline checks for the dataset downloader. No network: the digests, the
size test and the file list are exercised on small temporary files. Plain
asserts and the standard library only."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import fetch_stb as F


def test_git_blob_sha1():
    assert F.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_expected_digest():
    assert F.expected_digest({"oid": "abc", "lfs": {"oid": "def", "size": 3}}) == ("sha256", "def")
    assert F.expected_digest({"oid": "abc", "size": 3}) == ("git-sha1", "abc")


def test_verify():
    data = b"hello\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.txt"
        p.write_bytes(data)
        plain = {"oid": F.git_blob_sha1(data), "size": len(data)}
        lfs = {"oid": "x", "size": len(data),
               "lfs": {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}}
        assert F.verify(p, plain) and F.verify(p, lfs)
        assert not F.verify(p, {**plain, "oid": "0" * 40})
        assert not F.verify(p, {**lfs, "lfs": {**lfs["lfs"], "oid": "0" * 64}})
        assert not F.verify(p, {**plain, "size": len(data) + 1})
        assert not F.verify(Path(d) / "missing", plain)
        try:
            F.verify(p, {"path": "f.txt", "size": len(data)})
        except SystemExit as e:
            assert "f.txt" in str(e)
        else:
            raise AssertionError("an entry without a digest must stop the run")


def test_sample_files():
    files = F.sample_files(1)
    assert "DepthEstimate/Histogram/1.txt" in files
    assert "DepthEstimate/gt/1.csv" in files
    for name in ("IRF_global.csv", "config.yaml",
                 "final_extrinsic_sp_to_livox.txt", "offset.txt"):
        assert "config/" + name in files
    assert "README.md" in files


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {t.__name__}: {e!r}")
            return 1
    print(f"{len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
