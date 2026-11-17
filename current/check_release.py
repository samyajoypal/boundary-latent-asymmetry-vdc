"""Check publication exclusions and immutable source/data hashes before a push."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from run import ROOT, verify_source

REPO = ROOT.parent
MANIFEST = ROOT / "RELEASE_SHA256.json"
HISTORICAL_FIGURE = "results/paper2_dry_bean_confirmatory/analysis/dry_bean_boundary_margin.pdf"
HISTORICAL_FIGURE_SHA = "555432d94fbe349fff6699cf3c26e828b5ba44aa5dd076ec811aba02eb77b605"
FORBIDDEN_SUFFIXES = {".pdf", ".tex", ".bib", ".cls", ".sty", ".doc", ".docx", ".zip", ".tar", ".gz", ".pt", ".pth", ".npz", ".npy"}
FORBIDDEN_NAMES = {"papers", "manuscripts", "supplementary", "submission_packages", ".env", "id_rsa", "id_ed25519"}
SECRETS = [re.compile(x) for x in (
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    rb"gh[pousr]_[A-Za-z0-9]{36,}", rb"github_pat_[A-Za-z0-9_]{60,}",
    rb"AKIA[0-9A-Z]{16}",
)]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_file(name, data):
    path = Path(name)
    if name == HISTORICAL_FIGURE and digest(data) == HISTORICAL_FIGURE_SHA:
        return
    if path.suffix.lower() in FORBIDDEN_SUFFIXES or FORBIDDEN_NAMES.intersection(p.lower() for p in path.parts):
        raise ValueError(f"Document, private file, or training artifact is excluded: {name}")
    if data.startswith((b"%PDF-", b"PK\x03\x04")):
        raise ValueError(f"Unexpected binary document/archive: {name}")
    if any(pattern.search(data) for pattern in SECRETS):
        raise ValueError(f"Potential credential: {name}")
    if re.search(rb"\\(?:documentclass|begin\s*\{document\})", data):
        raise ValueError(f"Unexpected manuscript document content: {name}")
    if name.startswith("current/") and re.search(rb"/(?:home|scratch)/[A-Za-z0-9_]+/", data):
        raise ValueError(f"Private machine path in current release: {name}")


def selected_files(index=False):
    flags = ["--cached"] if index else ["--cached", "--others", "--exclude-standard"]
    reply = subprocess.check_output(["git", "ls-files", "-z", *flags], cwd=REPO)
    for name in sorted(set(reply.decode().split("\0")) - {""}):
        if index:
            data = subprocess.check_output(["git", "show", ":" + name], cwd=REPO)
        else:
            path = REPO / name
            if not path.exists():
                continue
            if path.is_symlink():
                raise ValueError(f"Release files must not be symlinks: {name}")
            data = path.read_bytes()
        yield name, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="store_true", help="Audit the exact staged Git content")
    parser.add_argument("--write-manifest", action="store_true", help="Maintainer: freeze current release file hashes")
    args = parser.parse_args()
    if args.index and args.write_manifest:
        parser.error("Do not modify the manifest while auditing the index")
    verify_source()
    files = dict(selected_files(args.index))
    for name, data in files.items():
        check_file(name, data)
    hashes = {name.removeprefix("current/"): digest(data) for name, data in files.items()
              if name.startswith("current/") and name != "current/RELEASE_SHA256.json"}
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    else:
        expected = json.loads(files["current/RELEASE_SHA256.json"])
        if hashes != expected:
            changed = sorted(k for k in hashes.keys() | expected.keys() if hashes.get(k) != expected.get(k))
            raise ValueError(f"Current release differs from its manifest: {changed}")
    print(f"Publication guard passed for {len(files)} files; {len(hashes)} current release files verified.")
    print("No manuscript, supplementary document, submission archive, or detected credential is included.")


if __name__ == "__main__":
    main()
