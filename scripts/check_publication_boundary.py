"""Limited fail-closed publication hygiene checks over files actually tracked by Git.

This does not replace GitHub secret scanning, a complete content audit,
license review, or validation that all included data is safe to disclose.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_EXTENSIONS = {".pem", ".key", ".p12", ".sqlite", ".sqlite3", ".db", ".parquet", ".feather", ".jsonl"}
# Conservative signatures only, to avoid misclassifying empty API key names.
SECRET_SIGNATURES = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"ghp_[a-zA-Z0-9]{36}"),
    re.compile(rb"github_pat_[a-zA-Z0-9_]{60,}"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)


def check_publication_boundary(repo: Path = ROOT) -> int:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo)
    paths = [Path(name.decode("utf-8")) for name in output.split(b"\0") if name]
    if not paths:
        raise ValueError("No tracked paths to audit")
    problems: list[str] = []
    for relative in paths:
        name = relative.name
        lower = name.lower()
        if (
            lower == "agents.md"
            or "prompt" in lower
            or lower.startswith("continuation_")
            or lower == ".env"
            or (lower.startswith(".env.") and lower != ".env.example")
            or relative.suffix.lower() in BANNED_EXTENSIONS
        ):
            problems.append(f"Restricted tracked filename: {relative}")
            continue
        path = repo / relative
        # Images are intentionally published graphical diagnostics.
        if relative.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            continue
        content = path.read_bytes()
        for signature in SECRET_SIGNATURES:
            if signature.search(content):
                problems.append(f"Credential-like content in tracked file: {relative}")
                break
    if problems:
        raise ValueError("\n".join(problems))
    print(f"Publication hygiene check PASSED for {len(paths)} tracked paths (limited signatures only).")
    return len(paths)


if __name__ == "__main__":
    check_publication_boundary()
