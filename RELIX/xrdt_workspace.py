"""Portable paths for the frozen inference components."""
from pathlib import Path as WorkspacePath
import hashlib

def workspace_root(source):
    root = next(p for p in WorkspacePath(source).resolve().parents
                if (p / "RELIX_PACKAGE.json").is_file())
    return root / "RELIX"

def canonical_relative(value):
    return str(value)

def frozen_source_digest(path):
    return hashlib.sha256(WorkspacePath(path).read_bytes()).hexdigest()

def source_digest_matches(path, expected):
    return frozen_source_digest(path) == expected
