"""Vendored dependency import helpers.

This file centralizes loading packages from `.vendor/` so the rest of the project does not need to know where optional dependencies are installed.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def ensure_vendor_path() -> None:
    root = Path(__file__).resolve().parents[1]
    vendor_dir = root / '.vendor'
    if vendor_dir.exists():
        vendor_path = os.fspath(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.append(vendor_path)


def import_onnx():
    ensure_vendor_path()
    import onnx  # type: ignore

    return onnx


def import_numpy():
    ensure_vendor_path()
    import numpy  # type: ignore

    return numpy


def import_onnxruntime():
    ensure_vendor_path()
    import onnxruntime  # type: ignore

    return onnxruntime
