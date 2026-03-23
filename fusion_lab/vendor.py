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