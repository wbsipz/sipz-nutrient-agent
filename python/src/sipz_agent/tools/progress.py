from __future__ import annotations

import json
import os
import sys
from typing import Any


def emit_progress(message: str, **details: Any) -> None:
    """Stream machine-readable progress without contaminating bridge stdout."""
    if os.getenv("SIPZ_PROGRESS_JSONL") != "1":
        return
    event = {"type": "progress", "message": message, "details": details}
    print(json.dumps(event, ensure_ascii=True), file=sys.stderr, flush=True)
