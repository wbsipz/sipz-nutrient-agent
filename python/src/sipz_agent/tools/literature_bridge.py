from __future__ import annotations

import json
import sys

from datetime import UTC, datetime

from sipz_agent.tools import literature, workflow


def tool_response(name: str, payload: dict[str, object]) -> dict[str, object]:
    started_at = datetime.now(UTC).isoformat()
    try:
        if name in literature.TOOL_INPUTS:
            result = literature.execute_tool(name, payload)
        elif name in workflow.TOOL_INPUTS:
            result = workflow.execute_tool(name, payload)
        else:
            raise ValueError(f"unknown_tool:{name}")
        return {
            "ok": True,
            "tool": name,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "result": result.model_dump(mode="json"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "tool": name,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": {"message": "expected one tool name"}}))
        return 2
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 2
    response = tool_response(sys.argv[1], payload)
    print(json.dumps(response, ensure_ascii=True))
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
