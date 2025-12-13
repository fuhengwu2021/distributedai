#!/usr/bin/env python3
"""Small router example mimicking session affinity selection.

This is a simplified runner selector for unit testing routing logic.
Replace heuristics with real cluster state and KV-location metadata.
"""
from typing import Dict, Optional


RUNNERS = {
    "runner-a": {"available": True, "has_kv_sessions": {"s1"}},
    "runner-b": {"available": True, "has_kv_sessions": set()},
    "runner-c": {"available": False, "has_kv_sessions": set()},
}


def session_has_hot_kv(session_id: str) -> Optional[str]:
    for r, meta in RUNNERS.items():
        if session_id in meta.get("has_kv_sessions", set()):
            return r
    return None


def select_best_runner(request: Dict) -> Optional[str]:
    # Very simple capacity/locality heuristic: pick first available runner
    for r, meta in RUNNERS.items():
        if meta.get("available"):
            return r
    return None


def route_request(request: Dict) -> Optional[str]:
    session = request.get("session_id")
    r_with_kv = session_has_hot_kv(session)
    if r_with_kv:
        return r_with_kv
    return select_best_runner(request)


def main():
    req1 = {"session_id": "s1", "payload": "hello"}
    req2 = {"session_id": "s2", "payload": "other"}
    print("Routing req1 ->", route_request(req1))
    print("Routing req2 ->", route_request(req2))


if __name__ == "__main__":
    main()
