from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .prepare import read_skills, routing_records, session_edges


@dataclass
class RoutingData:
    candidates: list[str]
    descriptions: dict[str, str]
    train: list[dict]
    test: list[dict]
    total_sessions: int
    coactivation_edges: dict[tuple[str, str], int]
    coactivation_counts: Counter
    sequential_edges: dict[tuple[str, str], int]
    sequential_out_counts: Counter


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_data(path: str | Path) -> RoutingData:
    root = Path(path)
    candidates, descriptions = read_skills(_read_json(root / "skills.json"))
    train_sessions = _read_json(root / "train_sessions.json")
    test_sessions = _read_json(root / "test_sessions.json")
    candidate_set = set(candidates)

    # graph statistics come from training sessions only
    coactivation, sequential = session_edges(train_sessions, candidate_set)

    coactivation_edges: dict[tuple[str, str], int] = {}
    coactivation_counts = Counter()
    for (left, right), count in coactivation.items():
        coactivation_edges[left, right] = count
        coactivation_edges[right, left] = count
        coactivation_counts[left] += count
        coactivation_counts[right] += count

    sequential_out_counts = Counter()
    for (source, _), count in sequential.items():
        sequential_out_counts[source] += count

    return RoutingData(
        candidates=candidates,
        descriptions=descriptions,
        train=routing_records(train_sessions, candidate_set),
        test=routing_records(test_sessions, candidate_set),
        total_sessions=len(train_sessions),
        coactivation_edges=coactivation_edges,
        coactivation_counts=coactivation_counts,
        sequential_edges=sequential,
        sequential_out_counts=sequential_out_counts,
    )
