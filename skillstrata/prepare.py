from __future__ import annotations

from collections import Counter

MAX_QUERY_LENGTH = 500


def read_skills(skills: list[dict]) -> tuple[list[str], dict[str, str]]:
    descriptions = {}
    for skill in skills:
        name = str(skill["name"]).strip()
        if name in descriptions:
            raise ValueError(f"duplicate skill: {name}")
        descriptions[name] = str(skill.get("description", "")).strip() or name
    return sorted(descriptions), descriptions


def routing_records(sessions: list[dict], candidates: set[str]) -> list[dict]:
    # one record per turn: the query and earlier skills in the session predict the turn's skill
    records = []
    for session in sessions:
        prior_skills = []
        for turn_idx, turn in enumerate(session["turns"]):
            skill = str(turn.get("skill", "")).strip()
            query = str(turn.get("query", "")).strip()
            if skill not in candidates or not query:
                continue
            records.append(
                {
                    "query": query[:MAX_QUERY_LENGTH],
                    "skill": skill,
                    "session_id": str(session.get("session_id", "")),
                    "turn_idx": turn_idx,
                    "prior_skills": list(prior_skills),
                }
            )
            prior_skills.append(skill)
    return records


def session_edges(sessions: list[dict], candidates: set[str]) -> tuple[dict, dict]:
    # co-activation: skill pairs used in the same session (order-free, counted once per session)
    # sequential: skill pairs called in adjacent turns (directed)
    coactivation = Counter()
    sequential = Counter()
    for session in sessions:
        sequence = [str(turn.get("skill", "")).strip() for turn in session["turns"]]
        sequence = [skill for skill in sequence if skill in candidates]
        used = sorted(set(sequence))
        for position, left in enumerate(used):
            for right in used[position + 1 :]:
                coactivation[left, right] += 1
        for source, target in zip(sequence, sequence[1:]):
            if source != target:
                sequential[source, target] += 1
    # Louvain depends on edge insertion order, so edges are returned in a fixed
    # order: count descending, ties by first appearance
    return dict(coactivation.most_common()), dict(sequential.most_common())
