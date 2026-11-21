from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from community import community_louvain
from sklearn.preprocessing import normalize

from .data import RoutingData


class LayeredSkillGraph:
    """Scene assignments and skill-layer workflow transitions mined from training sessions."""

    def __init__(
        self,
        data: RoutingData,
        candidate_embeddings: np.ndarray,
        scene_resolution: float,
        scene_coactivation_min: int,
        scene_npmi_min: float,
        seed: int,
    ):
        self.candidates = data.candidates
        self.candidate_index = {name: i for i, name in enumerate(self.candidates)}
        self.candidate_embeddings = candidate_embeddings
        self.descriptions = data.descriptions
        self.coactivation_edges = data.coactivation_edges
        self.sequential_edges = data.sequential_edges
        self.sequential_out_counts = data.sequential_out_counts
        self.total_sessions = data.total_sessions
        self.scene_resolution = scene_resolution
        self.scene_coactivation_min = scene_coactivation_min
        self.scene_npmi_min = scene_npmi_min
        self.seed = seed

        self.transitions = self._build_transitions()
        self.scene_map = self._build_scenes(data)
        self.candidate_scenes = np.array(
            [self.scene_map.get(name, -1) for name in self.candidates]
        )
        self.scene_ids, self.scene_columns, self.scene_centroids = self._build_centroids()

    def _build_transitions(self) -> dict[str, dict[str, float]]:
        # P(b | a): row-normalized adjacent-turn transition counts
        transitions = defaultdict(dict)
        for (source, target), count in self.sequential_edges.items():
            total = self.sequential_out_counts.get(source, 0)
            if total:
                transitions[source][target] = count / total
        return transitions

    def _build_scenes(self, data: RoutingData) -> dict[str, int]:
        described = {name for name in self.candidates if self.descriptions.get(name)}
        graph = nx.Graph()
        for (left, right), count in self.coactivation_edges.items():
            if left >= right or left not in described or right not in described:
                continue
            left_count = data.coactivation_counts.get(left, 0)
            right_count = data.coactivation_counts.get(right, 0)
            if not left_count or not right_count or count < self.scene_coactivation_min:
                continue
            npmi = np.log(self.total_sessions * count / (left_count * right_count))
            npmi /= -np.log(count / self.total_sessions + 1e-12)
            if npmi > self.scene_npmi_min:
                graph.add_edge(left, right, weight=float(npmi))

        if graph.number_of_edges() == 0:
            return {name: i for i, name in enumerate(sorted(described))}

        scenes = community_louvain.best_partition(
            graph,
            weight="weight",
            resolution=self.scene_resolution,
            random_state=self.seed,
        )

        # skills without co-activation edges go to the nearest scene centroid
        grouped = defaultdict(list)
        for name, scene_id in scenes.items():
            grouped[scene_id].append(self.candidate_embeddings[self.candidate_index[name]])
        centroids = {
            scene_id: normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
            for scene_id, vectors in grouped.items()
        }
        for name in sorted(described - set(scenes)):
            vector = self.candidate_embeddings[self.candidate_index[name]]
            scenes[name] = max(centroids, key=lambda scene_id: float(vector @ centroids[scene_id]))
        return scenes

    def _build_centroids(self):
        scene_ids = sorted({int(value) for value in self.candidate_scenes if value >= 0})
        columns = {scene_id: column for column, scene_id in enumerate(scene_ids)}
        centroids = np.zeros(
            (len(scene_ids), self.candidate_embeddings.shape[1]), dtype=np.float32
        )
        for scene_id in scene_ids:
            mask = self.candidate_scenes == scene_id
            centroid = self.candidate_embeddings[mask].mean(axis=0)
            centroids[columns[scene_id]] = centroid / (np.linalg.norm(centroid) + 1e-12)
        return scene_ids, columns, centroids

    def routing_case(self, prior_skills: list[str], target: str) -> str:
        if not prior_skills:
            return "first"
        if target in prior_skills:
            return "repeat"
        target_scene = self.scene_map.get(target, -1)
        prior_scenes = {self.scene_map.get(name, -2) for name in prior_skills}
        return "in_scene_novel" if target_scene in prior_scenes else "out_scene"
