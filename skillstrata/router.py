from __future__ import annotations

from collections import defaultdict

import lightgbm as lgb
import numpy as np

from .data import RoutingData
from .graph import LayeredSkillGraph


FEATURE_NAMES = (
    "intent_proximity",
    "workflow_recency",
    "scene_compatibility",
)


class SkillRouter:
    """Contextual candidate generation, stratum features, and LambdaMART reranking."""

    def __init__(self, graph: LayeredSkillGraph, data: RoutingData, parameters: dict):
        self.graph = graph
        self.data = data
        self.parameters = parameters
        self.neighbors = self._build_neighbors()
        self.ranker = None

    def _build_neighbors(self) -> dict[str, set[str]]:
        neighbors = defaultdict(set)
        for source, targets in self.graph.transitions.items():
            for target in targets:
                neighbors[source].add(target)
                neighbors[target].add(source)
        minimum = int(self.parameters["expansion_coactivation_min"])
        for (left, right), count in self.data.coactivation_edges.items():
            if count >= minimum:
                neighbors[left].add(right)
        return neighbors

    def dense_scores(self, query_embeddings: np.ndarray) -> np.ndarray:
        return query_embeddings @ self.graph.candidate_embeddings.T

    def _candidate_pool(
        self,
        dense_row: np.ndarray,
        prior_skills: list[str],
        target_index: int | None = None,
    ) -> np.ndarray:
        index = self.graph.candidate_index
        seeds = np.argsort(-dense_row)[: int(self.parameters["dense_seed_count"])]
        mandatory = set(seeds.tolist())
        expanded: set[int] = set()

        for candidate_index in seeds:
            name = self.graph.candidates[candidate_index]
            expanded.update(index[value] for value in self.neighbors.get(name, ()) if value in index)

        for name in prior_skills:
            if name in index:
                mandatory.add(index[name])
            expanded.update(index[value] for value in self.neighbors.get(name, ()) if value in index)
            expanded.update(
                index[value] for value in self.graph.transitions.get(name, {}) if value in index
            )

        prior_scenes = {self.graph.scene_map.get(name) for name in prior_skills}
        prior_scenes.discard(None)
        if prior_scenes:
            expanded.update(
                np.where(np.isin(self.graph.candidate_scenes, list(prior_scenes)))[0].tolist()
            )

        # dense seeds and prior skills are always kept; only expansion candidates are capped
        expanded -= mandatory
        pool_cap = self.parameters.get("pool_cap")
        if pool_cap and len(expanded) > max(pool_cap - len(mandatory), 0):
            budget = max(pool_cap - len(mandatory), 0)
            values = np.fromiter(expanded, dtype=np.int64, count=len(expanded))
            keep = values[np.argsort(-dense_row[values])[:budget]]
            expanded = set(keep.tolist())

        pool = mandatory | expanded
        if target_index is not None and target_index >= 0:
            pool.add(target_index)
        return np.fromiter(pool, dtype=np.int64, count=len(pool))

    def _features(
        self,
        query_embedding: np.ndarray,
        prior_skills: list[str],
        pool: np.ndarray,
        scene_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        index = self.graph.candidate_index
        embeddings = self.graph.candidate_embeddings
        prior_indices = [index[name] for name in prior_skills if name in index]

        if prior_indices:
            intent = (embeddings[pool] @ embeddings[prior_indices].T).max(axis=1)
        else:
            intent = np.zeros(len(pool), dtype=np.float32)

        # later prior skills get larger weights; the weights are not normalized
        workflow = defaultdict(float)
        for position, name in enumerate(prior_skills):
            recency = (position + 1) / len(prior_skills)
            for target, probability in self.graph.transitions.get(name, {}).items():
                candidate_index = index.get(target)
                if candidate_index is not None:
                    workflow[candidate_index] += recency * probability
        workflow_recency = np.array(
            [workflow.get(int(candidate_index), 0.0) for candidate_index in pool],
            dtype=np.float32,
        )

        if scene_scores is None:
            scene_scores = query_embedding @ self.graph.scene_centroids.T
        scene = np.zeros(len(pool), dtype=np.float32)
        for row, candidate_index in enumerate(pool):
            column = self.graph.scene_columns.get(int(self.graph.candidate_scenes[candidate_index]))
            if column is not None:
                scene[row] = scene_scores[column]

        return np.column_stack([intent, workflow_recency, scene]).astype(np.float32)

    def fit(self, records: list[dict], query_embeddings: np.ndarray, seed: int) -> int:
        sample_count = int(self.parameters["train_sample"])
        if sample_count and sample_count < len(records):
            rng = np.random.RandomState(seed)
            selected = rng.choice(len(records), sample_count, replace=False)
            records = [records[i] for i in selected]
            query_embeddings = query_embeddings[selected]

        dense = self.dense_scores(query_embeddings)
        scene_scores = query_embeddings @ self.graph.scene_centroids.T
        features, labels, groups = [], [], []
        for row, record in enumerate(records):
            prior_skills = record.get("prior_skills", [])
            target_index = self.graph.candidate_index[record["skill"]]
            # the positive is inserted into the pool during training only
            pool = self._candidate_pool(dense[row], prior_skills, target_index)
            features.append(
                self._features(query_embeddings[row], prior_skills, pool, scene_scores[row])
            )
            labels.append((pool == target_index).astype(np.int32))
            groups.append(len(pool))

        lightgbm_parameters = self.parameters["lightgbm"]
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=int(lightgbm_parameters["n_estimators"]),
            learning_rate=float(lightgbm_parameters["learning_rate"]),
            num_leaves=int(lightgbm_parameters["num_leaves"]),
            min_child_samples=int(lightgbm_parameters["min_child_samples"]),
            subsample=float(lightgbm_parameters["subsample"]),
            colsample_bytree=float(lightgbm_parameters["colsample_bytree"]),
            random_state=seed,
            n_jobs=int(lightgbm_parameters["n_jobs"]),
            verbose=-1,
            label_gain=[0, 1],
        )
        self.ranker.fit(np.concatenate(features), np.concatenate(labels), group=np.asarray(groups))
        return len(records)

    def score(self, records: list[dict], query_embeddings: np.ndarray) -> np.ndarray:
        if self.ranker is None:
            raise RuntimeError("router must be fitted before scoring")
        scores = self.dense_scores(query_embeddings).astype(np.float32)
        scene_scores = query_embeddings @ self.graph.scene_centroids.T
        ranker_weight = float(self.parameters["ranker_weight"])
        pool_bonus = float(self.parameters["pool_bonus"])
        for row, record in enumerate(records):
            prior_skills = record.get("prior_skills", [])
            pool = self._candidate_pool(scores[row], prior_skills)
            features = self._features(query_embeddings[row], prior_skills, pool, scene_scores[row])
            ranker_scores = self.ranker.predict(features).astype(np.float32)
            ranker_scores -= ranker_scores.min()
            ranker_scores /= ranker_scores.max() + 1e-9
            # final score = dense + ranker_weight * normalized ranker score + pool_bonus
            scores[row, pool] = scores[row, pool] + ranker_weight * ranker_scores + pool_bonus
        return scores
