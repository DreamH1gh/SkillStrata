from __future__ import annotations

import argparse
import json

import numpy as np

from skillstrata.config import dataset_config, load_config, model_config
from skillstrata.data import load_data
from skillstrata.encoder import TextEncoder
from skillstrata.graph import LayeredSkillGraph
from skillstrata.metrics import evaluate, evaluate_subset
from skillstrata.router import FEATURE_NAMES, SkillRouter


ROUTING_CASES = ("first", "repeat", "in_scene_novel", "out_scene")


def evaluate_by_case(scores, records, graph, top_k):
    targets = [record["skill"] for record in records]
    cases = {name: [] for name in ROUTING_CASES}
    for index, record in enumerate(records):
        cases[graph.routing_case(record.get("prior_skills", []), record["skill"])].append(index)
    return {
        "all": evaluate(scores, targets, graph.candidate_index, top_k),
        **{
            name: evaluate_subset(scores, indices, targets, graph.candidate_index, top_k)
            for name, indices in cases.items()
        },
    }


def run_experiment(config, dataset_name, model_name, device, cache_only):
    dataset_parameters = dataset_config(config, dataset_name)
    model_parameters = model_config(config, model_name)
    data = load_data(dataset_parameters["path"])

    encoder = TextEncoder(
        model_id=model_parameters["model_id"],
        model_name=model_name,
        cache_dir=config["root"] / config["paths"]["cache"] / model_name / dataset_name,
        device=device,
        batch_size=int(config["encoding"]["batch_size"]),
        max_sequence_length=int(config["encoding"]["max_sequence_length"]),
        query_prefix=model_parameters["query_prefix"],
        cache_only=cache_only,
    )
    candidate_texts = [data.descriptions.get(name, "") or name for name in data.candidates]
    candidate_embeddings = encoder.encode_candidates(candidate_texts)
    test_embeddings = encoder.encode_queries([record["query"] for record in data.test], "test")
    train_embeddings = encoder.encode_queries([record["query"] for record in data.train], "train")

    graph = LayeredSkillGraph(
        data=data,
        candidate_embeddings=candidate_embeddings,
        scene_resolution=float(dataset_parameters["scene_resolution"]),
        scene_coactivation_min=int(dataset_parameters["scene_coactivation_min"]),
        scene_npmi_min=float(dataset_parameters["scene_npmi_min"]),
        seed=int(config["seed"]),
    )

    ranking_parameters = dict(config["ranking"])
    ranking_parameters["expansion_coactivation_min"] = dataset_parameters["expansion_coactivation_min"]
    ranking_parameters["pool_cap"] = dataset_parameters["pool_cap"]
    router = SkillRouter(graph, data, ranking_parameters)
    ranker_train = router.fit(data.train, train_embeddings, int(config["seed"]))
    scores = router.score(data.test, test_embeddings)

    top_k = [int(value) for value in config["evaluation"]["top_k"]]
    results = evaluate_by_case(scores, data.test, graph, top_k)
    output = {
        "dataset": dataset_name,
        "model": model_name,
        "counts": {
            "candidates": len(data.candidates),
            "train": len(data.train),
            "test": len(data.test),
            "scenes": len(graph.scene_ids),
            "ranker_train": ranker_train,
        },
        "parameters": {
            "seed": config["seed"],
            "dataset": {key: value for key, value in dataset_parameters.items() if key != "path"},
            "encoding": config["encoding"],
            "ranking": config["ranking"],
            "features": list(FEATURE_NAMES),
        },
        "results": results,
    }

    result_dir = config["root"] / config["paths"]["results"] / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{dataset_name}.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    metrics = results["all"]
    print(
        f"[{model_name} / {dataset_name}] "
        + " ".join(f"Hit@{k}={metrics[f'Hit@{k}']:.4f}" for k in top_k)
        + f" MRR={metrics['MRR']:.4f}"
    )
    print(f"saved: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SkillStrata")
    parser.add_argument("--config", default="experiment_config.json")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model", default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(int(config["seed"]))
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    models = list(config["models"]) if args.model == "all" else [args.model]
    for model_name in models:
        for dataset_name in datasets:
            run_experiment(config, dataset_name, model_name, args.device, args.cache_only)


if __name__ == "__main__":
    main()
