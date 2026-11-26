"""Configuration-driven experiment stages; local checks never invoke a model API."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import yaml

from liars_agent.agents.classifier_bot import SmallClassifierBot
from liars_agent.agents.rule_bots import create_rule_agent, CardCounterBot
from liars_agent.engine.replay import GameReplayBundle, save_replays_jsonl
from liars_agent.engine.state import LiarsBarEngine, _derive_seed
from liars_agent.training.dataset_splitter import split_dataset_by_game_id, split_game_ids
from liars_agent.contracts import current_contract, require_contract

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/sft_qwen35_9b_lora.yaml"
MODEL_LABELS = {"zero_shot": "Qwen3.5-9B Zero-shot", "small_classifier": "Small classifier",
                "rule_counter": "Card counter", "lora_sft": "Qwen3.5-9B SFT", "lora_rs": "Qwen3.5-9B RS"}
OPTIONAL_MODEL_LABELS = {"lora_rl": "Qwen3.5-9B RL", "lora_rl_v2": "Qwen3.5-9B RL v2"}
ALL_MODEL_LABELS = {**MODEL_LABELS, **OPTIONAL_MODEL_LABELS}
DEFAULT_RL_CONFIG = {
    "loss_fn": "ppo",
    "num_steps": 30,
    "groups_per_step": 4,
    "group_size": 4,
    "learning_rate": 2.0e-5,
    "temperature": 0.7,
    "top_p": 0.9,
    "win_weight": 0.70,
    "round_survival_weight": 0.30,
    "gamma": 1.0,
    "challenge_hit_bonus": 0.0,
    "caught_bluff_penalty": 0.0,
    "fallback_penalty": -1.0,
    "clip_advantage": 3.0,
    "belief_aux_batch_size": 8,
    "belief_aux_weight": 0.25,
    "on_policy_belief_ratio": 0.0,
    "posterior_opponent_prob": 0.25,
    "switcher_opponent_prob": 0.0,
    "self_play_opponent_prob": 0.0,
    "rollout_workers": 1,
    "eval_every_steps": 0,
    "eval_blocks": 6,
    "eval_temperature": 0.2,
    "checkpoint_every_steps": 10,
}
DEFAULT_RL_V2_CONFIG = {
    "loss_fn": "ppo",
    "num_steps": 80,
    "groups_per_step": 5,
    "group_size": 4,
    "learning_rate": 3.0e-5,
    "temperature": 1.0,
    "top_p": 0.98,
    "win_weight": 0.70,
    "round_survival_weight": 0.30,
    "gamma": 0.90,
    "challenge_hit_bonus": 0.15,
    "caught_bluff_penalty": -0.20,
    "fallback_penalty": -1.0,
    "clip_advantage": 3.0,
    "belief_aux_batch_size": 8,
    "belief_aux_weight": 0.25,
    "on_policy_belief_ratio": 0.50,
    "posterior_opponent_prob": 0.25,
    "switcher_opponent_prob": 0.15,
    "self_play_opponent_prob": 0.10,
    "rollout_workers": 4,
    "eval_every_steps": 10,
    "eval_blocks": 6,
    "eval_temperature": 0.2,
    "checkpoint_every_steps": 10,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def config_method_version(config: dict) -> str:
    return str(config.get("method_version", "v2"))


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = read_json(path) if path.suffix == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["platform"] != "tinker":
        raise ValueError("The implemented real training backend is Tinker.")
    mv = config_method_version(config)
    if "contract" in config:
        require_contract(config, "saved experiment configuration", method_version=mv)
    if "opponents" not in config:
        pool_path = path.parent / config.pop("opponent_config", "opponent_pools.yaml")
        config["opponents"] = yaml.safe_load(pool_path.read_text(encoding="utf-8"))
    # Instantiate configured agents once to catch misspelled or unused parameters.
    if mv == "adaptive_v4":
        from liars_agent.training.adaptive_opponents import create_v4_opponent
        for pool in ("seen_pool", "unseen_pool"):
            for spec in config["opponents"][pool]["agents"]:
                create_v4_opponent(spec["class"], name=spec["id"], **spec.get("params", {}))
    else:
        for pool in ("seen_pool", "unseen_pool"):
            for spec in config["opponents"][pool]["agents"]:
                create_rule_agent(spec["class"], name=spec["id"], **spec.get("params", {}))
    config["contract"] = current_contract(mv)
    from liars_agent.training.action_supervision import normalize_action_policy
    config["training"]["action_supervision"] = normalize_action_policy(config["training"].get("action_supervision"))
    if config["dataset"].get("history_ablation_augmentation_ratio", 0) != 0:
        raise ValueError("Training history augmentation is disabled in this method version.")
    if mv == "v2" and {s["class"] for s in config["opponents"]["seen_pool"]["agents"]} != {"HonestConservativeBot", "AggressiveBlufferBot", "CardCounterBot"}:
        raise ValueError("Training requires the three configured seen classes only.")
    return config


def opponent_specs(config: dict, pool: str = "seen") -> list[dict]:
    pools = config["opponents"]
    specs = {a["id"]: a for key in ("seen_pool", "unseen_pool") for a in pools[key]["agents"]}
    return [specs[name] for name in pools["default_matchups"][f"{pool}_lineup"]]


def teacher_config(config: dict) -> dict:
    specs = opponent_specs(config)
    actor = next(s for s in specs if s["class"] == "CardCounterBot")
    return {"actor": actor, "opponents": specs}


def opponent_factories(config: dict, pool: str = "seen") -> list:
    pools = config["opponents"]
    specs = {a["id"]: a for key in ("seen_pool", "unseen_pool") for a in pools[key]["agents"]}
    if config_method_version(config) == "adaptive_v4":
        from liars_agent.training.adaptive_opponents import create_v4_opponent
        return [partial(create_v4_opponent, specs[name]["class"], name, **specs[name].get("params", {}))
                for name in pools["default_matchups"][f"{pool}_lineup"]]
    return [partial(create_rule_agent, specs[name]["class"], name, **specs[name].get("params", {}))
            for name in pools["default_matchups"][f"{pool}_lineup"]]


def _generate_data_batch(job: tuple) -> dict:
    from liars_agent.training.data_generator import generate_rule_replays, extract_action_samples, extract_belief_samples
    config, output, batch_id, games, target, belief_target = job
    path = Path(output)
    completed = path / f"batch_{batch_id:04d}.json"
    if completed.exists():
        result = read_json(completed)
        require_contract(result, "data batch")
        return result
    ds = config["dataset"]
    seed = _derive_seed(config["seed"], "dataset_batch", batch_id)
    replay_path = path / f"replays_{batch_id:04d}.jsonl"
    if replay_path.exists():
        from liars_agent.engine.replay import verify_replay_determinism
        replays = list(read_jsonl(replay_path))
        for replay in replays:
            verify_replay_determinism(replay, raise_on_error=True)
    else:
        replays = generate_rule_replays(games, seed, opponent_factories(config))
        save_replays_jsonl(replays, replay_path)
        replays = [r.to_dict() for r in replays]
    # Assign every whole game before either task selects its decision points.
    partition = split_game_ids((r["game_id"] for r in replays), ds["train_ratio"], ds["val_ratio"], seed)
    write_json(path / f"partition_{batch_id:04d}.json", {"contract": current_contract(), "game_splits": partition})
    actions = extract_action_samples(replays, target, ds["rollout_trajectories_per_state"],
        ds["min_advantage_margin"], seed, teacher_config(config))
    beliefs = extract_belief_samples(replays, belief_target, seed)
    samples_path = path / f"samples_{batch_id:04d}.jsonl"
    write_jsonl(samples_path, actions + beliefs)
    result = {"contract": current_contract(), "batch_id": batch_id, "num_samples": len(actions),
        "num_belief_samples": len(beliefs), "num_games": len(replays), "game_splits": partition,
        "samples_path": str(samples_path), "replays_path": str(replay_path)}
    write_json(completed, result)
    return result


def _generate_v4_replay_batch(job: tuple) -> dict:
    from liars_agent.engine.replay import verify_replay_determinism
    from liars_agent.training.adaptive_teacher import extract_v4_belief_samples, generate_adaptive_v4_replays

    config, output, batch_id, games, belief_target = job
    path = Path(output)
    meta_path = path / f"v4_replays_batch_{batch_id:04d}.json"
    if meta_path.exists():
        result = read_json(meta_path)
        require_contract(result, "v4 replay batch", method_version="adaptive_v4")
        return result
    ds = config["dataset"]
    seed = _derive_seed(config["seed"], "dataset_batch", batch_id)
    replay_path = path / f"replays_{batch_id:04d}.jsonl"
    if replay_path.exists():
        replays = list(read_jsonl(replay_path))
        for replay in replays:
            verify_replay_determinism(replay, raise_on_error=True)
    else:
        replays = generate_adaptive_v4_replays(games, seed)
        save_replays_jsonl(replays, replay_path)
        replays = [r.to_dict() for r in replays]
    partition = split_game_ids((r["game_id"] for r in replays), ds["train_ratio"], ds["val_ratio"], seed)
    write_json(path / f"partition_{batch_id:04d}.json", {"contract": current_contract("adaptive_v4"), "game_splits": partition})
    beliefs = extract_v4_belief_samples(replays, belief_target, seed)
    belief_path = path / f"beliefs_{batch_id:04d}.jsonl"
    write_jsonl(belief_path, beliefs)
    result = {
        "contract": current_contract("adaptive_v4"),
        "batch_id": batch_id,
        "num_belief_samples": len(beliefs),
        "num_games": len(replays),
        "game_splits": partition,
        "beliefs_path": str(belief_path),
        "replays_path": str(replay_path),
    }
    write_json(meta_path, result)
    return result


def _extract_v4_action_batch(job: tuple) -> dict:
    from liars_agent.training.adaptive_teacher import CalibratedPosteriorBelief, extract_v4_action_samples

    config, output, batch_meta, target, posterior_path = job
    path = Path(output)
    batch_id = int(batch_meta["batch_id"])
    completed = path / f"batch_{batch_id:04d}.json"
    if completed.exists():
        result = read_json(completed)
        require_contract(result, "v4 data batch", method_version="adaptive_v4")
        return result
    ds = config["dataset"]
    seed = _derive_seed(config["seed"], "dataset_batch", batch_id)
    calibrated = CalibratedPosteriorBelief.load(posterior_path)
    replays = list(read_jsonl(Path(batch_meta["replays_path"])))
    beliefs = list(read_jsonl(Path(batch_meta["beliefs_path"])))
    actions = extract_v4_action_samples(
        replays,
        target,
        belief_fn=calibrated.predict,
        num_rollouts=int(ds["rollout_trajectories_per_state"]),
        margin=float(ds["min_advantage_margin"]),
        seed=seed,
        history_critical_ratio=float(ds.get("history_critical_ratio", 0.5)),
        action_supervision_policy=config["training"].get("action_supervision"),
    )
    samples_path = path / f"samples_{batch_id:04d}.jsonl"
    write_jsonl(samples_path, actions + beliefs)
    result = {
        "contract": current_contract("adaptive_v4"),
        "batch_id": batch_id,
        "num_samples": len(actions),
        "num_belief_samples": len(beliefs),
        "num_games": int(batch_meta["num_games"]),
        "game_splits": batch_meta["game_splits"],
        "samples_path": str(samples_path),
        "replays_path": batch_meta["replays_path"],
    }
    write_json(completed, result)
    print(f"[v4 batch {batch_id:04d}] games={result['num_games']}, action={result['num_samples']}, belief={result['num_belief_samples']}", flush=True)
    return result


def generate_adaptive_v4_data(config: dict, output: Path) -> dict:
    """Two-pass v4 generator: symmetric pool replays + train-only calibrated posterior + posterior rollout teacher."""
    from liars_agent.engine.observation import PlayerObservation
    from liars_agent.training.adaptive_teacher import (
        CalibratedPosteriorBelief,
        build_v4_counterfactual_train_samples,
    )

    ds = config["dataset"]
    batches_dir = output / "dataset/batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    games, target = int(ds["num_games"]), int(ds["total_target_samples"])
    belief_target = int(ds["belief_target_samples"])
    if min(games, target, belief_target) < 3 or ds["workers"] < 1:
        raise ValueError("Data generation needs at least three games and examples per task.")
    batch_size = int(ds["batch_games"])
    if batch_size < 3:
        raise ValueError("Batches need at least three games for shared train/val/test partitions.")

    replay_jobs = []
    action_quotas = []
    for batch_id, start in enumerate(range(0, games, batch_size)):
        count = min(batch_size, games - start)
        quota = target * (start + count) // games - target * start // games
        belief_quota = belief_target * (start + count) // games - belief_target * start // games
        replay_jobs.append((config, str(batches_dir), batch_id, count, belief_quota))
        action_quotas.append(quota)

    if int(ds["workers"]) == 1:
        replay_metas = list(map(_generate_v4_replay_batch, replay_jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(int(ds["workers"]), len(replay_jobs))) as pool:
            replay_metas = list(pool.map(_generate_v4_replay_batch, replay_jobs))

    # Fit CalibratedPosteriorBelief strictly on train-split belief rows (zero val/test leakage!)
    posterior_path = output / "models/v4_calibrated_posterior.json"
    if posterior_path.exists():
        calibrated = CalibratedPosteriorBelief.load(posterior_path)
    else:
        train_game_set = {gid for meta in replay_metas for gid in meta["game_splits"]["train"]}
        train_belief_pairs = []
        for meta in replay_metas:
            for row in read_jsonl(Path(meta["beliefs_path"])):
                if row["game_id"] in train_game_set:
                    train_belief_pairs.append((PlayerObservation.from_dict(row["observation"]), int(row["label"])))
        if not train_belief_pairs:
            raise ValueError("No train-split belief samples available to fit v4 calibrated posterior.")
        calibrated = CalibratedPosteriorBelief()
        calibrated.fit(train_belief_pairs)
        calibrated.save(posterior_path)

    action_jobs = [
        (config, str(batches_dir), meta, quota, str(posterior_path))
        for meta, quota in zip(replay_metas, action_quotas)
    ]
    if int(ds["workers"]) == 1:
        results = list(map(_extract_v4_action_batch, action_jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(int(ds["workers"]), len(action_jobs))) as pool:
            results = list(pool.map(_extract_v4_action_batch, action_jobs))

    max_games = int(ds.get("max_games", games * 2))
    completed_games = sum(item["num_games"] for item in results)
    sample_count = sum(item["num_samples"] for item in results)
    belief_count = sum(item["num_belief_samples"] for item in results)
    while (sample_count < target or belief_count < belief_target) and completed_games < max_games:
        count = min(batch_size, max_games - completed_games)
        next_id = len(results)
        meta = _generate_v4_replay_batch((
            config, str(batches_dir), next_id, count,
            min(belief_target - belief_count, math.ceil(belief_target / games) * count),
        ))
        item = _extract_v4_action_batch((
            config, str(batches_dir), meta,
            min(target - sample_count, math.ceil(target / games) * count),
            str(posterior_path),
        ))
        results.append(item)
        completed_games += item["num_games"]
        sample_count += item["num_samples"]
        belief_count += item["num_belief_samples"]

    samples = [s for item in results for s in read_jsonl(Path(item["samples_path"]))]
    game_ids = {key: [gid for item in results for gid in item["game_splits"][key]] for key in ("train", "val", "test")}

    # Optional counterfactual history-contrast pairs strictly into the train split
    cf_pairs = int(ds.get("counterfactual_train_pairs", 0))
    cf_samples, cf_gids = build_v4_counterfactual_train_samples(
        cf_pairs,
        belief_fn=calibrated.predict,
        seed=int(config["seed"]),
        num_rollouts=int(ds["rollout_trajectories_per_state"]),
        action_supervision_policy=config["training"].get("action_supervision"),
    )
    if cf_samples:
        samples.extend(cf_samples)
        game_ids["train"].extend(cf_gids)
        sample_count += len(cf_samples)

    splits = split_dataset_by_game_id(samples, game_splits=game_ids)
    for key, rows in splits.items():
        for task in ("action", "belief"):
            subset = [r for r in rows if r["task"] == task]
            if not subset:
                raise ValueError(f"{key} has no {task} samples; generate more whole games.")
            write_jsonl(output / f"dataset/{task}_{key}.jsonl", subset)
        write_jsonl(output / f"dataset/{key}.jsonl", rows)
    write_json(output / "dataset/game_splits.json", game_ids)
    write_jsonl(output / "dataset/sft_all_samples.jsonl", samples)
    classifier = SmallClassifierBot(name="small_classifier")
    classifier_stats = classifier.fit_from_belief_samples(
        (r for r in splits["train"] if r["task"] == "belief"), method_version="adaptive_v4"
    )
    classifier.save_weights(output / "models/small_classifier_weights.json", method_version="adaptive_v4")
    summary = {
        "contract": current_contract("adaptive_v4"),
        "method_version": "adaptive_v4",
        "target_samples": target,
        "actual_samples": sample_count,
        "belief_target_samples": belief_target,
        "actual_belief_samples": belief_count,
        "counterfactual_train_samples": len(cf_samples),
        "calibrated_posterior_path": str(posterior_path),
        "shortfall": max(0, target - sample_count) + max(0, belief_target - belief_count),
        "games_generated": completed_games,
        "samples_per_split": {k: len(v) for k, v in splits.items()},
        "task_samples_per_split": {k: {t: sum(r["task"] == t for r in v) for t in ("action", "belief")} for k, v in splits.items()},
        "games_per_split": {k: len(v) for k, v in game_ids.items()},
        "classifier_fit": classifier_stats,
        "batches": results,
        "teacher": "calibrated_posterior_continuation_v4",
    }
    write_json(output / "dataset/data_summary.json", summary)
    print(f"Data (adaptive_v4): action {sample_count}/{target} (incl. {len(cf_samples)} cf), belief {belief_count}/{belief_target}", flush=True)
    return summary


def generate_data(config: dict, output: Path) -> dict:
    if config_method_version(config) == "adaptive_v4":
        return generate_adaptive_v4_data(config, output)
    ds = config["dataset"]
    batches_dir = output / "dataset/batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    games, target = int(ds["num_games"]), int(ds["total_target_samples"])
    belief_target = int(ds["belief_target_samples"])
    if min(games, target, belief_target) < 3 or ds["workers"] < 1:
        raise ValueError("Data generation needs at least three games and examples per task.")
    batch_size = int(ds["batch_games"])
    if batch_size < 3:
        raise ValueError("Batches need at least three games for shared train/val/test partitions.")
    jobs = []
    for batch_id, start in enumerate(range(0, games, batch_size)):
        count = min(batch_size, games - start)
        quota = target * (start + count) // games - target * start // games
        belief_quota = belief_target * (start + count) // games - belief_target * start // games
        jobs.append((config, str(batches_dir), batch_id, count, quota, belief_quota))
    if int(ds["workers"]) == 1:
        results = list(map(_generate_data_batch, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(int(ds["workers"]), len(jobs))) as pool:
            results = list(pool.map(_generate_data_batch, jobs))
    max_games = int(ds.get("max_games", games * 2))
    completed_games = sum(item["num_games"] for item in results)
    sample_count = sum(item["num_samples"] for item in results)
    belief_count = sum(item["num_belief_samples"] for item in results)
    while (sample_count < target or belief_count < belief_target) and completed_games < max_games:
        count = min(batch_size, max_games - completed_games)
        item = _generate_data_batch((config, str(batches_dir), len(results), count,
            min(target - sample_count, math.ceil(target / games) * count),
            min(belief_target - belief_count, math.ceil(belief_target / games) * count)))
        results.append(item)
        completed_games += item["num_games"]
        sample_count += item["num_samples"]
        belief_count += item["num_belief_samples"]
    samples = [s for item in results for s in read_jsonl(Path(item["samples_path"]))]
    game_ids = {key: [gid for item in results for gid in item["game_splits"][key]] for key in ("train", "val", "test")}
    splits = split_dataset_by_game_id(samples, game_splits=game_ids)
    for key, rows in splits.items():
        for task in ("action", "belief"):
            subset = [r for r in rows if r["task"] == task]
            if not subset:
                raise ValueError(f"{key} has no {task} samples; generate more whole games.")
            write_jsonl(output / f"dataset/{task}_{key}.jsonl", subset)
        write_jsonl(output / f"dataset/{key}.jsonl", rows)
    write_json(output / "dataset/game_splits.json", game_ids)
    write_jsonl(output / "dataset/sft_all_samples.jsonl", samples)
    classifier = SmallClassifierBot(name="small_classifier")
    classifier_stats = classifier.fit_from_belief_samples(r for r in splits["train"] if r["task"] == "belief")
    classifier.save_weights(output / "models/small_classifier_weights.json")
    summary = {"contract": current_contract(), "target_samples": target, "actual_samples": sample_count,
        "belief_target_samples": belief_target, "actual_belief_samples": belief_count,
        "shortfall": max(0, target - sample_count) + max(0, belief_target - belief_count),
        "games_generated": completed_games, "samples_per_split": {k: len(v) for k, v in splits.items()},
        "task_samples_per_split": {k: {t: sum(r["task"] == t for r in v) for t in ("action", "belief")} for k, v in splits.items()},
        "games_per_split": {k: len(v) for k, v in game_ids.items()}, "classifier_fit": classifier_stats,
        "batches": results, "teacher": "pool_prior_counter_continuation"}
    write_json(output / "dataset/data_summary.json", summary)
    print(f"Data: action {sample_count}/{target}, belief {belief_count}/{belief_target}", flush=True)
    return summary


def make_backend(config: dict, output: Path):
    from liars_agent.training.tinker_backend import TinkerBackend
    return TinkerBackend(base_model=config["base_model"], budget_path=ROOT / "outputs/budget_usage.jsonl",
                         budget_limit_usd=config["budget_limit_usd"])


def trial_settings(config: dict) -> tuple[dict, dict]:
    from liars_agent.training.trial_verification import normalize_trial_verification_config
    verification = normalize_trial_verification_config(config["training"].get("trial_verification"))
    sampling = {key: config["sampling"][key] for key in ("temperature", "top_p", "max_new_tokens")}
    return verification, sampling


def require_verified_trial(config: dict, metadata: dict, splits: dict | None = None, *,
                           split_paths: dict[str, Path] | None = None) -> None:
    """A saved trial is usable only if its complete verification actually passed."""
    from liars_agent.training.action_supervision import require_training_objective
    from liars_agent.training.tinker_train import split_paths_sha256, splits_sha256
    mv = config_method_version(config)
    verification, sampling = trial_settings(config)
    require_contract(metadata, "trial metadata", method_version=mv)
    require_training_objective(metadata, config["training"].get("action_supervision"))
    closure = metadata.get("trial_verification") or {}
    if (metadata.get("training_status") != "trial_complete" or closure.get("status") != "passed"
            or closure.get("version") != verification["version"]
            or closure.get("config") != verification or closure.get("sampling_config") != sampling
            or closure.get("sampling", {}).get("status") != "passed"
            or closure.get("training", {}).get("status") != "passed"):
        raise ValueError("A passed update/sampler/state/optimizer trial verification is required; run sft_trial first.")
    if (metadata.get("base_model") != config["base_model"] or metadata.get("lora_config") != config["lora"]
            or closure.get("checkpoints") != {key: metadata.get(key) for key in ("state_path", "sampler_path")}):
        raise ValueError("Trial model, LoRA configuration or verified checkpoint paths differ.")
    training = config["training"]
    expected = {key: training[key] for key in ("learning_rate", "batch_size", "num_epochs", "max_seq_length")}
    expected.update(seed=training.get("seed", config["seed"]), warmup_ratio=.05)
    if any(metadata.get("training_metrics", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Trial training configuration differs from the requested formal training.")
    fingerprint = None
    if split_paths is not None:
        fingerprint = split_paths_sha256(split_paths)
    elif splits is not None:
        fingerprint = splits_sha256(splits)
    if fingerprint is not None:
        if metadata.get("dataset_sha256") != fingerprint or closure.get("dataset_sha256") != fingerprint:
            raise ValueError("Trial was verified with a different dataset.")


def endgame_directory(config: dict, output: Path) -> Path:
    path = Path(config["evaluation"].get("endgame_coverage_dir", "endgame_coverage"))
    return path if path.is_absolute() else output / path


def estimate_endgame_evaluation(config: dict, output: Path, backend) -> dict:
    """Upper bound for one action + two label scores per applicable observation."""
    directory = endgame_directory(config, output)
    if not (directory / "observations.jsonl").exists():
        return {"status": "not_available", "path": str(directory)}
    action_prefill, belief_prefill, claim_count, count = 0, 0, 0, 0
    for filename in ("observations.jsonl", "no_pending_claim.jsonl"):
        for row in read_jsonl(directory / filename):
            count += 1
            action_prefill += len(backend.encode_messages([
                {"role": "system", "content": row["action_system"]},
                {"role": "user", "content": row["action_prompt"]}], generation=True))
            if row["observation"]["last_claim"] is not None:
                claim_count += 1
                belief_prefill += 2 * (len(backend.encode_messages([
                    {"role": "system", "content": row["belief_system"]},
                    {"role": "user", "content": row["belief_prompt"]}], generation=True)) + 1)
    generated = count * config["sampling"]["max_new_tokens"] + 2 * claim_count
    cost = ((action_prefill + belief_prefill) * backend.rates["prefill"]
            + generated * backend.rates["sample"]) / 1_000_000
    return {"status": "estimated", "path": str(directory), "num_observations": count,
        "num_claim_observations": claim_count, "prefill_tokens_per_llm": action_prefill + belief_prefill,
        "generated_tokens_upper_bound_per_llm": generated, "usd_upper_bound_per_llm": cost,
        "usd_upper_bound_three_llms": 3 * cost,
        "basis": "one action sample and two single-label score requests per claim; no cache discount; local baselines cost zero"}


def estimate_trial_closure(config: dict, backend, splits: dict, max_train_steps: int) -> dict:
    """Tokenize only the fixed probes and one shared continuation batch."""
    from liars_agent.training.tinker_train import build_sft_datum, next_logical_batch_indices, training_diagnostic_plan
    from liars_agent.training.action_supervision import action_target_completions, mean_action_weight, resolve_action_supervision
    from liars_agent.training.trial_verification import select_trial_probe_rows, build_trial_probe_data, trial_verification_plan
    from liars_agent.training.trial_sampling import estimate_saved_sampler_verification
    verification, sampling = trial_settings(config)
    training = config["training"]
    policy = training["action_supervision"]
    seed = training.get("seed", config["seed"])
    grouped = {task: [row for row in splits["train"] if row["task"] == task] for task in ("action", "belief")}
    counts = {task: len(rows) for task, rows in grouped.items()}
    plan = training_diagnostic_plan(counts, training["num_epochs"], training["batch_size"],
        training.get("diagnostic_steps", [50, 100]), max_train_steps)
    probes = select_trial_probe_rows(splits, seed, verification["probe_per_task"])
    probe_data = build_trial_probe_data(probes, backend, training["max_seq_length"], policy)
    indices = next_logical_batch_indices(counts, training["batch_size"], seed, plan["planned_steps"])
    mean = mean_action_weight(grouped["action"], policy)
    batch = []
    for index in indices["action"]:
        row = grouped["action"][index]
        completions = action_target_completions(row, policy)
        weight = resolve_action_supervision(row, policy)["loss_weight"] / mean / len(completions)
        batch.extend(build_sft_datum(row, backend, training["max_seq_length"], completion_override=answer,
            sample_weight=weight, action_supervision_policy=policy) for answer in completions)
    batch.extend(build_sft_datum(grouped["belief"][index], backend, training["max_seq_length"],
        action_supervision_policy=policy) for index in indices["belief"])
    state = trial_verification_plan(probe_data, batch)
    sampler = estimate_saved_sampler_verification(backend, probes,
        max_seq_length=training["max_seq_length"], max_new_tokens=sampling["max_new_tokens"])
    tokens = state["total_processed_tokens"] + sampler["training_forward_tokens"]
    return {"version": verification["version"], "config": verification, "sampling_config": sampling,
        "probe_ids": {task: [row["sample_id"] for row in rows] for task, rows in probes.items()},
        "next_batch_sample_ids": {task: [grouped[task][i]["sample_id"] for i in ids] for task, ids in indices.items()},
        "state_and_optimizer": state, "sampler": sampler, "total_training_tokens": tokens,
        "total_usd_upper_bound": backend.estimate_training_cost(tokens) + sampler["sampling_usd_upper_bound"],
        "scope": "Extra verification only: seven probe forwards, three branch updates including optimizer-reset control, label-reference forward, sampler action/belief calls; checkpoint storage excluded."}


def estimate_cost(config: dict, output: Path, *, max_train_steps: int | None = None) -> dict:
    """Count all accepted answers per state visit and every validation candidate."""
    from liars_agent.training.tinker_train import build_sft_datum, select_diagnostic_rows, training_diagnostic_plan
    from liars_agent.training.tinker_backend import PRICING_SOURCE, PRICING_DATE
    from liars_agent.training.action_supervision import normalize_action_policy, action_target_completions, mean_action_weight
    from liars_agent.agents.prompt_builder import belief_prompt_for_method
    from liars_agent.engine.observation import PlayerObservation
    mv = config_method_version(config)
    backend = make_backend(config, output)
    policy = normalize_action_policy(config["training"].get("action_supervision"))
    lengths, counts, score_costs, grouped = {}, {}, [], {}
    token_cache = {}
    maximum = 0
    def token_length(row, completion=None):
        nonlocal maximum
        key = (row["sample_id"], completion or row["completion"])
        if key not in token_cache:
            datum = build_sft_datum(row, backend, config["training"]["max_seq_length"],
                                    completion_override=completion, action_supervision_policy=policy)
            token_cache[key] = len(datum.model_input.to_ints())
            maximum = max(maximum, token_cache[key])
        return token_cache[key]
    for key in ("train", "val", "test"):
        lengths[key] = {"action": [], "belief": []}
        grouped[key] = {"action": [], "belief": []}
        for row in read_jsonl(output / f"dataset/{key}.jsonl"):
            grouped[key][row["task"]].append(row)
            lengths[key][row["task"]].append(token_length(row))
            if row["task"] == "belief":
                obs = PlayerObservation.from_dict(row["observation"])
                b_sys, b_usr = belief_prompt_for_method(obs, mv)
                prefix = backend.encode_messages([{"role": "system", "content": b_sys},
                    {"role": "user", "content": b_usr}], generation=True)
                score_costs.append((2 * (len(prefix) + 1) * backend.rates["prefill"] + 2 * backend.rates["sample"]) / 1_000_000)
        counts[key] = {task: sum(sizes) for task, sizes in lengths[key].items()}
    if any(not sizes for sizes in lengths["train"].values()):
        raise ValueError("Both training tasks need samples for balanced cost estimation.")
    training = config["training"]
    half = training["batch_size"] // 2
    if half < 1 or training["batch_size"] % 2:
        raise ValueError("Balanced training needs an even batch size.")
    steps = math.ceil(max(map(len, lengths["train"].values())) / half)
    plan = training_diagnostic_plan(
        {task: len(rows) for task, rows in grouped["train"].items()},
        training["num_epochs"], training["batch_size"],
        diagnostic_steps=training.get("diagnostic_steps", [50, 100]), max_train_steps=max_train_steps)
    training_tokens = 0
    state_tokens, answer_counts = {}, {}
    logical_visits = {task: 0 for task in ("action", "belief")}
    expanded_sequences = {task: 0 for task in ("action", "belief")}
    for task in ("action", "belief"):
        state_tokens[task], answer_counts[task] = [], []
        for row in grouped["train"][task]:
            completions = action_target_completions(row, policy) if task == "action" else [None]
            state_tokens[task].append(sum(token_length(row, answer) for answer in completions))
            answer_counts[task].append(len(completions))
    for epoch in range(training["num_epochs"]):
        epoch_steps = min(steps, max(0, plan["planned_steps"] - epoch * steps))
        for index, task in enumerate(("action", "belief")):
            rows = grouped["train"][task]
            order = list(range(len(rows)))
            random.Random(training.get("seed", config["seed"]) + 2 * epoch + index).shuffle(order)
            visits = [order[i % len(order)] for i in range(epoch_steps * half)]
            training_tokens += sum(state_tokens[task][i] for i in visits)
            logical_visits[task] += len(visits)
            expanded_sequences[task] += sum(answer_counts[task][i] for i in visits)
    from liars_agent.training.action_validation import candidate_completion
    validation_action_tokens = sum(token_length(row, candidate_completion(row, action["action_id"]))
        for row in grouped["val"]["action"] for action in row["observation"]["legal_actions"])
    validation_candidates = sum(len(row["observation"]["legal_actions"]) for row in grouped["val"]["action"])
    diagnostics = select_diagnostic_rows(
        {split: [row for task in tasks.values() for row in task] for split, tasks in grouped.items()},
        training.get("seed", config["seed"]), training.get("diagnostic_samples_per_task", 64))
    diagnostic_action_tokens = sum(token_length(row, candidate_completion(row, action["action_id"]))
        for row in diagnostics["action"] for action in row["observation"]["legal_actions"])
    diagnostic_belief_tokens = sum(token_length(row) for row in diagnostics["belief"])
    diagnostic_tokens = (diagnostic_action_tokens + diagnostic_belief_tokens) * len(plan["diagnostic_steps"])
    full_validation_tokens = (validation_action_tokens + counts["val"]["belief"]) * plan["full_validation_epochs"]
    validation_tokens = full_validation_tokens + diagnostic_tokens
    result = {"contract": current_contract(mv), "model": config["base_model"], "tokens_by_split_and_task": counts,
        "max_input_tokens": maximum, "balanced_training_tokens": training_tokens,
        "training_logical_state_visits": logical_visits,
        "training_expanded_sequences": expanded_sequences,
        "training_optimizer_steps": plan["planned_steps"],
        "training_optimizer_steps_per_epoch": steps,
        "training_cost_basis": "state-based task-balanced batches; all accepted action answers at every state visit; one optimizer step per logical batch",
        "validation_tokens": validation_tokens,
        "full_validation_tokens": full_validation_tokens,
        "diagnostic_tokens": diagnostic_tokens,
        "diagnostic_action_tokens_per_check": diagnostic_action_tokens,
        "diagnostic_belief_tokens_per_check": diagnostic_belief_tokens,
        "diagnostic_sample_ids": {task: [row["sample_id"] for row in rows] for task, rows in diagnostics.items()},
        "training_plan": plan,
        "action_supervision_policy": policy,
        "training_action_raw_weight_mean": mean_action_weight(grouped["train"]["action"], policy),
        "validation_action_candidate_sequences_per_epoch": validation_candidates,
        "validation_action_tokens_per_epoch": validation_action_tokens,
        "validation_belief_tokens_per_epoch": counts["val"]["belief"],
        "validation_cost_basis": "all legal action JSON sequences at completed epochs plus a fixed validation subset at step 0 and configured early steps; no test model calls",
        "sft_train_and_validation_usd": backend.estimate_training_cost(training_tokens + validation_tokens),
        "belief_scoring_mean_usd_upper_bound_on_dataset": sum(score_costs) / len(score_costs),
        "belief_scoring_max_usd_upper_bound_on_dataset": max(score_costs),
        "belief_scoring_accounting": "two complete prefills plus at most two generated tokens; no cache discount assumed",
        "rates_per_million_tokens": backend.rates, "budget": backend.ledger.snapshot(),
        "pricing_source": PRICING_SOURCE, "pricing_date": PRICING_DATE,
        "scope": "SFT exact dataset token estimate; inference/RS trajectory volume remains to be estimated before paid execution."}
    result["endgame_evaluation"] = estimate_endgame_evaluation(config, output, backend)
    if max_train_steps is not None:
        splits = {split: [row for rows in tasks.values() for row in rows] for split, tasks in grouped.items()}
        result["trial_verification"] = estimate_trial_closure(config, backend, splits, max_train_steps)
        result["trial_total_usd_upper_bound"] = result["sft_train_and_validation_usd"] + result["trial_verification"]["total_usd_upper_bound"]
        result["scope"] = "Bounded trial plus complete update/sampler/state/optimizer verification; excludes checkpoint storage and separate endgame evaluation."
    write_json(output / ("trial_cost_estimate.json" if max_train_steps is not None else "cost_estimate.json"), result)
    return result


def train_model(config: dict, output: Path, backend, stage: str = "sft", *, max_train_steps: int | None = None) -> dict:
    from liars_agent.training.tinker_train import run_tinker_lora_sft
    from liars_agent.training.action_supervision import normalize_action_policy, require_training_objective
    mv = config_method_version(config)
    policy = normalize_action_policy(config["training"].get("action_supervision"))
    is_trial = stage == "sft_trial"
    if not is_trial and max_train_steps is not None:
        raise ValueError("A step limit belongs to the separate sft_trial stage.")
    split_paths = {key: output / f"dataset/{key}.jsonl" for key in ("train", "val", "test")}
    path = output / f"models/{stage}" / ("trial_metadata.json" if is_trial else "adapter_metadata.json")
    if path.exists():
        metadata = read_json(path)
        require_contract(metadata, "model metadata", method_version=mv)
        require_training_objective(metadata, policy)
        if is_trial:
            if metadata.get("max_train_steps") != max_train_steps:
                raise ValueError("This saved trial has a different step limit; use a new run directory.")
            require_verified_trial(config, metadata, split_paths=split_paths)
        return metadata
    ds_summary = read_json(output / "dataset/data_summary.json")
    require_contract(ds_summary, "dataset summary", method_version=mv)
    if ds_summary["shortfall"]:
        raise ValueError(f"Dataset shortfall: {ds_summary['shortfall']}; increase num_games in a new run before formal training.")
    training = config["training"]
    verification, trial_sampling = trial_settings(config)
    if stage == "sft":
        trial_path = output / "models/sft_trial/trial_metadata.json"
        if not trial_path.exists():
            raise ValueError("Formal SFT requires the passed sft_trial verification on this dataset first.")
        require_verified_trial(config, read_json(trial_path), split_paths=split_paths)
    resume_from = None
    resume_metadata = None
    if stage == "rs":
        split_paths["train"] = output / "dataset/rs_train.jsonl"
        counts = {"action": 0, "belief": 0}
        for row in read_jsonl(split_paths["train"]):
            if row["task"] in counts:
                counts[row["task"]] += 1
        if any(counts[task] < config["rs"][key]
               for task, key in (("action", "target_samples"), ("belief", "belief_target_samples"))):
            raise ValueError("RS dataset has not reached its configured target.")
        resume_metadata = read_json(output / "models/sft/adapter_metadata.json")
        require_contract(resume_metadata, "SFT resume checkpoint", method_version=mv)
        require_training_objective(resume_metadata, policy)
        resume_from = resume_metadata["state_path"]
    train_stub = [{"game_id": row["game_id"], "task": row["task"]} for row in read_jsonl(split_paths["train"])]
    val_rows = list(read_jsonl(split_paths["val"]))
    splits = {"train": train_stub, "val": val_rows, "test": []}
    return run_tinker_lora_sft(
        samples=[], splits=splits, split_paths=split_paths, output_dir=path.parent,
        backend=backend, base_model=config["base_model"], lora_rank=config["lora"]["r"],
        learning_rate=config["rs"]["learning_rate"] if stage == "rs" else training["learning_rate"],
        num_epochs=config["rs"]["num_epochs"] if stage == "rs" else training["num_epochs"],
        batch_size=training["batch_size"], max_seq_length=training["max_seq_length"],
        seed=training.get("seed", config["seed"]), resume_from=resume_from, resume_metadata=resume_metadata, stage=stage,
        resume=(path.parent / "training_progress.json").exists(),
        checkpoint_every_steps=training.get("checkpoint_every_steps"),
        train_attn=config["lora"]["train_attn"], train_mlp=config["lora"]["train_mlp"],
        train_unembed=config["lora"]["train_unembed"],
        action_supervision_policy=policy,
        diagnostic_steps=training.get("diagnostic_steps", [50, 100]),
        diagnostic_samples_per_task=training.get("diagnostic_samples_per_task", 64),
        max_train_steps=max_train_steps,
        trial_verification_config=verification, trial_sampling_config=trial_sampling,
        method_version=mv,
    )


def train_rl(config: dict, output: Path, backend, *, variant: str = "rl") -> dict:
    from liars_agent.agents.base import BaseAgent
    from liars_agent.agents.llm_agent import LLMAgent
    from liars_agent.training.tinker_rl import run_tinker_lora_rl
    if variant not in ("rl", "rl_v2"):
        raise ValueError(f"Unsupported RL variant: {variant}")
    mv = config_method_version(config)
    rl_dir = output / f"models/{variant}"
    meta_path = rl_dir / "adapter_metadata.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        require_contract(meta, "RL adapter metadata", method_version=mv)
        if meta.get("formal_training_complete") is True:
            return meta
    sft_meta_path = output / "models/sft/adapter_metadata.json"
    if not sft_meta_path.exists():
        raise ValueError("RL stage requires completed SFT adapter_metadata.json.")
    sft_meta = read_json(sft_meta_path)
    require_contract(sft_meta, "RL source SFT metadata", method_version=mv)
    if sft_meta.get("formal_training_complete") is not True:
        raise ValueError("RL requires a completed formal SFT checkpoint.")

    base_defaults = DEFAULT_RL_V2_CONFIG if variant == "rl_v2" else DEFAULT_RL_CONFIG
    rl_cfg = {**base_defaults, **config.get(variant, {})}
    training = config["training"]
    posterior_opp_prob = float(rl_cfg.get("posterior_opponent_prob", 0.25))
    switcher_opp_prob = float(rl_cfg.get("switcher_opponent_prob", 0.0))
    self_play_opp_prob = float(rl_cfg.get("self_play_opponent_prob", 0.0))

    class _FrozenSFTActionBot(BaseAgent):
        """Frozen SFT neural opponent using action-only sampling (no extra belief calls)."""

        def __init__(self, seed: int) -> None:
            super().__init__(name="frozen_sft_opponent", style_tag="frozen_sft")
            self._agent = LLMAgent(
                name="frozen_sft_opponent",
                mode="lora_sft",
                model_version=config["base_model"],
                backend=backend,
                adapter_path=sft_meta_path,
                seed=seed,
                temperature=float(config["sampling"].get("temperature", 0.35)),
                top_p=float(config["sampling"].get("top_p", 0.9)),
                max_new_tokens=int(config["sampling"]["max_new_tokens"]),
                method_version=mv,
            )

        def select_action(self, obs):
            return self._agent.sample_candidates(obs, num_samples=1, include_belief=False)[0]

    eval_opp_sampler = None
    if mv == "adaptive_v4":
        from liars_agent.training.adaptive_opponents import (
            MidGameSwitcherBot,
            ParameterizedArchetypeBot,
            V4_TRAIN_ARCHETYPES,
        )
        from liars_agent.training.adaptive_teacher import CalibratedPosteriorBelief, PosteriorAdaptiveBot
        post_path = output / "models/v4_calibrated_posterior.json"
        cal_belief_fn = CalibratedPosteriorBelief.load(post_path).predict if post_path.exists() else None

        def opp_sampler(group_seed: int):
            rng = random.Random(group_seed)
            makers = []
            for _ in range(3):
                u = rng.random()
                if posterior_opp_prob > 0 and u < posterior_opp_prob:
                    makers.append(lambda s, bfn=cal_belief_fn: PosteriorAdaptiveBot(seed=s, belief_fn=bfn))
                elif switcher_opp_prob > 0 and u < posterior_opp_prob + switcher_opp_prob:
                    early_spec, late_spec = rng.sample(V4_TRAIN_ARCHETYPES, 2)
                    sw_round = rng.choice((3, 4, 5))
                    makers.append(
                        lambda s, e=early_spec, l=late_spec, r=sw_round: MidGameSwitcherBot(
                            e, l, switch_round=r, seed=s
                        )
                    )
                elif self_play_opp_prob > 0 and u < posterior_opp_prob + switcher_opp_prob + self_play_opp_prob:
                    makers.append(lambda s: _FrozenSFTActionBot(seed=s))
                else:
                    spec = rng.choice(V4_TRAIN_ARCHETYPES)
                    makers.append(lambda s, sp=spec: ParameterizedArchetypeBot(sp, seed=s))
            return makers

        def eval_opp_sampler(block_seed: int):
            """Fixed train-side held-out validation mix (strictly zero unseen test archetypes)."""
            rng = random.Random(block_seed)
            early_spec, late_spec, third_spec = rng.sample(V4_TRAIN_ARCHETYPES, 3)
            makers = [
                lambda s, bfn=cal_belief_fn: PosteriorAdaptiveBot(seed=s, belief_fn=bfn),
                lambda s, e=early_spec, l=late_spec: MidGameSwitcherBot(e, l, switch_round=4, seed=s),
                lambda s, sp=third_spec: ParameterizedArchetypeBot(sp, seed=s),
            ]
            rng.shuffle(makers)
            return makers
    else:
        factories = opponent_factories(config)

        def opp_sampler(group_seed: int):
            order = list(range(3))
            random.Random(group_seed).shuffle(order)
            return [factories[i] for i in order]

    belief_path = output / "dataset/belief_train.jsonl"
    return run_tinker_lora_rl(
        output_dir=rl_dir,
        source_metadata=sft_meta,
        opponent_pool_sampler=opp_sampler,
        belief_train_path=belief_path if belief_path.exists() else None,
        base_model=config["base_model"],
        lora_rank=config["lora"]["r"],
        train_attn=config["lora"]["train_attn"],
        train_mlp=config["lora"]["train_mlp"],
        train_unembed=config["lora"]["train_unembed"],
        num_steps=int(rl_cfg["num_steps"]),
        groups_per_step=int(rl_cfg["groups_per_step"]),
        group_size=int(rl_cfg["group_size"]),
        learning_rate=float(rl_cfg["learning_rate"]),
        loss_fn=str(rl_cfg["loss_fn"]),
        loss_fn_config=rl_cfg.get("loss_fn_config"),
        temperature=float(rl_cfg["temperature"]),
        top_p=float(rl_cfg["top_p"]),
        max_new_tokens=int(config["sampling"]["max_new_tokens"]),
        max_seq_length=int(training["max_seq_length"]),
        win_weight=float(rl_cfg["win_weight"]),
        round_survival_weight=float(rl_cfg["round_survival_weight"]),
        gamma=float(rl_cfg["gamma"]),
        challenge_hit_bonus=float(rl_cfg["challenge_hit_bonus"]),
        caught_bluff_penalty=float(rl_cfg["caught_bluff_penalty"]),
        fallback_penalty=float(rl_cfg["fallback_penalty"]),
        clip_advantage=float(rl_cfg["clip_advantage"]),
        belief_aux_batch_size=int(rl_cfg["belief_aux_batch_size"]),
        belief_aux_weight=float(rl_cfg["belief_aux_weight"]),
        on_policy_belief_ratio=float(rl_cfg["on_policy_belief_ratio"]),
        rollout_workers=int(rl_cfg["rollout_workers"]),
        eval_every_steps=int(rl_cfg["eval_every_steps"]),
        eval_blocks=int(rl_cfg["eval_blocks"]),
        eval_temperature=float(rl_cfg["eval_temperature"]),
        eval_opponent_sampler=eval_opp_sampler,
        checkpoint_every_steps=int(rl_cfg["checkpoint_every_steps"]),
        seed=int(rl_cfg.get("seed", config["seed"])),
        method_version=mv,
        stage_name=variant,
        budget_path=ROOT / "outputs/budget_usage.jsonl",
        budget_limit_usd=float(config["budget_limit_usd"]),
        backend=backend,
    )


def llm_factory(config: dict, output: Path, backend, key: str):
    from liars_agent.agents.llm_agent import LLMAgent
    mv = config_method_version(config)
    subdir_map = {"lora_sft": "sft", "lora_rs": "rs", "lora_rl": "rl", "lora_rl_v2": "rl_v2"}
    adapter = None if key == "zero_shot" else output / f"models/{subdir_map[key]}/adapter_metadata.json"
    sampling = config["sampling"]
    mode = "lora_rs_improved" if key == "lora_rs" else key
    return lambda seed: LLMAgent(name=key, mode=mode,
        model_version=config["base_model"], backend=backend, adapter_path=adapter, seed=seed,
        temperature=sampling["temperature"], top_p=sampling["top_p"], max_new_tokens=sampling["max_new_tokens"],
        method_version=mv)


def generate_rs_data(config: dict, output: Path, backend) -> dict:
    from liars_agent.training.data_generator import evaluate_observation_action_values, make_action_sample, extract_belief_samples
    from liars_agent.training.action_supervision import make_action_supervision, normalize_action_policy
    from liars_agent.engine.observation import PlayerObservation
    from liars_agent.engine.replay import verify_replay_determinism
    mv = config_method_version(config)
    rs = config["rs"]
    policy = normalize_action_policy(config["training"].get("action_supervision"))
    candidate_temp = float(rs.get("candidate_temperature", max(0.7, float(config["sampling"]["temperature"]))))
    shard_dir = output / "dataset/rs_games"
    shard_dir.mkdir(parents=True, exist_ok=True)
    accepted, beliefs = [], []
    factory = llm_factory(config, output, backend, "lora_sft")
    factories = opponent_factories(config)
    v4_calibrated = None
    if mv == "adaptive_v4":
        from liars_agent.training.adaptive_opponents import ParameterizedArchetypeBot, V4_TRAIN_ARCHETYPES
        from liars_agent.training.adaptive_teacher import (
            CalibratedPosteriorBelief,
            evaluate_posterior_teacher_values,
            extract_v4_belief_samples,
            make_v4_action_sample,
        )
        v4_calibrated = CalibratedPosteriorBelief.load(output / "models/v4_calibrated_posterior.json")
    target, belief_target = rs["target_samples"], rs["belief_target_samples"]
    for index in range(rs.get("max_games", rs["num_games"] * 5)):
        if len(accepted) >= target and len(beliefs) >= belief_target:
            break
        shard = shard_dir / f"game_{index:05d}.jsonl"
        if shard.exists():
            rows = list(read_jsonl(shard))
            for row in rows:
                require_contract(row, "RS sample", method_version=mv)
            accepted.extend(r for r in rows if r["task"] == "action")
            beliefs.extend(r for r in rows if r["task"] == "belief")
            continue
        game_seed = _derive_seed(config["seed"], "rs_environment", index)
        game_id = f"rs_train_{index:05d}"
        actor = index % 4
        agent = factory(_derive_seed(config["seed"], "rs_actor", index))
        lineup = [None] * 4
        lineup[actor] = agent
        if mv == "adaptive_v4":
            opp_rng = random.Random(_derive_seed(config["seed"], "rs_v4_opps", index))
            picks = [opp_rng.choice(V4_TRAIN_ARCHETYPES) for _ in range(3)]
            for offset, spec in enumerate(picks, 1):
                seat = (actor + offset) % 4
                lineup[seat] = ParameterizedArchetypeBot(spec, seed=_derive_seed(config["seed"], "rs_policy", index, seat))
        else:
            order = list(range(3))
            random.Random(_derive_seed(config["seed"], "rs_order", index)).shuffle(order)
            for offset, spec_index in enumerate(order, 1):
                seat = (actor + offset) % 4
                lineup[seat] = factories[spec_index](seed=_derive_seed(config["seed"], "rs_policy", index, seat))
        replay_path = shard.with_suffix(".replay.jsonl")
        if replay_path.exists():
            replay = next(read_jsonl(replay_path))
            verify_replay_determinism(replay, raise_on_error=True)
        else:
            engine = LiarsBarEngine(game_id=game_id, game_seed=game_seed)
            while not engine.is_game_over():
                obs = engine.get_observation()
                action, record = lineup[obs.observer_seat].select_action(obs)
                engine.step(action, record)
            bundle = GameReplayBundle.from_engine(engine, [a.name for a in lineup])
            verify_replay_determinism(bundle, raise_on_error=True)
            save_replays_jsonl([bundle], replay_path)
            replay = bundle.to_dict()
        points_cache_path = shard.with_suffix(".points_cache.jsonl")
        cached_points = {}
        if points_cache_path.exists():
            for entry in read_jsonl(points_cache_path):
                cached_points[(entry["round_id"], entry["step_id"])] = entry
        if mv == "adaptive_v4":
            belief_rows = extract_v4_belief_samples([replay], min(belief_target - len(beliefs),
                math.ceil(belief_target / rs["num_games"])), _derive_seed(config["seed"], "rs_belief", index), actor)
        else:
            belief_rows = extract_belief_samples([replay], min(belief_target - len(beliefs),
                math.ceil(belief_target / rs["num_games"])), _derive_seed(config["seed"], "rs_belief", index), actor)
        points = [p for p in replay["steps"] if p["actor_seat"] == actor and len(p["observation"]["legal_actions"]) > 1]
        random.Random(_derive_seed(config["seed"], "rs_points", index)).shuffle(points)
        samples = []
        quota = min(target - len(accepted), math.ceil(target / rs["num_games"]))
        for point in points:
            if len(samples) >= quota:
                break
            obs = PlayerObservation.from_dict(point["observation"])
            point_key = (obs.round_id, obs.step_id)
            if point_key in cached_points:
                cached_entry = cached_points[point_key]
                if cached_entry.get("accepted") and cached_entry.get("sample") is not None:
                    sample = cached_entry["sample"]
                    require_contract(sample, "RS sample", method_version=mv)
                    samples.append(sample)
                continue
            try:
                proposals = agent.sample_candidates(obs, num_samples=rs["candidates_per_state"],
                                                    include_belief=False, temperature=candidate_temp)
            except TypeError:
                proposals = agent.sample_candidates(obs, num_samples=rs["candidates_per_state"], include_belief=False)
            candidates = {a.action_id: a for a, record in proposals if not record.was_fallback}
            if len(candidates) < 2:
                cached_points[point_key] = {"round_id": obs.round_id, "step_id": obs.step_id, "accepted": False}
                write_jsonl(points_cache_path, cached_points.values())
                continue
            if mv == "adaptive_v4":
                values = evaluate_posterior_teacher_values(
                    obs,
                    num_rollouts=rs["rollouts_per_candidate"],
                    rollout_seed=_derive_seed(config["seed"], "rs_teacher", index, obs.round_id, obs.step_id),
                    belief_fn=v4_calibrated.predict,
                    candidate_actions=list(candidates.values()),
                )
            else:
                values = evaluate_observation_action_values(obs, teacher_config(config), rs["rollouts_per_candidate"],
                    _derive_seed(config["seed"], "rs_teacher", index, obs.round_id, obs.step_id), list(candidates.values()))
            best = max(candidates.values(), key=lambda a: (values[a.action_id], a.action_type, a.cards))
            advantage = values[best.action_id] - sum(values.values()) / len(values)
            if advantage < rs["min_advantage_margin"]:
                cached_points[point_key] = {"round_id": obs.round_id, "step_id": obs.step_id, "accepted": False}
                write_jsonl(points_cache_path, cached_points.values())
                continue
            if mv == "adaptive_v4":
                sample = make_v4_action_sample(
                    obs, best, game_id,
                    rollout_candidate_values=values,
                    action_supervision_policy=policy,
                    rollout_advantage=advantage,
                    rollout_best_value=values[best.action_id],
                    candidate_count=len(values),
                    source="sft_rejection_sampling",
                    teacher="calibrated_posterior_continuation_v4",
                )
            else:
                sample = make_action_sample(obs, best, game_id, rollout_advantage=advantage,
                    rollout_best_value=values[best.action_id], candidate_count=len(values), source="sft_rejection_sampling",
                    teacher="pool_prior_counter_continuation", rollout_candidate_values=values)
                sample["action_supervision"] = make_action_supervision(sample, values, policy,
                                                                       source="sft_rejection_sampling")
            cached_points[point_key] = {"round_id": obs.round_id, "step_id": obs.step_id, "accepted": True, "sample": sample}
            write_jsonl(points_cache_path, cached_points.values())
            samples.append(sample)
        write_jsonl(shard, samples + belief_rows)
        if points_cache_path.exists():
            points_cache_path.unlink()
        accepted.extend(samples)
        beliefs.extend(belief_rows)
        print(f"RS data: action {len(accepted)}/{target}, belief {len(beliefs)}/{belief_target}", flush=True)
    write_jsonl(output / "dataset/rs_train.jsonl", accepted + beliefs)
    summary = {"contract": current_contract(mv), "actual_samples": len(accepted), "target_samples": target,
        "actual_belief_samples": len(beliefs), "belief_target_samples": belief_target,
        "shortfall": max(0, target - len(accepted)) + max(0, belief_target - len(beliefs))}
    write_json(output / "dataset/rs_summary.json", summary)
    return summary


def frozen_belief_samples(config: dict, output: Path, pool: str) -> list[dict]:
    from liars_agent.training.data_generator import extract_belief_samples
    from liars_agent.engine.replay import verify_replay_determinism
    mv = config_method_version(config)
    path = output / f"dataset/frozen_belief_{pool}.jsonl"
    if path.exists():
        rows = list(read_jsonl(path))
        for row in rows:
            require_contract(row, "frozen belief observations", method_version=mv)
        return rows
    if pool == "seen":
        rows = list(read_jsonl(output / "dataset/belief_test.jsonl"))
    else:
        target = config["evaluation"]["unseen_belief_samples"]
        factories = opponent_factories(config, "unseen")
        extract_fn = extract_belief_samples
        if mv == "adaptive_v4":
            from liars_agent.training.adaptive_teacher import extract_v4_belief_samples
            extract_fn = extract_v4_belief_samples
        rows = []
        for index in range(config["evaluation"]["unseen_belief_max_games"]):
            if len(rows) >= target:
                break
            seat = index % 4
            order = list(range(3))
            random.Random(_derive_seed(config["seed"], "probe_order", index)).shuffle(order)
            agents = [None] * 4
            agents[seat] = CardCounterBot(seed=_derive_seed(config["seed"], "probe_actor", index))
            for offset, spec in enumerate(order, 1):
                opp = (seat + offset) % 4
                agents[opp] = factories[spec](seed=_derive_seed(config["seed"], "probe_policy", index, opp))
            engine = LiarsBarEngine(game_id=f"unseen_probe_{index:05d}",
                game_seed=_derive_seed(config["seed"], "probe_environment", index))
            while not engine.is_game_over():
                obs = engine.get_observation()
                action, record = agents[obs.observer_seat].select_action(obs)
                engine.step(action, record)
            replay = GameReplayBundle.from_engine(engine, [a.name for a in agents])
            verify_replay_determinism(replay, raise_on_error=True)
            save_replays_jsonl([replay], output / f"dataset/frozen_belief_replays/game_{index:05d}.jsonl")
            rows.extend(extract_fn([replay], min(10, target - len(rows)),
                _derive_seed(config["seed"], "probe_selection", index), seat))
        if len(rows) < target:
            raise ValueError(f"Frozen unseen belief set has {len(rows)}/{target} observations.")
    if not rows:
        raise ValueError("Frozen belief evaluation requires held-out observations.")
    write_jsonl(path, rows)
    return rows


def evaluate_frozen_beliefs(config, output, backend, model_keys):
    from liars_agent.agents.base import estimate_heuristic_bluff_belief
    from liars_agent.engine.observation import PlayerObservation
    from liars_agent.evaluation.metrics import compute_probability_calibration, compute_paired_probability_metrics
    mv = config_method_version(config)
    results = {}
    for pool in ("seen", "unseen"):
        rows = frozen_belief_samples(config, output, pool)
        results[pool] = {}
        predictions = {}
        for key in model_keys:
            path = output / f"evaluation/belief_{pool}_{key}.json"
            if path.exists():
                stored = read_json(path)
                require_contract(stored, "frozen belief predictions", method_version=mv)
                if [r["sample_id"] for r in stored["predictions"]] != [r["sample_id"] for r in rows]:
                    raise ValueError("Frozen belief prediction IDs differ from the held-out set.")
                predictions[key] = stored["predictions"]
                results[pool][key] = stored["metrics"]
                continue
            if key == "small_classifier":
                agent = SmallClassifierBot.load_weights(output / "models/small_classifier_weights.json", method_version=mv)
            elif key == "rule_counter":
                agent = None
            else:
                agent = llm_factory(config, output, backend, key)(_derive_seed(config["seed"], "belief_eval"))
            partial_path = output / f"evaluation/belief_{pool}_{key}.partial.jsonl"
            cached_records = {}
            if partial_path.exists():
                for rec in read_jsonl(partial_path):
                    cached_records[rec["sample_id"]] = rec
            records = []
            for row in rows:
                if row["sample_id"] in cached_records:
                    records.append(cached_records[row["sample_id"]])
                    continue
                obs = PlayerObservation.from_dict(row["observation"])
                probability = estimate_heuristic_bluff_belief(obs)[0] if agent is None else agent.predict_bluff_prob(obs)
                rec = {"sample_id": row["sample_id"], "game_id": row["game_id"],
                       "label": row["label"], "probability": probability}
                records.append(rec)
                cached_records[row["sample_id"]] = rec
                if agent is not None and hasattr(agent, "checkpoint"):
                    write_jsonl(partial_path, cached_records.values())
            if partial_path.exists():
                partial_path.unlink()
            metrics = compute_probability_calibration([r["probability"] for r in records], [r["label"] for r in records])
            metadata = {"model_key": key, "base_model": config["base_model"] if agent is not None and hasattr(agent, "checkpoint") else None,
                "checkpoint": getattr(agent, "checkpoint", None),
                "belief_source": {"small_classifier": "classifier", "rule_counter": "heuristic"}.get(key, "model")}
            stored = {"contract": current_contract(mv), "predictions": records, "metrics": metrics,
                "model_metadata": metadata, "usage": agent.tracker.to_dict() if hasattr(agent, "tracker") else {"api_calls": 0}}
            write_json(path, stored)
            predictions[key], results[pool][key] = records, metrics
        results[pool]["paired"] = {}
        for a, b in (("lora_sft", "zero_shot"), ("lora_rs", "lora_sft"), ("lora_rl", "lora_sft"), ("lora_rl", "zero_shot"), ("lora_rl_v2", "lora_sft"), ("lora_rl_v2", "lora_rl"), ("lora_rl_v2", "zero_shot"), ("small_classifier", "rule_counter")):
            if a in predictions and b in predictions:
                results[pool]["paired"][f"{a}_vs_{b}"] = compute_paired_probability_metrics(predictions[a], predictions[b],
                    config["evaluation"]["bootstrap_samples"], config["seed"])
    return results


def evaluate_endgames(config: dict, output: Path, backend, model_keys: list[str], *, required=False) -> dict:
    from liars_agent.evaluation.endgame_model import evaluate_endgame_models
    from liars_agent.evaluation.endgame_coverage import validate_endgame_coverage, _validate_endgame_coverage
    from liars_agent.training.dataset_release import validate_dataset_release
    mv = config_method_version(config)
    release = validate_dataset_release(output)
    directory = endgame_directory(config, output)
    if not (directory / "coverage_report.json").exists():
        if required:
            raise ValueError(f"Independent endgame check set is missing: {directory}")
        return {"status": "not_available", "path": str(directory)}
    if release is not None:
        identity = release["source_identity"]
        coverage = _validate_endgame_coverage(directory, (set(identity["game_ids"]), set(identity["game_seeds"])))
    else:
        source_run = read_json(directory / "private/generation_metadata.json")["source_run"]
        coverage = validate_endgame_coverage(directory, source_run)
    if coverage["status"] != "complete":
        raise ValueError("Independent endgame coverage has unfilled reachable cells.")
    results = {}
    for key in model_keys:
        target = output / f"evaluation/endgame_{key}.json"
        # The evaluator validates frozen files and their fingerprint before model calls.
        if key == "small_classifier":
            agent = SmallClassifierBot.load_weights(output / "models/small_classifier_weights.json", method_version=mv)
        elif key == "rule_counter":
            agent = CardCounterBot(name=key, seed=_derive_seed(config["seed"], "endgame_model"))
        else:
            agent = llm_factory(config, output, backend, key)(_derive_seed(config["seed"], "endgame_model"))
        cached = {key: read_json(target)} if target.exists() else None
        result = evaluate_endgame_models(directory, {key: agent}, cached_results=cached)
        write_json(target, result["models"][key])
        results[key] = result["models"][key]
    summary = {"contract": current_contract(mv), "status": "complete", "dataset": result["dataset"],
        "natural_distribution": False, "models": results,
        "interpretation": "Independent diagnostic strata; free-action results are protocol checks, not optimal-action accuracy or win rate."}
    write_json(output / "evaluation/endgame_summary.json", summary)
    return summary


def evaluate(config: dict, output: Path, backend, model_keys: list[str]) -> dict:
    from liars_agent.evaluation.duplicate_runner import run_duplicate_evaluation
    from liars_agent.evaluation.metrics import (compute_behavioral_profile, compute_belief_calibration,
        compute_risk_and_hand_stratified_diagnosis, compute_paired_bootstrap_ci)
    mv = config_method_version(config)
    results = {}
    for pool in ("seen", "unseen"):
        results[pool] = {}
        for key in model_keys:
            target = output / f"evaluation/{pool}_{key}_full.json"
            replay_path = target.with_suffix(".replays.jsonl")
            if target.exists():
                results[pool][key] = read_json(target)
                require_contract(results[pool][key], "evaluation result", method_version=mv)
                continue
            if key == "small_classifier":
                factory = lambda seed: SmallClassifierBot.load_weights(output / "models/small_classifier_weights.json", method_version=mv)
                metadata = {"backend": "local_classifier", "weights": str(output / "models/small_classifier_weights.json")}
            elif key == "rule_counter":
                factory = lambda seed: CardCounterBot(name="rule_counter", seed=seed)
                metadata = {"backend": "rule"}
            else:
                factory = llm_factory(config, output, backend, key)
                metadata = {"backend": "tinker", "base_model": config["base_model"]}
                if key != "zero_shot":
                    subdir_map = {"lora_sft": "sft", "lora_rs": "rs", "lora_rl": "rl", "lora_rl_v2": "rl_v2"}
                    metadata.update(read_json(output / f"models/{subdir_map[key]}/adapter_metadata.json"))
            result = run_duplicate_evaluation(factory, opponent_factories(config, pool),
                num_blocks=config["evaluation"]["num_blocks"], base_seed=config["seed"] + (1000 if pool == "seen" else 5000),
                pool_name=f"{pool}_pool", model_metadata=metadata, opponent_specs=opponent_specs(config, pool),
                num_resamples=config["evaluation"]["bootstrap_samples"], bootstrap_seed=config["seed"],
                cache_dir=output / f"evaluation/blocks/{pool}_{key}_full")
            data = result.to_dict(include_replays=False)
            data["contract"] = current_contract(mv)
            replay_dicts = [r.to_dict() for r in result.replays]
            data["profile"] = compute_behavioral_profile(replay_dicts, result.agent_name)
            data["calibration"] = compute_belief_calibration(replay_dicts, result.agent_name,
                belief_source={"small_classifier": "classifier", "rule_counter": "heuristic"}.get(key, "model"))
            data["stratified"] = compute_risk_and_hand_stratified_diagnosis(replay_dicts, result.agent_name)
            save_replays_jsonl(result.replays, replay_path)
            write_json(target, data)
            results[pool][key] = data
            print(f"Evaluation {pool}/{key}: {result.overall_win_rate:.2%}", flush=True)
    paired = {}
    for pool in ("seen", "unseen"):
        for a, b in (("lora_sft", "zero_shot"), ("lora_rs", "lora_sft"), ("lora_rl", "lora_sft"), ("lora_rl", "zero_shot"), ("lora_rl_v2", "lora_sft"), ("lora_rl_v2", "lora_rl"), ("lora_rl_v2", "zero_shot"), ("small_classifier", "rule_counter")):
            if a in results[pool] and b in results[pool]:
                paired[f"{pool}_{a}_vs_{b}"] = compute_paired_bootstrap_ci(results[pool][a]["block_records"],
                    results[pool][b]["block_records"], config["evaluation"]["bootstrap_samples"], config["seed"])
    ablations = []
    if "lora_sft" in model_keys:
        for mode in ("full", "current_round_only", "last_action_only"):
            if mode == "full":
                data = results["seen"]["lora_sft"]
            else:
                target = output / f"evaluation/ablation_{mode}.json"
                if target.exists():
                    data = read_json(target)
                    require_contract(data, "ablation result", method_version=mv)
                else:
                    result = run_duplicate_evaluation(llm_factory(config, output, backend, "lora_sft"),
                        opponent_factories(config), num_blocks=config["evaluation"]["num_blocks"],
                        base_seed=config["seed"] + 1000, pool_name="seen_pool", ablation_mode=mode,
                        model_metadata=results["seen"]["lora_sft"]["model_metadata"],
                        opponent_specs=opponent_specs(config), num_resamples=config["evaluation"]["bootstrap_samples"],
                        bootstrap_seed=config["seed"],
                        cache_dir=output / f"evaluation/blocks/ablation_{mode}")
                    data = result.to_dict(include_replays=False)
                    data["contract"] = current_contract(mv)
                    data["profile"] = compute_behavioral_profile([r.to_dict() for r in result.replays], "lora_sft")
                    save_replays_jsonl(result.replays, target.with_suffix(".replays.jsonl"))
                    write_json(target, data)
            ablations.append({"mode": mode, "label": mode, "win_rate": data["overall_win_rate"],
                "challenge_hit_rate": data["profile"]["challenge_hit_rate"], "bluff_rate": data["profile"]["bluff_rate"],
                "block_win_rates": data["block_win_rates"], "win_rate_ci": data["win_rate_ci"],
                "denominators": {"win_rate": data["total_games"],
                    "challenge_hit_rate": data["profile"]["denominators"]["challenge_hit_rate"]}})
    frozen = evaluate_frozen_beliefs(config, output, backend, model_keys)
    endgames = evaluate_endgames(config, output, backend, model_keys)
    summary = {"contract": current_contract(mv), "frozen_belief": frozen, "endgame_diagnostics": endgames, "status": "complete" if set(model_keys) == set(MODEL_LABELS) else "selected_models_only",
               "models_evaluated": model_keys, "results": results, "paired_bootstrap_tests": paired,
               "history_ablation": ablations, "config": config,
               "dataset_and_training": read_json(output / "dataset/data_summary.json")}
    write_json(output / "experiment_summary.json", summary)
    return summary


def generate_figures(output: Path, *, method_version: str = "v2") -> list[str]:
    from liars_agent.evaluation.plots import (generate_fig1_winrate_svg, generate_fig2_history_ablation_svg,
        generate_fig3_risk_stratified_svg, generate_fig4_calibration_and_profile_svg)
    summary = read_json(output / "experiment_summary.json")
    require_contract(summary, "experiment summary", method_version=method_version)
    pools = summary["results"]
    items = {}
    for pool in ("seen", "unseen"):
        items[pool] = [{"key": key, "label": ALL_MODEL_LABELS[key], "win_rate": value["overall_win_rate"],
                        "ci_95_low": value["win_rate_ci"]["ci_95_low"], "ci_95_high": value["win_rate_ci"]["ci_95_high"], "num_blocks": value["num_blocks"]}
                       for key, value in pools[pool].items()]
    directory = output / "figures"
    figures = [generate_fig1_winrate_svg(items["seen"], items["unseen"], directory / "fig1_winrate_comparison.svg")]
    if summary["history_ablation"]:
        figures.append(generate_fig2_history_ablation_svg(summary["history_ablation"], directory / "fig2_history_ablation.svg"))
    if "lora_sft" in pools["seen"]:
        figures.append(generate_fig3_risk_stratified_svg(pools["seen"]["lora_sft"]["stratified"], directory / "fig3_risk_hand_stratified.svg"))
    if {"zero_shot", "lora_sft"} <= pools["seen"].keys():
        figures.append(generate_fig4_calibration_and_profile_svg(summary["frozen_belief"]["seen"]["zero_shot"],
            summary["frozen_belief"]["seen"]["lora_sft"], {key: value["profile"] for key, value in pools["seen"].items()},
            directory / "fig4_calibration_and_profile.svg"))
    summary["generated_figures"] = [str(p) for p in figures]
    write_json(output / "experiment_summary.json", summary)
    return summary["generated_figures"]



def validate_run_artifacts(output: Path, *, action_supervision_policy: dict | None = None,
                           method_version: str = "v2") -> None:
    """Validate existing contracts and both task schemas before paid preflight."""
    from liars_agent.training.tinker_train import validate_training_sample
    from liars_agent.training.dataset_release import validate_dataset_release
    release = validate_dataset_release(output)
    paths = [output / "dataset/data_summary.json", output / "dataset/rs_summary.json", output / "experiment_summary.json"]
    paths.extend((output / "evaluation").glob("*.json"))
    paths.extend((output / "models").glob("*/adapter_metadata.json"))
    paths.extend((output / "models").glob("*/trial_metadata.json"))
    paths.extend((output / "models").glob("*/training_progress.json"))
    paths.extend((output / "dataset/batches").glob("batch_*.json"))
    for path in paths:
        if path.exists():
            artifact = read_json(path)
            require_contract(artifact, str(path), method_version=method_version)
            if action_supervision_policy is not None and path.name in ("adapter_metadata.json", "trial_metadata.json", "training_progress.json") and artifact.get("stage") != "rl":
                from liars_agent.training.action_supervision import require_training_objective
                require_training_objective(artifact, action_supervision_policy)
    classifier = output / "models/small_classifier_weights.json"
    if classifier.exists():
        require_contract(read_json(classifier), "classifier weights", method_version=method_version)
    v4_posterior = output / "models/v4_calibrated_posterior.json"
    if v4_posterior.exists():
        require_contract(read_json(v4_posterior), "v4 calibrated posterior", method_version=method_version)
    sample_paths = list((output / "dataset").glob("*.jsonl"))
    sample_paths.extend((output / "dataset/batches").glob("samples_*.jsonl"))
    sample_paths.extend(p for p in (output / "dataset/rs_games").glob("game_*.jsonl")
                        if not p.name.endswith(".replay.jsonl") and not p.name.endswith(".points_cache.jsonl"))
    for path in sample_paths:
        for row in read_jsonl(path):
            validate_training_sample(row, action_supervision_policy=action_supervision_policy, method_version=method_version)
    split_path = output / "dataset/game_splits.json"
    if split_path.exists():
        partition = read_json(split_path)
        owner = {}
        for name, ids in partition.items():
            for gid in ids:
                if gid in owner:
                    raise ValueError("A game belongs to multiple dataset partitions.")
                owner[gid] = name
        for name in ("train", "val", "test"):
            path = output / f"dataset/{name}.jsonl"
            if path.exists() and any(owner.get(r["game_id"]) != name for r in read_jsonl(path)):
                raise ValueError(f"{name} samples do not match the shared game partition.")
        path = output / "dataset/rs_train.jsonl"
        if path.exists() and any(r["game_id"] in owner for r in read_jsonl(path)):
            raise ValueError("RS must use fresh training-side games.")
    if release is not None:
        # The pinned input hashes bind the audited augmentation/refinement to
        # this release. Sample/partition checks above still run after relocation;
        # only provenance checks needing historical absolute source paths stop here.
        return
    snapshot = output / "config.json"
    dataset_config = read_json(snapshot).get("dataset", {}) if snapshot.exists() else {}
    if dataset_config.get("endgame_augmentation"):
        from liars_agent.training.endgame_augmentation import validate_endgame_augmented_run
        validate_endgame_augmented_run(output)
    elif dataset_config.get("action_refinement"):
        from liars_agent.training.dataset_quality import validate_refined_dataset
        validate_refined_dataset(output)
    if dataset_config.get("supervision_preparation"):
        from liars_agent.training.prepare_supervision import validate_prepared_supervision_run
        validate_prepared_supervision_run(output)

def run_pipeline(config: dict, output: Path, stage: str = "all", model_keys: list[str] | None = None,
                 *, max_train_steps: int | None = None) -> dict:
    mv = config_method_version(config)
    require_contract(config, "experiment configuration", method_version=mv)
    from liars_agent.training.action_supervision import normalize_action_policy
    policy = normalize_action_policy(config["training"].get("action_supervision"))
    if max_train_steps is not None and stage not in ("sft_trial", "estimate_trial"):
        raise ValueError("--max-train-steps requires sft_trial or estimate_trial.")
    if stage in ("sft_trial", "estimate_trial") and max_train_steps is None:
        max_train_steps = 5
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "config.json"
    if snapshot.exists() and read_json(snapshot) != config:
        raise ValueError("This output directory belongs to a different configuration; choose a new directory.")
    validate_run_artifacts(output, action_supervision_policy=policy, method_version=mv)
    write_json(snapshot, config)
    model_keys = model_keys or (list(MODEL_LABELS) if stage != "local" else ["small_classifier", "rule_counter"])
    if any(key not in ALL_MODEL_LABELS for key in model_keys):
        raise ValueError(f"Unknown model; choose from {list(ALL_MODEL_LABELS)}")
    if stage == "local" and set(model_keys) - {"small_classifier", "rule_counter"}:
        raise ValueError("The local stage only supports small_classifier and rule_counter; it never calls Tinker.")
    sequence = ["data", "sft_trial", "sft", "rs", "evaluate", "figures"] if stage == "all" else (["data", "evaluate", "figures"] if stage == "local" else [stage])
    backend = None
    status_path = output / "run_status.json"
    status = read_json(status_path) if status_path.exists() else {"stages": {}, "real_experiments": "pending"}
    for current in sequence:
        status["stages"][current] = "running"
        write_json(status_path, status)
        try:
            needs_api = current in ("sft", "sft_trial", "rs", "rl", "rl_v2", "preflight") or (current in ("evaluate", "endgame_evaluate") and any(k.startswith("lora") or k == "zero_shot" for k in model_keys))
            validate_run_artifacts(output, action_supervision_policy=policy, method_version=mv)
            if needs_api and backend is None:
                backend = make_backend(config, output)
            if current == "preflight":
                backend.preflight()
            if current == "data":
                if not (output / "dataset/data_summary.json").exists():
                    generate_data(config, output)
                summary = read_json(output / "dataset/data_summary.json")
                if summary["shortfall"]:
                    raise ValueError(f"Data generation reached max_games with {summary['shortfall']} samples still missing; artifacts were saved.")
            elif current in ("estimate", "estimate_trial"):
                estimate_cost(config, output, max_train_steps=max_train_steps)
            elif current == "sft":
                train_model(config, output, backend)
            elif current == "sft_trial":
                trial_steps = max_train_steps if max_train_steps is not None else 5
                train_model(config, output, backend, "sft_trial", max_train_steps=trial_steps)
            elif current == "rs":
                if not (output / "dataset/rs_train.jsonl").exists():
                    generate_rs_data(config, output, backend)
                train_model(config, output, backend, "rs")
            elif current == "rl":
                train_rl(config, output, backend, variant="rl")
            elif current == "rl_v2":
                train_rl(config, output, backend, variant="rl_v2")
            elif current == "evaluate":
                evaluate(config, output, backend, model_keys)
            elif current == "endgame_evaluate":
                evaluate_endgames(config, output, backend, model_keys, required=True)
            elif current == "figures":
                generate_figures(output, method_version=mv)
            status["stages"][current] = "trial_complete" if current == "sft_trial" else "complete"
        except Exception as exc:
            status["stages"][current] = "failed"
            status["error"] = {"stage": current, "type": type(exc).__name__, "message": str(exc)}
            write_json(status_path, status)
            raise
        status.pop("error", None)
        write_json(status_path, status)
    ui_config = {"contract": current_contract(mv), "method_version": mv, "base_model": config["base_model"],
                 "budget_path": str(ROOT / "outputs/budget_usage.jsonl"),
                 "budget_limit_usd": config["budget_limit_usd"], "sampling": config["sampling"]}
    for key, relative in (("sft_adapter", "models/sft/adapter_metadata.json"), ("rs_adapter", "models/rs/adapter_metadata.json"),
                          ("rl_adapter", "models/rl/adapter_metadata.json"),
                          ("rl_v2_adapter", "models/rl_v2/adapter_metadata.json"),
                          ("classifier_weights", "models/small_classifier_weights.json")):
        if (output / relative).exists():
            ui_config[key] = str(output / relative)
    (output / "ui_models.yaml").write_text(yaml.safe_dump(ui_config, allow_unicode=True), encoding="utf-8")
    status["real_experiments"] = "pending"
    if all(status["stages"].get(s) == "complete" for s in ("sft", "rs", "evaluate")):
        evaluated = read_json(output / "experiment_summary.json")["models_evaluated"]
        if set(evaluated) == set(MODEL_LABELS):
            status["real_experiments"] = "complete"
    write_json(status_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=("all", "local", "data", "estimate", "estimate_trial", "sft", "sft_trial", "rs", "rl", "rl_v2", "evaluate", "endgame_evaluate", "figures", "preflight"), default="all")
    parser.add_argument("--max-train-steps", type=int, help="Bounded trial only; default 5 steps for sft_trial/estimate_trial")
    parser.add_argument("--models", help="Comma-separated model keys for evaluation")
    parser.add_argument("--smoke", action="store_true", help="Small local data/check sizes; never changes model identities")
    parser.add_argument("--eval-blocks", type=int)
    parser.add_argument("--target-samples", type=int)
    parser.add_argument("--belief-target-samples", type=int)
    parser.add_argument("--train-games", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config["dataset"].update(total_target_samples=24, belief_target_samples=24, num_games=12, max_games=24, workers=2, batch_games=6,
                                 rollout_trajectories_per_state=2, min_advantage_margin=0.0)
        if config_method_version(config) == "adaptive_v4":
            config["dataset"]["counterfactual_train_pairs"] = 2
        config["evaluation"].update(num_blocks=2, unseen_belief_samples=24, unseen_belief_max_games=12)
    if args.eval_blocks is not None:
        config["evaluation"]["num_blocks"] = args.eval_blocks
    if args.target_samples is not None:
        config["dataset"]["total_target_samples"] = args.target_samples
    if args.belief_target_samples is not None:
        config["dataset"]["belief_target_samples"] = args.belief_target_samples
    if args.train_games is not None:
        config["dataset"]["num_games"] = args.train_games
    if args.workers is not None:
        config["dataset"]["workers"] = args.workers
    if args.seed is not None:
        config["seed"] = args.seed
    output = args.output_dir or ROOT / "outputs/runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_pipeline(config, output, args.stage, args.models.split(",") if args.models else None,
                          max_train_steps=args.max_train_steps)
    print(json.dumps({"output_dir": str(output.resolve()), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
