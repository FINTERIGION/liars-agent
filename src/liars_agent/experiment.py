"""Configuration-driven experiment stages; local checks never invoke a model API."""
from __future__ import annotations

import argparse
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


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["platform"] != "tinker":
        raise ValueError("The implemented real training backend is Tinker.")
    pool_path = path.parent / config.pop("opponent_config", "opponent_pools.yaml")
    config["opponents"] = yaml.safe_load(pool_path.read_text(encoding="utf-8"))
    # Instantiate configured agents once to catch misspelled or unused parameters.
    for pool in ("seen_pool", "unseen_pool"):
        for spec in config["opponents"][pool]["agents"]:
            create_rule_agent(spec["class"], name=spec["id"], **spec.get("params", {}))
    config["contract"] = current_contract()
    if config["dataset"].get("history_ablation_augmentation_ratio", 0) != 0:
        raise ValueError("Training history augmentation is disabled in this method version.")
    if {s["class"] for s in config["opponents"]["seen_pool"]["agents"]} != {"HonestConservativeBot", "AggressiveBlufferBot", "CardCounterBot"}:
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


def generate_data(config: dict, output: Path) -> dict:
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


def estimate_cost(config: dict, output: Path) -> dict:
    """Real-tokenizer estimates, including balanced cycling and binary scoring."""
    from liars_agent.training.tinker_train import build_sft_datum
    from liars_agent.agents.prompt_builder import BELIEF_SYSTEM_PROMPT, build_bluff_user_prompt
    from liars_agent.engine.observation import PlayerObservation
    backend = make_backend(config, output)
    lengths, counts, score_costs = {}, {}, []
    maximum = 0
    for key in ("train", "val", "test"):
        lengths[key] = {"action": [], "belief": []}
        for row in read_jsonl(output / f"dataset/{key}.jsonl"):
            datum = build_sft_datum(row, backend, config["training"]["max_seq_length"])
            length = len(datum.model_input.to_ints())
            maximum = max(maximum, length)
            lengths[key][row["task"]].append(length)
            if row["task"] == "belief":
                obs = PlayerObservation.from_dict(row["observation"])
                prefix = backend.encode_messages([{"role": "system", "content": BELIEF_SYSTEM_PROMPT},
                    {"role": "user", "content": build_bluff_user_prompt(obs)}], generation=True)
                score_costs.append((2 * (len(prefix) + 1) * backend.rates["prefill"] + 2 * backend.rates["sample"]) / 1_000_000)
        counts[key] = {task: sum(sizes) for task, sizes in lengths[key].items()}
    if any(not sizes for sizes in lengths["train"].values()):
        raise ValueError("Both training tasks need samples for balanced cost estimation.")
    training = config["training"]
    half = training["batch_size"] // 2
    if half < 1 or training["batch_size"] % 2:
        raise ValueError("Balanced training needs an even batch size.")
    steps = math.ceil(max(map(len, lengths["train"].values())) / half)
    training_tokens = 0
    for epoch in range(training["num_epochs"]):
        for index, task in enumerate(("action", "belief")):
            sizes = lengths["train"][task]
            order = list(range(len(sizes)))
            random.Random(training.get("seed", config["seed"]) + 2 * epoch + index).shuffle(order)
            training_tokens += sum(sizes[order[i % len(order)]] for i in range(steps * half))
    validation_tokens = sum(counts["val"].values()) * training["num_epochs"]
    result = {"contract": current_contract(), "model": config["base_model"], "tokens_by_split_and_task": counts,
        "max_input_tokens": maximum, "balanced_training_tokens": training_tokens,
        "validation_tokens": validation_tokens,
        "sft_train_and_validation_usd": backend.estimate_training_cost(training_tokens + validation_tokens),
        "belief_scoring_mean_usd_upper_bound_on_dataset": sum(score_costs) / len(score_costs),
        "belief_scoring_max_usd_upper_bound_on_dataset": max(score_costs),
        "belief_scoring_accounting": "two complete prefills plus at most two generated tokens; no cache discount assumed",
        "rates_per_million_tokens": backend.rates, "budget": backend.ledger.snapshot(),
        "scope": "SFT exact dataset token estimate; inference/RS trajectory volume remains to be estimated before paid execution."}
    write_json(output / "cost_estimate.json", result)
    return result


def train_model(config: dict, output: Path, backend, stage: str = "sft") -> dict:
    from liars_agent.training.tinker_train import run_tinker_lora_sft
    path = output / f"models/{stage}/adapter_metadata.json"
    if path.exists():
        metadata = read_json(path)
        require_contract(metadata, "model metadata")
        return metadata
    ds_summary = read_json(output / "dataset/data_summary.json")
    require_contract(ds_summary, "dataset summary")
    if ds_summary["shortfall"]:
        raise ValueError(f"Dataset shortfall: {ds_summary['shortfall']}; increase num_games in a new run before formal training.")
    splits = {key: list(read_jsonl(output / f"dataset/{key}.jsonl")) for key in ("train", "val", "test")}
    training = config["training"]
    resume_from = None
    resume_metadata = None
    if stage == "rs":
        splits["train"] = list(read_jsonl(output / "dataset/rs_train.jsonl"))
        if any(sum(r["task"] == task for r in splits["train"]) < config["rs"][key]
               for task, key in (("action", "target_samples"), ("belief", "belief_target_samples"))):
            raise ValueError("RS dataset has not reached its configured target.")
        resume_metadata = read_json(output / "models/sft/adapter_metadata.json")
        require_contract(resume_metadata, "SFT resume checkpoint")
        resume_from = resume_metadata["state_path"]
    return run_tinker_lora_sft(
        samples=splits["train"] + splits["val"] + splits["test"], splits=splits, output_dir=path.parent,
        backend=backend, base_model=config["base_model"], lora_rank=config["lora"]["r"],
        learning_rate=config["rs"]["learning_rate"] if stage == "rs" else training["learning_rate"],
        num_epochs=config["rs"]["num_epochs"] if stage == "rs" else training["num_epochs"],
        batch_size=training["batch_size"], max_seq_length=training["max_seq_length"],
        seed=training.get("seed", config["seed"]), resume_from=resume_from, resume_metadata=resume_metadata, stage=stage,
        resume=(path.parent / "training_progress.json").exists(),
        train_attn=config["lora"]["train_attn"], train_mlp=config["lora"]["train_mlp"],
        train_unembed=config["lora"]["train_unembed"],
    )


def llm_factory(config: dict, output: Path, backend, key: str):
    from liars_agent.agents.llm_agent import LLMAgent
    adapter = None if key == "zero_shot" else output / f"models/{'sft' if key == 'lora_sft' else 'rs'}/adapter_metadata.json"
    sampling = config["sampling"]
    return lambda seed: LLMAgent(name=key, mode="lora_rs_improved" if key == "lora_rs" else key,
        model_version=config["base_model"], backend=backend, adapter_path=adapter, seed=seed,
        temperature=sampling["temperature"], top_p=sampling["top_p"], max_new_tokens=sampling["max_new_tokens"])


def generate_rs_data(config: dict, output: Path, backend) -> dict:
    from liars_agent.training.data_generator import evaluate_observation_action_values, make_action_sample, extract_belief_samples
    from liars_agent.engine.observation import PlayerObservation
    from liars_agent.engine.replay import verify_replay_determinism
    rs = config["rs"]
    shard_dir = output / "dataset/rs_games"
    shard_dir.mkdir(parents=True, exist_ok=True)
    accepted, beliefs = [], []
    factory = llm_factory(config, output, backend, "lora_sft")
    factories = opponent_factories(config)
    target, belief_target = rs["target_samples"], rs["belief_target_samples"]
    for index in range(rs.get("max_games", rs["num_games"] * 5)):
        if len(accepted) >= target and len(beliefs) >= belief_target:
            break
        shard = shard_dir / f"game_{index:05d}.jsonl"
        if shard.exists():
            rows = list(read_jsonl(shard))
            for row in rows:
                require_contract(row, "RS sample")
            accepted.extend(r for r in rows if r["task"] == "action")
            beliefs.extend(r for r in rows if r["task"] == "belief")
            continue
        game_seed = _derive_seed(config["seed"], "rs_environment", index)
        game_id = f"rs_train_{index:05d}"
        actor = index % 4
        agent = factory(_derive_seed(config["seed"], "rs_actor", index))
        order = list(range(3))
        random.Random(_derive_seed(config["seed"], "rs_order", index)).shuffle(order)
        lineup = [None] * 4
        lineup[actor] = agent
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
            proposals = agent.sample_candidates(obs, num_samples=rs["candidates_per_state"], include_belief=False)
            candidates = {a.action_id: a for a, record in proposals if not record.was_fallback}
            if len(candidates) < 2:
                continue
            values = evaluate_observation_action_values(obs, teacher_config(config), rs["rollouts_per_candidate"],
                _derive_seed(config["seed"], "rs_teacher", index, obs.round_id, obs.step_id), list(candidates.values()))
            best = max(candidates.values(), key=lambda a: (values[a.action_id], a.action_type, a.cards))
            advantage = values[best.action_id] - sum(values.values()) / len(values)
            if advantage < rs["min_advantage_margin"]:
                continue
            samples.append(make_action_sample(obs, best, game_id, rollout_advantage=advantage,
                rollout_best_value=values[best.action_id], candidate_count=len(values), source="sft_rejection_sampling",
                teacher="pool_prior_counter_continuation"))
        write_jsonl(shard, samples + belief_rows)
        accepted.extend(samples)
        beliefs.extend(belief_rows)
        print(f"RS data: action {len(accepted)}/{target}, belief {len(beliefs)}/{belief_target}", flush=True)
    write_jsonl(output / "dataset/rs_train.jsonl", accepted + beliefs)
    summary = {"contract": current_contract(), "actual_samples": len(accepted), "target_samples": target,
        "actual_belief_samples": len(beliefs), "belief_target_samples": belief_target,
        "shortfall": max(0, target - len(accepted)) + max(0, belief_target - len(beliefs))}
    write_json(output / "dataset/rs_summary.json", summary)
    return summary


def frozen_belief_samples(config: dict, output: Path, pool: str) -> list[dict]:
    from liars_agent.training.data_generator import extract_belief_samples
    from liars_agent.engine.replay import verify_replay_determinism
    path = output / f"dataset/frozen_belief_{pool}.jsonl"
    if path.exists():
        rows = list(read_jsonl(path))
        for row in rows:
            require_contract(row, "frozen belief observations")
        return rows
    if pool == "seen":
        rows = list(read_jsonl(output / "dataset/belief_test.jsonl"))
    else:
        target = config["evaluation"]["unseen_belief_samples"]
        factories = opponent_factories(config, "unseen")
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
            rows.extend(extract_belief_samples([replay], min(10, target - len(rows)),
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
    results = {}
    for pool in ("seen", "unseen"):
        rows = frozen_belief_samples(config, output, pool)
        results[pool] = {}
        predictions = {}
        for key in model_keys:
            path = output / f"evaluation/belief_{pool}_{key}.json"
            if path.exists():
                stored = read_json(path)
                require_contract(stored, "frozen belief predictions")
                if [r["sample_id"] for r in stored["predictions"]] != [r["sample_id"] for r in rows]:
                    raise ValueError("Frozen belief prediction IDs differ from the held-out set.")
                predictions[key] = stored["predictions"]
                results[pool][key] = stored["metrics"]
                continue
            if key == "small_classifier":
                agent = SmallClassifierBot.load_weights(output / "models/small_classifier_weights.json")
            elif key == "rule_counter":
                agent = None
            else:
                agent = llm_factory(config, output, backend, key)(_derive_seed(config["seed"], "belief_eval"))
            records = []
            for row in rows:
                obs = PlayerObservation.from_dict(row["observation"])
                probability = estimate_heuristic_bluff_belief(obs)[0] if agent is None else agent.predict_bluff_prob(obs)
                records.append({"sample_id": row["sample_id"], "game_id": row["game_id"],
                    "label": row["label"], "probability": probability})
            metrics = compute_probability_calibration([r["probability"] for r in records], [r["label"] for r in records])
            metadata = {"model_key": key, "base_model": config["base_model"] if agent is not None and hasattr(agent, "checkpoint") else None,
                "checkpoint": getattr(agent, "checkpoint", None),
                "belief_source": {"small_classifier": "classifier", "rule_counter": "heuristic"}.get(key, "model")}
            stored = {"contract": current_contract(), "predictions": records, "metrics": metrics,
                "model_metadata": metadata, "usage": agent.tracker.to_dict() if hasattr(agent, "tracker") else {"api_calls": 0}}
            write_json(path, stored)
            predictions[key], results[pool][key] = records, metrics
        results[pool]["paired"] = {}
        for a, b in (("lora_sft", "zero_shot"), ("lora_rs", "lora_sft"), ("small_classifier", "rule_counter")):
            if a in predictions and b in predictions:
                results[pool]["paired"][f"{a}_vs_{b}"] = compute_paired_probability_metrics(predictions[a], predictions[b],
                    config["evaluation"]["bootstrap_samples"], config["seed"])
    return results


def evaluate(config: dict, output: Path, backend, model_keys: list[str]) -> dict:
    from liars_agent.evaluation.duplicate_runner import run_duplicate_evaluation
    from liars_agent.evaluation.metrics import (compute_behavioral_profile, compute_belief_calibration,
        compute_risk_and_hand_stratified_diagnosis, compute_paired_bootstrap_ci)
    results = {}
    for pool in ("seen", "unseen"):
        results[pool] = {}
        for key in model_keys:
            target = output / f"evaluation/{pool}_{key}_full.json"
            replay_path = target.with_suffix(".replays.jsonl")
            if target.exists():
                results[pool][key] = read_json(target)
                require_contract(results[pool][key], "evaluation result")
                continue
            if key == "small_classifier":
                factory = lambda seed: SmallClassifierBot.load_weights(output / "models/small_classifier_weights.json")
                metadata = {"backend": "local_classifier", "weights": str(output / "models/small_classifier_weights.json")}
            elif key == "rule_counter":
                factory = lambda seed: CardCounterBot(name="rule_counter", seed=seed)
                metadata = {"backend": "rule"}
            else:
                factory = llm_factory(config, output, backend, key)
                metadata = {"backend": "tinker", "base_model": config["base_model"]}
                if key != "zero_shot":
                    metadata.update(read_json(output / f"models/{'sft' if key == 'lora_sft' else 'rs'}/adapter_metadata.json"))
            result = run_duplicate_evaluation(factory, opponent_factories(config, pool),
                num_blocks=config["evaluation"]["num_blocks"], base_seed=config["seed"] + (1000 if pool == "seen" else 5000),
                pool_name=f"{pool}_pool", model_metadata=metadata, opponent_specs=opponent_specs(config, pool),
                num_resamples=config["evaluation"]["bootstrap_samples"], bootstrap_seed=config["seed"])
            data = result.to_dict(include_replays=False)
            data["contract"] = current_contract()
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
        for a, b in (("lora_sft", "zero_shot"), ("lora_rs", "lora_sft"), ("small_classifier", "rule_counter")):
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
                    require_contract(data, "ablation result")
                else:
                    result = run_duplicate_evaluation(llm_factory(config, output, backend, "lora_sft"),
                        opponent_factories(config), num_blocks=config["evaluation"]["num_blocks"],
                        base_seed=config["seed"] + 1000, pool_name="seen_pool", ablation_mode=mode,
                        model_metadata=results["seen"]["lora_sft"]["model_metadata"],
                        opponent_specs=opponent_specs(config), num_resamples=config["evaluation"]["bootstrap_samples"],
                        bootstrap_seed=config["seed"])
                    data = result.to_dict(include_replays=False)
                    data["contract"] = current_contract()
                    data["profile"] = compute_behavioral_profile([r.to_dict() for r in result.replays], "lora_sft")
                    save_replays_jsonl(result.replays, target.with_suffix(".replays.jsonl"))
                    write_json(target, data)
            ablations.append({"mode": mode, "label": mode, "win_rate": data["overall_win_rate"],
                "challenge_hit_rate": data["profile"]["challenge_hit_rate"], "bluff_rate": data["profile"]["bluff_rate"],
                "block_win_rates": data["block_win_rates"], "win_rate_ci": data["win_rate_ci"],
                "denominators": {"win_rate": data["total_games"],
                    "challenge_hit_rate": data["profile"]["denominators"]["challenge_hit_rate"]}})
    frozen = evaluate_frozen_beliefs(config, output, backend, model_keys)
    summary = {"contract": current_contract(), "frozen_belief": frozen, "status": "complete" if set(model_keys) == set(MODEL_LABELS) else "selected_models_only",
               "models_evaluated": model_keys, "results": results, "paired_bootstrap_tests": paired,
               "history_ablation": ablations, "config": config,
               "dataset_and_training": read_json(output / "dataset/data_summary.json")}
    write_json(output / "experiment_summary.json", summary)
    return summary


def generate_figures(output: Path) -> list[str]:
    from liars_agent.evaluation.plots import (generate_fig1_winrate_svg, generate_fig2_history_ablation_svg,
        generate_fig3_risk_stratified_svg, generate_fig4_calibration_and_profile_svg)
    summary = read_json(output / "experiment_summary.json")
    require_contract(summary, "experiment summary")
    pools = summary["results"]
    items = {}
    for pool in ("seen", "unseen"):
        items[pool] = [{"key": key, "label": MODEL_LABELS[key], "win_rate": value["overall_win_rate"],
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



def validate_run_artifacts(output: Path) -> None:
    """Validate existing contracts and both task schemas before paid preflight."""
    from liars_agent.training.tinker_train import validate_training_sample
    paths = [output / "dataset/data_summary.json", output / "dataset/rs_summary.json", output / "experiment_summary.json"]
    paths.extend((output / "evaluation").glob("*.json"))
    paths.extend((output / "models").glob("*/adapter_metadata.json"))
    paths.extend((output / "models").glob("*/training_progress.json"))
    paths.extend((output / "dataset/batches").glob("batch_*.json"))
    for path in paths:
        if path.exists():
            require_contract(read_json(path), str(path))
    classifier = output / "models/small_classifier_weights.json"
    if classifier.exists():
        require_contract(read_json(classifier), "classifier weights")
    sample_paths = list((output / "dataset").glob("*.jsonl"))
    sample_paths.extend((output / "dataset/batches").glob("samples_*.jsonl"))
    sample_paths.extend(p for p in (output / "dataset/rs_games").glob("game_*.jsonl") if not p.name.endswith(".replay.jsonl"))
    for path in sample_paths:
        for row in read_jsonl(path):
            validate_training_sample(row)
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

def run_pipeline(config: dict, output: Path, stage: str = "all", model_keys: list[str] | None = None) -> dict:
    require_contract(config, "experiment configuration")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "config.json"
    if snapshot.exists() and read_json(snapshot) != config:
        raise ValueError("This output directory belongs to a different configuration; choose a new directory.")
    validate_run_artifacts(output)
    write_json(snapshot, config)
    model_keys = model_keys or (list(MODEL_LABELS) if stage != "local" else ["small_classifier", "rule_counter"])
    if any(key not in MODEL_LABELS for key in model_keys):
        raise ValueError(f"Unknown model; choose from {list(MODEL_LABELS)}")
    if stage == "local" and set(model_keys) - {"small_classifier", "rule_counter"}:
        raise ValueError("The local stage only supports small_classifier and rule_counter; it never calls Tinker.")
    sequence = ["data", "sft", "rs", "evaluate", "figures"] if stage == "all" else (["data", "evaluate", "figures"] if stage == "local" else [stage])
    backend = None
    status_path = output / "run_status.json"
    status = read_json(status_path) if status_path.exists() else {"stages": {}, "real_experiments": "pending"}
    for current in sequence:
        status["stages"][current] = "running"
        write_json(status_path, status)
        try:
            needs_api = current in ("sft", "rs", "preflight") or (current == "evaluate" and any(k.startswith("lora") or k == "zero_shot" for k in model_keys))
            validate_run_artifacts(output)
            if needs_api and backend is None:
                backend = make_backend(config, output)
                backend.preflight()
            if current == "data":
                if not (output / "dataset/data_summary.json").exists():
                    generate_data(config, output)
                summary = read_json(output / "dataset/data_summary.json")
                if summary["shortfall"]:
                    raise ValueError(f"Data generation reached max_games with {summary['shortfall']} samples still missing; artifacts were saved.")
            elif current == "estimate":
                estimate_cost(config, output)
            elif current == "sft":
                train_model(config, output, backend)
            elif current == "rs":
                if not (output / "dataset/rs_train.jsonl").exists():
                    generate_rs_data(config, output, backend)
                train_model(config, output, backend, "rs")
            elif current == "evaluate":
                evaluate(config, output, backend, model_keys)
            elif current == "figures":
                generate_figures(output)
            status["stages"][current] = "complete"
        except Exception as exc:
            status["stages"][current] = "failed"
            status["error"] = {"stage": current, "type": type(exc).__name__, "message": str(exc)}
            write_json(status_path, status)
            raise
        status.pop("error", None)
        write_json(status_path, status)
    ui_config = {"contract": current_contract(), "base_model": config["base_model"], "budget_path": str(ROOT / "outputs/budget_usage.jsonl"),
                 "budget_limit_usd": config["budget_limit_usd"], "sampling": config["sampling"]}
    for key, relative in (("sft_adapter", "models/sft/adapter_metadata.json"), ("rs_adapter", "models/rs/adapter_metadata.json"),
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
    parser.add_argument("--stage", choices=("all", "local", "data", "estimate", "sft", "rs", "evaluate", "figures", "preflight"), default="all")
    parser.add_argument("--models", help="Comma-separated model keys for evaluation")
    parser.add_argument("--smoke", action="store_true", help="Small local data/check sizes; never changes model identities")
    parser.add_argument("--eval-blocks", type=int)
    parser.add_argument("--target-samples", type=int)
    parser.add_argument("--belief-target-samples", type=int)
    parser.add_argument("--train-games", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config["dataset"].update(total_target_samples=24, belief_target_samples=24, num_games=12, max_games=24, workers=2, batch_games=6,
                                 rollout_trajectories_per_state=2, min_advantage_margin=0.0)
        config["evaluation"].update(num_blocks=2, unseen_belief_samples=24, unseen_belief_max_games=12)
    if args.eval_blocks is not None:
        config["evaluation"]["num_blocks"] = args.eval_blocks
    if args.target_samples is not None:
        config["dataset"]["total_target_samples"] = args.target_samples
    if args.belief_target_samples is not None:
        config["dataset"]["belief_target_samples"] = args.belief_target_samples
    if args.train_games is not None:
        config["dataset"]["num_games"] = args.train_games
    if args.seed is not None:
        config["seed"] = args.seed
    output = args.output_dir or ROOT / "outputs/runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_pipeline(config, output, args.stage, args.models.split(",") if args.models else None)
    print(json.dumps({"output_dir": str(output.resolve()), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
