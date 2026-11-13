"""One complete-game partition shared by samples, replay baselines and metrics."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any


def split_game_ids(
    game_ids: Iterable[str], train_ratio: float = .8, val_ratio: float = .1, seed: int = 42,
) -> dict[str, list[str]]:
    """Partition unique IDs deterministically; never split decisions of one game.

    With at least three games, each requested partition receives a game. Smaller
    smoke datasets may have empty validation/test partitions; training callers
    decide whether their requested operation requires those partitions.
    """
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Invalid split ratios.")
    ids = sorted(set(str(gid) for gid in game_ids))
    random.Random(seed).shuffle(ids)
    count = len(ids)
    if count == 0:
        return {"train": [], "val": [], "test": []}
    n_train = max(1, int(round(count * train_ratio)))
    n_val = max(1, int(round(count * val_ratio))) if val_ratio and count >= 3 else 0
    reserve_test = 1 if count >= 2 else 0
    n_val = min(n_val, count - 1 - reserve_test)
    n_train = min(n_train, count - n_val - reserve_test)
    return {"train": ids[:n_train], "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:]}


def split_dataset_by_game_id(
    samples: list[dict[str, Any]], train_ratio: float = .8, val_ratio: float = .1,
    seed: int = 42, game_splits: dict[str, list[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Use an explicit shared partition, or derive one from the sample game IDs."""
    if any(sample.get("historical_example") for sample in samples):
        raise ValueError("Historical examples cannot be used for training; regenerate data with current rules.")
    partition = game_splits if game_splits is not None else split_game_ids(
        (sample["game_id"] for sample in samples), train_ratio, val_ratio, seed)
    owner: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for game_id in partition[name]:
            if str(game_id) in owner:
                raise ValueError(f"Game {game_id} belongs to multiple partitions.")
            owner[str(game_id)] = name
    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        game_id = str(sample["game_id"])
        if game_id not in owner:
            raise ValueError(f"Game {game_id} is absent from the shared partition.")
        result[owner[game_id]].append(sample)
    return result
