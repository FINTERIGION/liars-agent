"""按完整对局 (`game_id`) 划分训练/验证/测试集，防止同局步间数据泄漏."""

from __future__ import annotations

import random
from typing import Any


def split_dataset_by_game_id(
    samples: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """严格以 `game_id` 为单位切分数据集，确保同一局游戏的所有决策步只出现在唯一划分中."""
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Invalid split ratios.")

    game_to_samples: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        gid = str(s["game_id"])
        game_to_samples.setdefault(gid, []).append(s)

    unique_games = sorted(game_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_games)

    n_games = len(unique_games)
    n_train = max(1, int(round(n_games * train_ratio)))
    n_val = max(1, int(round(n_games * val_ratio))) if n_games >= 3 else 0
    if n_train + n_val >= n_games and n_games >= 3:
        n_train = n_games - 2
        n_val = 1

    train_games = set(unique_games[:n_train])
    val_games = set(unique_games[n_train : n_train + n_val])
    test_games = set(unique_games[n_train + n_val :])

    # 校验无交集
    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for gid, group in game_to_samples.items():
        if gid in train_games:
            splits["train"].extend(group)
        elif gid in val_games:
            splits["val"].extend(group)
        else:
            splits["test"].extend(group)

    return splits
