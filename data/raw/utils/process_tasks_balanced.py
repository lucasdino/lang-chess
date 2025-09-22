import os
import ast
import json
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .exceptions import DiscardedSample
from .verl_prompts import user_prompt_bank
from .board import convert_board, get_piece_name_at_location
from utils.prompts.chat_to_prompt import ChatProcessor



# ───────────────────────────────────────────
# Balancing configs (keep piece‑agnostic)
# ───────────────────────────────────────────
FULLMOVE_DEFAULT_BUCKETS = [
    ((0, 9), 0.1),
    ((10, 19), 0.3),
    ((20, 29), 0.3),
    ((30, 39), 0.15),
    ((40, 150), 0.15)
]

TASK_BUCKETING_MAP = {
    "bestmove":   FULLMOVE_DEFAULT_BUCKETS,
    "worstmove":  FULLMOVE_DEFAULT_BUCKETS,
    "legalmoves": FULLMOVE_DEFAULT_BUCKETS,
    "predictmove": FULLMOVE_DEFAULT_BUCKETS,
}

DEFAULT_OVERSAMPLE_FACTOR = 4

# ==========================================
# Helper functions for distribution balancing
# ==========================================
def _fullmove_from_fen(fen: str) -> int:
    """Return the full‑move number (6th field) from a FEN string."""
    try:
        return int(fen.split()[5])
    except Exception:
        return 0


def _bucket_fullmove(n: int,
                     bucket_def: List[Tuple[Tuple[int, int], float]]) -> str:
    """Map a full‑move number to its bucket label (e.g. '10‑19')."""
    for (lo, hi), _ in bucket_def:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return ">max"   # should not happen with correct definitions


def _desired_dict(bucket_def: List[Tuple[Tuple[int, int], float]]
                  ) -> Dict[str, float]:
    """Convert bucket definition → {label: ratio} dict."""
    return {f"{lo}-{hi}": r for (lo, hi), r in bucket_def}


def _scale_targets(ratios: Dict[str, float],
                   capacities: Dict[str, int],
                   total: int) -> Dict[str, int]:
    """
    Round desired counts to integers and distribute remainders
    (à la largest‑fraction method) while respecting bucket capacity.
    """
    desired_float = {k: ratios[k] * total for k in ratios}
    floored       = {k: math.floor(v) for k, v in desired_float.items()}
    leftovers     = total - sum(floored.values())

    # Grant remaining counts to buckets with biggest fractional parts
    order = sorted(((desired_float[k] - floored[k], k) for k in ratios),
                   reverse=True)
    for _, k in order:
        if leftovers == 0:
            break
        if floored[k] < capacities.get(k, 0):
            floored[k] += 1
            leftovers  -= 1
    return floored


def _balance_to_distribution(samples: List[dict],
                             bucket_def: List[Tuple[Tuple[int, int], float]],
                             target_total: int) -> List[dict]:
    """
    Down‑sample *samples* to size *target_total* so that the full‑move
    histogram matches *bucket_def* as closely as possible.
    """
    desired_ratio = _desired_dict(bucket_def)

    # Group sample indices by bucket
    bucket_to_idxs: Dict[str, List[int]] = {b: [] for b in desired_ratio}
    for idx, obj in enumerate(samples):
        fen   = obj["extra_info"]["board"]
        label = _bucket_fullmove(_fullmove_from_fen(fen), bucket_def)
        if label in bucket_to_idxs:
            bucket_to_idxs[label].append(idx)

    capacities = {b: len(v) for b, v in bucket_to_idxs.items()}
    targets    = _scale_targets(desired_ratio, capacities, target_total)

    chosen: List[int] = []
    for b, need in targets.items():
        avail = bucket_to_idxs[b]
        random.shuffle(avail)
        chosen.extend(avail[:need])

    # If some buckets ran short, top‑up with random leftovers
    if len(chosen) < target_total:
        leftovers = [i for i in range(len(samples)) if i not in chosen]
        random.shuffle(leftovers)
        chosen.extend(leftovers[: target_total - len(chosen)])

    random.shuffle(chosen)
    return [samples[i] for i in chosen]


# ==========================================
# Main entry points
# ==========================================
def process_tasks_balanced(tasks, generator_args, output_folder,
                           model_version, output_type="parquet"):
    chat_processor = ChatProcessor(tokenizer_version=model_version)

    for task in tasks:
        # ── Read & shuffle raw data ─────────────────────────────────
        df = pd.read_csv(task["data_source"])
        df["Move"]            = df["Move"].apply(ast.literal_eval)
        df["Win Probability"] = df["Win Probability"].apply(ast.literal_eval)
        df = df.sample(frac=1).reset_index(drop=True)

        # ── Create (over‑)sampled RL dataset ───────────────────────
        rl_dataset, gen_meta = create_rl_dataset(
            df=df,
            chat_processor=chat_processor,
            task=task,
            generator_args=generator_args,
        )

        # ── Persist ────────────────────────────────────────────────
        split_dir = os.path.join(output_folder, task["split"])
        os.makedirs(split_dir, exist_ok=True)

        fname = f"{task['type']}.{ 'parquet' if output_type=='parquet' else 'jsonl' if output_type=='jsonl' else 'json' }"
        out_path = os.path.join(split_dir, fname)

        if output_type == "parquet":
            pd.DataFrame(rl_dataset).to_parquet(out_path)
        elif output_type == "jsonl":
            with open(out_path, "w", encoding="utf-8") as f:
                for item in rl_dataset:
                    f.write(json.dumps(item) + "\n")
        elif output_type == "json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rl_dataset, f, indent=2)
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        print(f"✔ Saved {len(rl_dataset)} samples → {out_path}. "
              f"Discarded: {gen_meta['discarded_samples']}, "
              f"raw generated: {gen_meta['raw_generated']})")


def create_rl_dataset(df, chat_processor, task, generator_args,
                      board_notation="visual"):
    """
    Generate many samples, then down‑sample so the final set matches the
    prescribed full‑move distribution for *task['type']*.
    """
    oversample_factor = generator_args.get("oversample_factor",
                                           DEFAULT_OVERSAMPLE_FACTOR)
    target_n  = task["samples"]
    raw_goal  = target_n * oversample_factor
    bucket_def = TASK_BUCKETING_MAP[task["type"]]

    raw_samples: List[dict] = []
    meta = {"discarded_samples": 0, "raw_generated": 0}

    # ── First pass: over‑generate ──────────────────────────────────
    for _, row in df.iterrows():
        if len(raw_samples) >= raw_goal:
            break
        try:
            sys_p, usr_p, gt = _generate_sample(
                df_row=row,
                task_type=task["type"],
                generator_args=generator_args,
                chat_processor=chat_processor,
                board_notation=board_notation,
            )
        except DiscardedSample:
            meta["discarded_samples"] += 1
            continue

        # Build sample object
        raw_samples.append({
            "data_source": f"chess_{task['type']}",
            "prompt": [
                {"role": "system", "content": sys_p},
                {"role": "user",   "content": usr_p},
            ],
            "ability": "chess",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "board": row["FEN"],
                "split": task["split"],
                "data_source": task["data_source"],
            },
        })
        meta["raw_generated"] += 1

    # ── Down‑sample to match distribution ──────────────────────────
    if len(raw_samples) < target_n:
        raise RuntimeError(f"Insufficient samples ({len(raw_samples)}) to "
                           f"meet requested {target_n}. "
                           f"Increase oversample_factor or check data source.")

    final_samples = _balance_to_distribution(raw_samples, bucket_def, target_n)
    return final_samples, meta


# ==========================================
# Sample generation (logic unchanged)
# ==========================================
def _generate_sample(df_row, task_type, generator_args,
                     chat_processor, board_notation):
    """
    Generate a single RL sample (raises DiscardedSample to skip rows
    that don't meet heuristics).
    """
    board      = df_row["FEN"]
    moveset    = df_row["Move"]
    win_probs  = df_row["Win Probability"]
    mp_dict    = dict(zip(moveset, win_probs))
    mp_list    = list(zip(moveset, win_probs))

    if task_type == "predictmove":
        if len(moveset) < generator_args["predictmove_min_possible_moves"]:
            raise DiscardedSample()

        
        sys_p  = chat_processor._get_cached_prompt("chess_task_sysprompt.txt")
        usr_p  = user_prompt_bank[task_type].format(
            formatted_board=convert_board(board, board_notation))
        gt     = str(_score_scaling(
            mp_dict,
            score_scaling=generator_args["predictmove_score_scaling"],
            score_cut=generator_args["predictmove_score_cut"],
        ))

    elif task_type == "bestmove":
        best_move, best_wp = max(mp_list, key=lambda x: x[1])
        filt = [m for m, p in mp_list
                if p < best_wp - generator_args["bestmove_move_threshold"]]
        if len(filt) < generator_args["bestmove_provided_moves"] - 1:
            raise DiscardedSample()
        cand = random.sample(filt,
                             generator_args["bestmove_provided_moves"] - 1)
        cand.append(best_move)
        random.shuffle(cand)

        gt     = str({"answer": best_move, "candidates": cand})
        sys_p  = chat_processor._get_cached_prompt("chess_task_sysprompt.txt")
        usr_p  = user_prompt_bank[task_type].format(
            formatted_board=convert_board(board, board_notation),
            move_candidates=cand,
        )

    elif task_type == "worstmove":
        worst_move, worst_wp = min(mp_list, key=lambda x: x[1])
        filt = [m for m, p in mp_list
                if p > worst_wp + generator_args["worstmove_move_threshold"]]
        if len(filt) < generator_args["worstmove_provided_moves"] - 1:
            raise DiscardedSample()
        cand = random.sample(filt,
                             generator_args["worstmove_provided_moves"] - 1)
        cand.append(worst_move)
        random.shuffle(cand)

        gt     = str({"answer": worst_move, "candidates": cand})
        sys_p  = chat_processor._get_cached_prompt("chess_task_sysprompt.txt")
        usr_p  = user_prompt_bank[task_type].format(
            formatted_board=convert_board(board, board_notation),
            move_candidates=cand,
        )

    elif task_type == "legalmoves":
        pos_cnt = {}
        for m in moveset:
            pos_cnt[m[:2]] = pos_cnt.get(m[:2], 0) + 1
        valid = [k for k, v in pos_cnt.items()
                 if v >= generator_args["legalmoves_min_moves"]]
        if not valid:
            raise DiscardedSample()

        piece_pos  = random.choice(valid)
        piece_name = get_piece_name_at_location(board, piece_pos)
        if piece_name is None:
            raise DiscardedSample()

        gt    = str([m for m in moveset if m.startswith(piece_pos)])
        sys_p = chat_processor._get_cached_prompt("chess_task_sysprompt.txt")
        usr_p = user_prompt_bank[task_type].format(
            formatted_board=convert_board(board, board_notation),
            piece_name=piece_name,
            piece_pos=piece_pos,
        )

    else:
        raise ValueError(f"Unknown task_type '{task_type}'")

    return sys_p, usr_p, gt


# ==========================================
# Misc. utility
# ==========================================
def _score_scaling(score_dict, score_scaling="normalize", score_cut=0.3):
    """
    Transform {move: prob} dict, keeping only values ≥ score_cut after scaling.
    Modes: "normalize" | "linear".
    """
    moves, vals = zip(*score_dict.items())
    vals = np.asarray(vals, dtype=np.float64)

    if score_scaling == "normalize":
        vmin, vmax = vals.min(), vals.max()
        rng  = vmax - vmin
        scaled = (vals - vmin) / rng if rng else np.ones_like(vals)
    elif score_scaling == "linear":
        order  = np.argsort(-vals)           # descending
        linear = np.linspace(1, 0, len(vals))
        scaled = np.empty_like(vals)
        scaled[order] = linear
    else:
        raise ValueError(f"Invalid score_scaling '{score_scaling}'")

    scaled = np.where(scaled < score_cut, 0, scaled)
    return dict(zip(moves, scaled.astype(float).tolist()))