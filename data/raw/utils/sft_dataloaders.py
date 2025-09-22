import os, csv, json, random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets, Dataset

from utils.prompts.chat_to_prompt import LlamaFactoryChatProcessor, TokenizerCounter

# ------------------------------ config structs ------------------------------

@dataclass(frozen=True)
class DatasetSource:
    name: str
    file_paths: List[str]
    weight: float

    def load(self) -> Dataset:
        dsets = [load_dataset("json", data_files=fp, split="train") for fp in self.file_paths]
        dataset = concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]
        return dataset

# ------------------------------ helpers ------------------------------------

def _process_examples(examples: Iterable[dict], chat_processor: LlamaFactoryChatProcessor) -> List[Dict[str, str]]:
    out = []
    for ex in examples:
        sys, usr, ast = chat_processor.process_chat(ex["chat"])
        sample = {"system": sys, "user": usr, "assistant": ast}
        out.append(sample)
    return out

# ------------------------------ loader 1: weighted by *samples* ------------

def load_weighted_by_samples(
    sources: List[DatasetSource],
    max_samples: int,
    rng: random.Random | None = None,
) -> List[Dict[str, str]]:
    rng = rng or random.Random()
    chat_processor = LlamaFactoryChatProcessor()

    loaded = [s.load() for s in sources]

    # Compute the largest total that still respects per-set weights without depletion
    caps = [
        (len(ds) / s.weight) if s.weight > 0 else float("inf")
        for ds, s in zip(loaded, sources)
    ]
    total = int(min(max_samples, *caps))

    # Target counts by weight (last bucket gets the remainder)
    counts = [int(total * s.weight) for s in sources]
    counts[-1] = total - sum(counts[:-1])

    # If any dataset is already too small at these counts, stop early (cannot keep proportions)
    if any(len(ds) < c for ds, c in zip(loaded, counts)):
        # Reduce total to the limiting factor (recompute once)
        limiting_total = min(int(len(ds) / max(s.weight, 1e-12)) for ds, s in zip(loaded, sources))
        total = min(total, limiting_total)
        counts = [int(total * s.weight) for s in sources]
        counts[-1] = total - sum(counts[:-1])

    picked_examples = []
    for ds, c in zip(loaded, counts):
        if c <= 0:
            continue
        idxs = rng.sample(range(len(ds)), min(c, len(ds)))
        picked_examples.extend(ds.select(idxs))

    rng.shuffle(picked_examples)
    return _process_examples(picked_examples[:total], chat_processor)

# ------------------------------ loader 2: weighted by *assistant tokens* ----

def load_weighted_by_tokens(
    sources: List[DatasetSource],
    max_tokens: int,
    token_counter: TokenizerCounter,
    rng: random.Random | None = None,
) -> List[Dict[str, str]]:
    """
    Greedy proportional-fair scheduler on *assistant token mass*:
      at each step pick i = argmin_i (tokens_i / weight_i), add one example from i.
    Stop at max_tokens or when any positive-weight bucket depletes (can’t keep proportions).
    """
    rng = rng or random.Random()
    chat_processor = LlamaFactoryChatProcessor()

    # Load once
    source_names = [s.name for s in sources]
    loaded = [s.load() for s in sources]
    weights = [max(s.weight, 0.0) for s in sources]
    assert any(w > 0 for w in weights), "At least one positive weight required."

    # Assistant token length helper (exact, via TokenizerCounter)
    def ast_len(chat: List[Tuple[str, str]]) -> int:
        ast = ""
        for role, content in chat:
            if role == "assistant":
                ast = content
                break
        return token_counter.count(ast, add_special_tokens=False)

    # Build per-dataset pools: list of (index, token_len). Shuffle for randomness.
    pools: List[List[Tuple[int, int]]] = []
    for ds in loaded:
        lens = [(i, ast_len(ds[i]["chat"])) for i in range(len(ds))]
        rng.shuffle(lens)
        pools.append(lens)

    # Early termination if any positive-weight bucket is empty
    for w, pool in zip(weights, pools):
        if w > 0 and len(pool) == 0:
            return []  # cannot keep proportions at all

    # Running tallies
    token_tally = [0] * len(sources)
    total_tokens = 0
    result_indices: List[Tuple[int, int, int]] = []  # (dataset_id, row_index, tlen)

    # Main loop
    while total_tokens < max_tokens:
        # Choose dataset with minimal (tokens / weight). Datasets with w==0 are never chosen.
        best_i = None
        best_score = None
        for i, (w, pool) in enumerate(zip(weights, pools)):
            if w <= 0:
                continue
            if not pool:
                # If any positive-weight pool is empty, we cannot keep proportions → stop.
                best_i = None
                best_score = None
                break
            score = token_tally[i] / w
            if best_score is None or score < best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break  # cannot keep proportions

        idx, tlen = pools[best_i].pop()  # pop is O(1)
        token_tally[best_i] += tlen
        total_tokens += tlen
        result_indices.append((best_i, idx, tlen))

    # If the last example pushes us over, drop it (to avoid significant overshoot)
    if result_indices and total_tokens > max_tokens:
        best_i, idx, tlen = result_indices.pop()
        token_tally[best_i] -= tlen
        total_tokens -= tlen

    # Materialize and process chats
    final_examples = []
    for ds_id, row_idx, _ in result_indices:
        final_examples.append(loaded[ds_id][row_idx])

    # Zip and print the name -> Amount tokens -> % of total for sanity check
    for i, (name, tokens) in enumerate(zip(source_names, token_tally)):
        pct = 100.0 * tokens / total_tokens if total_tokens > 0 else 0.0
        print(f"  {name:<20} {tokens:>8,} tokens ({pct:5.1f}%)")

    rng.shuffle(final_examples)
    return _process_examples(final_examples, chat_processor)


# ------------------------------ Helper to get token stats on datasets ------------

def write_token_csv_and_stats(
    sources: list[DatasetSource],
    token_counter: TokenizerCounter,
    csv_path: str,
    log_every: int = 100_000,
) -> None:
    per_file_rows: list[dict] = []
    group_totals   = defaultdict(int)
    grand_total    = 0

    for src in sources:
        for fp in src.file_paths:
            token_sum = 0
            line_n    = 0
            print(f"→ Scanning {fp} …")
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line_n += 1
                    if line_n % log_every == 0:
                        print(f"   {line_n:,} lines processed…")
                    obj = json.loads(line)
                    assistant_text = None
                    chat = obj.get("chat", [])
                    for entry in chat:
                        if entry[0] == "assistant":
                            assistant_text = entry[1]
                            break
                    token_sum += token_counter.count(assistant_text)
            print(f"   done: {line_n:,} lines, {token_sum:,} tokens\n")

            per_file_rows.append(
                {"group": src.name, "file": os.path.basename(fp), "tokens": token_sum}
            )
            group_totals[src.name] += token_sum
            grand_total            += token_sum

    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        w = csv.DictWriter(csvf, fieldnames=["group", "file", "tokens"])
        w.writeheader()
        w.writerows(per_file_rows)

    print("\nToken distribution by group (assistant text only):")
    for grp, tk in group_totals.items():
        pct = 100.0 * tk / grand_total if grand_total else 0.0
        print(f"  {grp:<20} {tk:>12,}  ({pct:6.2f}%)")
