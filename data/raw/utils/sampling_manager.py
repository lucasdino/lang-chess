import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional


# ------------- helper -------------------------------------------------------- #
def _normalise_spec(spec: Dict, total: int) -> Dict[str, float]:
    """
    Accept {label: p, …} where p may sum to anything; return a
    {label: p_norm} that sums to 1, multiplied by *total* later.
    Supports tuple ranges (lo, hi|None) – converted to strings "lo‑hi"/"lo+".
    """
    def label(k):
        if isinstance(k, tuple):    # (lo, hi|None)
            lo, hi = k
            return f"{lo}+" if hi is None else f"{lo}-{hi}"
        return str(k)

    clean = {label(k): v for k, v in spec.items()}
    s = sum(clean.values())
    if s <= 0:
        raise ValueError("Criterion proportions must sum to a positive number.")
    return {k: v / s for k, v in clean.items()}


def _allocate_counts(props, total):
    """Greedy rounding so ints sum to *total*."""
    exact  = np.array(props) * total
    floors = np.floor(exact).astype(int)
    rem    = total - floors.sum()
    idx    = np.argsort(-(exact - floors))      # largest remainder first
    floors[idx[:rem]] += 1
    return floors.tolist()


# ------------- main class ---------------------------------------------------- #
class SamplingManager:
    """
    Generic sampler that hits arbitrary marginals via raking
    (iterative proportional fitting).

    Built‑in writers: movecount / player.
    Add any new criterion by creating the bucket column yourself or
    passing a writer via `extra_writers`.
    """

    # --------------------------------------------------------------------- #
    # construction                                                          #
    # --------------------------------------------------------------------- #
    def __init__(self,
                 df: pd.DataFrame,
                 criteria: Optional[Dict[str, Dict]] = None,
                 extra_writers: Optional[Dict[str, Callable[[pd.DataFrame], None]]] = None):
        """
        Parameters
        ----------
        df : DataFrame with at least a 'FEN' column.
        criteria : default criterion dict {criterion: {bucket: p, …}, …}.
        extra_writers : optional mapping {criterion: writer_fn(df)} to
                        auto‑generate bucket columns for new built‑ins.
        """
        self.df = df.copy()
        self.default_criteria = criteria or {}

        # built‑in writers (auto‑materialise columns)
        self._writers: Dict[str, Callable[[], None]] = {
            "movecount": self._write_movecount,
            "player":    self._write_player,
        }
        if extra_writers:
            self._writers.update(extra_writers)

        # ensure bucket columns for the *default* criteria
        self._ensure_columns(self.default_criteria)

    # --------------------------------------------------------------------- #
    # public face                                                           #
    # --------------------------------------------------------------------- #
    def set_df(self, new_df: pd.DataFrame):
        """Swap in a different DataFrame (e.g. after you add columns)."""
        self.df = new_df.copy()

    def get_samples(self,
                    n: int,
                    criteria: Optional[Dict[str, Dict]] = None,
                    replace: bool = False,
                    random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Draw `n` rows to match `criteria` (defaults to `self.default_criteria`).
        Accepts any *new* criterion so long as the DataFrame already
        contains a column either called `<criterion>` or `<criterion>_bucket`.
        """
        crits = criteria or self.default_criteria
        if not crits:
            return self.df.sample(n=n, replace=replace, random_state=random_state)

        # auto‑materialise built‑in columns if needed
        self._ensure_columns(crits)

        # -------- normalise proportions & gather column names ----------- #
        norm = {c: _normalise_spec(spec, n) for c, spec in crits.items()}

        cols = {}
        for c in norm:
            if f"{c}_bucket" in self.df.columns:
                cols[c] = f"{c}_bucket"
            elif c in self.df.columns:
                cols[c] = c
            else:
                raise KeyError(f"No column for criterion '{c}'.")

        # -------- build capacity table ---------------------------------- #
        tbl = (self.df
               .groupby(list(cols.values()), observed=True).size()
               .rename("capacity").to_frame().reset_index())

        # make sure every requested bucket exists in the MultiIndex
        full_levels = [list(norm[c].keys()) for c in norm]
        full_index  = pd.MultiIndex.from_product(full_levels, names=list(cols.values()))
        tbl = tbl.set_index(list(cols.values())) \
                 .reindex(full_index, fill_value=0) \
                 .reset_index()

        # -------- raking (iterative proportional fitting) --------------- #
        tbl["weight"] = np.where(tbl["capacity"] > 0, 1.0, 0.0)

        targets = {
            c: {lbl: p * n for lbl, p in norm[c].items()}
            for c in norm
        }

        def marginal(c):
            return (tbl.groupby(cols[c])["weight"].sum()).to_dict()

        for _ in range(8):          # plenty for ≤6 criteria
            for c in norm:
                cur = marginal(c)
                for lbl, tgt in targets[c].items():
                    w = cur.get(lbl, 0.0)
                    if w == 0 and tgt > 0:
                        raise ValueError(f"Impossible: no rows for {c}={lbl}")
                    if w > 0:
                        tbl.loc[tbl[cols[c]] == lbl, "weight"] *= tgt / w

        # -------- exact integer allocation (never > capacity if !replace) #
        tbl["desired"] = tbl["weight"] * n / tbl["weight"].sum()

        floors = np.floor(tbl["desired"]).astype(int)
        tbl["take"] = floors
        rem = n - floors.sum()

        order = np.argsort(-(tbl["desired"] - floors))
        tbl.loc[order[:rem], "take"] += 1

        # cap to capacity instead of erroring
        if not replace:
            tbl["take"] = np.minimum(tbl["take"], tbl["capacity"])

        # -------- draw --------------------------------------------------- #
        pieces = []
        rng = np.random.default_rng(random_state)
        for _, row in tbl.iterrows():
            k = int(row["take"])
            if k == 0:
                continue

            mask = pd.Series(True, index=self.df.index)
            for c, col in cols.items():
                mask &= self.df[col] == row[col]
            subset = self.df[mask]

            pieces.append(subset.sample(
                n=k,
                replace=replace or len(subset) < k,
                random_state=rng.integers(1e9)
            ))

        return (pd.concat(pieces)
                .sample(frac=1, random_state=random_state)
                .reset_index(drop=True))

    # --------------------------------------------------------------------- #
    # internal: column materialisation                                     #
    # --------------------------------------------------------------------- #
    def _ensure_columns(self, crits: Dict[str, Dict]):
        """Call writer if column missing and writer exists."""
        for c in crits:
            if (c in self.df.columns) or (f"{c}_bucket" in self.df.columns):
                continue
            if c in self._writers:
                self._writers[c]()        # adds the bucket column
            else:
                raise KeyError(f"No column and no writer for criterion '{c}'.")

    # --------------------------------------------------------------------- #
    # built‑in writers                                                     #
    # --------------------------------------------------------------------- #
    def _write_movecount(self):
        def bucket(full):
            for lo, hi in [(0, 9), (10, 19), (20, 29), (30, 39), (40, None)]:
                if full >= lo and (hi is None or full <= hi):
                    return f"{lo}+" if hi is None else f"{lo}-{hi}"
        self.df["movecount_bucket"] = (
            self.df["FEN"].str.split().str[-1].astype(int).apply(bucket)
        )

    def _write_player(self):
        self.df["player_bucket"] = self.df["FEN"].str.split().str[1]