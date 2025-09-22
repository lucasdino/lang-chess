import chess
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass(slots=True)
class VariationNode:
    """One ply in the variation tree."""

    move: chess.Move
    score: int                     # Static evaluation **after** this move
    delta_score: Optional[int]     # Change in static evaluation from parent node
    minimax: int                   # Result of minimax (+αβ) below this node
    depth: int                     # Depth of this node in the tree (root = 0)
    is_mate: bool = False          # Engine says position is forced mate
    mate_in: Optional[int] = None  # Moves until mate (sign ‑ for us to move)
    parent: "VariationNode" = None # Also have pointer to parent (None -> root)
    children: List["VariationNode"] = field(default_factory=list)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------

    def uci(self) -> str:
        """Return move in UCI notation."""
        return self.move.uci()

    def visualize(self, depth: int = 0) -> str:
        """Pretty‑print subtree for debugging."""
        indent = "  " * depth
        delta_str = f" (Δ={self.delta_score:+})" if self.delta_score is not None else ""
        line = f"{indent}{self.uci()} (score={self.score}{delta_str}, minimax={self.minimax})"
        if self.is_mate:
            line += f" (mate in {self.mate_in})"
        for child in self.children:
            line += "\n" + child.visualize(depth + 1)
        return line