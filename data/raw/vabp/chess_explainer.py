from __future__ import annotations
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
import chess
import chess.engine

from .variation_node import VariationNode
from .generate_reasoning import generate_reasoning



###############################################################################
# Main driver class
###############################################################################
class ChessExplainer:
    """Wrapper around Stockfish to build explanation trees and generate prose."""

    # Tunables – tweak for speed/quality trade‑off --------------------------
    ROOT_SAMPLING = {
        "min_k": 2,
        "max_k": 3,
        "min_p": 0.00,
        "cum_p": 1.0,
        "temp": 500
    }
    
    TREE_PLAYER_SAMPLING = {
        "min_k": 1,
        "max_k": 2,
        "min_p": 0.2,
        "cum_p": 0.8,
        "temp": 20
    }

    TREE_OPP_SAMPLING = {
        "min_k": 1,
        "max_k": 2,
        "min_p": 0.2,
        "cum_p": 0.8,
        "temp": 10
    }

    TREE_NODES_MAX = 10              # Hard cap to keep engine calls bounded
    TREE_DEPTH_MAX = 2               # Plies *after* the root move
    
    INF = 10_000_000                 # Sentinel for minimax initialisation
    MATE_SCORE = 10_000              # Normalised mate value (≫ any cp score)

    # Default engine path finding
    STOCKFISH_PATH = str((Path(__file__).resolve().parent / ".." / "stockfish").resolve())

    # ------------------------------------------------------------------
    # Construction / teardown
    # ------------------------------------------------------------------

    def __init__(
        self,
        engine_path: str | Path | None = None,
        depth: int = 15,
        multipv: int = 15,
        think_time: float = 0.2,
    ) -> None:
        resolved_engine_path = str(engine_path or self.STOCKFISH_PATH)
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(resolved_engine_path)
        except FileNotFoundError:
             print(f"Error: Engine executable not found at '{resolved_engine_path}'.")
             print("Please install Stockfish and ensure it's in your PATH or set STOCKFISH_EXECUTABLE env var.")
             raise
        except Exception as e:
             print(f"Error starting engine '{resolved_engine_path}': {e}")
             raise

        # Store config used for root analysis and tree building defaults
        self._root_cfg: Dict[str, Any] = {
            "depth": depth,
            "multipv": multipv,
            "think_time": think_time,
        }
        self._nodes_created = 0
        self._root_color: Optional[chess.Color] = None # Will be set in analyze_position

    def close(self) -> None:
        """Shut down Stockfish subprocess."""
        self._engine.quit()

    # Context manager support
    def __enter__(self) -> ChessExplainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------

    def analyze_position(
        self, fen: str, generate_explanation: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyzes the top moves from a FEN position, builds variation trees,
        and optionally generates natural language explanations.

        Args:
            fen: The FEN string of the board state.
            generate_explanation: If True, generate prose using MoveExplanation.

        Returns:
            A list of dictionaries, each containing:
             - 'uci': The move in UCI notation.
             - 'score': The engine's evaluation of the move (static score after the move).
             - 'tree': The VariationNode root of the calculated variation tree.
             - 'explanation': (Optional) The natural language explanation string.
        """
        board = chess.Board(fen)
        self._root_color = board.turn
        self._eval_cache = dict()

        # Return values for initial baseline analysis (using shallower approx.)
        initial_baseline_analysis = self._analyze(
            board,
            depth=self._root_cfg['depth'],
            multipv=1,
            think_time=self._root_cfg['think_time'] / 2
        )
        initial_board_score = initial_baseline_analysis[0]['score']

        # Get root moves to build off of
        root_lines = self._analyze(board, **self._root_cfg)
        sampled_moves = self._sample_moves(
            lines=root_lines,
            **self.ROOT_SAMPLING,
            opponent_turn=False,
            shuffle=True
        )

        # Now generate trajectories and explanations for each sampled move
        results: List[Dict[str, Any]] = []
        for move_info in sampled_moves:
            self._nodes_created = 0 # Reset counter for each tree
            tree = self._build_tree(
                board.copy(),
                move=move_info["move"],
                parent_score=initial_board_score,
                ply_left=self.TREE_DEPTH_MAX,
                alpha=-self.INF,
                beta=self.INF,
                current_depth=0 # Start depth at 0 for root moves
            )
            
            # Append our tree and optionally generate a natural language explanation for this trajectory
            if tree is not None:
                entry: Dict[str, Any] = {
                    "uci": move_info["move"].uci(),
                    "score": tree.score,
                    "tree": tree,
                    "explanation": None
                }
            results.append(entry)

        if generate_explanation:
            results = generate_reasoning(board.copy(), results, initial_board_score)
            
        return results

    def visualize_tree(self, tree: VariationNode) -> str:
        """Utility for quick CLI inspection."""
        return tree.visualize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    
    # -------- Stockfish Engine / sampling helpers ----------
    def _analyze(
        self, board: chess.Board, *, depth: int, multipv: int, think_time: float
    ) -> List[Dict[str, Any]]:
        """Run Stockfish and structure its output (POV = *self._root_color*)."""
        limit = (
            chess.engine.Limit(time=think_time)
            if think_time > 0
            else chess.engine.Limit(depth=depth)
        )
        if self._root_color is None:
            raise ValueError("Root color must be set before calling _analyze.")

        try:
            raw = self._engine.analyse(board, limit=limit, multipv=multipv)
            return self._structure_analysis(raw, self._root_color)
        except chess.engine.EngineTerminatedError:
            raise ValueError("Engine terminated unexpectedly during analysis.")
        except Exception:
            raise ValueError("Error during engine analysis.")

    @classmethod
    def _structure_analysis(
        cls, raw: List[Dict[str, Any]], perspective: chess.Color
    ) -> List[Dict[str, Any]]:
        """Convert engine JSON blobs → sorted move dictionaries."""
        moves: List[Dict[str, Any]] = []
        for entry in raw:
             pv = entry.get("pv")
             if not pv:
                 move = entry.get("move")
                 if move is None: continue
             else:
                 move = pv[0]
    
             score_obj = entry["score"].pov(perspective)
             if score_obj.is_mate():
                 score = cls.MATE_SCORE * (1 if score_obj.mate() > 0 else -1)
                 mate_in = score_obj.mate()
             else:
                 score = score_obj.score(mate_score=cls.MATE_SCORE) or 0
                 mate_in = None

             moves.append({
                 "move": move,
                 "score": score, # This is the static evaluation AFTER the move 'move'
                 "is_mate": score_obj.is_mate(),
                 "mate_in": mate_in,
             })
        
        if len(moves) == 0:
            raise ValueError("No moves found in analysis.")
        return sorted(moves, key=lambda d: d["score"], reverse=True)

    def _sample_moves(
        self, 
        lines: List[Dict[str, Any]], 
        min_k: int,
        max_k: int,
        min_p: float,
        cum_p: float,
        temp: float,
        opponent_turn: bool,
        shuffle: bool = True
    ) -> List[Dict[str, Any]]:
        """ Given a list of lines, sample a random number of moves from this. """
        scores = np.array([line["score"] for line in lines])
        # print(f"Opp Turn:{opponent_turn}\n{scores}")
        
        # Need to handle case when sampling for opponent (min is better)
        if opponent_turn:
            scores = -scores
        normalized_scores = scores - np.max(scores)
        
        # Apply softmax
        exp_scores = np.exp(normalized_scores / temp)
        probabilities = exp_scores / np.sum(exp_scores)
        # print(f"Ps: {probabilities}")
        
        # Sample moves based on probabilities
        selected_indices = [i for i, p in enumerate(probabilities) if p > min_p]
        if len(selected_indices) < min_k:
            selected_indices = np.argsort(probabilities)[-min_k:]
        elif len(selected_indices) > max_k:
            selected_indices = np.random.choice(selected_indices, size=max_k, replace=False, p=probabilities[selected_indices] / probabilities[selected_indices].sum())
        selected_indices = sorted(selected_indices, key=lambda i: probabilities[i], reverse=True)

        # Ensure cumulative probability constraint
        cumulative_prob = 0.0
        final_indices = []
        for i in selected_indices:
            if cumulative_prob >= cum_p and len(final_indices) >= min_k:
                break
            final_indices.append(int(i))
            cumulative_prob += probabilities[i]
        # print(f"FI: {final_indices}")

        # Return the sampled moves
        samples = [lines[i] for i in final_indices]
        if shuffle:
            random.shuffle(samples)
        # print(f"Samples:\n{samples}"")
        return samples

    # -------- Recursive tree creation helpers ----------
    def _leaf_score(self, board: chess.Board) -> int:
        """Cheap fallback evaluation when node/ply limits hit."""
        fen = board.fen()
        if fen in self._eval_cache:
            return self._eval_cache[fen][0]["score"]
        quick_analysis = self._analyze(board, depth=6, multipv=1, think_time=0.05)
        if quick_analysis:
            self._eval_cache[fen] = quick_analysis
            return quick_analysis[0]["score"]
        if board.is_checkmate():
            return self.MATE_SCORE if board.turn != self._root_color else -self.MATE_SCORE
        # Standard evaluation for draws is 0 centipawns
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        # Should not happen if called correctly, but return 0 as fallback
        return 0

    def _terminal_score(self, board: chess.Board) -> int:
        """Return score for game-ending positions from root perspective."""
        if self._root_color is None:
            raise ValueError("Root color must be set before calling _terminal_score.")
        if board.is_checkmate():
            return self.MATE_SCORE if board.turn != self._root_color else -self.MATE_SCORE
        return 0

    def _build_tree(
        self,
        board: chess.Board,
        move: chess.Move,
        parent_score: int,
        ply_left: int,
        alpha: int,
        beta: int,
        current_depth: int, # Added depth parameter
        child_eval: Optional[int] = None,
    ) -> Optional[VariationNode]:
        """Depth-limited α-β search using Stockfish for evaluations."""
        indent = "  " * (self.TREE_DEPTH_MAX - ply_left)

        # --- 1. Make the move ---
        board.push(move)
        self._nodes_created += 1
        fen = board.fen()

        # --- 2. Check for immediate game over ---
        if board.is_game_over(claim_draw=True):
            final_score = self._terminal_score(board)
            delta_score = final_score - parent_score
            is_mate = board.is_checkmate()
            mate_in = 0 if is_mate else None
            node = VariationNode(
                move=move,
                score=final_score,
                delta_score=delta_score,
                minimax=final_score,
                depth=current_depth, # Set depth
                is_mate=is_mate,
                mate_in=mate_in,
                parent=None,
                children=[]
            )
            board.pop()
            return node

        # --- 3. Get analysis lines (next moves) for current position ---
        lines = self._eval_cache.get(fen)
        if lines is None:
            lines = self._analyze(board, **self._root_cfg)
            self._eval_cache[fen] = lines


        # --- 4. Determine current static score ---
        current_score: int
        if child_eval is not None:
            # Prefer the score passed from the parent analysis if available
            current_score = child_eval
        elif lines:
            # If no child_eval, use the top move's score from the analysis results
            current_score = lines[0]["score"]
        else:
            # If analysis returned no lines (e.g., unexpected terminal state) or cache contained empty list
            # Use a fallback evaluation - this ideally shouldn't be reached often
            current_score = self._leaf_score(board)


        delta_score = current_score - parent_score

        # --- 5. Check depth/node limits ---
        if ply_left <= 0:
            node = VariationNode(
                move=move,
                score=current_score,
                delta_score=delta_score,
                minimax=current_score,
                depth=current_depth, # Set depth
                parent=None,
                children=[]
            )
            board.pop() # Pop the move that led to this leaf node
            return node
        if self._nodes_created >= self.TREE_NODES_MAX:
            node = VariationNode(
                move=move,
                score=current_score,
                delta_score=delta_score,
                minimax=current_score,
                depth=current_depth, # Set depth
                parent=None,
                children=[]
            )
            board.pop() # Pop the move that led to this leaf node
            return node

        # --- 6. Handle case where analysis yielded no moves ---
        # This might happen if the position is terminal but wasn't caught by is_game_over
        if not lines:
            final_score = self._terminal_score(board) # Use definitive terminal score
            delta_score = final_score - parent_score
            is_mate = board.is_checkmate()
            mate_in = 0 if is_mate else None
            node = VariationNode(
                move=move,
                score=final_score,
                delta_score=delta_score,
                minimax=final_score,
                depth=current_depth, # Set depth
                is_mate=is_mate,
                mate_in=mate_in,
                parent=None,
                children=[]
            )
            board.pop()
            return node

        # --- 7. Check for distant mate ---
        current_is_mate = lines[0]["is_mate"]
        current_mate_in = lines[0]["mate_in"]
        if current_is_mate and current_mate_in is not None and abs(current_mate_in) > ply_left:
            current_is_mate = False
            current_mate_in = None

        # --- 8. Sample moves for recursion ---
        maximizing = board.turn == self._root_color
        sampling_params = self.TREE_OPP_SAMPLING if not maximizing else self.TREE_PLAYER_SAMPLING
        sampled_moves = self._sample_moves(
            lines=lines,
            **sampling_params,
            opponent_turn=not maximizing,
            shuffle=True
        )

        # --- 9. Recursive calls ---
        best_minimax_val = -self.INF if maximizing else self.INF
        children: List[VariationNode] = []
        a, b = alpha, beta
        for i, move_info in enumerate(sampled_moves):
            child_node = self._build_tree(
                board,
                move=move_info["move"],
                parent_score=current_score, # Current node's static score is parent score for children
                ply_left=ply_left - 1,
                alpha=a,
                beta=b,
                current_depth=current_depth + 1, # Increment depth for children
                child_eval=move_info["score"], # Pass the known evaluation *after* child move
            )

            if child_node:
                children.append(child_node)
                child_minimax = child_node.minimax

                if maximizing:
                    best_minimax_val = max(best_minimax_val, child_minimax)
                    a = max(a, best_minimax_val)
                else: # Minimizing
                    best_minimax_val = min(best_minimax_val, child_minimax)
                    b = min(b, best_minimax_val)

                if a >= b:
                    break


        # --- 10. Finalize node ---
        board.pop() # Pop the move made in step 1

        if not children and sampled_moves:
             # This case means sampling returned moves, but all recursive calls failed or were pruned immediately.
             # Fallback to static score.
            best_minimax_val = current_score
        elif not children and not sampled_moves:
            # This case means sampling returned 0 moves. Should be rare if lines were valid.
            best_minimax_val = current_score

        # Create the parent node
        node = VariationNode(
            move=move,
            score=current_score, # The static score determined in step 4
            delta_score=delta_score,
            minimax=best_minimax_val, # The result of minimax search below this node
            depth=current_depth, # Set depth
            is_mate=current_is_mate,
            mate_in=current_mate_in,
            parent=None,
            children=children,
        )
        
        # Set parent reference for all child nodes
        for child in children:
            child.parent = node
            
        return node