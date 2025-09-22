from __future__ import annotations

import re
import math
import chess
import random
from typing import List, Dict, Any, Optional, Tuple
from itertools import groupby

from .variation_node import VariationNode
from .phrase_banks import phrase_banks


# ------------------------------------------------------------------
# Main API for turning tree to natural language
# ------------------------------------------------------------------
def generate_reasoning(
        initial_board: chess.Board,
        root_entries: List[Dict[str, Any]],
        initial_score: int,
        append_move_delta: bool = True,
        append_move_value_all_moves: bool = True,
        bracket_value_type: str = "value",  # 'value' or 'minimax'
    ) -> List[Dict[str, Any]]:
        """ External-facing function to generate an explanation. Wraps around the MoveExplanation class """
        explainer = MoveExplanation(
            initial_board,
            root_entries,
            initial_score,
            append_move_delta=append_move_delta,
            append_move_value_all_moves=append_move_value_all_moves,
            bracket_value_type=bracket_value_type,
        )
        return explainer.generate_explanations()


# ------------------------------------------------------------------
# Helper object for move explanations 
# ------------------------------------------------------------------


class MoveExplanation:
    """
    Build an uncertain, human‑sounding commentary over a *list* of analysis
    entries (dictionaries containing VariationNode trees) produced by ChessExplainer.

    Key ideas
    ---------
    • Works from our side's point of view (initial_board.turn).
    • Iterates through each provided analysis entry.
    • For each entry with a valid tree, generates a narrative explanation.
    • First picks the best root move by minimax across all valid trees.
    • "Writes off" clearly inferior lines using two tunable cut‑offs:
         ROOT_WRITE_OFF_CP   – compared to the best root (within its own explanation)
         BRANCH_WRITE_OFF_CP – compared to the best sibling at that depth
    • Recurses until leaf nodes or until a branch falls under a write‑off.
    • Never shows raw centipawn numbers, only natural‑language verdicts.
    • Stores the generated explanation string back into the input dictionary.
    • Returns the list of dictionaries, now including explanations.
    """

    # ------------------------------------------------------------------ #
    #                        TUNABLE HYPERPARAMETERS                      #
    # ------------------------------------------------------------------ #
    INF = 10_000_000              # Sentinel for minimax initialisation
    MATE_CP   = 10_000            # Stockfish convention
    
    ROOT_WRITE_OFF_CP   = 100     # "bad strategy" threshold at root (cp)
    ROOT_WRITE_OFF_EPS  = 0.2     # Probability we still check this root even if 'written-off'
    BRANCH_WRITE_OFF_CP = 60      # same idea for sub‑branches   (cp)


    # Determinants for move value
    GOOD_MOVE_CP = 50
    BAD_MOVE_CP = 50
    EXCELLENT_MOVE_CP = 100
    BLUNDER_CP = 100
    
    # RL-theory specific tuners
    NARRATE_BOARD_VALUE = True
    NARRATE_MOVE_VALUE = False
    SHOW_MOVE_VALUE = False
    SHOW_BOARD_VALUE = False

    # Output text hyperparams
    OUTPUT_TEXT_FORMAT = "depth_paragraph"    # {'list', 'paragraph', 'depth_paragraph'}
    
    PIECE_NAMES = {
        chess.PAWN:   "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK:   "rook",
        chess.QUEEN:  "queen",
        chess.KING:   "king",
    }

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        initial_board: chess.Board,
        root_entries: List[Dict[str, Any]],
        initial_score: int,
        append_move_delta: bool = True,
        append_move_value_all_moves: bool = False,
        bracket_value_type: str = "value",
    ):
        self.initial_board = initial_board.copy()
        self.root_entries = root_entries
        self.initial_score = initial_score
        self.root_color = self.initial_board.turn
        self.append_move_delta = append_move_delta
        # If True, append value brackets for every move (branches and leaves). If False, only at leaves.
        self.append_move_value_all_moves = append_move_value_all_moves
        # Choose label/value source for brackets: 'value' (node.score) or 'minimax' (node.minimax)
        self.bracket_value_type = bracket_value_type if bracket_value_type in ("value", "minimax") else "value"

        # Extract valid VariationNode roots from the dictionaries
        self.roots: List[VariationNode] = []
        for entry in self.root_entries:
            tree = entry.get('tree')
            if isinstance(tree, VariationNode):
                self.roots.append(tree)

        # Determine the best root move by minimax from OUR point of view IF roots exist
        self.best_root: Optional[VariationNode] = None
        self.best_root_value: Optional[int] = None
        if self.roots:
            self.best_root = (
                max(self.roots, key=lambda n: n.minimax))
            self.best_root_value = self.best_root.minimax

        # Need this to avoid confusion in the narration when jumping multiple depths
        self.previous_narration_depth = 0

    # ------------------------------------------------------------------ #
    #                         PUBLIC ENTRY POINT                         #
    # ------------------------------------------------------------------ #
    def generate_explanations(self) -> List[Dict[str, Any]]:
        """Generates explanations for each root entry and returns the updated list."""
        explanations = []
        depth_values = [self.initial_score]
        board = self.initial_board.copy()

        # Generate explanation for each entry
        for entry in self.root_entries:
            node = entry.get('tree')
            self.previous_narration_depth = 0   # Reset since always starting at root
            narrations, is_writeoff = self._generate_recursive_explanation(board, node, depth_values)
            board_value = None if is_writeoff else self._narrate_board_value(node)
            narrations.append((board_value, node.depth))
            entry['explanation'] = self.format_explanation(narrations)
            explanations.append(entry)

        # Generate our final response value
        final_statement, best_move_uci = self._generate_final_choice([node.get('tree') for node in self.root_entries])

        return explanations, final_statement, best_move_uci

    # ------------------------------------------------------------------ #
    #                          PRIVATE HELPERS                           #
    # ------------------------------------------------------------------ #
    def _generate_recursive_explanation(
            self, 
            board: chess.Board, 
            node: VariationNode,
            depth_values: List[int],
        ) -> Tuple[List[Tuple[str, int]], bool]:
        """
        Primary function to recursively generate programmatic explanations using DFS technique.
        Returns:
            explanation_parts: List of explanations in natural language with depth.
            is_writeoff: If true, means this node was written off (note that if this node's child is written off we don't propagate written off up)
        """
        explanation_parts: List[Tuple[str, int]] = []
        our_move = board.turn == self.root_color
        is_root = node.parent is None

        # For all moves we will want to narrate
        if is_root:
            narration, is_writeoff = self._narrate_root(board, node)
        else:
            narration, is_writeoff = self._narrate_branch(board, node, depth_values, our_move)
        
        # Add to our list of explanation parts with current depth
        for text in narration:
            if text is not None:
                explanation_parts.append((text, node.depth))

        # Base case 1: Written off
        if is_writeoff:
            return explanation_parts, True
        
        # Recursive case
        children_considered = 0
        board.push(node.move)   # Update board for this move
        depth_values.append(node.score)
        for child in node.children:
            narrations, is_writeoff = self._generate_recursive_explanation(
                board=board,
                node=child,
                depth_values=depth_values
            )
            children_considered += 0 if is_writeoff else 1
            explanation_parts.extend(narrations)

        # If multiple children considered, need to generate final 'best move' narration
        if children_considered > 1:
            # Note: We need to do 'not our_move' since we're narrating for our child (and based on how this recursion was defined)
            best_move_narrations = self._narrate_best_move(board, node.children, not our_move)
            for text in best_move_narrations:
                if text is not None:
                    explanation_parts.append((text, node.children[0].depth))
            
        _ = depth_values.pop()
        _ = board.pop()
        
        return explanation_parts, False

    # ................................................................. #    
    def _narrate_root(self, board: chess.Board, root: VariationNode) -> Tuple[List[str], bool]:
        """Narrate the root node and its children."""
        root_narration = []
        # First narrate the first move.
        move_description, _ = self._describe_move(board, root)
        root_narration.extend([
            random.choice(phrase_banks['root_consider_phrases']).format(
            move_description = move_description)
        ])

        # Now check if the root is a write-off
        if root.minimax < self.best_root_value - self.ROOT_WRITE_OFF_CP:
            # Even if it's below threshold, we still have a chance to explore it
            if random.random() < self.ROOT_WRITE_OFF_EPS:
                return root_narration, False
            
            # Append minimax value to the write-off phrase (trim trailing period, add bracket + period)
            writeoff_phrase = random.choice(phrase_banks['write_off_root_phrases'])
            if writeoff_phrase and writeoff_phrase.endswith('.'):
                writeoff_phrase = writeoff_phrase[:-1]
            writeoff_phrase = f"{writeoff_phrase} [mm{root.minimax:+d}]."
            root_narration.append(writeoff_phrase)
            return root_narration, True
        
        return root_narration, False
    

    def _narrate_branch(self, board, node, depth_values, our_move) -> Tuple[List[str, Any], bool]:
        """ Helper function to generate text for a branch node. """
        branch_text = []
        first_child = node == node.parent.children[0]

        prefix = self._get_depth_jump_prefix(self.previous_narration_depth, node.depth, our_move)

        move_description, value_narration = self._describe_move(board, node, depth_values)
        if our_move:
            if first_child:
                branch_text.extend([
                    prefix + random.choice(phrase_banks['our_move_first_child_phrases']).format(move_description=move_description),
                    value_narration
                ])
            else:
                branch_text.extend([
                    prefix + random.choice(phrase_banks['our_move_sibling_phrases']).format(move_description=move_description),
                    value_narration
                ])
        else:
            if first_child:
                branch_text.extend([
                    prefix + random.choice(phrase_banks['opponent_move_first_child_phrases']).format(move_description=move_description),
                    value_narration
                ])
            else:
                branch_text.extend([
                    prefix + random.choice(phrase_banks['opponent_move_sibling_phrases']).format(move_description=move_description),
                    value_narration
                ])
        
        prune_text, is_writeoff = self._consider_branch(
            node=node, 
            our_move=our_move
        )
        branch_text.append(prune_text)

        return branch_text, is_writeoff


    def _consider_branch(self, node, our_move) -> Tuple[List[str], bool]:
        """ Checks if we should consider this branch or if we should prune. """
        if our_move:
            if node.minimax < node.parent.minimax - self.BRANCH_WRITE_OFF_CP:
                prune_text = random.choice(phrase_banks['us_prune_branch_phrases'])
                return prune_text, True
        else:
            if node.minimax > node.parent.minimax + self.BRANCH_WRITE_OFF_CP:
                prune_text = random.choice(phrase_banks['opponent_prune_branch_phrases'])    
                return prune_text, True

        # If this is returned, we'll continue exploring that branch
        return None, False


    def _narrate_best_move(self, board, children, our_move) -> List[str]:
        """ Given a list of children picks the best one and returns in natural language. """
        best_child = None
        optimal_minimax = -self.INF if our_move else self.INF
        best_move_narration = ""

        # Find the best child
        for child in children:
            if our_move:
                if child.minimax > optimal_minimax:
                    optimal_minimax = child.minimax
                    best_child = child
            else:
                if child.minimax < optimal_minimax:
                    optimal_minimax = child.minimax
                    best_child = child
        
        # Will use this regardless of if leaf or branch node
        # Force minimax bracket when verbalizing the best move summary
        move_description, _ = self._describe_move(board, best_child, bracket_force_type="minimax")
            
        # First case - this is a leaf node (assuming depth of leaf nodes all same - based on our build this should work)
        if len(best_child.children) == 0:        
            if our_move:
                best_move_narration = random.choice(phrase_banks['us_best_move_leaf_phrases']).format(
                    move_description=move_description)
            else:
                best_move_narration = random.choice(phrase_banks['opponent_best_move_leaf_phrases']).format(
                    move_description=move_description)
            return [best_move_narration]

        # Alternatively -- these are branch nodes. Need to add more context here
        uci_list = self._get_uci_list(children)
        if our_move:
            best_move_narration = random.choice(phrase_banks['us_best_move_branch_phrases']).format(
                uci_list=uci_list,
                move_description=move_description,
            )
        else:
            best_move_narration = random.choice(phrase_banks['opponent_best_move_branch_phrases']).format(
                uci_list=uci_list,
                move_description=move_description,
            )
        
        return [best_move_narration]


    def _describe_move(
            self, 
            board: chess.Board, 
            node: VariationNode, 
            depth_values: List[int] = None,
            bracket_force_type: Optional[str] = None,  # override 'value'/'minimax' for this call
        ) -> Tuple[str, bool]:
        """ Function to, given a move, describe the move (and optionally narrate move value). """
        color = "white" if board.turn == chess.WHITE else "black"
        self.previous_narration_depth = node.depth   # Set for current node

        # Castling
        if board.is_castling(node.move):
            side = "kingside" if chess.square_file(node.move.to_square) == 6 else "queenside"
            move_description = f"{color} castles {side} ({node.move.uci()})"
            move_description += self._format_delta_bracket(node, depth_values, force_type=bracket_force_type)
            return move_description, None

        piece = board.piece_at(node.move.from_square)
        piece_name = self.PIECE_NAMES.get(piece.piece_type, "piece")
        dest = chess.square_name(node.move.to_square)

        if board.is_capture(node.move):
            captured = (
                board.piece_at(node.move.to_square)
                if not board.is_en_passant(node.move)
                else board.piece_at(chess.square(node.move.to_square % 8, node.move.from_square // 8))
            )
            cap_name = self.PIECE_NAMES.get(captured.piece_type, "piece") if captured else "piece"
            action = f"captures the {cap_name} on {dest}"
        else:
            action = f"moves to {dest}"

        if node.move.promotion:
            promo = self.PIECE_NAMES[node.move.promotion]
            action += f", promoting to {promo}"
        
        # Clone the board and make the move to check for check/checkmate
        test_board = board.copy()
        test_board.push(node.move)
        
        if test_board.is_checkmate():
            action += " delivering checkmate"
        elif test_board.is_check():
            action += " putting the king in check"
        
        move_description = f"{color} {piece_name} {action} ({node.move.uci()})"
        move_description += self._format_delta_bracket(node, depth_values, force_type=bracket_force_type)

        # Narrate move value (if hyperparam set)
        value_narration = None
        if self.NARRATE_MOVE_VALUE and depth_values and len(depth_values) > 1:
            move_value = f"[{node.score-depth_values[-2]}]" if self.SHOW_MOVE_VALUE else ""
             # Case 1: Excellent move
            if node.score > (depth_values[-2] + self.EXCELLENT_MOVE_CP) and node.delta_score > self.EXCELLENT_MOVE_CP:
                value_narration = random.choice(phrase_banks['excellent_move_phrases']).format(move_value=move_value)
            # Case 2: Good move
            elif node.score > (depth_values[-2] + self.GOOD_MOVE_CP) and node.delta_score > self.GOOD_MOVE_CP:
                value_narration = random.choice(phrase_banks['good_move_phrases']).format(move_value=move_value)
            # Case 3: Blunder
            elif node.score < (depth_values[-2] - self.BLUNDER_CP) and node.delta_score < self.BLUNDER_CP:
                value_narration = random.choice(phrase_banks['blunder_phrases']).format(move_value=move_value)
            # Case 4: Bad move
            elif node.score < (depth_values[-2] - self.BAD_MOVE_CP) and node.delta_score > self.BAD_MOVE_CP:
                value_narration = random.choice(phrase_banks['bad_move_phrases']).format(move_value=move_value)
            # Case 5: Nothing to report, doesn't materially sway board
            else:
                pass
            return move_description, value_narration

        return move_description, value_narration

    def _format_delta_bracket(self, node: VariationNode, depth_values: Optional[List[int]], force_type: Optional[str] = None) -> str:
        """Append a bracket with the raw centipawn score (signed), with label.

        Behavior:
        - If append_move_delta is False: never append.
        - If append_move_value_all_moves is True: append on all moves that have a numeric value.
        - Otherwise (default/back-compat): append only at leaf nodes.
        - Label with 'v' for node.score or 'mm' for node.minimax. 'force_type' overrides global setting.
        """
        if not self.append_move_delta:
            return ""

        try:
            # Show on all moves when enabled
            if getattr(self, 'append_move_value_all_moves', False):
                val_type = force_type if force_type in ("value", "minimax") else self.bracket_value_type
                # Prefer chosen value; ensure it exists
                if val_type == "minimax" and getattr(node, 'minimax', None) is not None:
                    return f" [mm{int(node.minimax):+d}]"
                if getattr(node, 'score', None) is not None:
                    return f" [v{int(node.score):+d}]"
                return ""

            # Otherwise, only show at leaf nodes
            if getattr(node, "children", None) and len(node.children) > 0:
                return ""
        except Exception:
            # If children not accessible, fall through and try to show score if present
            pass

        try:
            val_type = force_type if force_type in ("value", "minimax") else self.bracket_value_type
            if val_type == "minimax" and getattr(node, 'minimax', None) is not None:
                return f" [mm{int(node.minimax):+d}]"
            if getattr(node, 'score', None) is not None:
                return f" [v{int(node.score):+d}]"
        except Exception:
            pass
        return ""


    def _narrate_board_value(self, root: VariationNode) -> str:
        """
        Generates a statement (for the root) that provides a natural language definition for the value of the board based on the minimax score.
        """
        if not self.NARRATE_BOARD_VALUE:
            return None

        # Always include signed centipawn score when narrating board value, labeled as minimax
        board_value = f"[mm{root.minimax:+d}]"
        phrases = phrase_banks['board_valuation_phrases']
        node_score = root.minimax

        # Compute softmax weights based on negative absolute difference
        temp = 10.0    # Tested this out -- 10-25 seems to be a good distribution 
        diffs = [abs(node_score - val) for (_, val) in phrases]
        logits = [-d / temp for d in diffs]
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]

        # Sample from the distribution
        idx = random.choices(range(len(phrases)), weights=probs, k=1)[0]
        phrase = phrases[idx][0]
        return phrase.format(board_value=board_value)


    def format_explanation(self, explanation_parts: List[Tuple[str, int]]) -> str:
        """ 
        Given your explanations (list of tuples of strings and depths),
        return your explanation as a single string for output.

        Format style based on hyperparameter 'OUTPUT_TEXT_FORMAT' 
        """
        output_text = ""
        if self.OUTPUT_TEXT_FORMAT == 'list':
            for (s, depth) in explanation_parts:
                if s is not None:
                    output_text += f"[{depth}] {s}\n"
        elif self.OUTPUT_TEXT_FORMAT == 'paragraph':
            strings_only = [s for s, _ in explanation_parts if s is not None]
            output_text = self._sentence_casing(strings_only)
        elif self.OUTPUT_TEXT_FORMAT == 'depth_paragraph':
            paragraphs = []
            valid_parts = [(s, d) for s, d in explanation_parts if s is not None]

            for depth, group in groupby(valid_parts, key=lambda x: x[1]):
                strings_at_depth = [item[0] for item in group]
                
                if strings_at_depth:
                    indent = "| " * depth
                    paragraph_content = ' '.join(strings_at_depth)
                    paragraph_content_cased = self._sentence_casing([paragraph_content])
                    paragraph_text = f"{indent}{paragraph_content_cased}"
                    paragraphs.append(paragraph_text)
                
            output_text = "\n".join(paragraphs)
        else:
            raise ValueError("OUTPUT_TEXT_FORMAT not one of the defined formats.")

        return output_text


    def _generate_final_choice(self, nodes: List[VariationNode]) -> Tuple[str, str]:
        # First need to find our best move
        best_node = None
        max_val = -self.INF
        for node in nodes:
            if node.minimax > max_val:
                max_val = node.minimax
                best_node = node

        # Now generate our final analysis statement
        # Build move description, but strip any trailing value/minimax bracket so only the final minimax is shown
        best_move, _ = self._describe_move(self.initial_board, best_node)
        best_move = re.sub(r"\s*\[(?:mm|v)[+-]?\d+\]$", "", best_move)
        phrase = random.choice(phrase_banks['final_statement_phrases'])
        # Force minimax in the final statement bracket
        board_value_bracket = f"[mm{best_node.minimax:+d}]"
        # Allow phrase to include {board_value} if desired; otherwise append at end.
        final_statement = phrase.format(best_move=best_move, board_value=board_value_bracket)
        if '{board_value}' not in phrase:
            final_statement = f"{final_statement} {board_value_bracket}"
        return final_statement, best_node.move.uci()        


    @staticmethod
    def _get_uci_list(children: List[VariationNode]) -> str:
        """ Given a list of children, return a string of their moves. """
        uci_list = ""
        for i in range(0, len(children)-1):
            uci_list += children[i].move.uci() + ", "
        
        # Formatting for the last element to ensure correct comma usage 
        if len(children) == 2:
            uci_list = uci_list [:-2] + " and " + children[-1].move.uci()
        else:
            uci_list += "and " + children[-1].move.uci()

        return uci_list


    @staticmethod
    def _get_depth_jump_prefix(prev_narration_depth, node_depth, our_move):
        depth_jump_size = (prev_narration_depth - node_depth)//2

        # Nominal - no need to prefix about which move we're returning to
        if depth_jump_size <= 0:
            return ""
        
        returning_to_move = node_depth // 2
        phrases = ['first', 'second', 'third'] # Won't take our tree deeper than this likely

        if our_move:
            return f"Returning to our {phrases[returning_to_move]} move, "
        else:
            return f"Returning to their {phrases[returning_to_move]} move, "


    @staticmethod
    def _sentence_casing(narrations: List[str]) -> str:
        """Converts a List of string/None with multiple sentences into sentence case."""
        filtered_narrations = [narration for narration in narrations if narration is not None]    
        text = " ".join(filtered_narrations)
        
        # First, lowercase everything
        text = text.lower()
        
        # Split text into sentences using regex to handle '.', '!', and '?'
        sentences = re.split(r'([.!?])', text)
        processed_sentences = []

        # Process sentence parts and capitalize the beginning of each sentence
        for i in range(0, len(sentences) - 1, 2):
            sentence_part = sentences[i].strip()
            if sentence_part: # Ensure not empty before capitalizing
                sentence_part = sentence_part[0].upper() + sentence_part[1:]
            punctuation = sentences[i + 1]
            processed_sentences.append(sentence_part + punctuation)

        # Handle any trailing text without punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            trailing_part = sentences[-1].strip()
            trailing_part = trailing_part[0].upper() + trailing_part[1:]
            processed_sentences.append(trailing_part)

        final_text = " ".join(processed_sentences)

        # Correct capitalization for 'I' and its contractions
        # Using more robust patterns without extra spaces and with better word boundaries
        i_patterns = [
            (r'(?<=\s)i(?=\s|$)', 'I'),  # Standalone I
            (r'(?<=\s)i\'m(?=\s|$)', 'I\'m'),  # I'm
            (r'(?<=\s)i\'ve(?=\s|$)', 'I\'ve'),  # I've
            (r'(?<=\s)i\'ll(?=\s|$)', 'I\'ll'),  # I'll
            (r'(?<=\s)i\'d(?=\s|$)', 'I\'d'),  # I'd
            (r'(?<=\s)i(?=\s|\'|$)', 'I'),  # Catches any remaining cases of 'i' before apostrophes
        ]
        
        for pattern, replacement in i_patterns:
            final_text = re.sub(pattern, replacement, final_text)

        return final_text