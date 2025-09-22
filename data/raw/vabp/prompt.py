import os
import random
import importlib.util
from typing import List, Tuple

from .phrase_banks import initial_think_phrase

# Use absolute imports to import our board conversion utility since calling from a jupyter notebook
spec = importlib.util.spec_from_file_location("board", os.path.abspath("./utils/board.py"))
board = importlib.util.module_from_spec(spec); spec.loader.exec_module(board)
convert_board = board.convert_board


# --------------------------------------------------
# |       Main Function for Explainer Data         |
# --------------------------------------------------
BOARD_REPRESENTATION = "uniform_visual"  # "fen", "spaced_fen", "visual", "uniform_visual"

def generate_data_sample(fen: str, explanations: List[str], final_statement: str, final_move_uci: str) -> Tuple[str, str, str]:
    """  
    Given a board (FEN notation), explanations, and a final evaluation, create a reasoning trace to train a model on.
    """
    sys_prompt = "chess_task_sysprompt.txt"
    user_prompt = f"""Here is a board in a game you're currently playing. I want you to think of possible moves you could play. If it seems like a good move, you should roll-out the line assuming optimal play from your opponent. For each move (you and opponent), predict the value of the board in the format: '[v+/-#]'. You should verbalize decisions when there are multiple branched moves using minimax logic -- list the minimax value as '[mm+/-#]'.\n\nAfter you think through your various moves, please end by telling me your chosen move (in UCI notation) within answer tags.\n\n{convert_board(fen, BOARD_REPRESENTATION)}"""

    model_response = f"""{random.choice(initial_think_phrase)}
<think> {_format_explanations(explanations, final_statement)} </think>

<answer> {final_move_uci} </answer>"""

    return sys_prompt, user_prompt, model_response


# --------------------------------------------------
# |               Helper Functions                 |
# --------------------------------------------------
def _format_explanations(explanations: List[str], final_statement: str) -> str:
    concat_exp = ""

    for exp in explanations:
        concat_exp += exp + "\n\n"

    return concat_exp + final_statement