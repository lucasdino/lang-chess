user_prompt_bank = {
    "predictmove": """Below is a chess board from your current game.

{formatted_board}

You must select the best move from this position and return it within answer tags. Your answer must be formatted as <answer> my_move </answer>, where my_move is a legal move in UCI notation.

Think step by step if necessary, but your response must be in UCI format and within answer tags. Only answers in the correct format will be accepted.""",
    "bestmove": """Below is a board in a game you're currently playing.

{formatted_board}
    
You must choose the best move from the following moves: {move_candidates}. 

You may want to think out loud to help finalize your answer. However, you must provide your answer within answer tags (e.g., <answer> move_choice </answer>).

The move must be provided in UCI notation and within answer tags in order to be accepted.""",
    "worstmove": """Below is a board in a game you're currently playing.

{formatted_board}
    
You must choose the worst move from the following moves: {move_candidates}. 

You may want to think out loud to help finalize your answer. However, you must provide your answer within answer tags (e.g., <answer> move_choice </answer>).

The move must be provided in UCI notation and within answer tags in order to be accepted.""",
    "legalmoves": """Below is a board in a game you're currently playing.

{formatted_board}

You must provide a list of all legal moves for the {piece_name} at {piece_pos}.

You may want to think out loud to help finalize your answer. However, you must provide your answer within answer tags (e.g., <answer> list_of_moves </answer>).

The moves must be provided as a list, in UCI notation, and within answer tags in order to be accepted."""
}