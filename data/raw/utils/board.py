import re
import chess
from typing import List


def convert_board(fen: str, board_representation: str) -> str:
    """
    Given an FEN board state, convert to a specific board representation.

    Various board representations:
    - "fen": Identity -- just returns FEN
    - "spaced_fen": FEN with spaces between pieces and empty squares
    - "visual": Visual representation of board laid out in 2D with columns and ranks
    - "uniform_visual": Visual representation of board laid out in 2D with columns and ranks but uniform tokenization for Qwen / Llama tokenizers
    """
    if board_representation == "fen":
        preamble = "The following is a board provided in Forsyth-Edwards Notation (FEN) where the rank decreases from 8 to 1 and white pieces are uppercase with black pieces being lowercase.\n"
        return preamble + fen
    elif board_representation == "spaced_fen":
        preamble = "The following is a board provided in Forsyth-Edwards Notation (FEN) where the rank decreases from 8 to 1 and white pieces are uppercase with black pieces being lowercase.\n"
        return preamble + _convert_fen_to_spaced_fen(fen)
    elif board_representation == "visual":
        preamble = "The following is a visual representation of a chess board where white pieces are uppercase with black pieces being lowercase.\n"
        return _convert_fen_to_visual(fen)
    elif board_representation == "uniform_visual":
        preamble = "The following is a visual representation of a chess board where white pieces are uppercase with black pieces being lowercase.\n"
        return _convert_fen_to_uniform_visual(fen)
    else:
        raise ValueError(f"Unknown board representation: {board_representation}")
    
def _convert_fen_to_spaced_fen(fen: str) -> str:
    """
    Sample: 
    r 2 q 2 k 1 / n p p 3 p 1 / p 3 b N 1 p / 4 p 2 Q / 4 P 3 / 3 P 4 / P P P 3 P P / R 4 R K 1  b - - 0 16
    """
    initial_fen = fen.split(" ")[0]
    board_details = fen[fen.find(" "):]
    return " ".join(initial_fen) + " " + board_details

def _convert_fen_to_visual(fen: str) -> str:
    """
    Sample:
    8| r . . q . . k .
    7| n p p . . . p .
    6| p . . . b N . p
    5| . . . . p . . Q
    4| . . . . P . . .
    3| . . . P . . . .
    2| P P P . . . P P
    1| R . . . . R K .
       _ _ _ _ _ _ _ _
       A B C D E F G H

    - It is Black’s turn to move.
    - No castling rights available.
    - No en passant target square.
    - Halfmove clock: 0
    - Fullmove number: 16
    """
    placement, active, castling, en_passant, halfmove, fullmove = fen.split()
    lines = []

    # 1) Board with '|' on left
    for i, rank in enumerate(placement.split('/')):
        row = []
        for c in rank:
            if c.isdigit():
                row.extend(['.'] * int(c))
            else:
                row.append(c)
        lines.append(f"{8 - i}| " + ' '.join(row))

    # 2) Bottom border of underscores and file labels
    lines.append("   " + ' '.join(['_' for _ in range(8)]))
    lines.append("   " + ' '.join(list("ABCDEFGH")))
    lines.append("")  # blank line before details

    # 3) Natural‑language details
    turn = 'White' if active == 'w' else 'Black'
    lines.append(f"- It is {turn}’s turn to move.")

    rights = []
    if 'K' in castling: rights.append('White can castle kingside')
    if 'Q' in castling: rights.append('White can castle queenside')
    if 'k' in castling: rights.append('Black can castle kingside')
    if 'q' in castling: rights.append('Black can castle queenside')
    if rights:
        lines.append(f"- Castling rights: {', '.join(rights)}.")
    else:
        lines.append("- No castling rights available.")

    if en_passant != '-':
        lines.append(f"- En passant target square: {en_passant}.")
    else:
        lines.append("- No en passant target square.")

    lines.append(f"- Halfmove clock: {halfmove}")
    lines.append(f"- Fullmove number: {fullmove}")

    return '\n'.join(lines)


def _convert_fen_to_uniform_visual(fen: str) -> str:
    """
    Sample:
    8| r . . q . . k . 
    7| n p p . . . p . 
    6| p . . . b N . p 
    5| . . . . p . . Q 
    4| . . . . P . . . 
    3| . . . P . . . . 
    2| P P P . . . P P 
    1| R . . . . R K . 
    #  _ _ _ _ _ _ _ _ 
    #  A B C D E F G H 

    - It is Black’s turn to move.
    - No castling rights available.
    - No en passant target square.
    - Halfmove clock: 0
    - Fullmove number: 16
    """
    placement, active, castling, en_passant, halfmove, fullmove = fen.split()
    lines = []

    # 1) Board with '|' on left
    for i, rank in enumerate(placement.split('/')):
        row = []
        for c in rank:
            if c.isdigit():
                row.extend(['.'] * int(c))
            else:
                row.append(c)
        lines.append(f"{8 - i}| " + ' '.join(row) + ' ')   # Add ' ' at end for uniformity

    # 2) Bottom border of underscores and file labels
    lines.append("#  " + ' '.join(['_' for _ in range(8)]) + ' ')
    lines.append("#  " + ' '.join(list("ABCDEFGH")) + ' ')
    lines.append("")  # blank line before details

    # 3) Natural‑language details
    turn = 'White' if active == 'w' else 'Black'
    lines.append(f"- It is {turn}’s turn to move.")

    rights = []
    if 'K' in castling: rights.append('White can castle kingside')
    if 'Q' in castling: rights.append('White can castle queenside')
    if 'k' in castling: rights.append('Black can castle kingside')
    if 'q' in castling: rights.append('Black can castle queenside')
    if rights:
        lines.append(f"- Castling rights: {', '.join(rights)}.")
    else:
        lines.append("- No castling rights available.")

    if en_passant != '-':
        lines.append(f"- En passant target square: {en_passant}.")
    else:
        lines.append("- No en passant target square.")

    lines.append(f"- Halfmove clock: {halfmove}")
    lines.append(f"- Fullmove number: {fullmove}")

    return '\n'.join(lines)


def get_piece_name_at_location(fen, location):
    board = chess.Board(fen)
    square = chess.parse_square(location)
    piece = board.piece_at(square)
    
    if piece is None:
        return None

    color = 'white' if piece.color == chess.WHITE else 'black'
    name = piece.piece_type  # this gives an int (1-6)
    name_str = chess.piece_name(name)  # maps int to string name like 'bishop'

    return f"{color} {name_str}"


# =========================================================
# Inverse / Parsing Functions to extract FEN from visual
# =========================================================
def visual_to_fen(visual: str) -> str:
    """
    Inverse of `_convert_fen_to_visual`.
    Accepts the *visual* chunk that starts with “8| …” and ends
    with “- Fullmove number: X”.
    Returns a legal FEN string.
    """
    lines: List[str] = [ln.rstrip() for ln in visual.splitlines() if ln.strip()]

    # ---- a) Piece placement ----------------------------------------
    board_rows = [ln for ln in lines if re.match(r"^[1-8]\|", ln)][:8]  # 8,7,…,1
    fen_ranks = []
    for row in board_rows:
        tokens = row.split("|", 1)[1].strip().split()
        fen_row, empties = "", 0
        for t in tokens:
            if t == ".":
                empties += 1
            else:
                if empties:
                    fen_row += str(empties)
                    empties = 0
                fen_row += t
        if empties:
            fen_row += str(empties)
        fen_ranks.append(fen_row)
    placement = "/".join(fen_ranks)

    # ---- b) Misc. fields -------------------------------------------
    def find_line(pat: str) -> str:
        return next(ln for ln in lines if pat in ln)

    active = "w" if "It is White" in find_line("It is") else "b"

    castling_ln = next((ln for ln in lines if ln.startswith("- Castling rights:")), "")
    if "No castling rights" in castling_ln or not castling_ln:
        castling = "-"
    else:
        castling = (
            ("K" if "White can castle kingside"  in castling_ln else "") +
            ("Q" if "White can castle queenside" in castling_ln else "") +
            ("k" if "Black can castle kingside"  in castling_ln else "") +
            ("q" if "Black can castle queenside" in castling_ln else "")
        ) or "-"

    ep_ln = next((ln for ln in lines if ln.lower().lstrip().startswith("- en passant")), "")
    parts = ep_ln.split(":", 1)
    if len(parts) == 2 and "target square" in parts[0].lower():
        ep = parts[1].strip().rstrip(".")
        en_passant = ep if re.match(r"^[a-h][36]$", ep) else "-"
    else:
        en_passant = "-"

    halfmove = find_line("Halfmove clock").split(":")[1].strip()
    fullmove = find_line("Fullmove number").split(":")[1].strip()

    return f"{placement} {active} {castling} {en_passant} {halfmove} {fullmove}"


def extract_visual(text: str) -> str:
    """
    Pulls the *visual* board block (board + details) out of an LLM prompt/
    completion string.
    """
    all_lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(all_lines) if re.match(r"^\s*8\|", ln))
        end   = next(i for i, ln in enumerate(all_lines[start:], start) 
                     if ln.lstrip().startswith("- Fullmove number"))
    except StopIteration:
        raise ValueError("Visual board block not found.")

    return "\n".join(all_lines[start:end + 1])