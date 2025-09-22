import ast
import time
import chess
import chess.engine
import random
import pandas as pd
import shutil
from typing import Dict
from pathlib import Path
import sys, asyncio, atexit, threading

from .prompt import convert_board
BOARD_REPRESENTATION = "uniform_visual"  # "fen", "spaced_fen", "visual", "uniform_visual"

# ===================================
# Setup our chess engine in advance 
# (only used in a couple of these generators)
# ===================================
# On Windows/Jupyter, use python-chess’ event loop policy so subprocess engines work
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    except Exception:
        pass

_ENGINE: chess.engine.SimpleEngine | None = None
_ENGINE_LOCK = threading.RLock()

def _get_engine_locked() -> chess.engine.SimpleEngine:
    """Must be called with _ENGINE_LOCK held."""
    global _ENGINE
    if _ENGINE is None:
        stockfish_dir = (Path(__file__).resolve().parent / ".." / "stockfish").resolve()
        # Pick the first *.exe (on Windows) or 'stockfish*' binary (on POSIX)
        if sys.platform == "win32":
            candidates = sorted(stockfish_dir.glob("stockfish*.exe"))
        else:
            candidates = sorted(stockfish_dir.glob("stockfish*"))
        if not candidates:
            raise FileNotFoundError(f"No Stockfish binary found in {stockfish_dir}")
        _ENGINE = chess.engine.SimpleEngine.popen_uci(str(candidates[0]))
    return _ENGINE

def _engine_analyse(board: chess.Board, limit: chess.engine.Limit, **kwargs):
    """Thread-safe wrapper around engine.analyse."""
    with _ENGINE_LOCK:
        eng = _get_engine_locked()
        return eng.analyse(board, limit, **kwargs)

def _shutdown_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            try:
                _ENGINE.quit()
            finally:
                _ENGINE = None

atexit.register(_shutdown_engine)


# ==================================================
# Prefill Helpers / Functions
# ==================================================
class ProgrammaticGenerationError(Exception):
    """ Exception raised when you get a failure from a Programmatic Generation. """
    pass


_PIECE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,
}


def _get_piece_values() -> str:
    return (f"The following are values for each piece: "
            f"Pawn={_PIECE_VALUE[chess.PAWN]}; "
            f"Knight={_PIECE_VALUE[chess.KNIGHT]}; "
            f"Bishop={_PIECE_VALUE[chess.BISHOP]}; "
            f"Rook={_PIECE_VALUE[chess.ROOK]}; "
            f"Queen={_PIECE_VALUE[chess.QUEEN]}; "
            f"King={_PIECE_VALUE[chess.KING]}. "
            f"You should respond with just an integer in the format '#,##0' or '-#,##0'.")


def _material_cp(board: chess.Board) -> int:
    """Static material score in centipawns (white‑positive, black‑negative)."""
    score = 0
    for ptype, val in _PIECE_VALUE.items():
        score += val * (len(board.pieces(ptype, chess.WHITE)) -
                        len(board.pieces(ptype, chess.BLACK)))
    return score


def _prob_bucket(p: float) -> str:
    if p < 0.2:   return "0-0.2"
    if p < 0.4:   return "0.2-0.4"
    if p < 0.6:   return "0.4-0.6"
    if p < 0.8:   return "0.6-0.8"
    return "0.8-1"


def _pick_move_prob(row: pd.Series) -> tuple[str, float]:
    """Return a (move, win‑probability) pair, parsing list‑strings if needed."""
    mv = ast.literal_eval(row["Move"]) if isinstance(row["Move"], str) and row["Move"].startswith("[") else row["Move"]
    wp = ast.literal_eval(row["Win Probability"]) if isinstance(row["Win Probability"], str) and row["Win Probability"].startswith("[") else row["Win Probability"]
    if isinstance(mv, list):
        idx = random.randrange(len(mv))
        return mv[idx], float(wp[idx])
    return mv, float(wp)


def _prefill_is_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'is_check_bucket' column:
        n : no side in check
        w : white king is in check
        b : black king is in check
    Returns a *copy* of the input DataFrame with the new column.
    """
    def status(fen: str) -> str:
        board = chess.Board(fen)
        if not board.is_check():
            return "n"
        return "w" if board.turn else "b"   # board.turn == True  -> white to move

    df = df.copy()
    df["is_check_bucket"] = df["FEN"].apply(status)
    return df


def _prefill_material_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      • 'mat_adv'              : centipawn material score (white positive)
      • 'large_mat_adv_bucket' : 'y' if |mat_adv| > 300 else 'n'
      • 'mat_bal_bucket' : 'y' if |mat_adv| < 120 else 'n'
    """
    def cp(fen): return _material_cp(chess.Board(fen))
    df = df.copy()
    df["mat_adv"] = df["FEN"].apply(cp)
    df["large_mat_adv_bucket"] = df["mat_adv"].abs().gt(300).map({True: "y", False: "n"})
    df["mat_bal_bucket"] = df["mat_adv"].abs().lt(120).map({True: "y", False: "n"})
    return df


def _prefill_mat_adv_value(df: pd.DataFrame) -> pd.DataFrame:
    df = _prefill_material_count(df)
    
    def _bucket(cp: int) -> str:
        a = abs(cp)
        if a <= 100:
            return "0-100"
        elif a <= 300:
            return "100-300"
        else:
            return "300+"
    
    df = df.copy()
    df["mat_adv_abs_bucket"] = df["mat_adv"].apply(_bucket)
    return df


def _mobility_bucket(n: int) -> str:
    if n <= 1:  return "0-1"
    if n <= 3:  return "2-3"
    if n <= 5:  return "4-5"
    return "6+"


def _prefill_identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _prefill_multisample(df):
    "Fills in both 'is_check' and the centipawn advantage values"
    def _is_check_status(fen: str) -> str:
        board = chess.Board(fen)
        if not board.is_check():
            return "n"
        return "w" if board.turn else "b"   # board.turn == True  -> white to move   
    def _centipawn_adv(fen): return _material_cp(chess.Board(fen))

    df = df.copy()
    df["is_check_bucket"] = df["FEN"].apply(_is_check_status)
    df["mat_adv"] = df["FEN"].apply(_centipawn_adv)
    return df

# ==================================================
# Generator Helpers / Functions
# ==================================================
LATENTS_SYSPROMPT = "chess_generic.txt"
LATENTS_USERPROMPT = (
    "Here is a board in a game you're currently playing:\n{board}{add_info}\n\n"
    "Answer the following - if multiple questions, include a space between each answer:\n{question}"
)
ADD_INFO_YN = "\n\nYou should respond with a single token -- 'Yes' or 'No' -- to the following."

_piece_letter_map = {            # already in file but shown for clarity
    chess.PAWN:"p", chess.KNIGHT:"n", chess.BISHOP:"b",
    chess.ROOK:"r", chess.QUEEN:"q", chess.KING:"k",
}
_letter_to_piece = {v:k for k,v in _piece_letter_map.items()}
_piece_word = {                  # for wording the question
    "p":"pawn", "n":"knight", "b":"bishop", "r":"rook", "q":"queen", "k":"king"
}


def _generate_is_check(df_row: pd.Series, config_args: Dict, just_qa: bool = False):
    # Cases to generate our data
    if df_row["is_check_bucket"] == "n":
        question = (
            "Is the black king in check {'Yes', 'No'}?"
            if random.random() < 0.5
            else "Is the white king in check {'Yes', 'No'}?"
        )
        answer = "No"
        cat = "tn"
    else:
        if random.random() < config_args["tp"]:
            answer = "Yes"
            cat = "tp"
        else:
            answer = "No"
            cat = "fp"

        play_black = (
            (answer == "Yes" and df_row["is_check_bucket"] == "b")
            or (answer == "No" and df_row["is_check_bucket"] == "w")
        )
        question = (
            "Is the black king in check {'Yes', 'No'}?" if play_black else "Is the white king in check {'Yes', 'No'}?"
        )

    if just_qa:
        return question, answer

    # Formatting outputs
    sys_prompt = LATENTS_SYSPROMPT
    user_prompt = LATENTS_USERPROMPT.format(
        board=convert_board(df_row["FEN"], BOARD_REPRESENTATION), 
        question=question, 
        add_info=""
    )
    df_info = {"is_check_gen_bucket": cat}
    chat = {"chat": [["system", sys_prompt], ["user", user_prompt], ["assistant", answer]]}
    return chat, df_info


def _generate_large_mat_adv(df_row: pd.Series, config_args: Dict):
    if df_row["large_mat_adv_bucket"] == "n":
        ask_white = random.random() < 0.5
        question  = f"Does {'white' if ask_white else 'black'} have a material advantage?"
        answer, cat = "No", "tn"
    else:
        advantaged = "white" if df_row["mat_adv"] > 0 else "black"
        disadvantaged = "black" if advantaged == "white" else "white"

        if random.random() < config_args["tp"]:
            question = f"Does {advantaged} have a material advantage?"
            answer, cat = "Yes", "tp"
        else:
            question = f"Does {disadvantaged} have a material advantage?"
            answer, cat = "No", "fp"

    sys_prompt = LATENTS_SYSPROMPT
    user_prompt = LATENTS_USERPROMPT.format(
        board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
        question=question,
        add_info=ADD_INFO_YN
    )
    df_info = {"large_mat_adv_gen_bucket": cat}
    chat = {"chat": [["system", sys_prompt], ["user", user_prompt], ["assistant", answer]]}
    return chat, df_info


def _generate_mat_bal(df_row: pd.Series, config_args: Dict):
    """
    Q: “Is the game materially balanced?”
       (balance = |mat_adv| < 120)
    """
    question = "Is the game materially balanced?"
    answer = "Yes" if df_row["mat_bal_bucket"] == "y" else "No"

    sys_prompt  = LATENTS_SYSPROMPT
    user_prompt = LATENTS_USERPROMPT.format(
        board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
        question=question,
        add_info=ADD_INFO_YN
    )
    chat    = {"chat": [["system", sys_prompt],
                        ["user",   user_prompt],
                        ["assistant", answer]]}
    return chat, None


def _pick_legal(board: chess.Board, color: bool, weights: Dict[str, float]):
    letters = [l for l in weights if board.pieces(_letter_to_piece[l], color)]
    random.shuffle(letters)    
    seen_letters = set()
    for l in letters:
        if l in seen_letters:
            continue
        seen_letters.add(l)
        ptype = _letter_to_piece[l]
        sqs   = list(board.pieces(ptype, color))
        random.shuffle(sqs)
        for sq in sqs:
            moves = [m for m in board.legal_moves if m.from_square == sq]
            if moves:
                return l, random.choice(moves).uci()
    raise RuntimeError("Side has no legal moves.")


def _plausible_move(board: chess.Board,
                    square: chess.Square,
                    ptype: chess.PieceType,
                    color: bool,
                    legal_check: bool = False) -> str:

    def empty_board_cands():
        b = chess.Board(None)
        b.clear()
        b.set_piece_at(square, chess.Piece(ptype, color))
        b.turn = color
        lst = list(b.generate_pseudo_legal_moves())
        random.shuffle(lst)
        return lst

    # 1) Non‑check: shape from empty board, skip any truly legal real‑board moves
    if not legal_check:
        for mv in empty_board_cands():
            # only skip your own pieces -- allow empty landings and opponent captures
            if board.color_at(mv.to_square) == color:
                continue
            if board.is_legal(mv):  # if it actually is legal → skip
                continue
            return mv.uci()

    # 2) In‑check: real‑board pseudo‑legals, skip real‑legals, pick ones that still leave you in check
    else:
        tmp = board.copy(stack=False)
        tmp.turn = color
        legal_now = set(tmp.legal_moves)
        cands = list(tmp.generate_pseudo_legal_moves())
        random.shuffle(cands)
        for mv in cands:
            if mv in legal_now:
                continue
            b2 = tmp.copy(stack=False)
            b2.push(mv)
            if b2.is_check():
                return mv.uci()

    # 3) Fallback → retry the empty‑board logic (so you never get a self‑capture from real board)
    for mv in empty_board_cands():
        if board.color_at(mv.to_square) == color:
            continue
        if not board.is_legal(mv):
            return mv.uci()

    raise RuntimeError("No plausible illegal move found.")


def _generate_is_legal(df_row: pd.Series, cfg: Dict, just_qa: bool = False):
    """
    cfg:
        piece_freq   = {"p": w, "n": …}
        choose_legal = {"legal_you": w1, "legal_opp": w2, "illegal": w3}
        # NEW: nothing else required – ‘in‑check’ handled internally.
    """
    board, you, opp = chess.Board(df_row["FEN"]), None, None
    you, opp        = board.turn, not board.turn
    weights         = cfg["piece_freq"]

    # ----- check status & weight override -----------------------------
    in_check = board.is_check()
    choose_w = ({"legal_you": 0.1, "legal_opp": 0.1, "illegal": 0.8}
                if in_check else
                cfg["choose_legal"])

    scenario = random.choices(
        ["legal_you", "legal_opp", "illegal"],
        weights=[choose_w["legal_you"],
                 choose_w["legal_opp"],
                 choose_w["illegal"]],
        k=1
    )[0]

    piece = move = answer = bucket = None

    # ---------------- attempt each scenario in order ------------------- #
    if scenario == "legal_you":
        try:
            piece, move = _pick_legal(board, you, weights)
            answer, bucket = "Yes", "tp"
        except RuntimeError:
            scenario = "legal_opp"     # Upon failure move to next

    if scenario == "legal_opp":
        board_opp = board.copy(); board_opp.turn = opp
        try:
            piece, move = _pick_legal(board_opp, opp, weights)
            answer, bucket = "No", "fp"
        except RuntimeError:
            scenario = "illegal"

    if scenario == "illegal":
        # try every piece type and every individual piece until an illegal move materialises
        letters = [l for l in weights if board.pieces(_letter_to_piece[l], you)]
        random.shuffle(letters)                       # avoid bias
        for piece in letters:
            ptype   = _letter_to_piece[piece]
            squares = list(board.pieces(ptype, you))
            random.shuffle(squares)
            for square in squares:
                try:
                    move = _plausible_move(board, square, ptype, you, legal_check=board.is_check())
                    answer, bucket = "No", "tn"
                    break           # success for this scenario
                except RuntimeError:
                    continue        # this square yielded no illegal move
            else:
                continue            # try next piece type
            break                   # outer loop breaks when inner found a move
        else:
            # every attempt failed → fall back to a legal move by you
            piece, move = _pick_legal(board, you, weights)
            answer, bucket = "Yes", "tp"

    # ---------------- build output ------------------------------------- #
    question = f"Can you legally play {move} {{'Yes', 'No'}}?"
    
    if just_qa:
        return question, answer
    
    chat = {
        "chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user", LATENTS_USERPROMPT.format(
                board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
                question=question,
                add_info="")],
            ["assistant", answer],
        ]
    }
    info = {
        "is_legal_gen_bucket":   bucket,
        "is_legal_piece_bucket": piece,
        "is_legal_in_check_bucket":     ("y" if in_check else "n"),
    }
    return chat, info


def _generate_under_attack(df_row: pd.Series, cfg: Dict, just_qa: bool = False):
    """
    Prompt: “Can your <piece> take their <piece>?”

    Buckets
      tp : your capture exists               (answer 'Yes')
      fp : only opponent capture exists      (answer 'No')
      tn : neither capture exists            (answer 'No')
    """
    board = chess.Board(df_row["FEN"])
    you, opp = board.turn, not board.turn
    nonking  = {chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN}

    opp_has_nonking = any(board.pieces(pt, opp) for pt in nonking)

    # --- helper: capture pair set --------------------------------------
    def capture_pairs(side, allow_king=False):
        b = board if side == board.turn else board.copy(stack=False); b.turn = side
        res = set()
        for m in b.legal_moves:
            if b.color_at(m.to_square) != (not side):
                continue
            if not allow_king and b.piece_type_at(m.to_square) == chess.KING:
                continue
            atk = _piece_letter_map[b.piece_type_at(m.from_square)]
            vic = _piece_letter_map[b.piece_type_at(m.to_square)]
            res.add((atk, vic))
        return res

    tp_set = capture_pairs(you, allow_king=not opp_has_nonking)
    opp_set = capture_pairs(opp, allow_king=True)
    fp_set = opp_set - tp_set                      # mirror capture only for opp

    # candidate piece types present on each side (exclude king victim if others)
    you_types = [l for l in cfg["piece_freq"]
                 if board.pieces(_letter_to_piece[l], you)]
    opp_types = [l for l in cfg["piece_freq"]
                 if board.pieces(_letter_to_piece[l], opp)
                    and (l != "k" or not opp_has_nonking)]

    tn_set = {(p, t) for p in you_types for t in opp_types
              if (p, t) not in tp_set and (p, t) not in fp_set}

    pools = {
        "attack_you": list(tp_set),
        "attack_opp": list(fp_set),
        "safe":       list(tn_set),
    }

    # ----- choose scenario among non‑empty pools -----------------------
    feasible = [k for k, v in pools.items() if v]
    if not feasible:                       # extremely static position
        piece, target, answer, bucket = "q", "q", "No", "tn"
    else:
        weights = [cfg["legal_attack"][k] for k in feasible]
        scenario = random.choices(feasible, weights=weights, k=1)[0]
        piece, target = random.choice(pools[scenario])
        if scenario == "attack_you":
            answer, bucket = "Yes", "tp"
        elif scenario == "attack_opp":
            answer, bucket = "No",  "fp"
        else:
            answer, bucket = "No",  "tn"

    question = f"Can your {_piece_word[piece]} take their {_piece_word[target]} {{'Yes', 'No'}}?"
    
    if just_qa:
        return question, answer
    
    chat = {
        "chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user",   LATENTS_USERPROMPT.format(
                    board=convert_board(df_row['FEN'], BOARD_REPRESENTATION), 
                    question=question, 
                    add_info=""
                )],
            ["assistant", answer],
        ]
    }
    info = {"under_attack_gen_bucket": bucket, "under_attack_piece_bucket": piece + target}
    return chat, info


def _generate_mat_adv_value(df_row: pd.Series, _cfg: Dict, just_qa: bool = False):
    ask_white = random.random() < 0.5
    color     = "white" if ask_white else "black"
    answer    = str(df_row["mat_adv"] if ask_white else -df_row["mat_adv"])
    question=f"What is the material advantage for {color} ({_get_piece_values()})?"

    if just_qa:
        return question, answer

    chat = {"chat": [
        ["system", LATENTS_SYSPROMPT],
        ["user", LATENTS_USERPROMPT.format(
            board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
            question=question,
            add_info=""
        )],
        ["assistant", answer],
    ]}
    info = {
        "mat_adv_abs_bucket": df_row["mat_adv_abs_bucket"],
        "mat_adv_question_color": color,
    }
    return chat, info


def _generate_win_prob(df_row: pd.Series, _cfg: Dict, just_qa: bool = False):
    move, prob     = _pick_move_prob(df_row)
    decs           = 1
    question       = f"If you play {move}, what is your expected probability of winning (answer with a float to {decs} decimal place(s))?"
    answer         = f"{prob:.{decs}f}"
    
    if just_qa:
        return question, answer
    
    chat = {
        "chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user", LATENTS_USERPROMPT.format(
                board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
                question=question,
                add_info=""
            )],
            ["assistant", answer],
        ]
    }
    info = {"win_prob_bucket": _prob_bucket(prob)}
    return chat, info


def _generate_mobility(df_row: pd.Series, cfg: Dict, just_qa: bool = False):
    board = chess.Board(df_row["FEN"])
    you   = board.turn

    # pick a piece TYPE you actually have, using cfg["piece_freq"] weights
    letters = [l for l in cfg["piece_freq"] if board.pieces(_letter_to_piece[l], you)]
    piece_l = random.choices(letters, weights=[cfg["piece_freq"][l] for l in letters], k=1)[0]
    ptype   = _letter_to_piece[piece_l]

    # pick a concrete piece (square) of that type
    square  = random.choice(list(board.pieces(ptype, you)))
    sq_name = chess.square_name(square)

    n_moves = sum(1 for mv in board.legal_moves if mv.from_square == square)

    question = f"How many legal moves does your {_piece_word[piece_l]} at {sq_name} have (answer with an integer)?"
    answer   = str(n_moves)

    if just_qa:
        return question, answer

    chat = {
        "chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user", LATENTS_USERPROMPT.format(
                board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
                question=question,
                add_info=""
            )],
            ["assistant", answer],
        ]
    }
    info = {
        "mobility_piece_bucket": piece_l,
        "mobility_moves_bucket": _mobility_bucket(n_moves),
    }
    return chat, info


def _generate_contrastive_ntp(df_row: pd.Series, cfg: Dict):
    """
    Ask which of two moves (same piece/square) is better.
    The assistant must answer with the *destination* square of the better move.
    Buckets: '1' best‑move listed first, '2' best listed second, 'None' = fail.
    """
    # ── parse list columns ───────────────────────────────────────────────
    mv = ast.literal_eval(df_row["Move"]) if isinstance(df_row["Move"], str) and df_row["Move"].startswith("[") else df_row["Move"]
    wp = ast.literal_eval(df_row["Win Probability"]) if isinstance(df_row["Win Probability"], str) and df_row["Win Probability"].startswith("[") else df_row["Win Probability"]
    if not (isinstance(mv, list) and isinstance(wp, list)):
        raise ValueError("Contrastive NTP requires list‑valued 'Move' and 'Win Probability'.")

    pairs  = list(zip(mv, map(float, wp)))                # [(uci, prob), …]
    board  = chess.Board(df_row["FEN"]); you = board.turn
    thr    = cfg.get("min_threshold", 0.25)
    pieces = [l for l in cfg.get("piece_freq", _piece_letter_map.keys()) if board.pieces(_letter_to_piece[l], you)]
    random.shuffle(pieces)

    for pl in pieces:
        for sq in random.sample(list(board.pieces(_letter_to_piece[pl], you)), k=len(board.pieces(_letter_to_piece[pl], you))):
            here = [(m, p) for m, p in pairs if chess.Move.from_uci(m).from_square == sq]
            if len(here) < 2: 
                continue
            best_m, best_p = max(here, key=lambda t: t[1])
            worst = [t for t in here if best_p - t[1] >= thr and t[0] != best_m]
            if not worst: 
                continue
            worst_m, _ = random.choice(worst)
            opts = [best_m, worst_m] if random.random() < 0.5 else [worst_m, best_m]
            answer = chess.Move.from_uci(best_m).uci()[2:4]      # destination only
            bucket = "1" if opts[0] == best_m else "2"

            q = (f"For the {'white' if you else 'black'} {_piece_word[pl]} on {chess.square_name(sq)}, "
                 f"which of the following moves are better between {opts[0]} and {opts[1]}?")

            chat = {"chat": [
                ["system", LATENTS_SYSPROMPT],
                ["user", LATENTS_USERPROMPT.format(
                    board=convert_board(df_row['FEN'], BOARD_REPRESENTATION),
                    question=q,
                    add_info="Respond immediately with just the destination square (e.g. 'e4') -- nothing else."
                )],
                ["assistant", answer],
            ]}
            info = {
                "contrastive_ntp_bucket": bucket,
                "contrastive_ntp_piece_bucket": pl,
                "contrastive_ntp_delta": round(best_p - min(t[1] for t in worst), 3),
            }
            return chat, info

    raise ProgrammaticGenerationError()   # Generation Failed


def _generate_cloze_capture(df_row: pd.Series, cfg: Dict, just_qa: bool = False):
    """
    Pick a *unique* capture (only one friendly piece can capture that target).
    The blank is the origin square.  Answer is that square (e.g. 'e4').

    cfg:
      • piece_freq : lottery weights like other tasks.
    Info columns (always present):
      • cloze_piece_bucket  : 'p','n','b','r','q','k' or 'None'
    """
    board = chess.Board(df_row["FEN"]); you, opp = board.turn, not board.turn
    weights = cfg.get("piece_freq", _piece_letter_map.keys())

    # --- collect legal capture moves, grouped by target square ----------
    targets = {}   # to_sq -> [(from_sq, p_letter, victim_letter)]
    for m in board.legal_moves:
        if not board.is_capture(m): 
            continue
        if board.is_en_passant(m):
            continue
        frm, to = m.from_square, m.to_square
        p_l = _piece_letter_map[board.piece_type_at(frm)]
        v_l = _piece_letter_map[board.piece_type_at(to)]
        targets.setdefault(to, []).append((frm, p_l, v_l))

    # --- keep only targets with exactly one attacker --------------------
    unique = [(frm, p_l, v_l, to)            # include to_sq for wording
              for to, lst in targets.items() if len(lst) == 1
              for frm, p_l, v_l in lst]

    if not unique:
        raise ProgrammaticGenerationError()
    
    # lottery over piece types present
    present = [u for u in unique if u[1] in weights]
    if not present:
        raise ProgrammaticGenerationError()
    else:
        types = [u[1] for u in present]
        chosen_type = random.choices(types,
                                        weights=[weights[t] for t in types],
                                        k=1)[0]
        cand = random.choice([u for u in present if u[1] == chosen_type])
        frm, p_l, v_l, to = cand
        color  = "white" if you else "black"
        question = (f"My piece on __ could take the opponent's "
                    f"{_piece_word[v_l]} on {chess.square_name(to)} (answer with square of only piece that makes this statement true -- e.g., 'e4').")
        answer   = chess.square_name(frm)

        if just_qa:
            return question, answer

        chat = {"chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user", LATENTS_USERPROMPT.format(
                board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
                question=question,
                add_info=""
            )],
            ["assistant", answer],
        ]}
        info = {"cloze_piece_bucket": p_l}
        return chat, info


def _generate_predict_bestmove(df_row: pd.Series, _cfg: Dict):
    """
    Dataset columns:  FEN • Move   (Move = best move in UCI)
    Prompt: ask for the best move, expecting UCI only.
    """
    move = df_row["Move"]
    chat = {
        "chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user", LATENTS_USERPROMPT.format(
                board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
                question="What is the best move?",
                add_info="Respond immediately with just the move in UCI notation (e.g. 'e2e4') — nothing else."
            )],
            ["assistant", move],
        ]
    }
    return chat, None


def _generate_multisample(df_row: pd.Series, cfg: Dict):
    """
    cfg:
      generation_samples: (low, high)
      tasks: {task: {"frequency": int, "max_samples": int, "args": dict}}
    """
    low, high = cfg["generation_samples"]
    target = random.randint(low, high)

    specs = {t: s for t, s in cfg["tasks"].items() if t in TASK_MAP and s.get("frequency", 0) > 0}
    if not specs:
        raise ValueError("No eligible tasks.")

    tickets = [t for t, s in specs.items() for _ in range(int(s["frequency"]))]
    if not tickets:
        raise ValueError("No tickets.")
    random.shuffle(tickets)

    quota = {t: int(specs[t].get("max_samples", 1)) for t in specs}

    qs, ans = [], []
    seen: set[tuple[str, str]] = set()
    failed: set[str] = set()

    while tickets and len(qs) < target:
        t = tickets.pop()
        if t in failed:
            continue
        if quota.get(t, 0) <= 0:
            if t in tickets:
                tickets = [x for x in tickets if x != t]  # hard-stop further draws for this task
            continue
        try:
            q, a = TASK_MAP[t](df_row, specs[t].get("args", {}), just_qa=True)
        except Exception:
            failed.add(t); continue

        if (q, a) == (None, None):
            continue
        if not (isinstance(q, str) and isinstance(a, str) and q and a):
            failed.add(t); continue
        if (q, a) in seen:
            continue

        seen.add((q, a)); qs.append(q); ans.append(a)
        quota[t] -= 1
        if quota[t] <= 0 and t in tickets:
            tickets = [x for x in tickets if x != t]  # purge remaining tickets for this task

    if not qs:
        raise ProgrammaticGenerationError("No samples generated.")

    chat = {
        "chat": [
            ["system", LATENTS_SYSPROMPT],
            ["user", LATENTS_USERPROMPT.format(
                board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
                question="\n".join(qs),
                add_info=""
            )],
            ["assistant", " ".join(ans)],
        ]
    }
    info = {"multi_sample_n": len(qs), "multi_failed_tasks": sorted(failed)}
    return chat, info


def _get_optimal_line(fen: str, line_plies: int = 5, search_depth: int = 14,
                      show_cp_delta: bool = True) -> str:
    board = chess.Board(fen)
    info0 = _engine_analyse(board, chess.engine.Limit(depth=search_depth))
    pv = (info0.get("pv") or [])[:line_plies]
    if not pv:
        raise ProgrammaticGenerationError()

    tmp = board.copy(stack=False)
    for m in pv:
        tmp.push(m)

    # must give full plies unless mate ends it
    if len(pv) < line_plies and not tmp.is_checkmate():
        raise ProgrammaticGenerationError()

    uci_str = " ".join(m.uci() for m in pv)
    if tmp.is_checkmate():
        return uci_str + " mate"

    if not show_cp_delta:
        return uci_str

    try:
        s0 = info0["score"].pov(board.turn).cp
        s1 = _engine_analyse(tmp, chess.engine.Limit(depth=search_depth))["score"].pov(board.turn).cp
        d = s1 - s0
        return f"{uci_str} [Δ{'+' if d >= 0 else ''}{d}]"
    except:
        raise ProgrammaticGenerationError()


def _generate_predict_line(df_row: pd.Series, cfg: Dict):
    low, high = cfg["plies"]
    target = random.randint(low, high)
    search_depth = cfg['search_depth'] 
    try:
        answer = _get_optimal_line(df_row['FEN'], line_plies=target, search_depth=search_depth)
        question = f"What are the likely next {target} plies that would play out? List all moves in UCI notation (e.g., e4f6) separated by spaces. If a checkmate occurs, end with 'mate'. Otherwise, end with the change in centipawns in the format '[Δ+/-#]'."
        chat = {
            "chat": [
                ["system", LATENTS_SYSPROMPT],
                ["user", LATENTS_USERPROMPT.format(
                    board=convert_board(df_row["FEN"], BOARD_REPRESENTATION),
                    question=question,
                    add_info=""
                )],
                ["assistant", answer],
            ]
        }
        return chat, None
    except:
        raise ProgrammaticGenerationError()


# ==================================================
# Router
# ==================================================
TASK_MAP = {
    "is_check":          _generate_is_check,
    "large_mat_adv":     _generate_large_mat_adv,
    "mat_bal":           _generate_mat_bal,
    "is_legal":          _generate_is_legal,
    "under_attack":      _generate_under_attack,
    "mat_adv_value":     _generate_mat_adv_value,
    "win_prob":          _generate_win_prob,
    "mobility":          _generate_mobility,
    "contrastive_ntp":   _generate_contrastive_ntp,
    "cloze_capture":     _generate_cloze_capture,
    "bestmove":          _generate_predict_bestmove,
    "multi_sample":      _generate_multisample,
    "bestline":          _generate_predict_line,
}

PREFILL_TASK_MAP = {
    "is_check":          _prefill_is_check,
    "large_mat_adv":     _prefill_material_count,
    "mat_bal":           _prefill_material_count,
    "is_legal":          _prefill_is_check,
    "under_attack":      _prefill_identity,
    "mat_adv_value":     _prefill_mat_adv_value,
    "win_prob":          _prefill_identity,
    "mobility":          _prefill_identity,
    "contrastive_ntp":   _prefill_identity,
    "cloze_capture":     _prefill_identity,
    "bestmove":          _prefill_identity,
    "multi_sample":      _prefill_multisample,
    "bestline":          _prefill_identity,
}


# ==================================================
# Main Prompt Generator
# ==================================================
def fba_generator(task: str, df: pd.DataFrame, config_args: Dict = None) -> pd.DataFrame:
    if task in PREFILL_TASK_MAP:
        df = PREFILL_TASK_MAP[task](df)

    chat_col = f"{task}_chat"
    chat_data_list, keep_idx = [], []
    info_buffers: Dict[str, list] = {}
    num_generation_errors = 0

    total_rows = len(df)
    print_interval = max(1, total_rows // 10)
    start_time = time.time()

    for i, (idx, row) in enumerate(df.iterrows(), 1):
        try:
            chat_data, df_info = TASK_MAP[task](row, config_args)
            chat_data_list.append(chat_data)
            keep_idx.append(idx)

            if df_info:
                for k, v in df_info.items():
                    info_buffers.setdefault(k, []).append(v)
        except ProgrammaticGenerationError:
            num_generation_errors += 1
        except Exception as e:
            raise e

        if i % print_interval == 0 or i == total_rows:
            elapsed = time.time() - start_time
            print(f"[{i}/{total_rows}] {i/elapsed:.2f} samples/s")

    # keep only successful rows
    df = df.loc[keep_idx].reset_index(drop=True)
    df[chat_col] = chat_data_list
    for k, vals in info_buffers.items():
        df[k] = vals

    print(f"Total Number of generation errors: {num_generation_errors}")
    _shutdown_engine()
    return df