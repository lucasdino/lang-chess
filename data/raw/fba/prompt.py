import os
import importlib.util

# Use absolute imports to import our board conversion utility since calling from a jupyter notebook
spec = importlib.util.spec_from_file_location("board", os.path.abspath("./utils/board.py"))
board = importlib.util.module_from_spec(spec); spec.loader.exec_module(board)

convert_board = board.convert_board