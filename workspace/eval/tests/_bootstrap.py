from pathlib import Path
import sys


EVAL_DIR = str(Path(__file__).resolve().parents[1])
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
