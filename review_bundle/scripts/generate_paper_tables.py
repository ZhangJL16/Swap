from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper_tables import generate_paper_tables


if __name__ == "__main__":
    generate_paper_tables()
