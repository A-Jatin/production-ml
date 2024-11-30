from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "resources" / "data"

# VGM settings
N_COMPONENTS = 10
RANDOM_STATE = 42
EPS = 1e-6

# Data processing settings
CHUNK_SIZE = 100_000
N_JOBS = -1  # Use all available cores 