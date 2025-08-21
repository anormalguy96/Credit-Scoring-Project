from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS / "models"
FIGS_DIR = OUTPUTS / "figs"

RANDOM_STATE = 42

for p in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)