import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

SOURCE_MODEL = MODELS_DIR / "tree_model.joblib"
BEST_MODEL = MODELS_DIR / "best_model.joblib"

def main():
    model = joblib.load(SOURCE_MODEL)
    joblib.dump(model, BEST_MODEL)
    print(f"Best model saved to: {BEST_MODEL}")

if __name__ == "__main__":
    main()