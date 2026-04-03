import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

SCRIPTS = [
    "src.train_logistic",
    "src.train_logistic_tuned",
    "src.train_tree",
    "src.train_tree_tuned",
    "src.train_forest",
    "src.train_forest_tuned",
]

def reset_mlruns() -> None:
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)
        print("Deleted existing mlruns folder.")
    MLRUNS_DIR.mkdir(exist_ok=True)
    print("Created fresh mlruns folder.")

def run_module(module_name: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"Running: python -m {module_name}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Module failed: {module_name}")

def main() -> None:
    clean = "--clean" in sys.argv

    if clean:
        print("Clean mode enabled.")
        reset_mlruns()

    print("Starting full training pipeline...")

    for module in SCRIPTS:
        run_module(module)

    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()