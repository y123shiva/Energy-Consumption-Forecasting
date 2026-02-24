"""
Training entrypoint for Energy Consumption Forecasting

Usage:
    python train.py

This script:
1. Loads dataset
2. Runs full training pipeline
3. Logs experiments to MLflow
4. Saves best model
"""

from pathlib import Path
import sys

# -------------------------------------------------
# Make src importable without PYTHONPATH headaches
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent


from src.pipelines.train_pipelines import run_training


def main():
    print("⚡ Starting Energy Forecast Training Pipeline...\n")

    run_training()

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
