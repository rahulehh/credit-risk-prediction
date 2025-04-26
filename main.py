import os
import argparse
import subprocess
from src import run_pipeline


def main():
    """
    Main entry point for the credit risk prediction pipeline.

    Orchestrates the entire pipeline including downloading the dataset,
    preprocessing, model training, evaluation and deployment.
    """
    parser = argparse.ArgumentParser(description="Credit Risk Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-app",
        action="store_true",
        help="Run the Streamlit inference app",
    )
    group.add_argument(
        "--run-build",
        action="store_true",
        help="Build and train the credit risk model",
    )

    os.makedirs("./artifacts", exist_ok=True)

    args = parser.parse_args()
    if args.run_app:
        model_path = "./artifacts/model.pkl"
        if not os.path.exists(model_path):
            print("🚫 Model not found. Please build the model first using:")
            print("   python main.py --run-build")
            return
        subprocess.run(["streamlit", "run", "app.py"])

    elif args.run_build:
        run_pipeline()


if __name__ == "__main__":
    main()
