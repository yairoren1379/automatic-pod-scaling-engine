"""Promotes a trained run to models/, where the API server loads it from.

Usage, from back/python:
    python promote_model.py --run-name experiments/lr0.0005_g0.99_s0

Copies the run's DQN and the LSTM it was trained with, so the server always pairs the agent
with the forecaster it learned from. Online learning on the cluster then continues from this
model (stage 3 of the sim-to-real pipeline). Delete data/online_dqn.pt to restart from it.
"""
import argparse
import json
import os
import shutil
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main():
    parser = argparse.ArgumentParser(description="Copy a trained run to models/ for the API server.")
    parser.add_argument("--run-name", required=True, help="folder under runs/, e.g. experiments/lr0.0005_g0.99_s0")
    args = parser.parse_args()

    run_dir = os.path.join(RUNS_DIR, args.run_name)
    if not os.path.exists(os.path.join(run_dir, "dqn.pt")):
        raise SystemExit(f"{run_dir}/dqn.pt not found")

    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copyfile(os.path.join(run_dir, "dqn.pt"), os.path.join(MODELS_DIR, "dqn.pt"))
    if os.path.exists(os.path.join(run_dir, "lstm.pt")):
        shutil.copyfile(os.path.join(run_dir, "lstm.pt"), os.path.join(MODELS_DIR, "lstm.pt"))
    else:
        print("Note: this run used naive forecasts (no lstm.pt); models/lstm.pt is unchanged.")

    with open(os.path.join(run_dir, "config.json")) as f:
        run_config = json.load(f)
    evaluation = None
    if os.path.exists(os.path.join(run_dir, "evaluation.json")):
        with open(os.path.join(run_dir, "evaluation.json")) as f:
            evaluation = json.load(f)

    info = {"run_name": args.run_name, "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": run_config.get("args"), "dqn": run_config["config"]["dqn"], "evaluation": evaluation}
    with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Promoted {args.run_name} to {MODELS_DIR}")


if __name__ == "__main__":
    main()
