#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from pathlib import Path


def run(cmd, env=None, check=False):
    print("\n[MANAGER] Running:")
    print(" ".join(cmd))
    return subprocess.run(cmd, env=env, check=check)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worlds",
        nargs="+",
        default=["office", "testzone", "ideal"],
        help="World cycle, e.g. office testzone ideal complex",
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--timesteps-per-world", type=int, default=128)
    parser.add_argument("--stack-wait-s", type=float, default=75.0)
    parser.add_argument("--save-name", type=str, default="ppo_frontier_overnight_001")
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--model-dir", type=str, default="~/Sam/fyp_ws/logs/ppo_models")
    args = parser.parse_args()

    home = Path.home()
    stack_script = home / "Sam/fyp_ws/scripts/start_rl_training_stack.sh"
    trainer = home / "Sam/fyp_ws/src/fyp_exploration/tools/ppo_frontier_trainer.py"

    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.save_name}.zip"

    base_env = os.environ.copy()
    base_env["HEADLESS"] = "1"
    base_env["DISABLE_ROSBAG"] = "1"
    base_env["DISABLE_RVIZ"] = "1"

    print("[MANAGER] Worlds:", args.worlds)
    print("[MANAGER] Cycles:", args.cycles)
    print("[MANAGER] Timesteps per world:", args.timesteps_per_world)
    print("[MANAGER] Model path:", model_path)
    print("[MANAGER] Rosbags disabled.")
    print("[MANAGER] RViz disabled.")

    try:
        for cycle in range(args.cycles):
            for world in args.worlds:
                print("\n" + "=" * 80)
                print(f"[MANAGER] Cycle {cycle + 1}/{args.cycles}, world={world}")
                print("=" * 80)

                env = base_env.copy()
                env["WORLD_NAME"] = world

                run([str(stack_script)], env=env, check=False)

                print(f"[MANAGER] Waiting {args.stack_wait_s:.1f}s for stack startup...")
                time.sleep(args.stack_wait_s)

                trainer_cmd = [
                    "python3",
                    str(trainer),
                    "--timesteps",
                    str(args.timesteps_per_world),
                    "--save-name",
                    args.save_name,
                    "--checkpoint-every",
                    str(args.checkpoint_every),
                    "--model-dir",
                    str(model_dir),
                ]

                if model_path.exists():
                    trainer_cmd.extend(["--load-model", str(model_path)])

                result = run(trainer_cmd, env=base_env, check=False)
                print(f"[MANAGER] Trainer exit code: {result.returncode}")

                print("[MANAGER] Cleaning stack before next world...")
                cleanup_env = base_env.copy()
                cleanup_env["CLEANUP_ONLY"] = "1"
                run([str(stack_script)], env=cleanup_env, check=False)
                time.sleep(5.0)

    except KeyboardInterrupt:
        print("\n[MANAGER] Interrupted. Cleaning stack...")
        cleanup_env = base_env.copy()
        cleanup_env["CLEANUP_ONLY"] = "1"
        run([str(stack_script)], env=cleanup_env, check=False)

    print("\n[MANAGER] Done.")
    print(f"[MANAGER] Final model should be at: {model_path}")


if __name__ == "__main__":
    main()
