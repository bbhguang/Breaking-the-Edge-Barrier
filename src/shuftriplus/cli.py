"""Command-line entry point."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from .config import DEFAULT_CONFIG, ROOT, load_config, validate
from .runner import run


def main():
    parser = argparse.ArgumentParser(description="ShufTri+ triangle-counting experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run a configured experiment")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    run_parser.add_argument("--datasets", nargs="+")
    run_parser.add_argument("--epsilons", nargs="+", type=float)
    run_parser.add_argument("--trials", type=int)
    run_parser.add_argument("--threads", type=int)
    run_parser.add_argument("--seed", type=int, help="optional replay seed; default uses current system time")
    sub.add_parser("show-config", help="print the default experiment configuration")
    args = parser.parse_args()
    try:
        if args.command == "show-config":
            print(json.dumps(load_config(), indent=2))
            return 0
        config = load_config(args.config)
        for key in ("epsilons", "trials", "threads"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if args.datasets:
            unknown = set(args.datasets) - set(config["datasets"])
            if unknown:
                raise ValueError("unknown dataset names: " + ", ".join(sorted(unknown)))
            config["datasets"] = {k: v for k, v in config["datasets"].items() if k in args.datasets}
        validate(config)
        output = args.output_dir or ROOT / "outputs" / f"run-{time.time_ns()}"
        run(config, args.data_root, output, args.cache_dir, args.seed)
        return 0
    except subprocess.CalledProcessError as exc:
        parser.exit(2, f"Error: backend exited with status {exc.returncode}; see its diagnostic above\n")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
