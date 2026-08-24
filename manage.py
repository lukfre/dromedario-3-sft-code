#!/usr/bin/env python3
import argparse

import utils.active_runs as active_runs
import utils.checkpoints as checkpoints
import utils.plot_losses as plot_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run management utilities.")
    sub = parser.add_subparsers(dest="command")

    upd = sub.add_parser("update", aliases=["u"], help="Update resume_from_checkpoint in all _configs/ YAML files to the latest saved checkpoint.")
    upd.add_argument("--dry-run", action="store_true", help="Preview changes without writing")

    sub.add_parser("runs", aliases=["run", "r"], help="Show which runs have an active SLURM job.")
    sub.add_parser("plot", aliases=["plots", "p"], help="Plot training loss curves from logs/ into plots/.")

    args = parser.parse_args()

    if args.command in ("update", "u"):
        checkpoints.main(dry_run=args.dry_run)
    elif args.command in ("runs", "run", "r"):
        active_runs.main()
    elif args.command in ("plot", "plots", "p"):
        plot_losses.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
