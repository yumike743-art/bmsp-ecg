"""Run the locked ECG experiment and regenerate its deliverables."""

import argparse
from pathlib import Path

import yaml

from .data import check_directory, fetch_database, read_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="Fetch or check PhysioNet data")
    data.add_argument("action", choices=["fetch", "check"])
    smoke = sub.add_parser("smoke", help="Train and evaluate on six real recordings")
    run = sub.add_parser("run", help="Run the complete pre-specified experiment")
    analyze = sub.add_parser("analyze", help="Recompute post-hoc tables from saved predictions")
    stress = sub.add_parser("stress", help="Run the 0 dB drift detector comparison with saved models")
    report = sub.add_parser("report", help="Rebuild figures and two PDF documents")
    report.add_argument("--identity", type=Path, help="Local YAML with name, student_id, date and repository")
    report.add_argument("--submission-dir", type=Path, default=Path("submission"))
    report.add_argument("--draft", action="store_true")
    for command in (data, smoke, run, analyze, stress, report):
        command.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = yaml.safe_load(args.config.read_text())
        dataset = cfg["dataset"]
        root = args.data_dir or Path(dataset["directory"])
        output = args.output_dir or Path(cfg["outputs"])
        rows = read_split(Path(dataset["split"]))
        if args.command == "data":
            action = fetch_database if args.action == "fetch" else check_directory
            checksums = action(root, rows, dataset["version"])
            detail = f"{len(checksums)} official SHA-256 values matched" if checksums else "no checksum file supplied"
            print(f"{root}: {len(rows)} record triplets present, {detail}")
        elif args.command in {"run", "smoke"}:
            from .experiment import run_experiment

            checksums = check_directory(root, rows, dataset["version"])
            run_experiment(cfg, root, rows, checksums, output / args.command, args.command)
        elif args.command == "analyze":
            from .stress import rebuild_posthoc

            rebuild_posthoc(cfg, output / "run")
        elif args.command == "stress":
            from .stress import run_drift_stress

            checksums = check_directory(root, rows, dataset["version"])
            run_drift_stress(cfg, root, rows, checksums, output / "run")
        else:
            from .report import build_reports

            build_reports(cfg, output / "run", args.submission_dir, args.identity, args.draft)
    except (OSError, ValueError) as error:
        parser.exit(2, f"bmsp_ecg: {error}\n")


if __name__ == "__main__":
    main()
