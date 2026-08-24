import re
import subprocess
import sys
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"


def parse_jobs(text: str) -> dict:
    """Returns {job_id: time_str} from squeue output."""
    jobs = {}
    lines = text.splitlines()
    if not lines:
        return jobs
    header = lines[0].split()
    try:
        jobid_col = header.index("JOBID")
        time_col = header.index("TIME")
    except ValueError:
        jobid_col, time_col = 0, 5
    for line in lines[1:]:
        parts = line.split()
        if not parts or not parts[jobid_col].isdigit():
            continue
        job_id = parts[jobid_col]
        time_str = parts[time_col] if len(parts) > time_col else "?"
        jobs[job_id] = time_str
    return jobs


def build_jobid_to_run(logs_dir: Path) -> dict:
    mapping = {}
    for log_file in logs_dir.glob("*/*"):
        if not log_file.is_file():
            continue
        stem = log_file.name.split(".")[0]
        job_id = stem.rsplit("_", 1)[-1]
        if job_id.isdigit():
            mapping[job_id] = log_file.parent.name
    return mapping


def main() -> None:
    if not sys.stdin.isatty():
        squeue_output = sys.stdin.read()
    else:
        user = subprocess.check_output(["whoami"]).decode().strip()
        result = subprocess.run(["squeue", "-u", user], capture_output=True, text=True)
        squeue_output = result.stdout

    active_jobs = parse_jobs(squeue_output)
    if not active_jobs:
        print("No jobs found in squeue output.")
        return

    jobid_to_run = build_jobid_to_run(LOGS_DIR)

    found: dict = {}
    for job_id in sorted(active_jobs):
        run = jobid_to_run.get(job_id)
        if run:
            found.setdefault(run, []).append(job_id)

    unmatched = set(active_jobs) - set(jobid_to_run)

    if found:
        header = f"{'JOBID':<12} {'TIME':<12} CONFIG"
        max_len = max(len(run) for run in found) + len(header)
        print("=" * max_len)
        print(header)
        print("-" * max_len)
        for run, ids in sorted(found.items()):
            for job_id in ids:
                print(f"{job_id:<12} {active_jobs[job_id]:<12} {run}")
    else:
        print("No active jobs matched any log directory.")

    if unmatched:
        print(f"\nQUEUED: {', '.join(sorted(unmatched))}")

    print()
