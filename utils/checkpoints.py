import re
from pathlib import Path
from typing import Optional

CONFIGS_DIR = Path(__file__).parent.parent / "_configs"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.is_dir():
        return None
    checkpoints = []
    for d in output_dir.iterdir():
        if d.is_dir():
            m = CHECKPOINT_PATTERN.match(d.name)
            if m:
                checkpoints.append((int(m.group(1)), d))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: x[0])[1]


def update_config(config_path: Path, dry_run: bool) -> None:
    text = config_path.read_text()

    output_dir_match = re.search(r"^output_dir:\s*(.+)$", text, re.MULTILINE)
    if not output_dir_match:
        print(f"  [SKIP] no output_dir found in {config_path.name}")
        return

    output_dir = Path(output_dir_match.group(1).strip())
    latest = find_latest_checkpoint(output_dir)

    if latest is None:
        print(f"  [SKIP] no checkpoints found in {output_dir}")
        return

    new_value = f"resume_from_checkpoint: {latest}"

    if re.search(r"^resume_from_checkpoint:", text, re.MULTILINE):
        new_text = re.sub(
            r"^resume_from_checkpoint:.*$", new_value, text, flags=re.MULTILINE
        )
    else:
        new_text = re.sub(
            r"(^output_dir:.+$)",
            rf"\1\n{new_value}",
            text,
            flags=re.MULTILINE,
        )

    if new_text == text:
        print(f"  [OK]   {config_path.name} — already up to date ({latest.name})")
        return

    print(f"  [UPD]  {config_path.name} — set to {latest.name}")
    if not dry_run:
        config_path.write_text(new_text)


def main(dry_run: bool = False) -> None:
    if dry_run:
        print("Dry run — no files will be modified.\n")

    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"No YAML files found in {CONFIGS_DIR}")

    for config_path in configs:
        update_config(config_path, dry_run)
