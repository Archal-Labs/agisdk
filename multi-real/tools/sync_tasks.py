#!/usr/bin/env python3
"""
Sync tasks from multi-real/tasks to package directory.

This handles the case where filenames don't match task IDs,
ensuring TaskConfig can find tasks by their ID.

Usage:
    uv run python multi-real/tools/sync_tasks.py
"""

import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # multi-real/


def sync_tasks(
    source_dir: Path | None = None,
    target_dir: Path | None = None,
):
    """
    Sync tasks from source to target directory.

    Creates/updates files in target_dir named {task_id}.json based on the
    "id" field in each task, not the source filename.
    """
    if source_dir is None:
        source_dir = BASE_DIR / "tasks"
    if target_dir is None:
        target_dir = BASE_DIR.parent / "src" / "agisdk" / "REAL" / "browsergym" / "webclones" / "multi" / "tasks"

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return

    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}")
        return

    print(f"Syncing tasks from {source_dir}/ to {target_dir}/")
    print()

    synced = 0
    errors = 0

    for source_file in sorted(source_dir.glob("*.json")):
        try:
            with open(source_file) as f:
                task = json.load(f)

            # Get task ID (use filename if not in JSON)
            task_id = task.get("id", source_file.stem)

            # Target filename is based on task ID, not source filename
            target_file = target_dir / f"{task_id}.json"

            # Write to target
            with open(target_file, "w") as f:
                json.dump(task, f, indent=2)

            if source_file.stem != task_id:
                print(f"✓ {source_file.name:40s} → {target_file.name} (ID mismatch)")
            else:
                print(f"✓ {source_file.name:40s} → {target_file.name}")

            synced += 1

        except Exception as e:
            print(f"✗ {source_file.name}: {e}")
            errors += 1

    print()
    print(f"{'='*80}")
    print(f"Synced: {synced} tasks")
    if errors > 0:
        print(f"Errors: {errors}")
    print(f"{'='*80}")


if __name__ == "__main__":
    sync_tasks()
