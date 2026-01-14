"""
Multi-REAL Task Suite Viewer

Lists and summarizes tasks in the benchmark.

Usage:
    uv run python run_suite.py --list
"""

import argparse
import glob
import json
import os
from pathlib import Path


def load_task_suite(tasks_dir: str) -> list[dict]:
    """Load all task JSON files from a directory."""
    tasks = []
    for task_file in glob.glob(os.path.join(tasks_dir, "*.json")):
        with open(task_file, 'r') as f:
            task = json.load(f)
            task["_path"] = task_file
            tasks.append(task)
    return tasks


def get_task_websites(task: dict) -> list[dict]:
    """Get websites from task (handles both single and multi-app)."""
    if "websites" in task:
        return task["websites"]
    return [task["website"]]


def print_suite_summary(tasks: list[dict]):
    """Print summary of task suite."""
    print(f"\n{'='*60}")
    print(f"Task Suite: {len(tasks)} tasks")
    print(f"{'='*60}")
    
    # Group by difficulty
    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for task in tasks:
        diff = task.get("difficulty", "unknown")
        if diff in by_difficulty:
            by_difficulty[diff].append(task)
    
    for diff, task_list in by_difficulty.items():
        if task_list:
            print(f"\n{diff.upper()} ({len(task_list)} tasks):")
            for t in task_list:
                websites = get_task_websites(t)
                site_ids = [w["id"] for w in websites]
                multi = "[MULTI]" if len(websites) > 1 else ""
                print(f"  - {t['id']}: {site_ids} {multi}")
    
    # Count multi-app tasks
    multi_app = [t for t in tasks if "websites" in t]
    print(f"\nMulti-app tasks: {len(multi_app)}/{len(tasks)}")
    print(f"Total points: {sum(t.get('points', 1) for t in tasks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default="tasks", help="Directory containing task JSONs")
    parser.add_argument("--list", action="store_true", help="Just list tasks, don't run")
    args = parser.parse_args()
    
    tasks = load_task_suite(args.tasks_dir)
    print_suite_summary(tasks)
    
    if not args.list:
        print("\nTo run these tasks, use the OpenAI runner (see section 5)")
