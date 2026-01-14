#!/usr/bin/env python3
"""
Auto-fill expected_value=true for boolean queries.

For benchmark tasks checking success, boolean queries should return true
when the task is completed correctly. This script fills TODO expected values
for queries that are clearly boolean (length >= N, contains(), etc.).

Usage:
    uv run python multi-real/tools/auto_fill_boolean_expected.py --dry-run
    uv run python multi-real/tools/auto_fill_boolean_expected.py
"""

import argparse
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TASKS_DIR = BASE_DIR / "tasks"

# Patterns that indicate boolean-returning queries
BOOLEAN_PATTERNS = [
    r"length\s*\(",           # length() function
    r"contains\s*\(",         # contains() function
    r"!=\s*null",             # != null check
    r"==\s*null",             # == null check
    r"\s+&&\s+",              # logical AND
    r"\s+\|\|\s+",            # logical OR
    r">=\s*`\d+`",            # >= `N` comparison
    r"<=\s*`\d+`",            # <= `N` comparison
    r">\s*`\d+`",             # > `N` comparison
    r"<\s*`\d+`",             # < `N` comparison
    r"==\s*`\d+`",            # == `N` comparison
    r"==\s*`true`",           # == `true`
    r"==\s*`false`",          # == `false`
    r"!=\s*`",                # != `value`
]


def is_boolean_query(query: str) -> bool:
    """Check if query returns a boolean value."""
    for pattern in BOOLEAN_PATTERNS:
        if re.search(pattern, query):
            return True
    return False


def is_todo(expected_value) -> bool:
    """Check if expected_value is a TODO placeholder."""
    if expected_value is None:
        return True
    if isinstance(expected_value, str) and "TODO" in expected_value:
        return True
    return False


def process_tasks(dry_run: bool = False) -> dict:
    """Process all tasks and fill boolean expected values."""
    stats = {
        "tasks_processed": 0,
        "evals_filled": 0,
        "evals_skipped_non_boolean": 0,
        "evals_already_set": 0,
        "non_boolean_todos": [],
    }

    for task_file in sorted(TASKS_DIR.glob("*.json")):
        with open(task_file, encoding="utf-8") as f:
            task = json.load(f)

        stats["tasks_processed"] += 1
        modified = False

        for i, eval_item in enumerate(task.get("evals", [])):
            if eval_item.get("type") != "jmespath":
                continue

            query = eval_item.get("query", "")
            expected = eval_item.get("expected_value")

            if not is_todo(expected):
                stats["evals_already_set"] += 1
                continue

            if is_boolean_query(query):
                stats["evals_filled"] += 1
                if not dry_run:
                    eval_item["expected_value"] = True
                    modified = True
            else:
                stats["evals_skipped_non_boolean"] += 1
                stats["non_boolean_todos"].append({
                    "task": task["id"],
                    "eval_index": i,
                    "description": eval_item.get("description", "")[:60],
                    "query": query[:80],
                })

        if modified and not dry_run:
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Auto-fill boolean expected values")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - no files will be modified\n")

    stats = process_tasks(dry_run=args.dry_run)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tasks processed: {stats['tasks_processed']}")
    print(f"Evals already set: {stats['evals_already_set']}")
    print(f"Evals filled (boolean → true): {stats['evals_filled']}")
    print(f"Evals skipped (non-boolean): {stats['evals_skipped_non_boolean']}")

    if stats["non_boolean_todos"]:
        print(f"\nNon-boolean TODOs requiring manual review ({len(stats['non_boolean_todos'])}):")
        for item in stats["non_boolean_todos"]:
            print(f"  [{item['task']}] {item['description']}")
            print(f"    Query: {item['query']}...")

    if args.dry_run and stats["evals_filled"] > 0:
        print(f"\nRun without --dry-run to fill {stats['evals_filled']} expected values.")


if __name__ == "__main__":
    main()
