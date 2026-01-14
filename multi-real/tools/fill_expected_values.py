#!/usr/bin/env python3
"""
Fill expected_value fields in task JSONs using ground truth.

For publication-quality benchmarks:
- Boolean queries get expected_value = true (ground truth = successful completion)
- Field extraction queries are flagged for review (should use contains())
- Full provenance report generated for reproducibility

Usage:
    uv run python multi-real/tools/fill_expected_values.py
    uv run python multi-real/tools/fill_expected_values.py --dry-run
    uv run python multi-real/tools/fill_expected_values.py --task dashdish-gomail-1
    uv run python multi-real/tools/fill_expected_values.py --verbose
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jmespath

BASE_DIR = Path(__file__).parent.parent  # multi-real/
TASKS_DIR = BASE_DIR / "tasks"
DEFAULT_GT_DIR = BASE_DIR / "final_states" / "manual"


# =============================================================================
# Query Classification
# =============================================================================

def is_boolean_query(query: str) -> bool:
    """
    Determine if a JMESPath query returns a boolean value.

    Boolean patterns:
    - contains(path, 'value')
    - length(path) >= `N` (or >, ==, <, <=)
    - path != null / path == null
    - expr1 && expr2 / expr1 || expr2
    - path == 'literal' / path == `number`

    Returns:
        True if the query is expected to return a boolean
    """
    # Normalize whitespace for pattern matching
    q = query.strip()

    # Pattern 1: contains() function
    if "contains(" in q:
        return True

    # Pattern 2: length() with comparison
    # Match: length(...) >= `N`, length(...) > `N`, etc.
    if re.search(r"length\s*\([^)]+\)\s*(>=|>|==|<|<=)\s*`\d+`", q):
        return True

    # Also match: | length(@) >= `N` at the end
    if re.search(r"\|\s*length\s*\(\s*@\s*\)\s*(>=|>|==|<|<=)\s*`\d+`", q):
        return True

    # Also match: | length(@) (returns number, but often used as truthy check)
    # This is actually NOT boolean - it returns an integer
    # Only flag as boolean if there's a comparison

    # Pattern 3: != null or == null
    if "!= null" in q or "== null" in q:
        return True

    # Pattern 4: Logical operators (high confidence of boolean)
    if " && " in q or " || " in q:
        return True

    # Pattern 5: Equality comparison with literals
    # Match: == 'string' or == `number` or == `true` or == `false`
    if re.search(r"==\s*'[^']*'", q):  # == 'string'
        return True
    if re.search(r"==\s*`[^`]+`", q):  # == `number` or `true`/`false`
        return True

    # Pattern 6: Inequality comparison with literals
    if re.search(r"!=\s*'[^']*'", q):  # != 'string'
        return True
    if re.search(r"!=\s*`[^`]+`", q):  # != `number`
        return True

    # Pattern 7: >= or <= comparisons (often with dates, prices)
    if re.search(r"(>=|<=|>|<)\s*`[^`]+`", q):
        return True

    return False


def get_query_type(query: str) -> str:
    """
    Classify the query type for reporting.

    Returns:
        'boolean' | 'field_extraction' | 'filter' | 'unknown'
    """
    if is_boolean_query(query):
        return "boolean"

    q = query.strip()

    # Field extraction: ends with .field or ['field']
    if re.search(r"\.\w+$", q):
        return "field_extraction"

    # Array index: ends with [0] or [N]
    if re.search(r"\[\d+\]$", q):
        return "field_extraction"

    # Filter: ends with [?condition]
    if re.search(r"\[\?[^\]]+\]$", q):
        return "filter"

    # Pipe to index: | [0]
    if re.search(r"\|\s*\[\d+\]$", q):
        return "field_extraction"

    return "unknown"


# =============================================================================
# Ground Truth Loading
# =============================================================================

def load_json(path: Path) -> dict:
    """Load JSON from file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Save JSON to file with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def find_ground_truth(task_id: str, gt_dir: Path) -> Path | None:
    """
    Find ground truth file for a task.

    Tries exact match first, then partial match on task ID components.
    """
    # Exact match
    exact = gt_dir / f"{task_id}.json"
    if exact.exists():
        return exact

    # Partial match: task ID might have different suffix
    # e.g., gomail-marrsuite-1 vs gomail-marrisuite-1 (typo in GT)
    for gt_file in gt_dir.glob("*.json"):
        # Check if the main components match
        gt_id = gt_file.stem
        task_parts = set(task_id.split("-")[:-1])  # Remove number suffix
        gt_parts = set(gt_id.split("-")[:-1])

        if task_parts == gt_parts:
            return gt_file

    return None


# =============================================================================
# Expected Value Derivation
# =============================================================================

def execute_query(query: str, data: dict) -> tuple[Any, str | None]:
    """
    Execute a JMESPath query against data.

    Returns:
        (result, error_message)
        result is None if query failed
    """
    try:
        result = jmespath.search(query, data)
        return result, None
    except Exception as e:
        return None, str(e)


def derive_expected_value(
    query: str,
    actual_result: Any,
    query_error: str | None,
) -> tuple[Any, str]:
    """
    Derive the expected_value for a query based on ground truth execution.

    Args:
        query: The JMESPath query
        actual_result: Result from executing query against ground truth
        query_error: Error message if query failed

    Returns:
        (expected_value, derivation_method)

    derivation_method is one of:
        - "boolean_true": Boolean query, result was truthy
        - "boolean_false": Boolean query, result was falsy
        - "field_extraction_flagged": Non-boolean query, needs review
        - "query_error": Query failed to execute
        - "null_result": Query returned None/null
    """
    # Query execution error
    if query_error:
        return "TODO: Query error - " + query_error[:50], "query_error"

    # Check if boolean query
    if is_boolean_query(query):
        # For boolean queries, the result should be True/False
        if isinstance(actual_result, bool):
            return actual_result, "boolean_true" if actual_result else "boolean_false"

        # Some boolean-like queries return the matched value (truthy) or None (falsy)
        if actual_result is not None:
            return True, "boolean_true"
        else:
            return False, "boolean_false"

    # Non-boolean query
    query_type = get_query_type(query)

    if actual_result is None:
        return "TODO: Query returned null", "null_result"

    # Field extraction or filter - flag for review
    # We could return the actual value, but it's better to flag for human review
    # since the task might accept multiple valid completions
    return (
        f"TODO: Review - query returns {type(actual_result).__name__}",
        "field_extraction_flagged",
    )


# =============================================================================
# Main Processing
# =============================================================================

def process_task(
    task_path: Path,
    gt_data: dict,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Process a single task, filling expected_value fields.

    Returns:
        Provenance record for this task
    """
    task = load_json(task_path)
    task_id = task.get("id", task_path.stem)

    record = {
        "task_id": task_id,
        "task_file": str(task_path.relative_to(BASE_DIR)),
        "evals": [],
        "updated_count": 0,
        "flagged_count": 0,
        "skipped_count": 0,
    }

    evals = task.get("evals", [])
    modified = False

    for i, eval_item in enumerate(evals):
        if eval_item.get("type") != "jmespath":
            continue

        query = eval_item.get("query", "")
        description = eval_item.get("description", f"Eval {i}")
        old_expected = eval_item.get("expected_value")

        # Check if already has a non-TODO value
        if old_expected is not None and not (
            isinstance(old_expected, str) and "TODO" in old_expected
        ):
            record["skipped_count"] += 1
            if verbose:
                print(f"    [{i}] SKIP: Already has expected_value")
            continue

        # Execute query against ground truth
        actual_result, query_error = execute_query(query, gt_data)

        # Derive expected value
        new_expected, method = derive_expected_value(query, actual_result, query_error)

        # Build eval record
        eval_record = {
            "index": i,
            "description": description,
            "query": query,
            "query_type": get_query_type(query),
            "old_expected_value": old_expected,
            "new_expected_value": new_expected,
            "derivation_method": method,
            "actual_query_result": (
                str(actual_result)[:100] if actual_result is not None else None
            ),
        }
        record["evals"].append(eval_record)

        # Update counts
        if method in ("boolean_true", "boolean_false"):
            record["updated_count"] += 1
        else:
            record["flagged_count"] += 1

        # Print progress
        if verbose:
            status = "UPDATED" if method.startswith("boolean") else "FLAGGED"
            print(f"    [{i}] {status}: {description[:50]}...")
            print(f"        Query: {query[:60]}...")
            print(f"        Type: {get_query_type(query)} → {new_expected}")

        # Update the eval item
        if not dry_run:
            eval_item["expected_value"] = new_expected
            modified = True

    # Save updated task
    if modified and not dry_run:
        save_json(task_path, task)

    return record


def main():
    parser = argparse.ArgumentParser(
        description="Fill expected_value fields in task JSONs using ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run - show what would change
    uv run python multi-real/tools/fill_expected_values.py --dry-run

    # Process single task
    uv run python multi-real/tools/fill_expected_values.py --task dashdish-gomail-1

    # Process all tasks with verbose output
    uv run python multi-real/tools/fill_expected_values.py --verbose

    # Use different ground truth directory
    uv run python multi-real/tools/fill_expected_values.py --ground-truth-dir final_states/synthetic
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Process single task by ID",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help=f"Directory with ground truth finish JSONs (default: {DEFAULT_GT_DIR.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=BASE_DIR / "provenance_report.json",
        help="Path for provenance report (default: provenance_report.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing non-TODO expected_value fields",
    )

    args = parser.parse_args()

    # Validate ground truth directory
    if not args.ground_truth_dir.exists():
        print(f"Error: Ground truth directory not found: {args.ground_truth_dir}")
        sys.exit(1)

    # Find ground truth files
    gt_files = list(args.ground_truth_dir.glob("*.json"))
    if not gt_files:
        print(f"Error: No ground truth files found in {args.ground_truth_dir}")
        sys.exit(1)

    print(f"Found {len(gt_files)} ground truth files in {args.ground_truth_dir}")

    # Find task files to process
    if args.task:
        task_path = TASKS_DIR / f"{args.task}.json"
        if not task_path.exists():
            print(f"Error: Task file not found: {task_path}")
            sys.exit(1)
        task_files = [task_path]
    else:
        task_files = sorted(TASKS_DIR.glob("*.json"))
        # Filter out backup files
        task_files = [f for f in task_files if not f.name.endswith(".backup")]

    print(f"Processing {len(task_files)} task files")
    if args.dry_run:
        print("DRY RUN - no files will be modified")
    print()

    # Process tasks
    provenance_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_dir": str(args.ground_truth_dir.relative_to(BASE_DIR)),
        "dry_run": args.dry_run,
        "tasks_processed": 0,
        "tasks_with_gt": 0,
        "tasks_without_gt": 0,
        "total_evals_updated": 0,
        "total_evals_flagged": 0,
        "total_evals_skipped": 0,
        "details": [],
    }

    tasks_without_gt = []

    for task_path in task_files:
        task_id = task_path.stem

        # Find matching ground truth
        gt_path = find_ground_truth(task_id, args.ground_truth_dir)

        if not gt_path:
            tasks_without_gt.append(task_id)
            provenance_report["tasks_without_gt"] += 1
            continue

        print(f"[{provenance_report['tasks_with_gt'] + 1}] {task_id}")
        print(f"    Ground truth: {gt_path.name}")

        # Load ground truth
        gt_data = load_json(gt_path)

        # Process task
        record = process_task(
            task_path,
            gt_data,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        record["ground_truth_file"] = str(gt_path.relative_to(BASE_DIR))

        # Update totals
        provenance_report["tasks_with_gt"] += 1
        provenance_report["total_evals_updated"] += record["updated_count"]
        provenance_report["total_evals_flagged"] += record["flagged_count"]
        provenance_report["total_evals_skipped"] += record["skipped_count"]
        provenance_report["details"].append(record)

        # Print summary for this task
        print(f"    Updated: {record['updated_count']}, Flagged: {record['flagged_count']}, Skipped: {record['skipped_count']}")
        print()

    provenance_report["tasks_processed"] = len(task_files)

    # Save provenance report
    if not args.dry_run:
        save_json(args.output_report, provenance_report)
        print(f"Provenance report saved to: {args.output_report}")
    else:
        # In dry run, save to a temp location for review
        dry_run_report = args.output_report.parent / "provenance_report_dry_run.json"
        save_json(dry_run_report, provenance_report)
        print(f"Dry run report saved to: {dry_run_report}")

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tasks processed:     {provenance_report['tasks_processed']}")
    print(f"Tasks with GT:       {provenance_report['tasks_with_gt']}")
    print(f"Tasks without GT:    {provenance_report['tasks_without_gt']}")
    print()
    print(f"Evals updated:       {provenance_report['total_evals_updated']}")
    print(f"Evals flagged:       {provenance_report['total_evals_flagged']}")
    print(f"Evals skipped:       {provenance_report['total_evals_skipped']}")

    if tasks_without_gt:
        print()
        print(f"Tasks without ground truth ({len(tasks_without_gt)}):")
        for tid in tasks_without_gt[:10]:
            print(f"  - {tid}")
        if len(tasks_without_gt) > 10:
            print(f"  ... and {len(tasks_without_gt) - 10} more")

    if provenance_report["total_evals_flagged"] > 0:
        print()
        print("NOTE: Some evals were flagged for review. Check the provenance report")
        print("      and consider converting field extraction queries to contains() checks.")


if __name__ == "__main__":
    main()
