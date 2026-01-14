#!/usr/bin/env python3
"""
Validate JMESPath queries against ground truth final states.

This tool measures the reliability of our evaluation queries by testing them
against known-good final states (both manual and synthetic).

Usage:
    uv run python validate_ground_truth.py
    uv run python validate_ground_truth.py --verbose
    uv run python validate_ground_truth.py --task gocalendar-gomail-omnizon-1
    uv run python validate_ground_truth.py --no-synthetic  # Only use manual ground truth
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jmespath

from core.registry import registry


MANUAL_GT_DIR = Path(__file__).parent.parent / "final_states" / "manual"
SYNTHETIC_GT_DIR = Path(__file__).parent.parent / "final_states" / "synthetic"


def load_ground_truth(include_synthetic: bool = True) -> dict[str, dict]:
    """Load all ground truth final states from manual and optionally synthetic dirs.

    Manual ground truth takes precedence over synthetic (won't be overwritten).
    """
    states = {}

    # Load manual ground truth first (higher confidence)
    if MANUAL_GT_DIR.exists():
        for state_file in MANUAL_GT_DIR.glob("*.json"):
            task_id = state_file.stem
            with open(state_file) as f:
                states[task_id] = json.load(f)

    # Load synthetic ground truth if requested
    if include_synthetic and SYNTHETIC_GT_DIR.exists():
        for state_file in SYNTHETIC_GT_DIR.glob("*.json"):
            task_id = state_file.stem
            # Don't overwrite manual ground truth (higher confidence)
            if task_id not in states:
                with open(state_file) as f:
                    states[task_id] = json.load(f)

    return states


def validate_task_evals(task_id: str, ground_truth: dict, verbose: bool = False) -> dict:
    """Validate all evals for a task against ground truth."""
    task = registry.get(task_id)
    if not task:
        return {"error": f"Task not found: {task_id}"}

    results = {
        "task_id": task_id,
        "total_evals": len(task.evals),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "unverified": 0,
        "details": [],
    }

    for i, eval_item in enumerate(task.evals):
        if eval_item.get("type") != "jmespath":
            continue

        query = eval_item.get("query", "")
        expected = eval_item.get("expected_value")
        description = eval_item.get("description", f"Eval {i+1}")

        detail = {
            "description": description,
            "query": query,
            "expected": expected,
        }

        try:
            actual = jmespath.search(query, ground_truth)
            detail["actual"] = actual

            # Check if matches
            if expected is None:
                # No expected value - just check query runs and returns something
                detail["status"] = "pass" if actual is not None else "fail"
            elif isinstance(expected, bool):
                # Boolean comparison
                detail["status"] = "pass" if actual == expected else "fail"
            elif isinstance(expected, str) and "TODO" in expected:
                # Unverified eval - report actual value for manual review
                detail["status"] = "unverified"
                detail["note"] = f"Actual value: {actual}"
                results["unverified"] += 1
            else:
                # Value comparison
                detail["status"] = "pass" if actual == expected else "fail"

            if detail["status"] == "pass":
                results["passed"] += 1
            elif detail["status"] == "fail":
                results["failed"] += 1

        except Exception as e:
            detail["status"] = "error"
            detail["error"] = str(e)
            results["errors"] += 1

        results["details"].append(detail)

        if verbose:
            status_icon = {"pass": "OK", "fail": "FAIL", "error": "ERR", "unverified": "TODO"}
            print(f"  [{status_icon.get(detail['status'], '?')}] {description}")
            if detail["status"] == "fail":
                print(f"       Expected: {expected}")
                print(f"       Actual:   {detail.get('actual')}")
            elif detail["status"] == "error":
                print(f"       Error: {detail.get('error')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate JMESPath queries against ground truth")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output for each eval")
    parser.add_argument("--task", help="Validate specific task ID")
    parser.add_argument("--no-synthetic", action="store_true", help="Only use manual ground truth")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    ground_truth = load_ground_truth(include_synthetic=not args.no_synthetic)

    if not ground_truth:
        print("No ground truth states found.")
        print(f"  Manual GT dir: {MANUAL_GT_DIR}")
        print(f"  Synthetic GT dir: {SYNTHETIC_GT_DIR}")
        return 1

    if not args.json:
        print(f"Loaded {len(ground_truth)} ground truth states")
        manual_count = len(list(MANUAL_GT_DIR.glob("*.json"))) if MANUAL_GT_DIR.exists() else 0
        synthetic_count = len(ground_truth) - manual_count
        print(f"  Manual: {manual_count}, Synthetic: {synthetic_count}")
        print()

    if args.task:
        task_ids = [args.task]
    else:
        task_ids = list(ground_truth.keys())

    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_unverified = 0
    total_evals = 0
    all_results = []

    for task_id in sorted(task_ids):
        if task_id not in ground_truth:
            if not args.json:
                print(f"No ground truth for: {task_id}")
            continue

        if not args.json:
            print(f"Validating: {task_id}")

        result = validate_task_evals(task_id, ground_truth[task_id], args.verbose)
        all_results.append(result)

        if "error" in result:
            if not args.json:
                print(f"  Error: {result['error']}")
            continue

        total_passed += result["passed"]
        total_failed += result["failed"]
        total_errors += result["errors"]
        total_unverified += result.get("unverified", 0)
        total_evals += result["total_evals"]

        if not args.json and not args.verbose:
            status = "PASS" if result["failed"] == 0 and result["errors"] == 0 else "ISSUES"
            unverified_note = f" ({result.get('unverified', 0)} TODO)" if result.get("unverified", 0) > 0 else ""
            print(f"  {status}: {result['passed']}/{result['total_evals']} evals passed{unverified_note}")

        if not args.json:
            print()

    # Output
    if args.json:
        output = {
            "ground_truth_count": len(ground_truth),
            "total_evals": total_evals,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "unverified": total_unverified,
            "pass_rate": total_passed / total_evals if total_evals else 0,
            "tasks": all_results,
        }
        print(json.dumps(output, indent=2))
    else:
        # Summary
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Ground truth tasks: {len(ground_truth)}")
        print(f"Total evals: {total_evals}")
        if total_evals:
            print(f"Passed: {total_passed} ({100*total_passed/total_evals:.1f}%)")
        else:
            print("Passed: 0")
        print(f"Failed: {total_failed}")
        print(f"Errors: {total_errors}")
        print(f"Unverified (TODO): {total_unverified}")

        if total_failed > 0 or total_errors > 0:
            print("\nRECOMMENDATION: Fix failing queries before running benchmark")
            return 1
        elif total_unverified > 0:
            print(f"\nNOTE: {total_unverified} evals have TODO placeholders - consider updating expected values")
            return 0
        else:
            print("\nAll queries validated successfully!")
            return 0


if __name__ == "__main__":
    exit(main())
