#!/usr/bin/env python3
"""
Analyze JMESPath queries from tasks and cross-reference with known schemas.

Identifies:
- Queries using paths that don't exist in schema
- Queries with syntax errors
- App prefixes that don't match known apps
- Common pattern mismatches (e.g., .differences vs .bookingDetailsDiff)

INCREMENTAL DESIGN:
- Re-run as you add more ground truth / fix schemas
- Uses schemas.json from discover_schemas.py
- Outputs actionable fixes for fix_query_patterns.py

Usage:
    uv run python multi-real/tools/analyze_queries.py
    uv run python multi-real/tools/analyze_queries.py --verbose
    uv run python multi-real/tools/analyze_queries.py --task gomail-marrisuite-1
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jmespath

BASE_DIR = Path(__file__).parent.parent  # multi-real/
TASKS_DIR = BASE_DIR / "tasks"
SCHEMAS_FILE = BASE_DIR / "docs" / "schemas.json"
OUTPUT_FILE = BASE_DIR / "docs" / "query_analysis.json"


def load_schemas() -> dict:
    """Load schemas from discover_schemas.py output."""
    if not SCHEMAS_FILE.exists():
        print(f"Warning: {SCHEMAS_FILE} not found. Run discover_schemas.py first.")
        return {}
    with open(SCHEMAS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("schemas", {})


def load_tasks() -> list[dict]:
    """Load all tasks with their queries."""
    tasks = []
    for task_file in sorted(TASKS_DIR.glob("*.json")):
        with open(task_file, encoding="utf-8") as f:
            task = json.load(f)
        tasks.append({
            "id": task["id"],
            "file": task_file.name,
            "websites": [w["id"] for w in task.get("websites", [])],
            "evals": task.get("evals", []),
        })
    return tasks


def extract_app_prefix(query: str) -> str | None:
    """Extract the app prefix from a query (e.g., 'gomail' from 'gomail.differences...')."""
    match = re.match(r"^(\w+)\.", query)
    if match:
        return match.group(1)
    # Also check inside functions like contains(gomail....)
    match = re.search(r"contains\((\w+)\.", query)
    if match:
        return match.group(1)
    match = re.search(r"length\((\w+)\.", query)
    if match:
        return match.group(1)
    return None


def extract_all_paths(query: str) -> list[str]:
    """Extract all dot-paths from a query."""
    paths = []
    # Match patterns like app.path.to.field
    for match in re.finditer(r"\b(\w+(?:\.\w+)+)", query):
        path = match.group(1)
        # Skip things that look like function calls
        if not path.startswith(("length", "contains", "values", "keys")):
            paths.append(path)
    return paths


def check_syntax(query: str) -> tuple[bool, str | None]:
    """Check if query has valid JMESPath syntax."""
    try:
        jmespath.compile(query)
        return True, None
    except Exception as e:
        return False, str(e)


def check_path_against_schema(path: str, schemas: dict) -> dict:
    """Check if a path exists in known schemas."""
    parts = path.split(".")
    if not parts:
        return {"valid": False, "reason": "empty path"}

    app_id = parts[0]
    if app_id not in schemas:
        return {"valid": False, "reason": f"unknown app: {app_id}", "app_id": app_id}

    app_schema = schemas[app_id]
    known_paths = [p["path"] for p in app_schema.get("diff_paths", [])]

    # Check if the path (or a prefix of it) matches known paths
    for known in known_paths:
        if path.startswith(known) or known.startswith(path):
            return {"valid": True, "matched_path": known}

    # Check for common mismatches
    if ".differences." in path:
        # Check if this app uses differences
        if not app_schema.get("has_differences"):
            alt_diffs = app_schema.get("app_specific_diffs", [])
            if alt_diffs:
                return {
                    "valid": False,
                    "reason": f"{app_id} has no 'differences' key",
                    "suggestion": f"Use {app_id}.{alt_diffs[0]} instead",
                    "alternatives": alt_diffs,
                }
            else:
                return {
                    "valid": False,
                    "reason": f"{app_id} has no 'differences' key",
                    "suggestion": f"Use {app_id}.initialfinaldiff instead",
                }

    return {
        "valid": False,
        "reason": "path not found in schema",
        "known_paths": known_paths[:5],
    }


def analyze_query(query: str, task_websites: list[str], schemas: dict) -> dict:
    """Analyze a single query."""
    result = {
        "query": query,
        "syntax_valid": True,
        "syntax_error": None,
        "paths_analyzed": [],
        "issues": [],
        "suggestions": [],
    }

    # Check syntax
    valid, error = check_syntax(query)
    if not valid:
        result["syntax_valid"] = False
        result["syntax_error"] = error
        result["issues"].append(f"Syntax error: {error}")
        return result

    # Extract and check paths
    paths = extract_all_paths(query)
    for path in paths:
        path_result = check_path_against_schema(path, schemas)
        path_result["path"] = path
        result["paths_analyzed"].append(path_result)

        if not path_result.get("valid"):
            reason = path_result.get("reason", "unknown")
            result["issues"].append(f"Invalid path '{path}': {reason}")
            if path_result.get("suggestion"):
                result["suggestions"].append(path_result["suggestion"])
            if path_result.get("alternatives"):
                result["suggestions"].append(f"Alternatives for {path_result.get('app_id', 'app')}: {path_result['alternatives']}")

    # Check app prefix matches task websites
    app_prefix = extract_app_prefix(query)
    if app_prefix and app_prefix not in task_websites:
        result["issues"].append(f"App '{app_prefix}' not in task websites: {task_websites}")

    return result


def analyze_task(task: dict, schemas: dict, verbose: bool = False) -> dict:
    """Analyze all queries in a task."""
    result = {
        "task_id": task["id"],
        "file": task["file"],
        "websites": task["websites"],
        "total_queries": 0,
        "valid_queries": 0,
        "invalid_queries": 0,
        "queries": [],
    }

    for i, eval_item in enumerate(task["evals"]):
        if eval_item.get("type") != "jmespath":
            continue

        query = eval_item.get("query", "")
        result["total_queries"] += 1

        analysis = analyze_query(query, task["websites"], schemas)
        analysis["eval_index"] = i
        analysis["description"] = eval_item.get("description", "")
        analysis["expected_value"] = eval_item.get("expected_value")

        if analysis["issues"]:
            result["invalid_queries"] += 1
        else:
            result["valid_queries"] += 1

        result["queries"].append(analysis)

        if verbose and analysis["issues"]:
            print(f"  [{i}] {analysis['description'][:50]}...")
            for issue in analysis["issues"]:
                print(f"      ❌ {issue}")
            for suggestion in analysis["suggestions"]:
                print(f"      💡 {suggestion}")

    return result


def generate_fix_patterns(all_results: list[dict]) -> list[dict]:
    """Generate suggested patterns for fix_query_patterns.py."""
    pattern_counts = defaultdict(int)
    pattern_suggestions = {}

    for task_result in all_results:
        for query_result in task_result["queries"]:
            for path_info in query_result.get("paths_analyzed", []):
                if not path_info.get("valid"):
                    reason = path_info.get("reason", "")
                    path = path_info.get("path", "")
                    suggestion = path_info.get("suggestion", "")

                    if "has no 'differences' key" in reason:
                        # Extract the pattern
                        parts = path.split(".")
                        if len(parts) >= 3:
                            app = parts[0]
                            pattern = f"{app}.differences"
                            pattern_counts[pattern] += 1
                            if suggestion and pattern not in pattern_suggestions:
                                pattern_suggestions[pattern] = suggestion

    fixes = []
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        fixes.append({
            "pattern": pattern,
            "occurrences": count,
            "suggestion": pattern_suggestions.get(pattern, "Check schema documentation"),
        })

    return fixes


def main():
    parser = argparse.ArgumentParser(description="Analyze queries against known schemas")
    parser.add_argument("--task", type=str, help="Analyze single task by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output JSON file")

    args = parser.parse_args()

    # Load schemas
    schemas = load_schemas()
    if not schemas:
        print("No schemas available. Run discover_schemas.py first.")
        return 1

    print(f"Loaded schemas for {len(schemas)} apps: {', '.join(sorted(schemas.keys()))}")

    # Load tasks
    tasks = load_tasks()
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"Task not found: {args.task}")
            return 1

    print(f"Analyzing {len(tasks)} tasks...")

    # Analyze all tasks
    all_results = []
    total_queries = 0
    total_valid = 0
    total_invalid = 0
    tasks_with_issues = []

    for task in tasks:
        if args.verbose:
            print(f"\n=== {task['id']} ===")

        result = analyze_task(task, schemas, verbose=args.verbose)
        all_results.append(result)

        total_queries += result["total_queries"]
        total_valid += result["valid_queries"]
        total_invalid += result["invalid_queries"]

        if result["invalid_queries"] > 0:
            tasks_with_issues.append((task["id"], result["invalid_queries"]))

    # Generate fix patterns
    fix_patterns = generate_fix_patterns(all_results)

    # Save results
    output_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "tasks_analyzed": len(tasks),
            "total_queries": total_queries,
            "valid_queries": total_valid,
            "invalid_queries": total_invalid,
            "tasks_with_issues": len(tasks_with_issues),
        },
        "fix_patterns": fix_patterns,
        "tasks": all_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAnalysis saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tasks analyzed: {len(tasks)}")
    print(f"Total queries: {total_queries}")
    print(f"  Valid: {total_valid} ({100*total_valid/total_queries:.0f}%)" if total_queries else "  Valid: 0")
    print(f"  Invalid: {total_invalid} ({100*total_invalid/total_queries:.0f}%)" if total_queries else "  Invalid: 0")
    print(f"Tasks with issues: {len(tasks_with_issues)}")

    if fix_patterns:
        print("\nSuggested fixes for fix_query_patterns.py:")
        for fix in fix_patterns[:10]:
            print(f"  {fix['pattern']} ({fix['occurrences']}x): {fix['suggestion']}")

    if tasks_with_issues and not args.verbose:
        print("\nTasks with invalid queries (run with --verbose for details):")
        for task_id, count in sorted(tasks_with_issues, key=lambda x: -x[1])[:10]:
            print(f"  {task_id}: {count} issues")

    return 0


if __name__ == "__main__":
    exit(main())
