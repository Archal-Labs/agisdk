"""
Validate JMESPath queries against raw finish JSON structure (no normalization).

This script tests queries directly on the raw JSON structure to identify
queries that use incorrect paths (like gomail.emails.added[0] instead of
values(gomail.initialfinaldiff.added.email.emails)[0]).

Usage:
    # Validate single task:
    uv run python multi-real/validate_query_structure.py multi-real/tasks/dashdish-gomail-2.json --finish-json multi-real/final_states/dashdish-gomail-1.json

    # Validate all tasks with available finish JSON:
    uv run python multi-real/validate_query_structure.py --all --finish-json-dir multi-real/final_states
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jmespath


def load_json(path: str) -> dict:
    """Load JSON from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_query_raw(query: str, data: dict) -> tuple[Any, bool, str | None]:
    """
    Test query on raw JSON (no normalization).
    
    Returns:
        (result, is_valid, error_message)
        is_valid: True if query executes without error and returns non-None
    """
    try:
        result = jmespath.search(query, data)
        if result is None:
            return None, False, "Query returned None (path not found)"
        return result, True, None
    except Exception as e:
        return None, False, str(e)


def validate_task_queries(task_path: str, finish_json: dict, verbose: bool = False):
    """
    Validate all queries in a task JSON against raw finish JSON.
    
    Returns:
        (passed, failed, broken_queries)
        broken_queries: List of (eval_index, query, error) tuples
    """
    task = load_json(task_path)
    evals = task.get("evals", [])
    
    print(f"\nValidating task: {task['id']}")
    print(f"Goal: {task['goal'][:80]}...")
    print("-" * 80)
    
    passed = 0
    failed = 0
    broken_queries = []
    
    for i, eval_item in enumerate(evals):
        eval_type = eval_item.get("type", "unknown")
        
        if eval_type != "jmespath":
            if verbose:
                print(f"\n[{i+1}] {eval_item.get('description', 'No description')}")
                print(f"    Type: {eval_type} (skipped - not jmespath)")
            continue
        
        query = eval_item.get("query", "")
        description = eval_item.get("description", "No description")
        
        result, is_valid, error = test_query_raw(query, finish_json)
        
        if is_valid:
            passed += 1
            if verbose:
                print(f"\n[{i+1}] ✅ {description}")
                print(f"    Query: {query}")
                print(f"    Result: {str(result)[:100]}...")
        else:
            failed += 1
            broken_queries.append((i + 1, query, error))
            print(f"\n[{i+1}] ❌ {description}")
            print(f"    Query: {query}")
            print(f"    Error: {error}")
    
    print("\n" + "=" * 80)
    print(f"Summary: {passed} passed, {failed} failed")
    
    return passed, failed, broken_queries


def find_finish_json_for_task(task_path: str, finish_json_dir: str) -> str | None:
    """
    Find matching finish JSON file for a task.
    
    Looks for files like: task_id.json or files containing task web apps.
    """
    task = load_json(task_path)
    task_id = task.get("id", "")
    task_dir = Path(finish_json_dir)
    
    # Try exact match first
    exact_match = task_dir / f"{task_id}.json"
    if exact_match.exists():
        return str(exact_match)
    
    # Try to find files that contain task web apps
    websites = task.get("websites", [])
    if not websites:
        return None
    
    website_ids = {w.get("id") for w in websites if w.get("id")}
    
    for json_file in task_dir.glob("*.json"):
        try:
            finish_data = load_json(str(json_file))
            finish_websites = set(finish_data.keys())
            if website_ids.issubset(finish_websites):
                return str(json_file)
        except:
            continue
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Validate JMESPath queries against raw finish JSON")
    parser.add_argument("task_json", nargs="?", help="Path to task JSON file")
    parser.add_argument("--finish-json", "-f", help="Path to finish JSON file")
    parser.add_argument("--finish-json-dir", "-d", help="Directory containing finish JSON files")
    parser.add_argument("--all", action="store_true", help="Validate all tasks with available finish JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.all:
        if not args.finish_json_dir:
            print("Error: --finish-json-dir required when using --all")
            sys.exit(1)
        
        task_dir = Path("multi-real/tasks")
        finish_json_dir = Path(args.finish_json_dir)
        
        task_files = list(task_dir.glob("*.json"))
        print(f"Found {len(task_files)} task files")
        print(f"Finish JSON directory: {finish_json_dir}\n")
        
        total_passed = 0
        total_failed = 0
        tasks_with_issues = []
        
        for task_file in sorted(task_files):
            finish_json_path = find_finish_json_for_task(str(task_file), str(finish_json_dir))
            if not finish_json_path:
                continue
            
            finish_json = load_json(finish_json_path)
            passed, failed, broken = validate_task_queries(str(task_file), finish_json, args.verbose)
            
            total_passed += passed
            total_failed += failed
            
            if failed > 0:
                tasks_with_issues.append((task_file.name, failed, broken))
        
        print("\n" + "=" * 80)
        print(f"Overall Summary: {total_passed} passed, {total_failed} failed")
        print(f"\nTasks with broken queries: {len(tasks_with_issues)}")
        for task_name, failed_count, broken in tasks_with_issues:
            print(f"  {task_name}: {failed_count} broken queries")
        
        return
    
    # Single task validation
    if not args.task_json:
        print("Error: task_json required (or use --all)")
        sys.exit(1)
    
    if not args.finish_json:
        if args.finish_json_dir:
            finish_json_path = find_finish_json_for_task(args.task_json, args.finish_json_dir)
            if finish_json_path:
                args.finish_json = finish_json_path
                print(f"Using finish JSON: {finish_json_path}")
            else:
                print("Error: Could not find matching finish JSON file")
                sys.exit(1)
        else:
            print("Error: --finish-json or --finish-json-dir required")
            sys.exit(1)
    
    finish_json = load_json(args.finish_json)
    validate_task_queries(args.task_json, finish_json, args.verbose)


if __name__ == "__main__":
    main()

