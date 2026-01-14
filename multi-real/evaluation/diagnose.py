#!/usr/bin/env python3
"""
Diagnose why JMESPath queries are failing.

Usage:
    uv run python multi-real/evaluation/diagnose.py

This script:
1. Runs validation on all tasks with finish JSONs
2. For each failing query, determines if it's:
   - Query bug (incorrect path/syntax)
   - Data bug (incomplete manual completion)
   - Schema bug (entity not tracked by app)
"""

import json
import jmespath
from pathlib import Path
from collections import defaultdict

# Base directory for multi-real
BASE_DIR = Path(__file__).parent.parent


def load_json(path: Path) -> dict:
    """Load JSON from file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_query_prefix(query: str) -> str:
    """Extract the app prefix from a query (e.g., 'gomail' from 'gomail.differences.emails')."""
    parts = query.split('.')
    if len(parts) > 1:
        return parts[0]
    return ""


def check_similar_paths(data: dict, failed_path: str, app_prefix: str) -> list:
    """Find similar paths that exist in the data."""
    similar = []

    # Split the failed path
    parts = failed_path.split('.')

    # Check if we're in multi-app mode
    if app_prefix and app_prefix in data:
        app_data = data[app_prefix]
    else:
        app_data = data

    # Check common variations
    if 'differences' in failed_path:
        if 'differences' in app_data:
            similar.append(f"{app_prefix}.differences" if app_prefix else "differences")

    if 'initialfinaldiff' in failed_path:
        if 'initialfinaldiff' in app_data:
            similar.append(f"{app_prefix}.initialfinaldiff" if app_prefix else "initialfinaldiff")

    # Check for email/emails variations
    if 'email' in failed_path:
        if 'initialfinaldiff' in app_data:
            added = app_data.get('initialfinaldiff', {}).get('added', {})
            if 'email' in added:
                similar.append(f"{app_prefix}.initialfinaldiff.added.email" if app_prefix else "initialfinaldiff.added.email")

        if 'differences' in app_data:
            diff = app_data.get('differences', {})
            if 'emails' in diff:
                similar.append(f"{app_prefix}.differences.emails" if app_prefix else "differences.emails")

    return similar


def diagnose_failure(query: str, finish_json: dict, error: str) -> dict:
    """
    Diagnose why a query failed.

    Returns:
        {
            'diagnosis': 'query_bug' | 'data_incomplete' | 'schema_missing',
            'explanation': str,
            'fix': str,
            'similar_paths': list
        }
    """
    app_prefix = get_query_prefix(query)

    # Check if path exists partially
    similar_paths = check_similar_paths(finish_json, query, app_prefix)

    # Diagnosis logic
    if "invalid type for value: None" in error or "Query returned None" in error:
        # Path doesn't exist at all
        if similar_paths:
            return {
                'diagnosis': 'data_incomplete',
                'explanation': f"Path '{query}' not found in finish JSON, but similar paths exist. Manual completion likely incomplete.",
                'fix': f"Re-run task manually and ensure all actions complete. Check that data appears at: {similar_paths[0] if similar_paths else 'N/A'}",
                'similar_paths': similar_paths
            }
        else:
            return {
                'diagnosis': 'schema_missing',
                'explanation': f"Path '{query}' not found and no similar paths exist. This entity may not be tracked by the app.",
                'fix': "Remove this query or change to check a tracked entity. Check /finish endpoint to see what's available.",
                'similar_paths': []
            }

    elif "syntax error" in error.lower() or "parse" in error.lower():
        return {
            'diagnosis': 'query_bug',
            'explanation': f"Query syntax error: {error}",
            'fix': "Fix JMESPath syntax. Common issues: comparison operators (use >= `1` not > 0), string escaping, null checks.",
            'similar_paths': []
        }

    else:
        return {
            'diagnosis': 'unknown',
            'explanation': f"Unknown error: {error}",
            'fix': "Manual investigation required.",
            'similar_paths': similar_paths
        }


def main():
    """Run diagnostics on all failing queries."""
    finish_dir = BASE_DIR / "final_states" / "manual"
    tasks_dir = BASE_DIR / "tasks"

    if not finish_dir.exists():
        print(f"Error: {finish_dir} not found")
        return

    # Get all finish JSONs
    finish_jsons = list(finish_dir.glob("*.json"))
    print(f"Found {len(finish_jsons)} finish JSONs\n")

    # Track issues by diagnosis type
    issues_by_type = defaultdict(list)

    for finish_path in sorted(finish_jsons):
        task_id = finish_path.stem
        task_path = tasks_dir / f"{task_id}.json"

        if not task_path.exists():
            continue

        # Load data
        task = load_json(task_path)
        finish_json = load_json(finish_path)

        # Check each eval
        evals = task.get('evals', [])
        failing_queries = []

        for i, eval_item in enumerate(evals):
            if eval_item.get('type') != 'jmespath':
                continue

            query = eval_item.get('query', '')

            # Test query
            try:
                result = jmespath.search(query, finish_json)
                if result is None:
                    error = "Query returned None (path not found)"
                    failing_queries.append((i, query, eval_item.get('description'), error))
            except Exception as e:
                error = str(e)
                failing_queries.append((i, query, eval_item.get('description'), error))

        # Diagnose failures
        if failing_queries:
            print(f"\n{'='*80}")
            print(f"Task: {task_id}")
            print(f"Goal: {task['goal'][:80]}...")
            print(f"Failing queries: {len(failing_queries)}")
            print(f"{'='*80}")

            for idx, query, description, error in failing_queries:
                diagnosis = diagnose_failure(query, finish_json, error)

                print(f"\n[Query {idx+1}] {description}")
                print(f"  Query: {query}")
                print(f"  Error: {error}")
                print(f"  Diagnosis: {diagnosis['diagnosis'].upper()}")
                print(f"  Explanation: {diagnosis['explanation']}")
                print(f"  Fix: {diagnosis['fix']}")

                if diagnosis['similar_paths']:
                    print(f"  Similar paths found: {', '.join(diagnosis['similar_paths'])}")

                issues_by_type[diagnosis['diagnosis']].append({
                    'task': task_id,
                    'query': query,
                    'description': description
                })

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY BY DIAGNOSIS TYPE")
    print(f"{'='*80}")

    for diagnosis_type, issues in issues_by_type.items():
        print(f"\n{diagnosis_type.upper()}: {len(issues)} issues")

        # Group by task
        tasks = defaultdict(int)
        for issue in issues:
            tasks[issue['task']] += 1

        for task_id, count in sorted(tasks.items()):
            print(f"  - {task_id}: {count} queries")

    # Recommendations
    print(f"\n\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    if 'data_incomplete' in issues_by_type:
        print(f"\n1. DATA INCOMPLETE ({len(issues_by_type['data_incomplete'])} queries):")
        print("   → Re-run these tasks manually, ensuring all steps complete:")
        tasks_to_rerun = set(issue['task'] for issue in issues_by_type['data_incomplete'])
        for task in sorted(tasks_to_rerun):
            print(f"     uv run python example/manual_agent.py {task}")

    if 'query_bug' in issues_by_type:
        print(f"\n2. QUERY BUGS ({len(issues_by_type['query_bug'])} queries):")
        print("   → Fix JMESPath syntax in these task JSONs:")
        for issue in issues_by_type['query_bug']:
            print(f"     {issue['task']}: {issue['query'][:60]}...")

    if 'schema_missing' in issues_by_type:
        print(f"\n3. SCHEMA MISSING ({len(issues_by_type['schema_missing'])} queries):")
        print("   → These entities aren't tracked by the apps. Either:")
        print("     a) Remove the query, or")
        print("     b) Change to query a tracked entity")
        for issue in issues_by_type['schema_missing']:
            print(f"     {issue['task']}: {issue['description']}")


if __name__ == '__main__':
    main()
