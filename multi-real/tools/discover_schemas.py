#!/usr/bin/env python3
"""
Discover and document finish JSON schemas from ground truth files.

Analyzes ground truth files to extract schema patterns for each web app,
identifying available diff paths, data types, and queryable fields.

INCREMENTAL DESIGN:
- Re-run this script as you add more ground truth files
- Schema knowledge accumulates across all GT files
- Coverage metrics show which apps/tasks still need GT
- Previous schema.json is merged with new discoveries

Usage:
    uv run python multi-real/tools/discover_schemas.py
    uv run python multi-real/tools/discover_schemas.py --output schemas.json
    uv run python multi-real/tools/discover_schemas.py --verbose
    uv run python multi-real/tools/discover_schemas.py --include-synthetic  # include final_states/synthetic/
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent.parent  # multi-real/
GT_DIRS = [
    BASE_DIR / "final_states" / "manual",
    BASE_DIR / "final_states" / "synthetic",
]
TASKS_DIR = BASE_DIR / "tasks"
OUTPUT_DIR = BASE_DIR / "docs"


def get_type_info(value: Any, max_sample_keys: int = 10) -> dict:
    """Get type information for a value."""
    if value is None:
        return {"type": "null"}
    elif isinstance(value, bool):
        return {"type": "bool"}
    elif isinstance(value, int):
        return {"type": "int"}
    elif isinstance(value, float):
        return {"type": "float"}
    elif isinstance(value, str):
        return {"type": "str", "sample": value[:50] if len(value) > 50 else value}
    elif isinstance(value, list):
        info = {"type": "list", "length": len(value)}
        if value and isinstance(value[0], dict):
            info["item_keys"] = list(value[0].keys())[:max_sample_keys]
        return info
    elif isinstance(value, dict):
        info = {"type": "dict", "keys": list(value.keys())[:max_sample_keys]}
        return info
    else:
        return {"type": type(value).__name__}


def extract_schema(data: dict, prefix: str = "", max_depth: int = 4, depth: int = 0) -> dict:
    """Recursively extract schema from data."""
    if depth >= max_depth or not isinstance(data, dict):
        return get_type_info(data)

    schema = {"type": "dict", "fields": {}}
    for key, value in data.items():
        if isinstance(value, dict):
            schema["fields"][key] = extract_schema(value, f"{prefix}.{key}", max_depth, depth + 1)
        elif isinstance(value, list):
            field_info = {"type": "list", "length": len(value)}
            if value:
                if isinstance(value[0], dict):
                    field_info["item_schema"] = extract_schema(value[0], f"{prefix}.{key}[]", max_depth, depth + 1)
                else:
                    field_info["item_type"] = type(value[0]).__name__
            schema["fields"][key] = field_info
        else:
            schema["fields"][key] = get_type_info(value)

    return schema


def find_diff_paths(data: dict, app_id: str) -> list[dict]:
    """Find all paths that could be used for evaluation queries."""
    diff_paths = []

    app_data = data.get(app_id, {})
    if not app_data:
        return diff_paths

    # Standard paths
    standard_paths = [
        ("differences", "Standard diff structure"),
        ("initialfinaldiff", "Initial vs final state diff"),
    ]

    for path, desc in standard_paths:
        if path in app_data:
            path_data = app_data[path]
            if isinstance(path_data, dict):
                for key, value in path_data.items():
                    full_path = f"{app_id}.{path}.{key}"
                    if isinstance(value, dict):
                        # Check for added/deleted/updated pattern
                        if any(k in value for k in ["added", "deleted", "updated"]):
                            for op in ["added", "deleted", "updated"]:
                                if op in value:
                                    op_value = value[op]
                                    diff_paths.append({
                                        "path": f"{full_path}.{op}",
                                        "type": type(op_value).__name__,
                                        "description": f"{desc} - {key} {op}",
                                        "has_data": bool(op_value) if isinstance(op_value, (list, dict)) else op_value is not None,
                                        "sample_keys": list(op_value.keys())[:5] if isinstance(op_value, dict) else (
                                            list(op_value[0].keys())[:5] if isinstance(op_value, list) and op_value and isinstance(op_value[0], dict) else None
                                        ),
                                    })
                        else:
                            diff_paths.append({
                                "path": full_path,
                                "type": "dict",
                                "description": desc,
                                "keys": list(value.keys())[:10],
                            })
                    elif isinstance(value, list):
                        diff_paths.append({
                            "path": full_path,
                            "type": "list",
                            "length": len(value),
                            "description": desc,
                        })

    # App-specific diff paths (e.g., bookingDetailsDiff, foodOrders)
    for key in app_data.keys():
        if key.endswith("Diff") or key in ["differences"]:
            if key == "differences":
                continue  # Already handled above
            value = app_data[key]
            if isinstance(value, dict):
                for op in ["added", "deleted", "updated"]:
                    if op in value:
                        op_value = value[op]
                        diff_paths.append({
                            "path": f"{app_id}.{key}.{op}",
                            "type": type(op_value).__name__,
                            "description": f"App-specific diff - {key}",
                            "has_data": bool(op_value) if isinstance(op_value, (list, dict)) else op_value is not None,
                            "sample_keys": list(op_value.keys())[:5] if isinstance(op_value, dict) else (
                                list(op_value[0].keys())[:5] if isinstance(op_value, list) and op_value and isinstance(op_value[0], dict) else None
                            ),
                        })

    return diff_paths


def analyze_ground_truth(gt_path: Path, verbose: bool = False) -> dict:
    """Analyze a single ground truth file."""
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    result = {
        "file": gt_path.name,
        "apps": {},
    }

    for app_id in data.keys():
        app_data = data[app_id]
        if not isinstance(app_data, dict):
            continue

        app_info = {
            "top_level_keys": list(app_data.keys()),
            "diff_paths": find_diff_paths(data, app_id),
            "has_differences": "differences" in app_data,
            "has_initialfinaldiff": "initialfinaldiff" in app_data,
            "app_specific_diffs": [k for k in app_data.keys() if k.endswith("Diff") and k != "initialfinaldiff"],
        }

        # Extract schema for key paths
        if "differences" in app_data:
            app_info["differences_schema"] = extract_schema(app_data["differences"], f"{app_id}.differences", max_depth=3)

        if "initialfinaldiff" in app_data:
            app_info["initialfinaldiff_schema"] = extract_schema(app_data["initialfinaldiff"], f"{app_id}.initialfinaldiff", max_depth=3)

        result["apps"][app_id] = app_info

        if verbose:
            print(f"\n  {app_id}:")
            print(f"    Top-level keys: {app_info['top_level_keys']}")
            print(f"    Has differences: {app_info['has_differences']}")
            print(f"    Has initialfinaldiff: {app_info['has_initialfinaldiff']}")
            print(f"    App-specific diffs: {app_info['app_specific_diffs']}")
            print(f"    Queryable paths ({len(app_info['diff_paths'])}):")
            for path_info in app_info["diff_paths"][:5]:
                print(f"      - {path_info['path']} ({path_info['type']})")

    return result


def merge_schemas(all_results: list[dict]) -> dict:
    """Merge schemas from multiple ground truth files."""
    merged = defaultdict(lambda: {
        "sources": [],
        "top_level_keys": set(),
        "diff_paths": {},
        "has_differences": False,
        "has_initialfinaldiff": False,
        "app_specific_diffs": set(),
    })

    for result in all_results:
        source_file = result["file"]
        for app_id, app_info in result["apps"].items():
            merged[app_id]["sources"].append(source_file)
            merged[app_id]["top_level_keys"].update(app_info["top_level_keys"])
            merged[app_id]["has_differences"] |= app_info["has_differences"]
            merged[app_id]["has_initialfinaldiff"] |= app_info["has_initialfinaldiff"]
            merged[app_id]["app_specific_diffs"].update(app_info.get("app_specific_diffs", []))

            for path_info in app_info["diff_paths"]:
                path = path_info["path"]
                if path not in merged[app_id]["diff_paths"]:
                    merged[app_id]["diff_paths"][path] = path_info
                elif path_info.get("has_data"):
                    # Prefer version with data
                    merged[app_id]["diff_paths"][path] = path_info

    # Convert sets to lists for JSON serialization
    final = {}
    for app_id, info in merged.items():
        final[app_id] = {
            "sources": info["sources"],
            "top_level_keys": sorted(info["top_level_keys"]),
            "has_differences": info["has_differences"],
            "has_initialfinaldiff": info["has_initialfinaldiff"],
            "app_specific_diffs": sorted(info["app_specific_diffs"]),
            "diff_paths": list(info["diff_paths"].values()),
        }

    return final


def load_tasks() -> dict[str, dict]:
    """Load all task definitions to compute coverage."""
    tasks = {}
    for task_file in TASKS_DIR.glob("*.json"):
        with open(task_file, encoding="utf-8") as f:
            task = json.load(f)
        tasks[task["id"]] = {
            "file": task_file.name,
            "websites": [w["id"] for w in task.get("websites", [])],
            "eval_count": len([e for e in task.get("evals", []) if e.get("type") == "jmespath"]),
            "todo_count": len([e for e in task.get("evals", []) if "TODO" in str(e.get("expected_value", ""))]),
        }
    return tasks


def compute_coverage(merged_schema: dict, tasks: dict, gt_files: list[str]) -> dict:
    """Compute coverage metrics showing what's validated vs pending."""
    # Which tasks have ground truth
    gt_task_ids = {Path(f).stem for f in gt_files}

    # All apps used across tasks
    all_apps = set()
    for task in tasks.values():
        all_apps.update(task["websites"])

    # Apps with schema knowledge
    known_apps = set(merged_schema.keys())

    # Per-task coverage
    task_coverage = {}
    for task_id, task_info in tasks.items():
        has_gt = task_id in gt_task_ids
        apps_known = all(app in known_apps for app in task_info["websites"])
        task_coverage[task_id] = {
            "has_ground_truth": has_gt,
            "all_apps_have_schema": apps_known,
            "missing_app_schemas": [a for a in task_info["websites"] if a not in known_apps],
            "eval_count": task_info["eval_count"],
            "todo_count": task_info["todo_count"],
        }

    return {
        "total_tasks": len(tasks),
        "tasks_with_gt": len(gt_task_ids),
        "tasks_without_gt": len(tasks) - len(gt_task_ids),
        "total_apps_in_tasks": len(all_apps),
        "apps_with_schema": len(known_apps),
        "apps_missing_schema": sorted(all_apps - known_apps),
        "apps_known": sorted(known_apps),
        "gt_files": sorted(gt_files),
        "task_coverage": task_coverage,
    }


def generate_markdown(merged_schema: dict, coverage: dict = None) -> str:
    """Generate markdown documentation from merged schema."""
    lines = [
        "# Multi-REAL Finish JSON Schema Documentation",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "This document describes the finish JSON schema for each web application.",
        "Re-run `tools/discover_schemas.py` as you add more ground truth files.",
        "",
    ]

    # Add coverage section if available
    if coverage:
        lines.extend([
            "## Coverage Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total tasks | {coverage['total_tasks']} |",
            f"| Tasks with ground truth | {coverage['tasks_with_gt']} ({100*coverage['tasks_with_gt']/coverage['total_tasks']:.0f}%) |",
            f"| Tasks without ground truth | {coverage['tasks_without_gt']} |",
            f"| Apps used in tasks | {coverage['total_apps_in_tasks']} |",
            f"| Apps with schema knowledge | {coverage['apps_with_schema']} |",
            "",
        ])
        if coverage['apps_missing_schema']:
            lines.extend([
                f"**Apps missing schema (need GT):** `{', '.join(coverage['apps_missing_schema'])}`",
                "",
            ])
        lines.extend([
            "### Tasks Needing Ground Truth",
            "",
            "Tasks without GT but with all app schemas known can still have queries validated for syntax/schema.",
            "",
            "| Status | Count |",
            "|--------|-------|",
            f"| ✅ Has GT | {coverage['tasks_with_gt']} |",
            f"| ⚠️ No GT, schema known | {sum(1 for t in coverage['task_coverage'].values() if not t['has_ground_truth'] and t['all_apps_have_schema'])} |",
            f"| ❌ No GT, missing app schema | {sum(1 for t in coverage['task_coverage'].values() if not t['has_ground_truth'] and not t['all_apps_have_schema'])} |",
            "",
        ])

    lines.extend([
        "## App Schema Overview",
        "",
        "| App | Has `differences` | Has `initialfinaldiff` | App-Specific Diffs | Sources |",
        "|-----|-------------------|------------------------|-------------------|---------|",
    ])

    for app_id in sorted(merged_schema.keys()):
        info = merged_schema[app_id]
        lines.append(
            f"| {app_id} | {'✅' if info['has_differences'] else '❌'} | "
            f"{'✅' if info['has_initialfinaldiff'] else '❌'} | "
            f"{', '.join(info['app_specific_diffs']) or 'None'} | "
            f"{len(info['sources'])} |"
        )

    lines.extend(["", "## Per-App Schema Details", ""])

    for app_id in sorted(merged_schema.keys()):
        info = merged_schema[app_id]
        lines.extend([
            f"### {app_id}",
            "",
            f"**Sources:** {', '.join(info['sources'])}",
            "",
            f"**Top-level keys:** `{', '.join(info['top_level_keys'])}`",
            "",
            "**Queryable Diff Paths:**",
            "",
            "| Path | Type | Has Data | Sample Keys |",
            "|------|------|----------|-------------|",
        ])

        for path_info in info["diff_paths"]:
            sample = path_info.get("sample_keys", path_info.get("keys", []))
            sample_str = ", ".join(sample[:5]) if sample else "-"
            has_data = "✅" if path_info.get("has_data") else "❌"
            lines.append(f"| `{path_info['path']}` | {path_info['type']} | {has_data} | {sample_str} |")

        lines.extend(["", "---", ""])

    lines.extend([
        "## Query Pattern Guidelines",
        "",
        "### Standard Pattern (when `differences` exists):",
        "```",
        "app_id.differences.collection.added[?filter_condition]",
        "app_id.differences.collection.added[?field == 'value'] | length(@) >= `1`",
        "```",
        "",
        "### InitialFinalDiff Pattern:",
        "```",
        "app_id.initialfinaldiff.added.collection != null",
        "values(app_id.initialfinaldiff.added.collection)[?condition]",
        "```",
        "",
        "### App-Specific Diff Pattern:",
        "```",
        "app_id.bookingDetailsDiff.added[?condition]",
        "app_id.foodOrders.added (note: may be dict, not array)",
        "```",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Discover finish JSON schemas from ground truth")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "schemas.json", help="Output JSON file")
    parser.add_argument("--markdown", type=Path, default=OUTPUT_DIR / "finish_schemas.md", help="Output markdown file")
    parser.add_argument("--include-synthetic", action="store_true", help="Include synthetic ground truth")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Find ground truth files from all directories
    gt_files = []
    gt_dirs_used = []
    for gt_dir in GT_DIRS:
        if not gt_dir.exists():
            continue
        if "synthetic" in str(gt_dir) and not args.include_synthetic:
            continue
        files = list(gt_dir.glob("*.json"))
        if files:
            gt_files.extend(files)
            gt_dirs_used.append(gt_dir)

    if not gt_files:
        print(f"No ground truth files found in: {GT_DIRS}")
        return 1

    print(f"Analyzing {len(gt_files)} ground truth files from {len(gt_dirs_used)} directories...")
    for gt_dir in gt_dirs_used:
        count = len(list(gt_dir.glob("*.json")))
        print(f"  - {gt_dir.relative_to(BASE_DIR)}: {count} files")

    # Analyze each file
    all_results = []
    for gt_file in sorted(gt_files):
        print(f"\n=== {gt_file.name} ===")
        result = analyze_ground_truth(gt_file, verbose=args.verbose)
        all_results.append(result)
        print(f"  Found {len(result['apps'])} apps: {', '.join(result['apps'].keys())}")

    # Merge schemas
    print("\n" + "=" * 60)
    print("Merging schemas...")
    merged = merge_schemas(all_results)

    # Load tasks and compute coverage
    print("Loading tasks and computing coverage...")
    tasks = load_tasks()
    gt_file_names = [f.name for f in gt_files]
    coverage = compute_coverage(merged, tasks, gt_file_names)

    # Save JSON with coverage info
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": coverage,
        "schemas": merged,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Schema JSON saved to: {args.output}")

    # Generate and save markdown
    markdown = generate_markdown(merged, coverage)
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Schema documentation saved to: {args.markdown}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tasks: {coverage['total_tasks']}")
    print(f"  With ground truth: {coverage['tasks_with_gt']} ({100*coverage['tasks_with_gt']/coverage['total_tasks']:.0f}%)")
    print(f"  Without ground truth: {coverage['tasks_without_gt']}")
    print()
    print(f"Apps discovered: {len(merged)}")
    for app_id in sorted(merged.keys()):
        info = merged[app_id]
        print(f"  {app_id}: {len(info['diff_paths'])} queryable paths")

    if coverage['apps_missing_schema']:
        print()
        print(f"Apps still needing ground truth: {', '.join(coverage['apps_missing_schema'])}")

    print()
    print("Re-run this script as you add more ground truth files to increase coverage.")

    return 0


if __name__ == "__main__":
    exit(main())
