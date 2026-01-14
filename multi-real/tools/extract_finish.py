#!/usr/bin/env python3
"""
Extract finish JSONs from benchmark results.

This script extracts the env_state from each task result and saves it
as a standalone finish JSON file. These can then be used for query validation.

Usage:
    uv run python multi-real/extract_finish_jsons.py results/openai_cua_20260111_123456
    uv run python multi-real/extract_finish_jsons.py results/anthropic_cua_20260111_123456

The script will:
1. Read all task_*.json files from results_dir/tasks/
2. Extract the env_state from each task result
3. Save as multi-real/final_states/{model_name}/{task_id}.json
4. Report statistics on successful extractions
"""

import argparse
import json
from pathlib import Path


def extract_finish_jsons(results_dir: Path, output_dir: Path | None = None, model_name: str = "auto") -> dict:
    """
    Extract env_state from all task result files.

    Args:
        results_dir: Directory containing benchmark results
        output_dir: Where to save extracted finish JSONs (default: multi-real/final_states/{model_name})
        model_name: Model name for organizing outputs (default: auto-detect from results)

    Returns:
        Dict with statistics: {'total': int, 'extracted': int, 'missing': int, 'tasks': list}
    """
    # Validate results directory
    tasks_dir = results_dir / "tasks"
    if not tasks_dir.exists():
        raise ValueError(f"Tasks directory not found: {tasks_dir}")

    # Auto-detect model name from results
    if model_name == "auto":
        summary_file = results_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                model_name = summary.get("model", "unknown")
                if model_name == "OpenAI-CUA":
                    model_name = "openai"
                elif model_name == "Anthropic-CUA":
                    model_name = "anthropic"
        else:
            # Parse from directory name
            dir_name = results_dir.name.lower()
            if "openai" in dir_name:
                model_name = "openai"
            elif "anthropic" in dir_name:
                model_name = "anthropic"
            else:
                model_name = "unknown"

    # Set output directory
    if output_dir is None:
        base_dir = Path("multi-real/final_states")
        output_dir = base_dir / model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all task result files
    task_files = list(tasks_dir.glob("task_*.json"))
    print(f"Found {len(task_files)} task result files")
    print(f"Extracting to: {output_dir}")
    print()

    # Track statistics
    stats = {
        'total': 0,
        'extracted': 0,
        'missing': 0,
        'empty': 0,
        'tasks': []
    }

    for task_file in sorted(task_files):
        stats['total'] += 1

        with open(task_file) as f:
            result = json.load(f)

        task_id = result.get("task_id", "unknown")
        env_state = result.get("env_state")
        success = result.get("success", False)

        # Check if env_state exists and is non-empty
        if env_state is None:
            print(f"✗ {task_id}: Missing env_state")
            stats['missing'] += 1
            stats['tasks'].append({
                'task_id': task_id,
                'status': 'missing',
                'success': success
            })
            continue

        if not env_state or (isinstance(env_state, dict) and not any(env_state.values())):
            print(f"⚠ {task_id}: Empty env_state")
            stats['empty'] += 1
            stats['tasks'].append({
                'task_id': task_id,
                'status': 'empty',
                'success': success
            })
            # Still save it for debugging
            output_file = output_dir / f"{task_id}.json"
            with open(output_file, "w") as f:
                json.dump(env_state, f, indent=2)
            continue

        # Save finish JSON
        output_file = output_dir / f"{task_id}.json"
        with open(output_file, "w") as f:
            json.dump(env_state, f, indent=2)

        status_icon = "✓" if success else "○"
        print(f"{status_icon} {task_id}: Extracted ({len(str(env_state))} bytes)")
        stats['extracted'] += 1
        stats['tasks'].append({
            'task_id': task_id,
            'status': 'extracted',
            'success': success,
            'size_bytes': len(str(env_state))
        })

    # Print summary
    print()
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total tasks: {stats['total']}")
    print(f"Successfully extracted: {stats['extracted']}")
    print(f"Empty env_state: {stats['empty']}")
    print(f"Missing env_state: {stats['missing']}")
    print(f"Extraction rate: {stats['extracted'] / stats['total'] * 100:.1f}%")
    print()

    # Save stats
    stats_file = output_dir / "_extraction_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Statistics saved to: {stats_file}")
    print(f"📁 Finish JSONs saved to: {output_dir}")

    return stats


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract finish JSONs from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from OpenAI results
  uv run python multi-real/extract_finish_jsons.py results/openai_cua_20260111_123456

  # Extract from Anthropic results
  uv run python multi-real/extract_finish_jsons.py results/anthropic_cua_20260111_123456

  # Specify custom output directory
  uv run python multi-real/extract_finish_jsons.py results/openai_cua_20260111_123456 -o custom/output/dir

  # Specify model name explicitly
  uv run python multi-real/extract_finish_jsons.py results/some_run -m openai
        """
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing benchmark results (e.g., results/openai_cua_20260111_123456)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: multi-real/final_states/{model_name})"
    )
    parser.add_argument(
        "-m", "--model",
        default="auto",
        help="Model name for organizing outputs (default: auto-detect)"
    )

    args = parser.parse_args()

    # Validate results directory exists
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    try:
        extract_finish_jsons(args.results_dir, args.output, args.model)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
