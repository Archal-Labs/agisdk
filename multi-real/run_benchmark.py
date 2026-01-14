#!/usr/bin/env python3
"""
Multi-REAL Benchmark Runner

Usage:
    # Run all tasks with a model
    uv run python run_benchmark.py --model gpt52

    # Run specific tasks
    uv run python run_benchmark.py --model claude_opus45 --tasks multi.gocalendar-gomail-1 multi.dashdish-gomail-1

    # Run with filters
    uv run python run_benchmark.py --model gemini25pro --websites gomail gocalendar

    # Show browser (default is headless)
    uv run python run_benchmark.py --model gpt52 --show-browser

    # Dry run (list tasks without running)
    uv run python run_benchmark.py --model gpt52 --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from core.registry import registry, MultiRealTask
from model_configs.schema import ModelConfig
from core.harness import MultiRealHarness, MultiRealResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "model_configs"
RESULTS_DIR = Path(__file__).parent / "results"


def load_model_config(model_name: str) -> ModelConfig:
    """Load model configuration by name."""
    config_path = CONFIGS_DIR / f"{model_name}.yaml"
    if not config_path.exists():
        available = [f.stem for f in CONFIGS_DIR.glob("*.yaml") if f.stem != "schema"]
        raise ValueError(
            f"Unknown model: {model_name}. Available: {', '.join(available)}"
        )
    return ModelConfig.from_yaml(config_path)


def filter_tasks(
    tasks: list[str] | None = None,
    websites: list[str] | None = None,
) -> list[MultiRealTask]:
    """Filter tasks based on criteria."""
    if tasks:
        # Specific task IDs
        result = []
        for task_id in tasks:
            task = registry.get(task_id)
            if task:
                result.append(task)
            else:
                logger.warning(f"Task not found: {task_id}")
        return result

    # Filter by criteria
    return list(registry.filter(websites=websites))


def save_run_summary(
    results: list[MultiRealResult],
    model_config: ModelConfig,
    output_dir: Path,
) -> Path:
    """Save summary of benchmark run."""
    summary = {
        "model": model_config.name,
        "model_id": model_config.model_id,
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(results),
        "passed": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
        "total_time": sum(r.elapsed_time for r in results),
        "total_cost": sum(r.total_cost for r in results),
        "by_confidence": {
            "high": sum(1 for r in results if r.confidence == "high" and r.success),
            "medium": sum(1 for r in results if r.confidence == "medium" and r.success),
            "low": sum(1 for r in results if r.confidence == "low" and r.success),
        },
        "by_eval_method": {
            "jmespath": sum(1 for r in results if r.eval_method == "jmespath"),
            "llm_judge": sum(1 for r in results if r.eval_method == "llm_judge"),
            "hybrid": sum(1 for r in results if r.eval_method == "hybrid"),
        },
        "tasks": [r.to_dict() for r in results],
    }

    output_file = output_dir / f"summary_{model_config.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run Multi-REAL benchmark")
    parser.add_argument("--model", required=True, help="Model config name (e.g., gpt52, claude_opus45)")
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs to run")
    parser.add_argument("--websites", nargs="+", help="Filter by websites (e.g., gomail gocalendar)")
    parser.add_argument("--show-browser", action="store_true", help="Show browser window (default: headless)")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid evaluation")
    parser.add_argument("--dry-run", action="store_true", help="List tasks without running")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory")

    args = parser.parse_args()

    # Load model config
    model_config = load_model_config(args.model)
    logger.info(f"Loaded model config: {model_config.name}")

    # Filter tasks
    tasks = filter_tasks(
        tasks=args.tasks,
        websites=args.websites,
    )
    logger.info(f"Selected {len(tasks)} tasks")

    if args.dry_run:
        print(f"\nDry run - would execute {len(tasks)} tasks with {model_config.name}:\n")
        for task in tasks:
            print(f"  {task.prefixed_id}: {task.goal[:60]}...")
        return

    # Run benchmark
    harness = MultiRealHarness(
        model_config=model_config,
        results_dir=args.output_dir / "raw",
        headless=not args.show_browser,
        use_hybrid_eval=not args.no_hybrid,
    )

    results = harness.run_all(tasks)

    # Save summary
    summary_path = save_run_summary(results, model_config, args.output_dir / "aggregated")

    # Print results
    passed = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE: {model_config.name}")
    print(f"{'='*60}")
    print(f"Tasks: {len(results)}")
    print(f"Passed: {passed} ({100*passed/len(results):.1f}%)")
    print(f"Failed: {len(results) - passed}")
    print(f"Total time: {sum(r.elapsed_time for r in results):.1f}s")
    print(f"Total cost: ${sum(r.total_cost for r in results):.2f}")
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
