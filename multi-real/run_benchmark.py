#!/usr/bin/env python3
"""
Multi-REAL Benchmark Runner

Usage:
    # Run all tasks with a model
    uv run python multi-real/run_benchmark.py --model claude_opus45

    # Run specific tasks
    uv run python run_benchmark.py --model claude_opus45 --tasks multi.gocalendar-gomail-1 multi.dashdish-gomail-1

    # Run with filters
    uv run python run_benchmark.py --model gemini25pro --websites gomail gocalendar

    # Show browser (default is headless)
    uv run python run_benchmark.py --model gpt52 --show-browser

    # Dry run (list tasks without running)
    uv run python run_benchmark.py --model gpt52 --dry-run

    # Resume from checkpoint (after pause/interrupt)
    uv run python run_benchmark.py --model claude_opus45 --resume

    # Clear checkpoint and start fresh
    uv run python run_benchmark.py --model claude_opus45 --clear-checkpoint
"""

import argparse
import json
import logging
import signal
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
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """
    Manages benchmark checkpoints for pause/resume functionality.
    
    Checkpoint file structure:
    {
        "model": "claude_opus45",
        "started_at": "2026-01-15T10:00:00",
        "last_updated": "2026-01-15T12:30:00",
        "completed_tasks": ["task-1", "task-2", ...],
        "results": [<MultiRealResult dicts>]
    }
    """
    
    def __init__(self, model_name: str, checkpoint_dir: Path = CHECKPOINTS_DIR):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_dir / f"{model_name}_checkpoint.json"
        self._data: dict | None = None
        
    def exists(self) -> bool:
        """Check if a checkpoint exists for this model."""
        return self.checkpoint_file.exists()
    
    def load(self) -> dict:
        """Load checkpoint data from file."""
        if self._data is None:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file) as f:
                    self._data = json.load(f)
            else:
                self._data = {
                    "model": self.model_name,
                    "started_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "completed_tasks": [],
                    "results": [],
                }
        return self._data
    
    def save(self) -> None:
        """Save checkpoint data to file."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = self.load()
        data["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_result(self, result: MultiRealResult) -> None:
        """Add a completed task result to the checkpoint."""
        data = self.load()
        task_id = result.task_id
        
        # Avoid duplicates
        if task_id not in data["completed_tasks"]:
            data["completed_tasks"].append(task_id)
            data["results"].append(result.to_dict())
            self.save()
    
    def get_completed_task_ids(self) -> set[str]:
        """Get set of completed task IDs."""
        data = self.load()
        return set(data["completed_tasks"])
    
    def get_results(self) -> list[dict]:
        """Get list of completed results."""
        data = self.load()
        return data["results"]
    
    def clear(self) -> None:
        """Delete the checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info(f"Cleared checkpoint: {self.checkpoint_file}")
        self._data = None
    
    def get_status(self) -> str:
        """Get a human-readable status string."""
        if not self.exists():
            return "No checkpoint found"
        data = self.load()
        completed = len(data["completed_tasks"])
        started = data.get("started_at", "unknown")
        updated = data.get("last_updated", "unknown")
        return f"{completed} tasks completed (started: {started}, last updated: {updated})"


# Global checkpoint manager for signal handling
_current_checkpoint: CheckpointManager | None = None
_interrupted = False


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _interrupted
    if _interrupted:
        # Second interrupt - force exit
        logger.warning("\nForce quit. Progress saved to checkpoint.")
        sys.exit(1)
    
    _interrupted = True
    logger.warning("\nInterrupt received. Finishing current task then saving checkpoint...")
    logger.warning("Press Ctrl+C again to force quit.")


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


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
    global _current_checkpoint, _interrupted
    
    parser = argparse.ArgumentParser(description="Run Multi-REAL benchmark")
    parser.add_argument("--model", required=True, help="Model config name (e.g., gpt52, claude_opus45)")
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs to run")
    parser.add_argument("--websites", nargs="+", help="Filter by websites (e.g., gomail gocalendar)")
    parser.add_argument("--show-browser", action="store_true", help="Show browser window (default: headless)")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid evaluation")
    parser.add_argument("--dry-run", action="store_true", help="List tasks without running")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--clear-checkpoint", action="store_true", help="Clear checkpoint and start fresh")
    parser.add_argument("--checkpoint-status", action="store_true", help="Show checkpoint status and exit")

    args = parser.parse_args()

    # Load model config
    model_config = load_model_config(args.model)
    logger.info(f"Loaded model config: {model_config.name}")
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager(args.model)
    _current_checkpoint = checkpoint
    
    # Handle checkpoint commands
    if args.checkpoint_status:
        print(f"\nCheckpoint status for {args.model}:")
        print(f"  {checkpoint.get_status()}")
        return
    
    if args.clear_checkpoint:
        checkpoint.clear()
        print(f"Checkpoint cleared for {args.model}")
        if not args.tasks and not args.websites:
            return  # Just clearing, not running

    # Filter tasks
    tasks = filter_tasks(
        tasks=args.tasks,
        websites=args.websites,
    )
    logger.info(f"Selected {len(tasks)} tasks")
    
    # Handle resume logic
    completed_ids = set()
    previous_results = []
    
    if args.resume and checkpoint.exists():
        completed_ids = checkpoint.get_completed_task_ids()
        previous_results = checkpoint.get_results()
        logger.info(f"Resuming from checkpoint: {len(completed_ids)} tasks already completed")
        
        # Filter out already completed tasks
        original_count = len(tasks)
        tasks = [t for t in tasks if t.prefixed_id not in completed_ids]
        logger.info(f"Remaining tasks: {len(tasks)} (skipping {original_count - len(tasks)} completed)")
        
        if not tasks:
            print(f"\nAll tasks already completed! Use --clear-checkpoint to start fresh.")
            # Still show results
            if previous_results:
                passed = sum(1 for r in previous_results if r.get("success"))
                print(f"Previous run: {passed}/{len(previous_results)} passed")
            return
    elif checkpoint.exists() and not args.clear_checkpoint:
        # Checkpoint exists but --resume not specified
        logger.warning(f"Checkpoint exists for {args.model}. Use --resume to continue or --clear-checkpoint to start fresh.")
        print(f"\nCheckpoint found: {checkpoint.get_status()}")
        print("Options:")
        print(f"  --resume          : Continue from where you left off")
        print(f"  --clear-checkpoint: Start fresh (deletes progress)")
        return

    if args.dry_run:
        print(f"\nDry run - would execute {len(tasks)} tasks with {model_config.name}:\n")
        for task in tasks:
            status = "(completed)" if task.prefixed_id in completed_ids else ""
            print(f"  {task.prefixed_id}: {task.goal[:60]}... {status}")
        return

    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()

    # Run benchmark
    harness = MultiRealHarness(
        model_config=model_config,
        results_dir=args.output_dir / "raw",
        headless=not args.show_browser,
        use_hybrid_eval=not args.no_hybrid,
    )

    # Run tasks one by one with checkpoint saves
    results = []
    total_tasks = len(tasks)
    
    logger.info("=" * 60)
    logger.info(f"Multi-REAL Benchmark Run")
    logger.info(f"Model: {model_config.name}")
    logger.info(f"Tasks: {total_tasks}" + (f" (+{len(completed_ids)} already completed)" if completed_ids else ""))
    logger.info("=" * 60)
    
    for i, task in enumerate(tasks):
        # Check for interrupt
        if _interrupted:
            logger.info(f"Stopping after {i} tasks due to interrupt. Progress saved.")
            break
        
        logger.info(f"\n[{i+1}/{total_tasks}] Starting: {task.prefixed_id}")
        logger.info("-" * 40)
        
        result = harness.run_task(task)
        results.append(result)
        
        # Save to checkpoint after each task
        checkpoint.add_result(result)
        
        # Log result
        status = "PASS" if result.success else "FAIL"
        logger.info(f"[{i+1}/{total_tasks}] {status}: {task.id}")
        
        # Running summary
        passed = sum(1 for r in results if r.success)
        total_completed = len(results) + len(completed_ids)
        total_passed = passed + sum(1 for r in previous_results if r.get("success"))
        logger.info(f"    Running: {total_passed}/{total_completed} passed")

    # Combine with previous results for final summary
    # Convert previous dict results back to MultiRealResult objects
    all_results: list[MultiRealResult] = []
    for r in previous_results:
        all_results.append(MultiRealResult(**r))
    all_results.extend(results)
    
    # Save summary
    summary_path = save_run_summary(all_results, model_config, args.output_dir / "aggregated")
    
    # Clear checkpoint on successful completion (all tasks done)
    if not _interrupted and len(results) == total_tasks:
        checkpoint.clear()
        logger.info("Benchmark complete - checkpoint cleared")

    # Print results
    total_passed = sum(1 for r in all_results if r.success)
    total_count = len(all_results)
    
    print(f"\n{'='*60}")
    if _interrupted:
        print(f"BENCHMARK PAUSED: {model_config.name}")
        print(f"Run with --resume to continue")
    else:
        print(f"BENCHMARK COMPLETE: {model_config.name}")
    print(f"{'='*60}")
    print(f"Tasks: {total_count}")
    print(f"Passed: {total_passed} ({100*total_passed/total_count:.1f}%)" if total_count > 0 else "Passed: 0")
    print(f"Failed: {total_count - total_passed}")
    
    session_results = results
    if session_results:
        print(f"\nThis session:")
        print(f"  Tasks run: {len(session_results)}")
        print(f"  Time: {sum(r.elapsed_time for r in session_results):.1f}s")
        print(f"  Cost: ${sum(r.total_cost for r in session_results):.2f}")
    
    print(f"\nResults saved to: {summary_path}")
    if _interrupted:
        print(f"Checkpoint saved to: {checkpoint.checkpoint_file}")


if __name__ == "__main__":
    main()
