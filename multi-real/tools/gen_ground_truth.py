#!/usr/bin/env python3
"""
Generate Synthetic Ground Truth

Runs a model on all benchmark tasks to collect finish states as ground truth.
Uses LLM judge with vision (screenshots + a-trees) to validate success.

Usage:
    # Generate using GPT-4o
    uv run python tools/gen_ground_truth.py --model gpt52

    # Generate using Claude Opus (recommended for quality)
    uv run python tools/gen_ground_truth.py --model claude_opus45

    # Specify minimum LLM confidence threshold
    uv run python tools/gen_ground_truth.py --model gpt52 --min-confidence 0.85

    # Use legacy JSON-only validation (not recommended)
    uv run python tools/gen_ground_truth.py --model gpt52 --no-vision
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src directory for browsergym imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.registry import registry
from model_configs.schema import ModelConfig
from core.harness import MultiRealHarness
from core.validator import HybridValidator

# Import ExpResult for loading step data
from agisdk.REAL.browsergym.experiments.loop import ExpResult
from agisdk.REAL.browsergym.utils.obs import flatten_axtree_to_str

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "final_states" / "synthetic"
REPORT_DIR = BASE_DIR / "results" / "synthetic_gt_reports"


def load_model_config(model_name: str) -> ModelConfig:
    """Load model configuration."""
    config_path = BASE_DIR / "model_configs" / f"{model_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Model config not found: {model_name}")
    return ModelConfig.from_yaml(config_path)


def image_to_base64_url(image: np.ndarray | Image.Image) -> str:
    """Convert a numpy array or PIL Image to a base64 encoded data URL."""
    import base64
    import io

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


def extract_vision_data_from_experiment(exp_dir: str | Path) -> tuple[list[tuple[str, str]], str | None]:
    """
    Extract screenshot and a-tree from experiment results.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        (screenshots, axtree_txt) where:
        - screenshots: List of (app_name, base64_data_url) tuples
        - axtree_txt: Formatted accessibility tree text, or None if unavailable
    """
    exp_dir = Path(exp_dir)
    screenshots = []
    axtree_txt = None

    try:
        exp_result = ExpResult(exp_dir)
        n_steps = exp_result.summary_info.get("n_steps", 0)

        if n_steps == 0:
            logger.warning(f"No steps found in experiment: {exp_dir}")
            return screenshots, axtree_txt

        # Get final step's screenshot
        final_step = n_steps - 1
        try:
            screenshot = exp_result.get_screenshot(final_step)
            if screenshot:
                # Convert PIL Image to numpy array if needed
                if isinstance(screenshot, Image.Image):
                    screenshot_array = np.array(screenshot)
                else:
                    screenshot_array = screenshot
                base64_url = image_to_base64_url(screenshot_array)
                screenshots.append(("final_state", base64_url))
        except FileNotFoundError:
            logger.warning(f"Screenshot not found for step {final_step}")

        # Get final step's a-tree
        try:
            step_info = exp_result.get_step_info(final_step)
            if step_info and step_info.obs:
                # Check for pre-formatted a-tree text
                if "axtree_txt" in step_info.obs and step_info.obs["axtree_txt"]:
                    axtree_txt = step_info.obs["axtree_txt"]
                # Otherwise try to format from raw a-tree object
                elif "axtree_object" in step_info.obs and step_info.obs["axtree_object"]:
                    axtree_txt = flatten_axtree_to_str(step_info.obs["axtree_object"])
        except Exception as e:
            logger.warning(f"Failed to extract a-tree: {e}")

    except Exception as e:
        logger.warning(f"Failed to load experiment data from {exp_dir}: {e}")

    return screenshots, axtree_txt


def validate_with_vision(
    task_goal: str,
    finish_state: dict,
    evals: list[dict],
    validator: HybridValidator,
    exp_dir: str | Path,
) -> tuple[bool, float, str]:
    """
    Validate task success using screenshots and a-tree (vision-based).

    This provides more reliable validation than truncated JSON by giving
    the LLM judge visual evidence and semantic UI state.

    Raises:
        RuntimeError: If screenshots cannot be extracted from the experiment

    Returns: (is_valid, confidence, reasoning)
    """
    # Extract screenshots and a-tree from experiment
    screenshots, axtree_txt = extract_vision_data_from_experiment(exp_dir)

    if not screenshots:
        raise RuntimeError(
            f"No screenshots found in experiment directory: {exp_dir}. "
            "Vision-based validation requires screenshots to be captured during task execution."
        )

    # Use vision-based evaluation
    result = validator.evaluate_with_vision(
        task_goal=task_goal,
        screenshots=screenshots,
        axtree_txt=axtree_txt,
        evals=evals,
        finish_state=finish_state,  # Also pass finish_state for concrete evidence extraction
    )

    return (
        result.get("overall_pass", False),
        result.get("confidence", 0.0),
        result.get("reasoning", "")
    )


def validate_finish_state_with_llm(
    task_goal: str,
    finish_state: dict,
    evals: list[dict],
    validator: HybridValidator,
) -> tuple[bool, float, str]:
    """
    Validate task success using truncated JSON (legacy method).

    Note: finish_state can be hundreds of thousands of lines (full action history
    from all websites). HybridValidator truncates to 10k chars before sending to LLM.

    Returns: (is_valid, confidence, reasoning)
    """
    result = validator.evaluate(task_goal, finish_state, evals)

    return (
        result.get("overall_pass", False),
        result.get("confidence", 0.0),
        result.get("reasoning", "")
    )


def generate_synthetic_ground_truth(
    model_name: str,
    min_confidence: float = 0.80,
    force_rerun: bool = False,
    use_vision: bool = True,
) -> dict:
    """
    Generate synthetic ground truth by running model on all tasks.

    Process:
    1. Run model on all tasks (or subset if some already have synthetic GT)
    2. Collect finish states from summary_info.json
       - Format: {"website_id": {"actionhistory": [...], "emails": [...], ...}, ...}
       - Captured via _get_multi_app_finish_json() from /finish endpoints
    3. Validate each finish state with LLM judge using:
       - Vision mode (default): Screenshots + a-tree + concrete evidence
       - Legacy mode: Truncated JSON (10k chars)
    4. Save high-confidence states (≥min_confidence) to final_states/synthetic/

    Args:
        model_name: Model config name to use
        min_confidence: Minimum LLM judge confidence to accept state
        force_rerun: Re-run tasks even if synthetic GT already exists
        use_vision: Use vision-based validation with screenshots and a-trees (default: True)

    Returns:
        Summary dict with statistics
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model config
    model_config = load_model_config(model_name)
    logger.info(f"Using model: {model_config.name}")
    logger.info(f"Validation mode: {'vision (screenshots + a-tree)' if use_vision else 'legacy (truncated JSON)'}")

    # Initialize validator
    validator = HybridValidator()

    # Get all tasks
    all_tasks = list(registry.all())
    logger.info(f"Total tasks: {len(all_tasks)}")

    # Filter tasks that don't already have synthetic GT (unless force_rerun)
    tasks_to_run = []
    if not force_rerun:
        for task in all_tasks:
            synthetic_path = OUTPUT_DIR / f"{task.id}.json"
            if not synthetic_path.exists():
                tasks_to_run.append(task)
    else:
        tasks_to_run = all_tasks

    logger.info(f"Tasks to run: {len(tasks_to_run)}")

    if not tasks_to_run:
        logger.info("All tasks already have synthetic ground truth. Use --force-rerun to regenerate.")
        return {"status": "skipped", "reason": "all tasks have synthetic GT"}

    # Run harness (without hybrid eval since we'll validate separately)
    harness = MultiRealHarness(
        model_config=model_config,
        results_dir=Path(__file__).parent / "results" / "synthetic_gt_runs",
        headless=True,
        use_hybrid_eval=False,  # We'll validate separately
    )

    results = harness.run_all(tasks_to_run, save_results=True)

    # Process results and validate with LLM
    stats = {
        "total_tasks_run": len(results),
        "completed": 0,
        "failed": 0,
        "validated_by_llm": 0,
        "rejected_by_llm": 0,
        "saved_as_gt": 0,
        "validation_mode": "vision" if use_vision else "legacy_json",
        "validation_details": [],
    }

    for result in results:
        task_id_clean = result.task_id.replace("multi.", "")
        task = registry.get(task_id_clean)

        if not task:
            logger.warning(f"Task not found in registry: {task_id_clean}")
            continue

        # Check if task completed and has finish state
        if result.error or not result.finish_state:
            stats["failed"] += 1
            stats["validation_details"].append({
                "task_id": task_id_clean,
                "status": "failed",
                "reason": result.error or "no finish state",
            })
            continue

        stats["completed"] += 1

        # Validate with LLM judge
        if use_vision:
            if not result.exp_dir:
                raise RuntimeError(
                    f"Vision validation enabled but no exp_dir for task {task_id_clean}. "
                    "Cannot extract screenshots and a-tree."
                )
            is_valid, confidence, reasoning = validate_with_vision(
                task.goal,
                result.finish_state,
                task.evals,
                validator,
                result.exp_dir,
            )
        else:
            # Legacy JSON-only validation (not recommended)
            is_valid, confidence, reasoning = validate_finish_state_with_llm(
                task.goal,
                result.finish_state,
                task.evals,
                validator,
            )

        validation_detail = {
            "task_id": task_id_clean,
            "status": "validated" if is_valid else "rejected",
            "llm_confidence": confidence,
            "reasoning": reasoning,
            "min_confidence": min_confidence,
        }

        if is_valid and confidence >= min_confidence:
            # Save as synthetic ground truth
            # Format: {"website_id": {"actionhistory": [...], ...}, ...}
            # This is the same format as manual ground truth in final_states/manual/
            output_path = OUTPUT_DIR / f"{task_id_clean}.json"
            with open(output_path, "w") as f:
                json.dump(result.finish_state, f, indent=2)

            logger.info(f"✓ {task_id_clean}: Saved as GT (confidence: {confidence:.2f})")
            stats["validated_by_llm"] += 1
            stats["saved_as_gt"] += 1
            validation_detail["saved"] = True
        else:
            logger.warning(f"✗ {task_id_clean}: Rejected (confidence: {confidence:.2f})")
            stats["rejected_by_llm"] += 1
            validation_detail["saved"] = False

        stats["validation_details"].append(validation_detail)

    # Save report
    report_path = REPORT_DIR / f"generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("SYNTHETIC GROUND TRUTH GENERATION COMPLETE")
    print("="*70)
    print(f"Total tasks run: {stats['total_tasks_run']}")
    print(f"Completed successfully: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Validated by LLM: {stats['validated_by_llm']}")
    print(f"Rejected by LLM: {stats['rejected_by_llm']}")
    print(f"Saved as ground truth: {stats['saved_as_gt']}")
    print(f"\nReport saved to: {report_path}")
    print(f"Ground truth states saved to: {OUTPUT_DIR}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ground truth")
    parser.add_argument("--model", required=True, help="Model to use (e.g., gpt52, claude_opus45)")
    parser.add_argument("--min-confidence", type=float, default=0.80, help="Minimum LLM confidence (0.0-1.0)")
    parser.add_argument("--force-rerun", action="store_true", help="Re-run tasks even if GT exists")
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision-based validation (use truncated JSON instead - not recommended)"
    )

    args = parser.parse_args()

    generate_synthetic_ground_truth(
        model_name=args.model,
        min_confidence=args.min_confidence,
        force_rerun=args.force_rerun,
        use_vision=not args.no_vision,
    )


if __name__ == "__main__":
    main()
