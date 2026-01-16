"""
Multi-REAL Benchmark Harness

Wraps the core REAL harness to run multi-app benchmark tasks.
"""

import base64
import gzip
import io
import jmespath
import json
import logging
import os
import pickle
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

# Add multi-real and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agisdk.REAL.browsergym.experiments.loop import EnvArgs, ExpArgs
from agisdk.REAL.demo_agent.basic_agent import DemoAgentArgs

from core.registry import MultiRealTask, registry
from model_configs.schema import ModelConfig

# HybridValidator import
try:
    from core.validator import HybridValidator
    HYBRID_VALIDATOR_AVAILABLE = True
except (ImportError, AttributeError):
    HYBRID_VALIDATOR_AVAILABLE = False
    HybridValidator = None

logger = logging.getLogger(__name__)


# Retry configuration
RETRYABLE_ERRORS = [
    "InternalServerError",
    "500",
    "502",
    "503",
    "504",
    "APIError",
    "RateLimitError",
    "Timeout",
    "ConnectionError",
    "overloaded",
]


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 5.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0


def is_retryable_error(error: str | Exception) -> bool:
    """Check if an error is retryable (API errors, rate limits, etc.)."""
    error_str = str(error).lower()
    for pattern in RETRYABLE_ERRORS:
        if pattern.lower() in error_str:
            return True
    return False


def calculate_retry_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay before next retry with exponential backoff."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    return min(delay, config.max_delay)


class ProgressMonitor:
    """Monitor experiment progress by watching the experiment directory."""

    def __init__(
        self,
        exp_dir: Path,
        task_id: str,
        check_interval: float = 5.0,
        stall_timeout: float = 300.0,  # 5 minutes without progress = stall
    ):
        self.exp_dir = Path(exp_dir)
        self.task_id = task_id
        self.check_interval = check_interval
        self.stall_timeout = stall_timeout
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_step = -1
        self._last_progress_time = time.time()
        self._stalled = False

    def start(self) -> None:
        """Start monitoring in a background thread."""
        self._stop_event.clear()
        self._last_progress_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def is_stalled(self) -> bool:
        """Check if the experiment appears to be stalled."""
        return self._stalled

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                current_step = self._get_current_step()
                current_time = time.time()

                if current_step > self._last_step:
                    # Progress was made
                    elapsed = current_time - self._last_progress_time
                    logger.info(
                        f"[{self.task_id}] Step {current_step} "
                        f"(+{elapsed:.1f}s since last step)"
                    )
                    self._last_step = current_step
                    self._last_progress_time = current_time
                    self._stalled = False
                else:
                    # Check for stall
                    time_since_progress = current_time - self._last_progress_time
                    if time_since_progress > self.stall_timeout:
                        if not self._stalled:
                            logger.warning(
                                f"[{self.task_id}] STALL DETECTED: "
                                f"No progress for {time_since_progress:.0f}s "
                                f"(stuck at step {self._last_step})"
                            )
                            self._stalled = True
                    elif time_since_progress > 60:
                        # Warn at 1 minute
                        logger.info(
                            f"[{self.task_id}] Waiting... "
                            f"{time_since_progress:.0f}s at step {self._last_step}"
                        )

            except Exception as e:
                logger.debug(f"Monitor error: {e}")

            self._stop_event.wait(self.check_interval)

    def _get_current_step(self) -> int:
        """Get the current step number from experiment directory."""
        if not self.exp_dir.exists():
            return -1

        # Look for step_*.pkl.gz files
        step_files = list(self.exp_dir.glob("step_*.pkl.gz"))
        if not step_files:
            # Also check for screenshot files as backup
            step_files = list(self.exp_dir.glob("screenshot_step_*.png"))

        if not step_files:
            return 0

        # Extract step numbers
        steps = []
        for f in step_files:
            match = re.search(r"step_(\d+)", f.name)
            if match:
                steps.append(int(match.group(1)))

        return max(steps) if steps else 0


@dataclass
class MultiRealResult:
    """Result of a single multi-app task execution."""
    task_id: str
    model_name: str
    success: bool
    score: float

    # Evaluation details
    jmespath_results: list[dict] = field(default_factory=list)
    llm_judge_results: dict | None = None
    eval_method: str = "jmespath"  # "jmespath", "llm_judge", or "hybrid"
    confidence: str = "high"  # "high", "medium", "low"

    # Execution metadata
    elapsed_time: float = 0.0
    num_steps: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # State capture
    finish_state: dict = field(default_factory=dict)
    error: str | None = None

    # Retry tracking
    retry_count: int = 0

    # Timestamps
    started_at: str = ""
    completed_at: str = ""

    # Experiment directory for debugging
    exp_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "success": self.success,
            "score": self.score,
            "jmespath_results": self.jmespath_results,
            "llm_judge_results": self.llm_judge_results,
            "eval_method": self.eval_method,
            "confidence": self.confidence,
            "elapsed_time": self.elapsed_time,
            "num_steps": self.num_steps,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "finish_state": self.finish_state,
            "error": self.error,
            "retry_count": self.retry_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exp_dir": self.exp_dir,
        }


class MultiRealHarness:
    """Harness for running multi-app benchmark tasks."""

    def __init__(
        self,
        model_config: ModelConfig,
        results_dir: Path | None = None,
        headless: bool = True,
        use_hybrid_eval: bool = True,
        retry_config: RetryConfig | None = None,
        task_timeout: float = 1800.0,  # 30 minutes default
        stall_timeout: float = 300.0,  # 5 minutes without progress = stall
        progress_check_interval: float = 10.0,
    ):
        self.model_config = model_config
        self.results_dir = results_dir or Path(__file__).parent.parent / "results" / "raw"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.use_hybrid_eval = use_hybrid_eval and HYBRID_VALIDATOR_AVAILABLE

        # Retry and timeout configuration
        self.retry_config = retry_config or RetryConfig()
        self.task_timeout = task_timeout
        self.stall_timeout = stall_timeout
        self.progress_check_interval = progress_check_interval

        if self.use_hybrid_eval:
            # HybridValidator uses Gemini 2.5 Pro via Vertex AI
            # Authentication is handled via gcloud auth (Application Default Credentials)
            try:
                self.hybrid_validator = HybridValidator()
                logger.info("HybridValidator initialized with Gemini 2.5 Pro via Vertex AI")
            except Exception as e:
                logger.warning(f"Failed to initialize HybridValidator: {e}. Hybrid evaluation disabled.")
                logger.warning("Ensure you've run 'gcloud auth application-default login'")
                self.use_hybrid_eval = False
                self.hybrid_validator = None
        elif use_hybrid_eval and not HYBRID_VALIDATOR_AVAILABLE:
            logger.warning("HybridValidator not available. Hybrid evaluation disabled.")
            self.hybrid_validator = None
        else:
            self.hybrid_validator = None

    def _create_agent_args(self) -> DemoAgentArgs:
        """Create agent args from model config.

        Note: DemoAgentArgs infers provider from model_name prefix:
        - gpt-*, o1*, o3* -> OpenAI
        - claude-*, sonnet-* -> Anthropic
        - openrouter/* -> OpenRouter
        - local/* -> Local vLLM
        """
        # Get API key from environment or use direct value
        api_key = os.environ.get(self.model_config.api_key_env)
        if not api_key:
            # If api_key_env looks like an actual key (starts with sk-), use it directly
            if self.model_config.api_key_env.startswith("sk-"):
                api_key = self.model_config.api_key_env
            else:
                raise ValueError(
                    f"API key not found in environment variable: {self.model_config.api_key_env}"
                )

        # Map API key to provider-specific field based on model_id prefix
        api_key_kwargs = {}
        model_id = self.model_config.model_id
        if model_id.startswith("gpt-") or model_id.startswith("o1") or model_id.startswith("o3"):
            api_key_kwargs["openai_api_key"] = api_key
        elif model_id.startswith("claude-") or model_id.startswith("sonnet-"):
            api_key_kwargs["anthropic_api_key"] = api_key
        elif model_id.startswith("openrouter/"):
            api_key_kwargs["openrouter_api_key"] = api_key
        # For Google/Gemini models, they may need special handling

        # Filter extra kwargs to only include valid DemoAgentArgs fields
        valid_extra_fields = {
            "chat_mode", "demo_mode", "use_html", "use_axtree", "use_screenshot",
            "system_message_handling", "thinking_budget",
            "openrouter_site_url", "openrouter_site_name",
        }
        filtered_extra = {k: v for k, v in self.model_config.extra.items() if k in valid_extra_fields}

        return DemoAgentArgs(
            model_name=self.model_config.model_id,
            **api_key_kwargs,
            **filtered_extra,
        )

    def _create_env_args(self, task: MultiRealTask) -> EnvArgs:
        """Create environment args for a task.

        Tasks are loaded from the 'multi' version directory which symlinks to
        multi-real/tasks/. The task_name format is 'browsergym/multi.{task_id}'
        (without 'webclones.' prefix since 'multi' is itself a version).
        """
        return EnvArgs(
            task_name=f"browsergym/multi.{task.id}",  # Multi-app task: multi.{task_id}
            task_seed=42,  # Fixed seed for reproducibility
            max_steps=self.model_config.max_steps,  # From model config
            headless=self.headless,
            viewport={"width": 1280, "height": 720},
        )

    def _evaluate_with_jmespath(
        self,
        task: MultiRealTask,
        finish_state: dict,
    ) -> tuple[list[dict], bool]:
        """
        Evaluate task result using JMESPath queries.

        Returns: (results_list, all_passed)
        """
        jmespath_results = []
        all_passed = True

        for eval_item in task.evals:
            if eval_item.get("type") != "jmespath":
                continue

            query = eval_item["query"]
            expected = eval_item.get("expected_value")

            try:
                # Use jmespath directly to evaluate the query
                actual = jmespath.search(query, finish_state)

                # Check if it matches expected
                if expected is None:
                    passed = actual is not None
                elif isinstance(expected, bool):
                    passed = actual == expected
                else:
                    passed = actual == expected

                jmespath_results.append({
                    "description": eval_item.get("description", ""),
                    "query": query,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                })

                if not passed:
                    all_passed = False

            except Exception as e:
                logger.warning(f"JMESPath query failed: {query}. Error: {e}")
                jmespath_results.append({
                    "description": eval_item.get("description", ""),
                    "query": query,
                    "expected": expected,
                    "actual": None,
                    "passed": False,
                    "error": str(e),
                })
                all_passed = False

        return jmespath_results, all_passed

    def _evaluate_with_hybrid(
        self,
        task: MultiRealTask,
        finish_state: dict,
        screenshots: list[tuple[str, str]] | None = None,
        axtree_txt: str | None = None,
    ) -> tuple[bool, float, dict]:
        """
        Evaluate task result using hybrid JMESPath + LLM judge.

        Args:
            task: The task being evaluated
            finish_state: The final state JSON from /finish endpoints
            screenshots: Optional list of (app_name, base64_data_url) for vision eval
            axtree_txt: Optional accessibility tree text for vision eval

        Returns: (success, score, eval_details)
        """
        # First pass: JMESPath evaluation
        jmespath_results, all_passed = self._evaluate_with_jmespath(task, finish_state)

        eval_details = {
            "jmespath_results": jmespath_results,
            "llm_judge_results": None,
            "eval_method": "jmespath",
            "confidence": "high",
        }

        # If JMESPath passes, we're done (high confidence)
        if all_passed:
            return True, float(task.points), eval_details

        # If JMESPath fails and hybrid is enabled, try LLM judge
        if self.use_hybrid_eval and self.hybrid_validator:
            try:
                # Prefer vision evaluation if screenshots are available
                if screenshots:
                    logger.info(f"Using LLM vision evaluation with {len(screenshots)} screenshot(s)")
                    llm_results = self.hybrid_validator.evaluate_with_vision(
                        task_goal=task.goal,
                        screenshots=screenshots,
                        axtree_txt=axtree_txt,
                        evals=task.evals,
                        finish_state=finish_state,
                    )
                    eval_details["eval_method_detail"] = "llm_vision"
                else:
                    # Fall back to text-only evaluation
                    logger.info("Using LLM text-only evaluation (no screenshots available)")
                    llm_results = self.hybrid_validator.evaluate(
                        task_goal=task.goal,
                        finish_state=finish_state,
                        evals=task.evals,
                    )
                    eval_details["eval_method_detail"] = "llm_text"

                eval_details["llm_judge_results"] = llm_results

                llm_passed = llm_results.get("overall_pass", False)
                llm_confidence = llm_results.get("confidence", 0.5)

                if llm_passed:
                    # JMESPath failed but LLM passed - likely query bug
                    eval_details["eval_method"] = "llm_judge"
                    eval_details["confidence"] = "medium" if llm_confidence >= 0.7 else "low"
                    return True, float(task.points), eval_details
                else:
                    # Both failed - high confidence failure
                    eval_details["eval_method"] = "hybrid"
                    eval_details["confidence"] = "high" if llm_confidence >= 0.7 else "medium"
                    return False, 0.0, eval_details

            except Exception as e:
                logger.error(f"LLM judge failed: {e}")
                # Fall through to JMESPath result with lower confidence

        # JMESPath failed, no hybrid or hybrid failed - medium confidence
        eval_details["confidence"] = "medium"
        return False, 0.0, eval_details

    def _update_summary_with_eval_results(
        self,
        summary_path: Path,
        result: MultiRealResult,
        task: MultiRealTask,
    ) -> None:
        """
        Update summary_info.json with evaluation results and remove bulky finish_state.

        This makes the summary file much more useful for debugging by showing
        which queries passed/failed instead of just the raw state.
        """
        try:
            with open(summary_path) as f:
                summary_info = json.load(f)

            # Remove the bulky finish_state (it's already in finish_state.json)
            summary_info.pop("finish_state", None)

            # Add evaluation results
            summary_info["eval_results"] = {
                "success": result.success,
                "score": result.score,
                "eval_method": result.eval_method,
                "confidence": result.confidence,
                "queries": result.jmespath_results,
                "llm_judge": result.llm_judge_results,
            }

            # Add task metadata for context
            summary_info["task"] = {
                "id": task.prefixed_id,
                "goal": task.goal,
                "websites": task.website_ids,
                "points": task.points,
            }

            # Add a summary line for quick viewing
            passed = sum(1 for q in result.jmespath_results if q.get("passed"))
            total = len(result.jmespath_results)
            summary_info["eval_summary"] = f"{passed}/{total} queries passed"

            with open(summary_path, "w") as f:
                json.dump(summary_info, f, indent=2)

            logger.debug(f"Updated summary_info.json with eval results: {passed}/{total} passed")

        except Exception as e:
            logger.warning(f"Failed to update summary_info.json with eval results: {e}")

    def _extract_final_step_data(
        self,
        exp_dir: Path,
    ) -> tuple[list[tuple[str, str]], str | None]:
        """
        Extract screenshot and axtree from the final step for LLM vision evaluation.

        Returns:
            (screenshots, axtree_txt) where:
            - screenshots: list of (app_name, base64_data_url) tuples
            - axtree_txt: formatted accessibility tree text or None
        """
        screenshots: list[tuple[str, str]] = []
        axtree_txt: str | None = None

        try:
            # Find the last step file
            step_files = sorted(exp_dir.glob("step_*.pkl.gz"))
            if not step_files:
                logger.warning("No step files found for vision evaluation")
                return screenshots, axtree_txt

            last_step_file = step_files[-1]
            logger.debug(f"Loading final step from {last_step_file}")

            with gzip.open(last_step_file, "rb") as f:
                step_info = pickle.load(f)

            # Extract screenshot
            if step_info.obs and "screenshot" in step_info.obs:
                screenshot_array = step_info.obs["screenshot"]
                # Convert numpy array to base64
                from PIL import Image
                img = Image.fromarray(screenshot_array)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                screenshots.append(("final_state", f"data:image/png;base64,{base64_data}"))
                logger.debug("Extracted final screenshot for vision evaluation")

            # Extract axtree
            if step_info.obs and "axtree_object" in step_info.obs:
                axtree_obj = step_info.obs["axtree_object"]
                if isinstance(axtree_obj, dict):
                    # Format the axtree as text
                    axtree_txt = json.dumps(axtree_obj, indent=2)[:20000]  # Truncate if too long
                elif isinstance(axtree_obj, str):
                    axtree_txt = axtree_obj[:20000]
                logger.debug("Extracted axtree for vision evaluation")

        except Exception as e:
            logger.warning(f"Failed to extract final step data: {e}")

        return screenshots, axtree_txt

    def _run_experiment_with_monitoring(
        self,
        exp_args: ExpArgs,
        task_id: str,
    ) -> tuple[bool, str | None]:
        """
        Run experiment with progress monitoring.

        Note: We can't use threading for timeout because Playwright uses greenlets
        and must run on the same thread where it was initialized.

        Returns: (success, error_message)
        """
        monitor = ProgressMonitor(
            exp_dir=exp_args.exp_dir,
            task_id=task_id,
            check_interval=self.progress_check_interval,
            stall_timeout=self.stall_timeout,
        )

        monitor.start()
        start_time = time.time()
        try:
            # Run directly on main thread (required for Playwright)
            exp_args.run()

            # Check if we exceeded soft timeout (for logging purposes)
            elapsed = time.time() - start_time
            if elapsed > self.task_timeout:
                logger.warning(
                    f"[{task_id}] Task completed but exceeded timeout "
                    f"({elapsed:.0f}s > {self.task_timeout}s)"
                )

            return True, None

        except KeyboardInterrupt:
            # Allow manual interruption
            error_msg = f"Task interrupted by user after {time.time() - start_time:.0f}s"
            logger.warning(f"[{task_id}] {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{task_id}] Experiment error: {error_msg}")
            return False, error_msg

        finally:
            monitor.stop()

    def run_task(self, task: MultiRealTask) -> MultiRealResult:
        """Run a single task with retries and return results."""
        started_at = datetime.now().isoformat()
        result = MultiRealResult(
            task_id=task.prefixed_id,
            model_name=self.model_config.name,
            success=False,
            score=0.0,
            started_at=started_at,
        )

        last_error: str | None = None
        attempt = 0

        while attempt <= self.retry_config.max_retries:
            if attempt > 0:
                delay = calculate_retry_delay(attempt - 1, self.retry_config)
                logger.info(
                    f"[{task.id}] Retry {attempt}/{self.retry_config.max_retries} "
                    f"after {delay:.1f}s delay..."
                )
                time.sleep(delay)

            try:
                # Create agent and environment args
                agent_args = self._create_agent_args()
                env_args = self._create_env_args(task)

                # Create experiment args
                # Enable screenshot and step saving for LLM vision evaluation
                exp_args = ExpArgs(
                    agent_args=agent_args,
                    env_args=env_args,
                    save_screenshot=True,
                    save_step_info_pkl=True,
                )

                # Prepare experiment (creates directory, saves args)
                exp_args.prepare(exp_root=self.results_dir / task.id)

                # Store exp_dir for later
                result.exp_dir = str(exp_args.exp_dir)

                # Run the experiment with monitoring
                logger.info(
                    f"[{task.id}] Starting experiment "
                    f"(attempt {attempt + 1}/{self.retry_config.max_retries + 1})"
                )
                success, run_error = self._run_experiment_with_monitoring(
                    exp_args, task.id
                )

                if run_error:
                    last_error = run_error
                    if is_retryable_error(run_error):
                        attempt += 1
                        continue
                    else:
                        # Non-retryable error
                        result.error = run_error
                        break

                # Load results from summary_info.json
                summary_path = exp_args.exp_dir / "summary_info.json"
                if not summary_path.exists():
                    raise FileNotFoundError(f"summary_info.json not found at {summary_path}")

                with open(summary_path) as f:
                    summary_info = json.load(f)

                # Extract finish state
                finish_state = summary_info.get("finish_state", {})
                result.finish_state = finish_state

                # Extract metadata
                result.elapsed_time = summary_info.get("stats.elapsed_time", 0.0)
                result.num_steps = summary_info.get("n_steps", 0)

                # Extract token usage and calculate cost
                input_tokens = summary_info.get("stats.cum_input_token", 0)
                output_tokens = summary_info.get("stats.cum_output_token", 0)
                result.total_tokens = input_tokens + output_tokens
                result.total_cost = (
                    input_tokens * self.model_config.input_price_per_1k / 1000 +
                    output_tokens * self.model_config.output_price_per_1k / 1000
                )

                # Check if there was an error in the experiment
                err_msg = summary_info.get("err_msg")
                if err_msg:
                    logger.warning(f"[{task.id}] Experiment error: {err_msg}")
                    if is_retryable_error(err_msg):
                        last_error = err_msg
                        attempt += 1
                        continue
                    else:
                        result.error = err_msg

                # Extract screenshots and axtree for LLM vision evaluation
                screenshots, axtree_txt = self._extract_final_step_data(exp_args.exp_dir)

                # Evaluate with hybrid validator
                if finish_state:
                    eval_success, score, eval_details = self._evaluate_with_hybrid(
                        task, finish_state, screenshots=screenshots, axtree_txt=axtree_txt
                    )
                    result.success = eval_success
                    result.score = score
                    result.jmespath_results = eval_details["jmespath_results"]
                    result.llm_judge_results = eval_details.get("llm_judge_results")
                    result.eval_method = eval_details["eval_method"]
                    result.confidence = eval_details["confidence"]
                else:
                    logger.warning(f"[{task.id}] No finish state available")
                    result.success = False
                    result.score = 0.0
                    result.confidence = "low"

                # Update summary_info.json with evaluation results (remove bulky finish_state)
                self._update_summary_with_eval_results(summary_path, result, task)

                # Success - exit retry loop
                result.retry_count = attempt
                logger.info(
                    f"[{task.id}] Completed in {result.num_steps} steps, "
                    f"{result.elapsed_time:.1f}s"
                    + (f" (after {attempt} retries)" if attempt > 0 else "")
                )
                break

            except Exception as e:
                error_str = str(e)
                logger.warning(f"[{task.id}] Exception: {error_str}")

                if is_retryable_error(e):
                    last_error = error_str
                    attempt += 1
                    continue
                else:
                    # Non-retryable error
                    logger.exception(f"[{task.id}] Non-retryable error")
                    result.error = error_str
                    result.success = False
                    result.score = 0.0
                    break

        # If we exhausted retries, set the last error
        if attempt > self.retry_config.max_retries and last_error:
            result.error = f"Failed after {self.retry_config.max_retries + 1} attempts. Last error: {last_error}"
            logger.error(f"[{task.id}] {result.error}")

        result.completed_at = datetime.now().isoformat()
        return result

    def run_all(
        self,
        tasks: list[MultiRealTask] | None = None,
        save_results: bool = True,
    ) -> list[MultiRealResult]:
        """Run all tasks (or specified subset)."""
        tasks = tasks or list(registry.all())
        results = []

        total_tasks = len(tasks)
        start_time = time.time()

        logger.info("=" * 60)
        logger.info(f"Multi-REAL Benchmark Run")
        logger.info(f"Model: {self.model_config.name}")
        logger.info(f"Tasks: {total_tasks}")
        logger.info(f"Timeout: {self.task_timeout}s per task")
        logger.info(f"Retries: {self.retry_config.max_retries}")
        logger.info("=" * 60)

        for i, task in enumerate(tasks):
            task_start = time.time()
            logger.info("")
            logger.info(f"[{i+1}/{total_tasks}] Starting: {task.prefixed_id}")
            logger.info("-" * 40)

            result = self.run_task(task)
            results.append(result)

            if save_results:
                self._save_result(result)

            # Log result
            task_elapsed = time.time() - task_start
            status = "✓ PASS" if result.success else "✗ FAIL"
            logger.info(f"[{i+1}/{total_tasks}] {status}: {task.id}")
            logger.info(
                f"    Score: {result.score}, Confidence: {result.confidence}, "
                f"Method: {result.eval_method}"
            )
            logger.info(f"    Steps: {result.num_steps}, Time: {task_elapsed:.1f}s")
            if result.error:
                logger.info(f"    Error: {result.error[:100]}...")

            # Running summary
            passed = sum(1 for r in results if r.success)
            logger.info(f"    Running: {passed}/{len(results)} passed")

        # Final summary
        total_elapsed = time.time() - start_time
        total_passed = sum(1 for r in results if r.success)
        total_score = sum(r.score for r in results)
        max_score = sum(task.points for task in tasks)

        logger.info("")
        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"Tasks Passed: {total_passed}/{total_tasks} ({100*total_passed/total_tasks:.1f}%)")
        logger.info(f"Total Score: {total_score}/{max_score}")
        logger.info("")

        # List failures
        failures = [r for r in results if not r.success]
        if failures:
            logger.info(f"Failed tasks ({len(failures)}):")
            for r in failures:
                error_summary = (r.error[:50] + "...") if r.error and len(r.error) > 50 else r.error
                logger.info(f"  - {r.task_id}: {error_summary or 'evaluation failed'}")

        return results

    def _save_result(self, result: MultiRealResult) -> None:
        """Save individual result to JSON."""
        # Create directory for this task
        task_id_clean = result.task_id.replace("multi.", "")
        output_dir = self.results_dir / task_id_clean
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save result JSON
        model_name_clean = self.model_config.name.lower().replace(" ", "_").replace(".", "")
        output_file = output_dir / f"{model_name_clean}.json"

        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.debug(f"Saved result to {output_file}")
