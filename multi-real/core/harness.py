"""
Multi-REAL Benchmark Harness

Wraps the core REAL harness to run multi-app benchmark tasks.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add multi-real and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agisdk.REAL.browsergym.experiments.loop import EnvArgs, ExpArgs
from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
from agisdk.REAL.demo_agent.basic_agent import DemoAgentArgs

from core.registry import MultiRealTask, registry
from model_configs.schema import ModelConfig, ModelProvider

# HybridValidator import
try:
    from core.validator import HybridValidator
    HYBRID_VALIDATOR_AVAILABLE = True
except (ImportError, AttributeError):
    HYBRID_VALIDATOR_AVAILABLE = False
    HybridValidator = None

logger = logging.getLogger(__name__)


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
    ):
        self.model_config = model_config
        self.results_dir = results_dir or Path(__file__).parent.parent / "results" / "raw"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.use_hybrid_eval = use_hybrid_eval and HYBRID_VALIDATOR_AVAILABLE

        if self.use_hybrid_eval:
            self.hybrid_validator = HybridValidator()
        elif use_hybrid_eval and not HYBRID_VALIDATOR_AVAILABLE:
            logger.warning("HybridValidator not available. Hybrid evaluation disabled.")

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

        evaluator = WebCloneEvaluator(task.evals)

        for eval_item in task.evals:
            if eval_item.get("type") != "jmespath":
                continue

            query = eval_item["query"]
            expected = eval_item.get("expected_value")

            try:
                # Use the evaluator's jmespath_verify method
                actual = evaluator.jmespath_search(query, finish_state)

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
    ) -> tuple[bool, float, dict]:
        """
        Evaluate task result using hybrid JMESPath + LLM judge.

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
        if self.use_hybrid_eval:
            try:
                llm_results = self.hybrid_validator.evaluate(
                    task_goal=task.goal,
                    finish_state=finish_state,
                    evals=task.evals,
                )
                eval_details["llm_judge_results"] = llm_results

                llm_passed = llm_results.get("overall_pass", False)

                if llm_passed:
                    # JMESPath failed but LLM passed - likely query bug
                    eval_details["eval_method"] = "llm_judge"
                    eval_details["confidence"] = "medium"
                    return True, float(task.points), eval_details
                else:
                    # Both failed - high confidence failure
                    eval_details["eval_method"] = "hybrid"
                    eval_details["confidence"] = "high"
                    return False, 0.0, eval_details

            except Exception as e:
                logger.error(f"LLM judge failed: {e}")
                # Fall through to JMESPath result with lower confidence

        # JMESPath failed, no hybrid or hybrid failed - medium confidence
        eval_details["confidence"] = "medium"
        return False, 0.0, eval_details

    def run_task(self, task: MultiRealTask) -> MultiRealResult:
        """Run a single task and return results."""
        started_at = datetime.now().isoformat()
        result = MultiRealResult(
            task_id=task.prefixed_id,
            model_name=self.model_config.name,
            success=False,
            score=0.0,
            started_at=started_at,
        )

        try:
            # Create agent and environment args
            agent_args = self._create_agent_args()
            env_args = self._create_env_args(task)

            # Create experiment args
            exp_args = ExpArgs(
                agent_args=agent_args,
                env_args=env_args,
            )

            # Prepare experiment (creates directory, saves args)
            exp_args.prepare(exp_root=self.results_dir / task.id)

            # Store exp_dir for later
            result.exp_dir = str(exp_args.exp_dir)

            # Run the experiment (saves results to disk)
            logger.info(f"Running experiment for {task.prefixed_id}")
            exp_args.run()

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

            # Check if there was an error
            if summary_info.get("err_msg"):
                result.error = summary_info["err_msg"]
                logger.warning(f"Task failed with error: {result.error}")

            # Evaluate with hybrid validator
            if finish_state:
                success, score, eval_details = self._evaluate_with_hybrid(task, finish_state)
                result.success = success
                result.score = score
                result.jmespath_results = eval_details["jmespath_results"]
                result.llm_judge_results = eval_details.get("llm_judge_results")
                result.eval_method = eval_details["eval_method"]
                result.confidence = eval_details["confidence"]
            else:
                logger.warning(f"No finish state available for {task.prefixed_id}")
                result.success = False
                result.score = 0.0
                result.confidence = "low"

        except Exception as e:
            logger.exception(f"Error running task {task.id}")
            result.error = str(e)
            result.success = False
            result.score = 0.0

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

        logger.info(f"Running {len(tasks)} tasks with {self.model_config.name}")

        for i, task in enumerate(tasks):
            logger.info(f"Task {i+1}/{len(tasks)}: {task.prefixed_id}")
            result = self.run_task(task)
            results.append(result)

            if save_results:
                self._save_result(result)

            # Log progress
            status = "PASS" if result.success else "FAIL"
            logger.info(
                f"  {status} (score={result.score}, confidence={result.confidence}, "
                f"method={result.eval_method})"
            )

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
