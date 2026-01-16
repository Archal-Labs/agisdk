import json
import logging
import os
import urllib.parse

import playwright.sync_api
import requests

from agisdk.REAL.browsergym.core.task import AbstractBrowserTask
from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
from agisdk.REAL.browsergym.webclones.task_config import (
    DEFAULT_VERSION,
    TaskConfig,
    Website,
    split_task_reference,
)
from agisdk.REAL.logging import logger as rich_logger

RAILWAY_API_BASE = "https://evaluate-production.up.railway.app/"

logger = logging.getLogger(__name__)


def get_run_id_from_api(api_key: str, model_id_name: str, run_name: str):
    """
    Get a run ID from the REAL evaluations API using an API key, model name, and run name.

    Args:
        api_key: REAL API key
        model_id_name: Name of the model being used
        run_name: Human-readable name for this run

    Returns:
        A run ID string if successful, None otherwise
    """
    try:
        # URL encode parameters
        encoded_model_id_name = urllib.parse.quote(model_id_name)
        encoded_run_name = urllib.parse.quote(run_name)

        # Construct the API URL
        # Prefer the REAL_API_BASE env override to support domain migrations (e.g., realevals.ai)
        base_url = os.getenv("REAL_API_BASE", "https://www.realevals.ai")
        url = f"{base_url.rstrip('/')}/api/runKey?api_key={api_key}&model_name={encoded_model_id_name}&run_name={encoded_run_name}"

        # Make the request
        response = requests.get(url, timeout=10)

        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            if "newRunId" in data:
                return data["newRunId"]
            else:
                logger.error(f"API response did not contain newRunId: {data}")
        else:
            logger.error(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

    except Exception as e:
        logger.error(f"Error getting run ID from API: {e}")

    return None


class AbstractWebCloneTask(AbstractBrowserTask):
    """
    Abstract class for all WebClones tasks
    """

    @classmethod
    def get_task_id(cls):
        return cls.task_id

    def __init__(
        self,
        seed: int,
        task_name: str = None,
        task_version: str = None,
        task_id: str = None,
        run_id: str = None,
        api_key: str = None,
        model_id_name: str = None,
        run_name: str = None,
    ) -> None:
        """
        Args:
            seed: Random seed for the task.
            task_name: Base task name (e.g. "dashdish-1").
            task_version: Version label (e.g. "v2").
            task_id: Canonical identifier ("v2.dashdish-1"). Deprecated in favour of
                     passing task_name and task_version explicitly.
            run_id: Optional run ID for the task. If provided, overrides the run_id in the task config.
                   This is used for leaderboard submissions.
            api_key: Optional REAL API key for automatic run_id generation.
            model_id_name: Optional model name for automatic run_id generation.
            run_name: Optional run name for automatic run_id generation.
            base_url: str (optional), the base URL where the task's HTML file is to be found.
                     If not provided, the WEBCLONES_URL environment variable will be used.
        """
        super().__init__(seed)

        self.seed = seed
        resolved_name: str
        resolved_version: str

        if task_name and task_version:
            resolved_name, resolved_version = task_name, task_version
        elif task_id:
            resolved_version, resolved_name = split_task_reference(task_id)
        elif task_name:
            resolved_name = task_name
            resolved_version = task_version or DEFAULT_VERSION
        else:
            raise ValueError("task_name and task_version are required.")

        self.task_config = TaskConfig(resolved_name, resolved_version)
        self.task_name = self.task_config.task_name
        self.task_version = self.task_config.version
        self.task_id = self.task_name
        self.canonical_task_id = self.task_config.canonical_id

        # Set run_id: prioritize RUNID environment variable,
        # then the explicitly provided run_id parameter,
        # then try to generate from API if api_key, model_id_name, and run_name are provided,
        # then check task config, finally default to '0'
        env_run_id = os.environ.get("RUNID")
        if env_run_id:
            self.run_id = env_run_id
            logger.info(f"Using run_id from environment variable: {self.run_id}")
        elif run_id is not None:
            self.run_id = run_id
            logger.info(f"Using explicitly provided run_id: {self.run_id}")
        else:
            if api_key is not None and model_id_name is not None and run_name is not None:
                # Try to get run_id from API
                logger.info(
                    f"Attempting to get run_id from API for model '{model_id_name}' and run '{run_name}'"
                )
                api_run_id = get_run_id_from_api(api_key, model_id_name, run_name)
                if api_run_id:
                    self.run_id = api_run_id
                    # Also set the environment variable for other components
                    os.environ["RUNID"] = api_run_id
                    logger.info(f"Successfully obtained run_id from API: {self.run_id}")
                else:
                    # Fall back to task config or default
                    if "run_id" in self.task_config.task.config:
                        self.run_id = self.task_config.task.config["run_id"]
                        logger.info(f"Using run_id from task config: {self.run_id}")
                    else:
                        self.run_id = "0"
                        logger.info(f"Using default run_id: {self.run_id}")
            elif "run_id" in self.task_config.task.config:
                self.run_id = self.task_config.task.config["run_id"]
                logger.info(f"Using run_id from task config: {self.run_id}")
            else:
                self.run_id = "0"
                logger.info(f"Using default run_id: {self.run_id}")

        self.evaluator = WebCloneEvaluator(task_config=self.task_config)
        self.goal = self.task_config.get_goal()

        # Multi-app support: store all websites
        self.websites: list[Website] = self.task_config.get_websites()
        self.is_multi_app = self.task_config.is_multi_app()

        # Primary URL for backward compatibility (first website)
        self.url = self.task_config.get_start_url()
        if not self.url:
            if "WEBCLONE_URL" in os.environ:
                self.url = os.environ["WEBCLONE_URL"]
            else:
                raise ValueError(
                    "Provide a WebClones base URL or set it up as WEBCLONES_URL env var."
                )

        # Dict to store background pages for each website (populated in setup)
        self._website_pages: dict[str, playwright.sync_api.Page] = {}

        rich_logger.info(f"⚙️ Initialized {self.canonical_task_id} task.")
        if self.is_multi_app:
            website_ids = [w.id for w in self.websites]
            rich_logger.info(f"🌐 Multi-app task with websites: {website_ids}")
        rich_logger.info(f"🎯 Goal: {self.goal}")

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        """
        Set up the task by configuring all websites.
        For multi-app tasks, creates a background page for each website and visits /config.
        """
        self.page = page

        # Historical v1 leaderboard expects bare task ids (e.g., "dashdish-3") rather than "v1.dashdish-3".
        config_task_id = self.canonical_task_id
        if self.task_version == "v1" and getattr(self, "run_id", "0") != "0":
            config_task_id = self.task_name

        # Configure each website
        for website in self.websites:
            # Create a background page for this website
            bg_page = page.context.new_page()
            self._website_pages[website.id] = bg_page

            # Visit config endpoint for this website
            config_url = f"{website.url}/config?run_id={self.run_id}&task_id={config_task_id}&latency=0"
            bg_page.goto(config_url)
            bg_page.wait_for_load_state("networkidle")

            # Pre-visit finish endpoint to initialize state tracking
            finish_url = f"{website.url}/finish"
            bg_page.goto(finish_url)

            logger.debug(f"Configured website: {website.id} at {website.url}")

        # Keep backward compatibility: self.background_page points to primary site
        self.background_page = self._website_pages[self.websites[0].id]

        # Navigate main page to primary website
        self.page.bring_to_front()
        self.page.goto(self.url)

        # For multi-app tasks, augment the goal with website URLs and navigation instructions
        # so the agent knows how to switch between websites
        goal_to_return = self.goal
        if self.is_multi_app:
            website_info = "\n\nAvailable websites for this task:\n"
            for website in self.websites:
                name = website.name or website.id
                similar = f" (similar to {website.similarTo})" if website.similarTo else ""
                website_info += f"- {name}{similar}: {website.url}\n"
            website_info += "\nTo switch between websites, use the goto(url) action. "
            website_info += "For example: goto('https://real-gomail.vercel.app')"
            goal_to_return = self.goal + website_info

        return goal_to_return, {"websites": [w.id for w in self.websites]}

    def teardown(self) -> None:
        """Close all browser pages including background pages for all websites."""
        # Close all website background pages
        for website_id, bg_page in self._website_pages.items():
            bg_page.close()
            logger.debug(f"Closed background page for website: {website_id}")
        self._website_pages.clear()
        self.page.close()

    def get_finish_json(self, timeout: int = 1000) -> dict:
        """
        Fetch final state JSON from all website /finish endpoints.

        For single-app tasks: returns the raw state dict from that website.
        For multi-app tasks: returns a dict keyed by website_id containing each site's state.
            e.g., {"networkin": {...}, "gocalendar": {...}}

        Args:
            timeout: Timeout in ms for page navigation

        Returns:
            Environment state dict (nested by website_id for multi-app tasks)
        """
        logger.debug("Fetching finish JSON from all websites...")

        if self.is_multi_app:
            return self._get_multi_app_finish_json(timeout)
        else:
            return self._get_single_app_finish_json(timeout)

    def _get_single_app_finish_json(self, timeout: int) -> dict:
        """Fetch finish JSON for a single-app task (backward compatible)."""
        env_state_json = {}
        error_message = ""

        bg_page = self._website_pages.get(self.websites[0].id, self.background_page)
        website_url = self.websites[0].url

        try:
            logger.debug(f"Navigating to finish endpoint: {website_url}/finish")
            bg_page.goto(f"{website_url}/finish", timeout=timeout)
            bg_page.wait_for_load_state("networkidle", timeout=timeout)
            pre_element = bg_page.wait_for_selector("pre")
            if pre_element:
                env_state = pre_element.inner_text()
                env_state_json = json.loads(env_state)
            else:
                error_message = "No state data available"
        except playwright.sync_api.TimeoutError:
            error_message = "Validation endpoint not yet available"
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON format: {str(e)}"
        except Exception as e:
            error_message = f"Validation error: {str(e)}"

        assert error_message == "", error_message
        return env_state_json

    def _get_multi_app_finish_json(self, timeout: int) -> dict:
        """
        Fetch finish JSON from all websites for a multi-app task.
        Returns a dict keyed by website_id.
        """
        merged_state: dict[str, dict] = {}
        errors: list[str] = []

        for website in self.websites:
            bg_page = self._website_pages.get(website.id)
            assert bg_page is not None, f"No background page for website: {website.id}"

            try:
                logger.debug(f"Navigating to finish endpoint: {website.url}/finish")
                bg_page.goto(f"{website.url}/finish", timeout=timeout)
                bg_page.wait_for_load_state("networkidle", timeout=timeout)
                pre_element = bg_page.wait_for_selector("pre")

                if pre_element:
                    env_state = pre_element.inner_text()
                    merged_state[website.id] = json.loads(env_state)
                    logger.debug(f"Collected state from {website.id}")
                else:
                    errors.append(f"No state data available from {website.id}")

            except playwright.sync_api.TimeoutError:
                errors.append(f"Timeout fetching state from {website.id}")
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON from {website.id}: {str(e)}")
            except Exception as e:
                errors.append(f"Error fetching state from {website.id}: {str(e)}")

        assert len(errors) == 0, "; ".join(errors)
        return merged_state

    def _has_script_eval(self) -> bool:
        """Return True if any evaluation uses a Python script."""
        logger.debug("Checking for script-based evals")
        try:
            evals = self.task_config.get_evals()
        except AttributeError:
            logger.debug("Task config missing evals list")
            return False
        return any(
            getattr(eval_config, "type", "") == "script" or getattr(eval_config, "script", "")
            for eval_config in evals
        )

    def _build_task_config_payload(self) -> dict:
        """Build a minimal task_config payload for remote evaluation."""
        logger.debug("Building task_config payload for Railway submission")
        task = getattr(self.task_config, "task", None)
        if not task:
            logger.warning("Task config missing task details; returning empty payload")
            return {"evals": [], "points": 0.0}

        evals_payload = []
        for eval_config in getattr(task, "evals", []):
            if hasattr(eval_config, "to_json"):
                evals_payload.append(eval_config.to_json())
            else:
                evals_payload.append(getattr(eval_config, "__dict__", {}))

        payload: dict[str, object] = {
            "evals": evals_payload,
            "points": getattr(task, "points", 0.0) or 0.0,
        }
        script_names = [
            eval_config.script
            for eval_config in getattr(task, "evals", [])
            if getattr(eval_config, "script", "")
        ]
        if script_names:
            payload["eval_scripts"] = script_names
        return payload

    def _submit_script_leaderboard(
        self, env_state_json: dict, model_response: str, info: dict, local_reward: float
    ) -> None:
        """Submit results for script-based tasks to the external evaluation service."""
        logger.info("Preparing Railway submission for script-based task")
        railway_url = f"{RAILWAY_API_BASE.rstrip('/')}/evaluate"
        payload = {
            "env_state": env_state_json,
            "model_response": model_response,
            "task_config": self._build_task_config_payload(),
            "run_id": self.run_id,
            "task_id": self.canonical_task_id,
        }

        logger.info(
            "🚂 Script task: sending to Railway for evaluation and leaderboard submission..."
        )
        try:
            logger.debug(f"POST {railway_url} with payload keys: {list(payload.keys())}")
            railway_response = requests.post(railway_url, json=payload, timeout=30)
        except requests.exceptions.Timeout:
            logger.error("❌ Railway request timed out")
            info["railway_verified"] = False
            info["leaderboard_submitted"] = False
            return
        except Exception as exc:
            logger.error(f"❌ Failed to send to Railway: {exc}")
            info["railway_verified"] = False
            info["leaderboard_submitted"] = False
            return

        if railway_response.status_code == 200:
            try:
                logger.debug("Railway responded with 200; parsing JSON")
                railway_result = railway_response.json()
            except json.JSONDecodeError as exc:
                logger.error(f"❌ Railway response was not valid JSON: {exc}")
                info["railway_verified"] = False
                info["leaderboard_submitted"] = False
                return

            railway_reward = railway_result.get("reward", 0.0)
            info["railway_reward"] = railway_reward
            info["railway_verified"] = True
            info["leaderboard_submitted"] = railway_result.get("leaderboard_submitted", False)
            logger.info(f"✅ Railway evaluation complete: reward={railway_reward}")
            logger.debug(f"Railway result payload: {railway_result}")

            if local_reward != railway_reward:
                logger.warning(
                    f"⚠️ Evaluation mismatch! Local: {local_reward}, Railway: {railway_reward}"
                )
        else:
            logger.error(
                f"❌ Railway returned status {railway_response.status_code}: {railway_response.text}"
            )
            info["railway_verified"] = False
            info["leaderboard_submitted"] = False

    def _submit_standard_leaderboard(self, model_response: str) -> None:
        """Submit results to the legacy WebClones leaderboard endpoint."""
        try:
            logger.info("Submitting result to legacy /submit endpoint")
            encoded_response = urllib.parse.quote(model_response)
            response = self.background_page.goto(
                self.url + "/submit?retrieved_answer=" + encoded_response
            )
            if response is None:
                print("Warning: No response received when submitting to leaderboard")
            else:
                status = response.status
                if status is not None and status >= 400:
                    status_text = response.status_text or "Unknown status"
                    print(f"Warning: Leaderboard submission returned HTTP {status} ({status_text})")
        except Exception as exc:
            print(f"Warning: Failed to submit response to server: {exc}")

    def validate(
        self,
        page: playwright.sync_api.Page,
        chat_messages: list[str],
        timeout: int = 1000,
        verbose: bool = True,
    ) -> tuple[float, bool, str, dict]:
        reward, done, message, info = 0.0, False, "", {}
        # Treat model response as a challenge solution submission
        assistant_messages = [m for m in chat_messages if m["role"] == "assistant"]
        model_response = assistant_messages[-1]["message"]

        # Determine termination threshold based on task type
        # First assistant message is the greeting, so:
        # - Single-app: terminate after 1 agent message (len > 1)
        # - Multi-app: terminate after 2 agent messages (len > 2) to allow navigation communication
        termination_threshold = 2 if self.is_multi_app else 1
        if len(assistant_messages) > termination_threshold:
            done = True
        logger.debug(
            f"Validation called. done={done}, leaderboard_run={getattr(self, 'run_id', '0')}"
        )
        if done:
            env_state_json = self.get_finish_json(timeout=timeout)
            reward, _, message, info = self.evaluator.evaluate(env_state_json, model_response)
            message = "Task completed!" if done else "Task still in progress"
            info = {"env_state": env_state_json, "local_reward": reward}
            if model_response is None or model_response == "":
                model_response = "Done"
            is_leaderboard_submission = getattr(self, "run_id", "0") != "0"
            logger.debug(f"Leaderboard submission? {is_leaderboard_submission}")
            if is_leaderboard_submission:
                if self._has_script_eval():
                    logger.debug("Detected script eval; using Railway submission path")
                    self._submit_script_leaderboard(env_state_json, model_response, info, reward)
                else:
                    logger.debug("No script eval; using legacy submit endpoint")
                    self._submit_standard_leaderboard(model_response)

        return reward, done, message, info
