import argparse
import base64
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from playwright.sync_api import sync_playwright

from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
from agisdk.REAL.browsergym.webclones.task_config import TaskConfig


client = OpenAI()
MODEL = "computer-use-preview"
WIDTH, HEIGHT = 1024, 768
MAX_ITERATIONS = 100


def load_tasks(tasks_dir: str, task_filter: str = None) -> list[dict]:
    """Load tasks from directory, optionally filtering by ID."""
    tasks = []
    for task_file in glob.glob(os.path.join(tasks_dir, "*.json")):
        with open(task_file, 'r') as f:
            task = json.load(f)
            task["_path"] = task_file
            if task_filter is None or task["id"] == task_filter:
                tasks.append(task)
    return tasks


def get_primary_url(task: dict) -> str:
    """Get the primary website URL for a task."""
    if "websites" in task:
        return task["websites"][0]["url"]
    return task["website"]["url"]


def collect_env_state(page, task: dict) -> dict:
    """Collect environment state from all websites."""
    if "websites" in task:
        # Multi-app: collect from each website
        env_state = {}
        for website in task["websites"]:
            page.goto(f"{website['url']}/finish")
            page.wait_for_selector("pre", timeout=10000)
            pre = page.query_selector("pre")
            if pre:
                env_state[website["id"]] = json.loads(pre.inner_text())
        return env_state
    else:
        # Single-app: collect directly
        base_url = task["website"]["url"]
        page.goto(f"{base_url}/finish")
        page.wait_for_selector("pre", timeout=10000)
        pre = page.query_selector("pre")
        if pre:
            return json.loads(pre.inner_text())
        return {}


def run_task(task: dict, headless: bool = True) -> dict:
    """Run a single task with OpenAI computer-use model."""
    task_id = task["id"]
    goal = task["goal"]
    base_url = get_primary_url(task)
    
    result = {
        "task_id": task_id,
        "start_time": datetime.now().isoformat(),
        "success": False,
        "reward": 0,
        "error": None,
        "response": None,
    }
    
    print(f"\n{'='*60}")
    print(f"Running: {task_id}")
    print(f"Goal: {goal[:100]}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[f"--window-size={WIDTH},{HEIGHT}"]
        )
        context = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT})
        page = context.new_page()
        
        # Configure and navigate
        config_url = f"{base_url}/config?task_id={task_id}"
        page.goto(config_url)
        page.goto(base_url)
        
        prev_response_id = None
        model_response = ""
        
        for iteration in range(MAX_ITERATIONS):
            # Take screenshot
            screenshot_b64 = base64.b64encode(page.screenshot()).decode()
            
            # Build input
            if iteration == 0:
                input_payload = [{"role": "user", "content": goal}]
            else:
                input_payload = [{
                    "call_id": last_call_id,
                    "type": "computer_call_output",
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                    },
                }]
            
            # Call model
            response = client.responses.create(
                model=MODEL,
                previous_response_id=prev_response_id,
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": WIDTH,
                    "display_height": HEIGHT,
                    "environment": "browser",
                }],
                input=input_payload,
                truncation="auto",
            )
            prev_response_id = response.id
            
            # Process computer calls
            computer_calls = [o for o in response.output if o.type == "computer_call"]
            if computer_calls:
                for call in computer_calls:
                    act = call.action
                    last_call_id = call.call_id
                    
                    print(f"  Step {iteration + 1}: {act.type}")
                    
                    if act.type == "click":
                        page.mouse.click(act.x, act.y)
                    elif act.type == "type":
                        page.keyboard.type(act.text)
                    elif act.type == "scroll":
                        page.mouse.wheel(
                            getattr(act, "delta_x", 0),
                            getattr(act, "delta_y", 0)
                        )
                    elif act.type == "keypress":
                        page.keyboard.press("+".join(act.keys))
                    
                    page.wait_for_timeout(500)
                continue
            
            # Check for text response (task complete)
            text_outputs = [o for o in response.output if o.type == "text"]
            if text_outputs:
                model_response = text_outputs[0].text
                print(f"  Model response: {model_response[:100]}...")
                break
        
        # Collect final state and evaluate
        env_state = collect_env_state(page, task)
        browser.close()
    
    # Evaluate
    config = TaskConfig(task["_path"], is_path=True)
    evaluator = WebCloneEvaluator(task_config=config)
    reward, done, message, info = evaluator.evaluate(
        env_state=env_state,
        model_response=model_response
    )
    
    elapsed = time.time() - start_time
    
    result.update({
        "success": reward > 0,
        "reward": reward,
        "response": model_response,
        "elapsed_time": elapsed,
        "message": message,
    })
    
    status = "PASS" if reward > 0 else "FAIL"
    print(f"\nResult: {status} (reward={reward}, time={elapsed:.1f}s)")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default="tasks")
    parser.add_argument("--filter", default=None, help="Run specific task ID")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    
    tasks = load_tasks(args.tasks_dir, args.filter)
    print(f"Found {len(tasks)} task(s)")
    
    results = []
    for task in tasks:
        result = run_task(task, headless=not args.no_headless)
        results.append(result)
    
    # Summary
    successful = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(successful)}/{len(results)} passed")
    print(f"Total reward: {sum(r['reward'] for r in results)}")
    print(f"{'='*60}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
