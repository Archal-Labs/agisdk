import argparse
import json
import glob
import os
from playwright.sync_api import sync_playwright
from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
from agisdk.REAL.browsergym.webclones.task_config import TaskConfig


def test_task(task_path: str, headless: bool = False):
    """Interactively test a task by completing it manually."""
    config = TaskConfig(task_path, is_path=True)
    
    # Get website URL(s)
    if config.is_multi_app():
        websites = config.get_websites()
        print(f"Multi-app task with {len(websites)} websites:")
        for w in websites:
            print(f"  - {w.id}: {w.url}")
        primary_url = websites[0].url
    else:
        primary_url = config.get_start_url()
        print(f"Single-app task: {primary_url}")
    
    print(f"\nGoal: {config.get_goal()}\n")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        
        # Navigate to website
        page.goto(primary_url)
        
        print("Complete the task in the browser, then press Enter...")
        input()
        
        # Collect state from all websites
        if config.is_multi_app():
            env_state = {}
            for website in config.get_websites():
                page.goto(f"{website.url}/finish")
                page.wait_for_selector("pre")
                pre = page.query_selector("pre")
                env_state[website.id] = json.loads(pre.inner_text())
        else:
            page.goto(f"{primary_url}/finish")
            page.wait_for_selector("pre")
            pre = page.query_selector("pre")
            env_state = json.loads(pre.inner_text())
        
        browser.close()
    
    # Evaluate
    print("\n--- Evaluation Results ---")
    evaluator = WebCloneEvaluator(task_config=config)
    reward, done, message, info = evaluator.evaluate(
        env_state=env_state,
        model_response=""
    )
    
    print(f"Reward: {reward}")
    print(f"Message: {message}")
    return reward > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_path", help="Path to task JSON file")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    
    success = test_task(args.task_path, args.headless)
    exit(0 if success else 1)
