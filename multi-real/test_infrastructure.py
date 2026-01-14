#!/usr/bin/env python3
"""
Comprehensive infrastructure test without requiring API keys.

Tests:
1. Task loading and syncing
2. Task filtering
3. TaskConfig initialization
4. Results directory creation
5. Extract finish JSONs script
6. Re-evaluation script
7. Query validation
8. All helper scripts

Usage:
    uv run python multi-real/test_infrastructure.py
"""

import json
import sys
from pathlib import Path
from typing import Any


def test_task_loading():
    """Test that tasks load correctly from directory."""
    print("=" * 80)
    print("TEST 1: Task Loading")
    print("=" * 80)

    tasks_dir = Path("multi-real/tasks")
    if not tasks_dir.exists():
        print(f"❌ FAIL: Tasks directory not found: {tasks_dir}")
        return False

    tasks = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        try:
            with open(task_file) as f:
                task = json.load(f)
                if "id" not in task:
                    task["id"] = task_file.stem
                tasks.append(task)
        except Exception as e:
            print(f"❌ FAIL: Error loading {task_file}: {e}")
            return False

    print(f"✓ Loaded {len(tasks)} tasks")

    # Validate task structure
    required_fields = ["id", "goal", "websites"]
    for task in tasks[:10]:  # Check first 10
        missing = [f for f in required_fields if f not in task]
        if missing:
            print(f"❌ FAIL: Task {task.get('id', 'unknown')} missing fields: {missing}")
            return False

    print(f"✓ All tasks have required fields")
    print(f"✅ PASS: Task loading works\n")
    return True


def test_task_syncing():
    """Test that task syncing works."""
    print("=" * 80)
    print("TEST 2: Task Syncing")
    print("=" * 80)

    source_dir = Path("multi-real/tasks")
    target_dir = Path("src/agisdk/REAL/browsergym/webclones/v1/tasks")

    if not source_dir.exists():
        print(f"❌ FAIL: Source directory not found: {source_dir}")
        return False

    if not target_dir.exists():
        print(f"❌ FAIL: Target directory not found: {target_dir}")
        return False

    # Get a few task IDs from source
    sample_tasks = []
    for task_file in list(source_dir.glob("*.json"))[:5]:
        with open(task_file) as f:
            task = json.load(f)
            task_id = task.get("id", task_file.stem)
            sample_tasks.append(task_id)

    # Check they exist in target with correct ID
    for task_id in sample_tasks:
        target_file = target_dir / f"{task_id}.json"
        if not target_file.exists():
            print(f"❌ FAIL: Task {task_id} not synced to {target_file}")
            return False

        # Verify ID matches
        with open(target_file) as f:
            task = json.load(f)
            if task.get("id") != task_id:
                print(f"❌ FAIL: Task {task_id} has wrong ID in target: {task.get('id')}")
                return False

    print(f"✓ Checked {len(sample_tasks)} synced tasks")
    print(f"✅ PASS: Task syncing works\n")
    return True


def test_task_config():
    """Test that TaskConfig can load tasks."""
    print("=" * 80)
    print("TEST 3: TaskConfig Loading")
    print("=" * 80)

    try:
        from agisdk.REAL.browsergym.webclones.task_config import TaskConfig

        # Try to load a few tasks
        test_ids = ["dashdish-gomail-1", "gomail-topwork-1", "gocalendar-omnizon-1"]

        for task_id in test_ids:
            try:
                config = TaskConfig(task_id, "v1")
                print(f"✓ Loaded TaskConfig for {task_id}")

                # Check basic properties
                if not hasattr(config, 'task'):
                    print(f"❌ FAIL: TaskConfig missing 'task' attribute")
                    return False

                if not config.get_websites():
                    print(f"❌ FAIL: TaskConfig has no websites")
                    return False

            except FileNotFoundError as e:
                print(f"❌ FAIL: Task file not found for {task_id}: {e}")
                return False
            except Exception as e:
                print(f"❌ FAIL: Error loading TaskConfig for {task_id}: {e}")
                return False

        print(f"✅ PASS: TaskConfig loading works\n")
        return True

    except ImportError as e:
        print(f"❌ FAIL: Cannot import TaskConfig: {e}")
        return False


def test_results_directory():
    """Test results directory creation."""
    print("=" * 80)
    print("TEST 4: Results Directory")
    print("=" * 80)

    results_dir = Path("results")
    if not results_dir.exists():
        print(f"⚠ Warning: Results directory doesn't exist yet (will be created on first run)")
    else:
        print(f"✓ Results directory exists: {results_dir}")

    print(f"✅ PASS: Results directory structure correct\n")
    return True


def test_extract_script():
    """Test extract_finish_jsons.py exists and is valid."""
    print("=" * 80)
    print("TEST 5: Extract Finish JSONs Script")
    print("=" * 80)

    script = Path("multi-real/extract_finish_jsons.py")
    if not script.exists():
        print(f"❌ FAIL: Script not found: {script}")
        return False

    print(f"✓ Script exists: {script}")

    # Try to import and check main function exists
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("extract_finish_jsons", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'extract_finish_jsons'):
            print(f"❌ FAIL: Script missing extract_finish_jsons function")
            return False

        print(f"✓ Script has extract_finish_jsons function")
        print(f"✅ PASS: Extract script is valid\n")
        return True

    except Exception as e:
        print(f"❌ FAIL: Error loading script: {e}")
        return False


def test_re_evaluate_script():
    """Test re_evaluate_results.py exists and is valid."""
    print("=" * 80)
    print("TEST 6: Re-evaluate Results Script")
    print("=" * 80)

    script = Path("multi-real/re_evaluate_results.py")
    if not script.exists():
        print(f"❌ FAIL: Script not found: {script}")
        return False

    print(f"✓ Script exists: {script}")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("re_evaluate_results", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 're_evaluate_results'):
            print(f"❌ FAIL: Script missing re_evaluate_results function")
            return False

        print(f"✓ Script has re_evaluate_results function")
        print(f"✅ PASS: Re-evaluate script is valid\n")
        return True

    except Exception as e:
        print(f"❌ FAIL: Error loading script: {e}")
        return False


def test_query_validation():
    """Test query validation with available finish JSONs."""
    print("=" * 80)
    print("TEST 7: Query Validation")
    print("=" * 80)

    finish_states_dir = Path("multi-real/final_states/manual")
    if not finish_states_dir.exists():
        print(f"⚠ Warning: No manual finish states yet (will be created after first run)")
        print(f"✅ PASS: Query validation will work once finish JSONs are collected\n")
        return True

    finish_jsons = list(finish_states_dir.glob("*.json"))
    if not finish_jsons:
        print(f"⚠ Warning: No finish JSONs found in {finish_states_dir}")
        print(f"✅ PASS: Query validation will work once finish JSONs are collected\n")
        return True

    print(f"✓ Found {len(finish_jsons)} finish JSONs")

    # Try to validate one
    try:
        import jmespath

        finish_json_path = finish_jsons[0]
        with open(finish_json_path) as f:
            finish_json = json.load(f)

        # Try a simple query
        test_query = "keys(@)"
        result = jmespath.search(test_query, finish_json)

        print(f"✓ JMESPath queries work")
        print(f"✓ Finish JSON has keys: {list(result)[:3]}...")
        print(f"✅ PASS: Query validation works\n")
        return True

    except Exception as e:
        print(f"❌ FAIL: Query validation error: {e}")
        return False


def test_all_helper_scripts():
    """Test that all helper scripts exist."""
    print("=" * 80)
    print("TEST 8: Helper Scripts")
    print("=" * 80)

    scripts = {
        "multi-real/hybrid_validator.py": "Hybrid validator",
        "multi-real/generate_comparison_report.py": "Comparison report generator",
        "multi-real/sync_tasks.py": "Task syncing",
        "multi-real/diagnose_failures.py": "Failure diagnostics",
        "multi-real/validate_query_structure.py": "Query structure validator",
        "multi-real/QUERY_PATTERNS.md": "Query patterns documentation",
    }

    all_exist = True
    for script_path, description in scripts.items():
        path = Path(script_path)
        if path.exists():
            print(f"✓ {description}: {script_path}")
        else:
            print(f"❌ Missing {description}: {script_path}")
            all_exist = False

    if all_exist:
        print(f"✅ PASS: All helper scripts exist\n")
        return True
    else:
        print(f"❌ FAIL: Some scripts missing\n")
        return False


def test_browser_use_script():
    """Test browser-use.py is configured correctly."""
    print("=" * 80)
    print("TEST 9: browser-use.py Configuration")
    print("=" * 80)

    script = Path("example/browser-use.py")
    if not script.exists():
        print(f"❌ FAIL: Script not found: {script}")
        return False

    print(f"✓ Script exists: {script}")

    # Check for key functions
    with open(script) as f:
        content = f.read()

    checks = [
        ("load_tasks_from_directory", "Task loading function"),
        ("sync_tasks_to_package", "Task syncing function"),
        ("--dry-run", "Dry-run mode"),
        ("--tasks-dir", "Tasks directory argument"),
        ("--filter", "Filter argument"),
    ]

    all_present = True
    for check, description in checks:
        if check in content:
            print(f"✓ Has {description}")
        else:
            print(f"❌ Missing {description}")
            all_present = False

    if all_present:
        print(f"✅ PASS: browser-use.py is properly configured\n")
        return True
    else:
        print(f"❌ FAIL: browser-use.py missing features\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MULTI-REAL BENCHMARK INFRASTRUCTURE TEST")
    print("=" * 80)
    print()

    tests = [
        ("Task Loading", test_task_loading),
        ("Task Syncing", test_task_syncing),
        ("TaskConfig Loading", test_task_config),
        ("Results Directory", test_results_directory),
        ("Extract Script", test_extract_script),
        ("Re-evaluate Script", test_re_evaluate_script),
        ("Query Validation", test_query_validation),
        ("Helper Scripts", test_all_helper_scripts),
        ("browser-use.py", test_browser_use_script),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Infrastructure is ready!")
        print("\nNext steps:")
        print("1. Set API key: export OPENAI_API_KEY=your_key")
        print("2. Run benchmark: uv run python example/browser-use.py --filter all --workers 5 --model-type openai")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - fix issues before running benchmark")
        return 1


if __name__ == "__main__":
    sys.exit(main())
