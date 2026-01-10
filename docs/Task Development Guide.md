# Task Development Guide
## Table of Contents

1. [Creating a Single-App Task](#1-creating-a-single-app-task)
2. [Creating a Multi-App Task](#2-creating-a-multi-app-task)
3. [Verifying Your Task](#3-verifying-your-task)
4. [Creating a Task Suite](#4-creating-a-task-suite)
5. [Running Tasks with OpenAI API](#5-running-tasks-with-openai-api)

---

## 1. Creating a Single-App Task

### Step 1: Explore the Website

1. Open any REAL website in a private browser window (e.g., `https://evals-omnizon.vercel.app`)
2. Perform the task you want to create (e.g., add an item to cart)
3. Navigate to `<BASE_URL>/finish` (e.g., `https://evals-omnizon.vercel.app/finish`)
4. Copy the JSON from the page - this is what your evaluators will check against

### Step 2: Write the Task JSON

### Task Fields Reference

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier (convention: `<website>-<number>`) |
| `goal` | Yes | Natural language description of what the agent should do |
| `website` | Yes | Website configuration object |
| `website.id` | Yes | Website identifier (e.g., "omnizon", "networkin") |
| `website.url` | Yes | Base URL of the website |
| `difficulty` | Yes | "easy", "medium", or "hard" |
| `challengeType` | Yes | "action", "retrieval", "retrieval-action", or "no-action" |
| `possible` | Yes | Whether the task is solvable (usually `true`) |
| `evals` | Yes | Array of evaluation checks |
| `points` | Yes | Points awarded for completion |
| `config` | No | Additional configuration (usually `{}`) |

### Challenge Types

The `challengeType` field categorizes tasks based on what the agent needs to accomplish:

- **`"action"`** - Tasks where the agent must perform actions that change the website state (e.g., add item to cart, create post, schedule event). Focus is on verifying the action was completed.
- **`"retrieval"`** - Tasks where the agent must find and report information (e.g., "How many unread emails?", "What is the price?"). Focus is on verifying the agent correctly retrieved and reported the information.
- **`"retrieval-action"`** - Tasks that combine both: the agent must retrieve information AND perform an action (e.g., "Find the price and add it to cart", "Search for jobs and apply to one").
- **`"no-action"`** - Tasks where the agent should recognize the task is infeasible or requires additional information and should NOT perform the action (e.g., "Book a ride home" when no address is provided). The agent should ask for clarification instead.

### Evaluation Types

**1. JMESPath (State Check)**
```json
{
  "description": "Cart has exactly 2 items",
  "type": "jmespath",
  "query": "length(cart.items)",
  "expected_value": 2
}
```

**2. LLM Boolean (Response Check)**
```json
{
  "description": "Agent reported the correct price",
  "type": "llm_boolean",
  "expected_value": true,
  "rubric": "Does the response mention that the price is $299.99?"
}
```

To enable `llm_boolean` evaluations, simply add an evaluation object with `"type": "llm_boolean"` to your `evals` array. You can use both `jmespath` and `llm_boolean` evals in the same task - they will all be checked.

**Note:** The standard REAL benchmark tasks use **both** evaluation types extensively:
- Many tasks use `llm_boolean` evals (typically for retrieval tasks where the agent must report information)
- Many tasks use `jmespath` evals (for state-based checks on actions performed)
- Some tasks use **both** types together (e.g., verify the action happened via state check AND verify the agent's response is correct)

**When to use each:**
- **jmespath**: Use for checking that specific actions occurred (item added to cart, post created, etc.)
- **llm_boolean**: Use for retrieval tasks where the agent must correctly report information they found
- **Both**: Use together when you want to verify both the action happened AND the agent reported it correctly

**3. Script (Custom Python)**
```json
{
  "description": "Custom validation logic",
  "type": "script",
  "script": "eval_custom_task.py"
}
```

### JMESPath Query Tips

Use the [Eval Genie tool](https://eval-genie-checks-maker.lovable.app/) to generate queries:
1. Paste your `/finish` JSON
2. Describe your task goal
3. Select generated queries and copy them

Common patterns:
```
length(array) > `0`           # Array has items
array[0].field                # First item's field
object.nested.value           # Nested value access
array[?field == 'value']      # Filter array
```

---

## 2. Creating a Multi-App Task

### Key Differences from Single-App

| Aspect | Single-App | Multi-App |
|--------|------------|-----------|
| Website field | `website: {...}` | `websites: [{...}, {...}]` |
| JMESPath queries | `cart.items[0]` | `omnizon.cart.items[0]` |
| State structure | Raw state | Nested by website ID |

### Multi-App JMESPath Queries

Queries must be prefixed with the website ID:
```
networkin.feedPostsDiff.modified[0]     # Networkin state
gocalendar.events.added[0].title        # GoCalendar state
```

---

## 3. Verifying Your Task

### Option A: Manual Verification with Browser

```python
import json
from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
from agisdk.REAL.browsergym.webclones.task_config import TaskConfig

# Load your task
task_path = "path/to/your/task.json"
config = TaskConfig(task_path, is_path=True)

# After manually completing the task, get the /finish state
env_state = {
    # Paste the JSON from /finish endpoint here
}

# Run evaluation
# Note: model_response is only used for "llm_boolean" type evaluations.
# - If your task has llm_boolean evals: provide the agent's text response
# - If your task only has jmespath or script evals: use empty string ""
evaluator = WebCloneEvaluator(task_config=config)
reward, done, message, info = evaluator.evaluate(
    env_state=env_state,
    model_response=""  # Empty string is safe for manual testing
)

print(f"Reward: {reward}")
print(f"Message: {message}")
print(f"Results: {info}")
```

### Option B: Interactive Testing Script

```bash
python test_task.py path/to/your/task.json
```

## 4. Creating a Task Suite

### Task Suite Runner

Run with:
```bash
python run_suite.py --tasks-dir ./tasks --list
```


## 5. Running Tasks with OpenAI API

```bash
pip install agisdk openai playwright
playwright install chromium
export OPENAI_API_KEY="your-api-key"
```

### Running Tasks

```bash
# Run all tasks in a directory
python run_openai.py --tasks-dir ./tasks
# Run a specific task
python run_openai.py --tasks-dir ./tasks --filter omnizon-1
# Run with visible browser
python run_openai.py --tasks-dir ./tasks --no-headless
# Save results to custom file
python run_openai.py --tasks-dir ./tasks --output my_results.json
```

### Using the Built-in OpenAI Runner

The SDK includes a production-ready runner at `example/openAI_cua.py`:

```bash
cd example
python openAI_cua.py --filter omnizon-1 --no-headless
```

---

## Appendix: Available REAL Websites

| Website ID | Similar To | URL |
|------------|------------|-----|
| omnizon | Amazon | https://evals-omnizon.vercel.app |
| networkin | LinkedIn | https://real-networkin.vercel.app |
| gocalendar | Google Calendar | https://real-gocalendar.vercel.app |
| gomail | Gmail | https://real-gomail.vercel.app |
| zilloft | Zillow | https://real-zilloft.vercel.app |
| dashdish | DoorDash | https://real-dashdish.vercel.app |
| staynb | Airbnb | https://real-staynb.vercel.app |
| topwork | Indeed | https://real-topwork.vercel.app |
| udriver | Uber | https://real-udriver.vercel.app |
| opendining | OpenTable | https://real-opendining.vercel.app |
| fly-unified | United Airlines | https://real-fly-unified.vercel.app |

