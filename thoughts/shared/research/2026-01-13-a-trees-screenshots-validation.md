---
date: 2026-01-13T09:21:27-07:00
researcher: Claude
git_commit: 05435f7f951e590a68fd6929fc557e5504643e82
branch: main
repository: agisdk
topic: "Using a-trees and screenshots for validation instead of truncated finish JSONs"
tags: [research, codebase, validation, a-trees, screenshots, multi-real]
status: complete
last_updated: 2026-01-13
last_updated_by: Claude
---

# Research: Using A-Trees and Screenshots for Validation

**Date**: 2026-01-13T09:21:27-07:00
**Researcher**: Claude
**Git Commit**: 05435f7f951e590a68fd6929fc557e5504643e82
**Branch**: main
**Repository**: agisdk

## Research Question
How to use a-trees and screenshots for validation instead of truncated finish JSONs in gen_ground_truth.py?

## Summary

The current `gen_ground_truth.py` uses `HybridValidator.evaluate()` which truncates finish state JSON to 10k characters before sending to the LLM judge. This truncation loses critical information from multi-app state (which can be 100k+ characters).

**Good news:** The infrastructure to use a-trees and screenshots already exists:
1. **A-trees** are captured per-step and stored in `step_N.pkl.gz` files
2. **Screenshots** are saved per-step as `screenshot_step_N.png` files
3. The `ExpResult` class provides lazy-loading methods to retrieve both

**Key approach to fix this:**
1. Load the last step's observation from `step_N.pkl.gz`
2. Extract `axtree_txt` (formatted accessibility tree text) from the observation
3. Load the final screenshot(s) from the experiment directory
4. Send the screenshot + a-tree text to the LLM judge instead of truncated JSON

## Detailed Findings

### 1. A-Tree Capture and Storage

#### Where A-Trees Are Captured
**File**: `src/agisdk/REAL/browsergym/core/observation.py`

- **Lines 412-482**: `extract_all_frame_axtrees()` - Extracts AXTree from all frames using Chrome DevTools Protocol's `Accessibility.getFullAXTree` command
- **Lines 485+**: `extract_merged_axtree()` - Merges all frame AXTrees into a single unified tree

#### A-Tree in Observations
**File**: `src/agisdk/REAL/browsergym/core/env.py`

- **Line 172**: `axtree_object` defined in observation space
- **Lines 753-754**: Stores extracted a-tree in observations

#### A-Tree Format Conversion
**File**: `src/agisdk/REAL/browsergym/utils/obs.py`

- **Lines 281-424**: `flatten_axtree_to_str()` - Converts raw AXTree dict to human-readable text:
  ```
  [bid] role_name "node_name", attribute1, attribute2=value
    [child_bid] child_role "child_name"
  ```

#### A-Tree in Experiment Data
**File**: `src/agisdk/REAL/browsergym/experiments/loop.py`

- **Lines 372-411**: `StepInfo` dataclass stores full observation including `axtree_object`
- **Lines 506-509**: Saved to `step_N.pkl.gz` per step

**File**: `src/agisdk/REAL/browsergym/experiments/agent.py`

- **Lines 14-23**: `default_obs_preprocessor()` converts `axtree_object` → `axtree_txt`:
  ```python
  obs["axtree_txt"] = flatten_axtree_to_str(obs["axtree_object"])
  del obs["axtree_object"]
  ```

### 2. Screenshot Capture and Storage

#### Where Screenshots Are Captured
**File**: `src/agisdk/REAL/browsergym/core/observation.py`

- **Lines 106-138**: `extract_screenshot()` - Uses Chrome CDP `Page.captureScreenshot` with PNG format
- Returns numpy array (height x width x RGB)

#### Screenshot Storage
**File**: `src/agisdk/REAL/browsergym/experiments/loop.py`

- **Lines 488-494**: Saves screenshots per step:
  ```python
  img = Image.fromarray(screenshot)
  img.save(exp_dir / f"screenshot_step_{self.step}.png")
  ```
- Format: PNG, RGB, typical 1280x720 viewport

#### Screenshot Retrieval
**File**: `src/agisdk/REAL/browsergym/experiments/loop.py`

- **Lines 803-830**: `ExpResult` provides lazy loading:
  ```python
  def get_screenshot(step, som=False) -> np.ndarray
  def get_screenshots(som=False) -> list[np.ndarray]
  ```
- Properties: `screenshots` and `screenshots_som`

### 3. Current HybridValidator Truncation

**File**: `multi-real/core/validator.py`

- **Lines 137-140**: Truncates finish_state to 10k characters:
  ```python
  finish_state_str = json.dumps(finish_state, indent=2)
  if len(finish_state_str) > 10000:
      finish_state_str = finish_state_str[:10000] + "\n...(truncated)"
  ```

This is the bottleneck causing loss of critical information.

### 4. Data Available Per Step

Each `step_N.pkl.gz` file contains a `StepInfo` with:
- `obs: dict` - Full observation including:
  - `axtree_object` (raw) or `axtree_txt` (formatted)
  - `screenshot` (numpy array)
  - `dom_txt` (HTML)
  - `last_action_error`
- `action: str` - Action taken
- `reward: float` - Reward received
- `agent_info: dict` - Agent metadata
- `task_info: dict` - Task-specific state info

### 5. Accessing Step Data

**File**: `src/agisdk/REAL/browsergym/experiments/loop.py`

```python
# Load experiment result
result = ExpResult.from_dir(exp_dir)

# Get step info for last step
last_step = result.get_step_info(result.summary_info["n_steps"] - 1)

# Access a-tree text
axtree_txt = last_step.obs.get("axtree_txt")

# Access screenshot
screenshot = result.get_screenshot(last_step.step)
```

## Code References

| Component | File | Lines |
|-----------|------|-------|
| A-tree extraction | `src/agisdk/REAL/browsergym/core/observation.py` | 412-530 |
| A-tree formatting | `src/agisdk/REAL/browsergym/utils/obs.py` | 281-424 |
| Screenshot capture | `src/agisdk/REAL/browsergym/core/observation.py` | 106-138 |
| Step storage | `src/agisdk/REAL/browsergym/experiments/loop.py` | 372-523 |
| Screenshot retrieval | `src/agisdk/REAL/browsergym/experiments/loop.py` | 803-830 |
| Current truncation | `multi-real/core/validator.py` | 137-140 |
| gen_ground_truth | `multi-real/tools/gen_ground_truth.py` | 50-71 |

## Architecture Documentation

### Current Validation Flow
```
gen_ground_truth.py
    └── HybridValidator.evaluate()
        └── Truncates finish_state JSON to 10k chars
        └── Sends to Claude LLM judge
        └── Returns (is_valid, confidence, reasoning)
```

### Proposed Validation Flow with A-Trees + Screenshots
```
gen_ground_truth.py
    └── Load ExpResult from experiment directory
    └── Extract last step's a-tree text + screenshots
    └── New validator method:
        └── Send screenshot(s) as vision input
        └── Send a-tree text as structured context
        └── Send task goal + eval criteria
        └── LLM judge evaluates visually + semantically
```

## Implementation Approach

To modify `gen_ground_truth.py` to use a-trees and screenshots:

### 1. Load Step Data After Harness Run

The harness already saves step data. After `harness.run_all()`:

```python
from src.agisdk.REAL.browsergym.experiments.loop import ExpResult

# result.exp_dir points to experiment directory
exp_result = ExpResult.from_dir(result.exp_dir)

# Get final step
final_step = exp_result.get_step_info(exp_result.summary_info["n_steps"] - 1)

# Get a-tree text
axtree_txt = final_step.obs.get("axtree_txt")

# Get final screenshot
final_screenshot = exp_result.get_screenshot(final_step.step)
```

### 2. Convert Screenshot for Vision API

Use existing utility from demo_agent:

**File**: `src/agisdk/REAL/demo_agent/basic_agent.py:24-37`

```python
def image_to_jpg_base64_url(img: np.ndarray, max_size: int = 1024) -> str:
    # Converts numpy array to base64 data URL for vision API
```

### 3. Create Vision-Enabled Validator

Modify `HybridValidator` or create new class that:
1. Takes screenshot + a-tree text instead of finish JSON
2. Uses Claude's vision capabilities
3. Sends structured prompt with:
   - Task goal
   - Eval criteria
   - A-tree text (full, not truncated)
   - Screenshot as vision input

### 4. Alternative: Use Per-App Screenshots

For multi-app tasks, could capture final screenshot of each app:
- Store in `finish_state.screenshots = {app_id: screenshot_b64}`
- Send all relevant screenshots to LLM judge

## Key Observations

1. **A-tree text is more compact than finish JSON** - The formatted a-tree (`axtree_txt`) captures UI state semantically without the verbose action history

2. **Screenshots provide visual ground truth** - The LLM can visually verify completed actions (filled forms, sent emails, etc.)

3. **Infrastructure exists** - No need to modify the core harness; just need to access existing step data differently in the validator

4. **Multi-app complexity** - Each app has its own final state; may need screenshots from multiple browser contexts

## Open Questions

1. Should validation use just the final step, or sequence of key steps?
2. How to handle multi-app scenarios where final state spans multiple browser pages?
3. Should a-tree be filtered/pruned to focus on relevant elements?
4. What's the optimal combination of: screenshot-only, a-tree-only, or both?
