# Baseline Evaluation Results

## Running models on Lambda

```bash
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 6: Clone and Setup Repository

```bash
ssh-keygen -t ed25519 -C "noahjsong705@gmail.com"
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:Archal-Labs/archival.git
cd bitter
uv sync
uv run playwright install
uv run playwright install-deps
```

### Running Qwen via vLLM Server

**Terminal 1: Start vLLM server**
```bash

# For Qwen-72B (4x H100 80GB recommended)
uv run vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --served-model-name gpt-qwen72b \
  --tensor-parallel-size 4 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 32768

# For Qwen-32B (2x H100 80GB recommended)
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
  --tensor-parallel-size 2 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 8192

```

**Terminal 2: Run evaluation**
```bash

http://localhost:8000/v1
or
export OPENAI_API_KEY=dummy

uv run python -c "
from agisdk import REAL

harness = REAL.harness(
    model='gpt-4o',
    task_type='omnizon',
    task_version='v2',
    max_steps=25,
    headless=True,
    use_axtree=True,
    use_screenshot=True
)
results = harness.run()
print(f'Results: {sum(1 for r in results.values() if r.get(\"success\"))}/{len(results)} tasks successful')
"
```


### Version with 72b
export OPENAI_BASE_URL=http://localhost:8000/v1 && export OPENAI_API_KEY=dummy

uv run python -c "
from agisdk import REAL

harness = REAL.harness(
    model='gpt-qwen72b',
    task_version='v2',
    max_steps=25,
    headless=True,
    use_axtree=True,
    use_screenshot=False,
    results_dir='./results/qwen72b'
)
results = harness.run()
"

## Running OpenCUA-72B with transformers + accelerate (no vLLM)

Use this path because OpenCUA-72B relies on the Kimi tokenizer/chat template and 1D RoPE changes that vLLM does not yet support.

### Prerequisites

  uv pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu124
  uv pip install "transformers>=4.45.0" "accelerate>=0.34.0" "einops" "pillow" "fastapi" "uvicorn" "pydantic" "huggingface_hub" "openai"

  ### Download weights

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="xlangai/OpenCUA-72B",
    local_dir="OpenCUA-72B",
    local_dir_use_symlinks=False,
)
PY

### Minimal OpenAI-compatible server (FastAPI)
script  in serve_opencua.py

### Start the server
cd /home/noahsong/work/archal/bitter/bitter
uv run uvicorn serve_opencua:app --host 0.0.0.0 --port 8000
uvicorn server:app --host 0.0.0.0 --port 8000
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy

uv run python -c "
from agisdk import REAL

harness = REAL.harness(
    model='gpt-opencua72b',
    task_version='v2',
    max_steps=25,
    headless=True,
    use_axtree=True,
    use_screenshot=True,
    results_dir='./results/opencua72b'
)
results = harness.run()
print(sum(1 for r in results.values() if r.get('success')), '/', len(results))
"

