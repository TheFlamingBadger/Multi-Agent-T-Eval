# Multi-Agent T-Eval

An evaluation harness for the [T-Eval benchmark](https://arxiv.org/abs/2312.14033) that focuses on tool-use capability of large language models. This repository began as a fork of the original [open-compass/T-Eval](https://github.com/open-compass/T-Eval) project; upstream documentation and assets still apply, but this fork layers on dual-agent orchestration, additional prompt frameworks, and workflow tooling tailored for multi-agent experiments.

## Upstream Project Snapshot

T-Eval measures performance along six tool-usage skills—**instruction**, **planning**, **reasoning**, **retrieval**, **understanding**, and **review**—by replaying multi-turn task traces and scoring model responses. For background and official results, see:

- Paper: [T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step](https://arxiv.org/abs/2312.14033)
- Project hub: [open-compass.github.io/T-Eval](https://open-compass.github.io/T-Eval/)
- Leaderboards: [English](https://open-compass.github.io/T-Eval/leaderboard.html) • [Chinese](https://open-compass.github.io/T-Eval/leaderboard_zh.html)
- Dataset: [Hugging Face – lovesnowbest/T-Eval](https://huggingface.co/datasets/lovesnowbest/T-Eval) (EN & ZH)

## What This Fork Adds

Highlights of the fork-specific functionality:

- **Configurable ReAct pipeline** – adds the `react` prompt framework plus a `test_all_en_react.sh` runner with overrideable system prompts.
- **Dual-agent evaluation loop** – introduces `test_all_en_react_dual.sh`, the `DualStageLLM` coordinator, and `SYSTEM_PROMPT_MODE` plumbing for planner/actor prompt injection.
- **Per-framework work_dirs** – caches now include a `framework_mode_display` suffix so repeated runs don’t collide.
- **GPU pinning & runtime improvements** – scripted `--gpus` selection alongside additional meta templates (e.g., QwQ) and Lagent overrides.

Additional quality-of-life changes span requirements updates, prompt parameter defaults, and improved resume semantics.

## Repository Layout

```
.
├── configs/                 # Prompt framework presets (dual_plan_react_default.json, etc.)
├── data/                    # Place benchmark JSON here (instruct_v2.json, …)
├── teval/
│   ├── llms/dual_agent.py   # DualStageLLM planner+actor coordinator
│   └── prompts/             # Prompt framework registry, ReAct & DualPlan implementations
├── test.py                  # Core evaluation driver
├── test_all_en.sh           # Single-model full benchmark runner
├── test_all_en_react.sh     # ReAct-optimized single-model runner
└── test_all_en_react_dual.sh# Planner+actor full benchmark runner
```

## Setup

```bash
git clone https://github.com/open-compass/T-Eval.git
# or clone this fork directly
git clone https://github.com/TheFlamingBadger/Multi-Agent-T-Eval.git
cd Multi-Agent-T-Eval
pip install -r requirements.txt

# Lagent (planner/actor backends for HuggingFace models)
git clone https://github.com/InternLM/lagent.git
cd lagent && pip install -e .
```

Download benchmark data and place it under `data/`:

```bash
mkdir -p data
# Hugging Face
python - <<'PY'
from datasets import load_dataset
dataset = load_dataset("lovesnowbest/T-Eval")
dataset.save_to_disk("data/raw")
PY

# Or grab the official ZIP from Google Drive and unpack into data/
```

Expect the following structure:

```
data/
  instruct_v2.json
  plan_json_v2.json
  ...
```

## Prompt Frameworks

This fork ships two higher-level prompt rewrites:

- `passthrough` – original behavior; dataset messages are forwarded unchanged.
- `react` – injects tool-trace cues following ReAct-style prompting.
- `dual_plan` – builds `planner_messages` and `actor_messages` and controls how the planner’s plan is inserted (prepend vs append) before actor execution.

Use `PROMPT_FRAMEWORK` to choose a framework and `PROMPT_PARAMS` to point to a JSON preset (see `configs/prompt_frameworks/`). `SYSTEM_PROMPT_MODE` (`overwrite` or `prepend`) determines how injected system prompts interact with dataset-provided system messages; `test.py` enforces those choices.

## Running Evaluations

### API Models

```bash
export OPENAI_API_KEY=sk-...
sh test_all_en.sh api gpt-4-1106-preview gpt4
```

### HuggingFace Models (single-LLM)

```bash
export MODEL_MODE=single
export PROMPT_FRAMEWORK=passthrough   # or react
sh test_all_en.sh hf /path/to/model vicuna-13b meta_template_name
```

### ReAct Workflow

The ReAct script preconfigures sensible defaults:

```bash
export PROMPT_FRAMEWORK=react
# Optional JSON for tool-specific instructions
export PROMPT_PARAMS=configs/prompt_frameworks/react_default.json
sh test_all_en_react.sh hf /path/to/model my-model meta_template
```

### Dual-Agent Planner + Actor

Setting `MODEL_MODE=dual` instructs `test.py` to spin up two backends. `test_all_en_react_dual.sh` wires the recommended environment variables:

```bash
export PROMPT_FRAMEWORK=dual_plan
export SYSTEM_PROMPT_MODE=prepend       # planner & actor inherit this unless overridden
export PLANNER_MODEL_PATH=/path/to/planner
export ACTOR_MODEL_PATH=/path/to/actor
export PLANNER_META_TEMPLATE=planner_template_name
export ACTOR_META_TEMPLATE=actor_template_name

sh test_all_en_react_dual.sh \
    hf "$PLANNER_MODEL_PATH" \
    "$PLANNER_MODEL_PATH" \
    hf "$ACTOR_MODEL_PATH" \
    actor-display-name \
    planner_template_name \
    actor_template_name
```

During inference the `DualStageLLM` (`teval/llms/dual_agent.py`) first queries the planner with `planner_messages`, stores the plan, then injects the plan into the actor prompt according to the directive supplied by the prompt framework (prepend vs append, custom role/template). The actor emits the final answer that is cached and scored.

### GPU Selection

All runner scripts honour `CUDA_VISIBLE_DEVICES`. Pass `--gpus 0,1` to pin devices at launch:

```bash
sh test_all_en.sh hf /path/to/model display-name meta_template --gpus 0,1
```

### Cache Layout

Outputs live under `work_dirs/<display_name>/`. File names now encode the prompt framework and model mode, e.g.:

```
work_dirs/my-model/
  instruct_dual_plan_dual_my-model.json
  plan_str_dual_plan_dual_my-model.json
  ...
```

This prevents single- vs dual-mode runs (or different frameworks) from overwriting each other.

## Inspecting Results

After all samples finish, aggregate metrics are written to `<out_dir>/<display_name>/<display_name>_-1.json` (and `_zh` for the Chinese set). Convert to leaderboard format with:

```bash
python teval/utils/convert_results.py \
    --result_path work_dirs/my-model/my-model_-1.json
```

## Contributing & Further Reading

- Track recent development with `git log --oneline` (see the highlighted commits above for a guide to the main fork features).
- Prompt frameworks live in `teval/prompts/`; meta templates are defined in `teval/utils/meta_template.py`.
- Dual-agent orchestration and trace capture are handled in `teval/llms/dual_agent.py`.

Pull requests are welcome—please open an issue describing the planned change before large refactors. This project inherits the original Apache 2.0 [LICENSE](./LICENSE).
