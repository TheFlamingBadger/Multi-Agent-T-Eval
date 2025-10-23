# Prompt Frameworks and Dual-Model Inference

## Overview
- **Prompt Frameworks** sit between the dataset and the model, transforming each sample before it reaches the LLM. Configure them via `--prompt_framework` and optional `--prompt_params` in `test.py`.
- **Dual-Model Mode** runs a planner model followed by an actor model using `DualStageLLM`. Enable with `--model_mode dual`.

## Prompt Framework Usage
1. Choose a framework:
   ```bash
   python test.py --prompt_framework passthrough ...
   python test.py --prompt_framework dual_plan --prompt_params configs/prompt_frameworks/dual_plan_default.json ...
   ```
2. Frameworks can emit meta-template hints. When present, the runner prefers them over CLI defaults.
3. Add custom frameworks by registering subclasses of `PromptFramework` in `teval/prompts`.

## Dual-Model Mode
1. Prepare planner and actor models:
   ```bash
   python test.py \
     --model_mode dual \
     --planner_model_path /path/to/planner \
     --actor_model_path /path/to/actor \
     --prompt_framework dual_plan \
     --prompt_params configs/prompt_frameworks/dual_plan_default.json \
     --planner_meta_template qwen \
     --actor_meta_template qwen
   ```
2. Override model types if needed (`--planner_model_type`, `--actor_model_type`). Defaults fall back to `--model_type` and `--model_path`.
3. Planner traces are saved per sample under `planner_plan` in the cached outputs.

## Batch Scripts
`test_all_en.sh` now respects the environment variables below (falls back to single-model defaults):
- `MODEL_MODE`
- `PROMPT_FRAMEWORK`
- `PROMPT_PARAMS`
- `PLANNER_MODEL_PATH`, `ACTOR_MODEL_PATH`
- `PLANNER_MODEL_TYPE`, `ACTOR_MODEL_TYPE`
- `PLANNER_META_TEMPLATE`, `ACTOR_META_TEMPLATE`

Example:
```bash
MODEL_MODE=dual \
PLANNER_MODEL_PATH=/models/planner \
ACTOR_MODEL_PATH=/models/actor \
PROMPT_FRAMEWORK=dual_plan \
PROMPT_PARAMS=configs/prompt_frameworks/dual_plan_default.json \
./test_all_en.sh hf /models/actor my-actor qwen
```
