# Prompt Framework & Dual-Model Orchestrator — Implementation Plan

## Goals
- Allow evaluation pipeline (`test.py`) to swap prompt “frameworks” without mutating datasets.
- Support a dual-model setup (planner + actor) that the evaluator can call like a single LLM.
- Avoid breaking existing single-model/CLI behaviour.

## Current Constraints
- `test.py` only understands one `BaseLLM` instance selected by `--model_type`.
- Prompt formatting lives inside the JSON datasets; template differences require editing the data.
- `Agent`/`BaseLLM` API is already flexible (`chat`, `stream_chat`); we can extend via subclasses/adapters rather than refactoring the agent core.

## High-Level Changes
1. **Prompt Framework Layer**
   - Introduce `teval/prompts/registry.py` with:
     - `PromptFramework` base class (method `build_sample(sample_dict) -> List[dict], meta_template_key`).
     - Registry helpers (`register_framework`, `build_framework(name, **kwargs)`).
   - Provide initial frameworks:
     - `DatasetPassthroughFramework` (maintains current behaviour).
     - `DualPlanFramework` (prep planner + actor prompts/tool listings).
   - Allow frameworks to inject system prompts, restructure roles, and choose meta-template keys.

2. **CLI & Runner Support**
   - Add `--prompt_framework` (default: `passthrough`) and optional `--prompt_params` (JSON string/path) to `test.py`.
   - Parse params once in `__main__`, instantiate the framework via the registry.
   - Before batching (`infer`), run `messages, meta_template_key = framework.build_sample(sample)`:
     - Replace `sample["origin_prompt"]`.
     - Track the selected `meta_template` (if provided) for the downstream model.

3. **Dual-Model Coordinator**
   - Create `teval/llms/dual_agent.py` with `DualStageLLM(BaseLLM)`:
     - Accept planner/actor configs (each resolved via `create_object` to `BaseLLM`).
     - `chat(messages, **kwargs)`:
       1. Split planner vs. actor prompts (using framework tags in `messages`).
       2. Call planner `chat`; parse plan/tool decisions.
       3. Build actor prompt (framework provides helper).
       4. Call actor `chat` and return final text.
     - Expose optional planner trace for logging.
   - Register class in `lagent/lagent/llms/__init__.py`.

4. **Wiring Into `test.py`**
   - Add CLI flags: `--model_mode` (`single` | `dual`), `--planner_model_path`, `--actor_model_path`.
   - When `model_mode == "dual"`, instantiate `DualStageLLM` with planner/actor settings; otherwise use current single-model flow.
   - Pass framework-selected `meta_template` to the chosen model instance.

5. **Support Scripts**
   - Update `test_all_en.sh` to forward new CLI arguments (`model_mode`, planner/actor paths, prompt framework name).
   - Provide sample framework configs under `configs/prompt_frameworks/`.

## Incremental Work Breakdown
1. **Scaffold prompt framework module**
   - Create registry, base class, and passthrough implementation.
   - Add unit-style smoke test for registry (if existing test infra allows).
2. **Integrate prompt_framework flag in `test.py`**
   - Parse options, instantiate framework, apply to dataset.
   - Ensure existing defaults preserve previous behaviour.
3. **Implement `DualStageLLM`**
   - Handle planner+actor chat flow; implement error handling/logging.
   - Add simple planner mock to validate.
4. **Finish `DualPlanFramework`**
   - Provide message annotations (e.g., mark planner vs actor segments) consumed by `DualStageLLM`.
5. **Update shell scripts & docs**
   - Extend `test_all_en.sh`.
   - Document usage in `README.md` or new `docs/prompt_frameworks.md`.

## Validation
- Run `python test.py` in single-model mode with default framework: expect identical outputs.
- Run `python test.py --model_mode dual --prompt_framework dual_plan ...` with small test subset; inspect cached JSON for planner trace and actor response.
- Optional: add regression tests for framework transformation helpers and `DualStageLLM.chat` (mock planner/actor to avoid heavy weights).

## Open Questions
- Planner output format: fixed JSON schema vs. free-form text? (Decide early; influences parser.)
- Do we need streaming support (`stream_chat`) for the dual coordinator? If yes, design a generator that yields planner trace first, then actor tokens.
- How to expose planner trace in evaluation outputs (store alongside `prediction`? separate file?).
