#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <model_type> <model_path> <display_name> [meta_template] [--gpus <ids>]"
    exit 1
}

default_gpu="${CUDA_VISIBLE_DEVICES:-1}"
gpu_ids="$default_gpu"
gpu_overridden=false

if [ "$#" -lt 3 ]; then
    usage
fi

model_type=$1
model_path=$2
display_name=$3
shift 3

meta_template="nan"
if [ "$#" -gt 0 ] && [ "$1" != "--gpus" ]; then
    meta_template=$1
    if [ -z "$meta_template" ]; then
        meta_template="nan"
    fi
    shift
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpus)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value for --gpus" >&2
                exit 1
            fi
            gpu_ids=$1
            gpu_overridden=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if $gpu_overridden; then
    if [[ ! "$gpu_ids" =~ ^[0-3](,[0-3])*$ ]]; then
        echo "Invalid GPU id list: '$gpu_ids'. Expected digits 0-3 optionally comma-separated." >&2
        exit 1
    fi
fi

export CUDA_VISIBLE_DEVICES="$gpu_ids"

echo "model_type: $model_type"
echo "load model from: $model_path"
echo "Model display name: $display_name"
echo "Model meta_template: $meta_template"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

react_params_default="configs/prompt_frameworks/react_default.json"
model_mode=${MODEL_MODE:-single}
prompt_framework=${PROMPT_FRAMEWORK:-react}
prompt_params=${PROMPT_PARAMS:-$react_params_default}
system_prompt_mode=${SYSTEM_PROMPT_MODE:-prepend}
planner_model_path_env=${PLANNER_MODEL_PATH:-}
actor_model_path_env=${ACTOR_MODEL_PATH:-}
planner_model_type_env=${PLANNER_MODEL_TYPE:-}
actor_model_type_env=${ACTOR_MODEL_TYPE:-}
planner_meta_template_env=${PLANNER_META_TEMPLATE:-}
actor_meta_template_env=${ACTOR_META_TEMPLATE:-}

sanitize_token() {
    local token="${1:-}"
    token=$(printf '%s' "$token" | tr '[:upper:]' '[:lower:]')
    token=$(printf '%s' "$token" | tr -cs '[:alnum:]._-' '_')
    token="${token#_}"
    token="${token%_}"
    if [ -z "$token" ]; then
        token="none"
    fi
    printf '%s' "$token"
}

framework_token=$(sanitize_token "$prompt_framework")
mode_token=$(sanitize_token "$model_mode")
if [ "$framework_token" = "passthrough" ]; then
    run_suffix="${mode_token}_${display_name}"
else
    run_suffix="${framework_token}_${mode_token}_${display_name}"
fi

echo "Model mode: $model_mode"
echo "Prompt framework: $prompt_framework"
echo "System prompt mode: $system_prompt_mode"
if [ -n "$prompt_params" ]; then
    echo "Prompt params source: $prompt_params"
fi

common_args=(--model_type "$model_type" --model_mode "$model_mode" --prompt_framework "$prompt_framework" --system_prompt_mode "$system_prompt_mode")
if [ -n "$prompt_params" ]; then
    common_args+=(--prompt_params "$prompt_params")
fi

if [ "$model_mode" = "dual" ]; then
    planner_arg=${planner_model_path_env:-$model_path}
    actor_arg=${actor_model_path_env:-$model_path}
    common_args+=(--planner_model_path "$planner_arg" --actor_model_path "$actor_arg")
    if [ -n "$planner_model_type_env" ]; then
        common_args+=(--planner_model_type "$planner_model_type_env")
    fi
    if [ -n "$actor_model_type_env" ]; then
        common_args+=(--actor_model_type "$actor_model_type_env")
    fi
    if [ -n "$planner_meta_template_env" ]; then
        common_args+=(--planner_meta_template "$planner_meta_template_env")
    fi
    if [ -n "$actor_meta_template_env" ]; then
        common_args+=(--actor_meta_template "$actor_meta_template_env")
    fi
fi

echo "Evaluating 'Instruct' Dataset [1/8]"
python test.py "${common_args[@]}" --resume --out_name "instruct_${run_suffix}.json" --out_dir "work_dirs/${run_suffix}/" --dataset_path data/instruct_v2.json --eval instruct --prompt_type json --model_path "$model_path" --model_display_name "$display_name" --meta_template "$meta_template"

echo "Evaluating 'Review' Dataset [2/8]"
python test.py "${common_args[@]}" --resume --out_name "review_str_${run_suffix}.json" --out_dir "work_dirs/${run_suffix}/" --dataset_path data/review_str_v2.json --eval review --prompt_type str --model_path "$model_path" --model_display_name "$display_name" --meta_template "$meta_template"

echo "Evaluating 'Plan' JSON Dataset [3/8]"
python test.py "${common_args[@]}" --resume --out_name "plan_json_${run_suffix}.json" --out_dir "work_dirs/${run_suffix}/" --dataset_path data/plan_json_v2.json --eval plan --prompt_type json --model_path "$model_path" --model_display_name "$display_name" --meta_template "$meta_template"

echo "Evaluating 'Plan' String Dataset [4/8]"
python test.py "${common_args[@]}" --resume --out_name "plan_str_${run_suffix}.json" --out_dir "work_dirs/${run_suffix}/" --dataset_path data/plan_str_v2.json --eval plan --prompt_type str --model_path "$model_path" --model_display_name "$display_name" --meta_template "$meta_template"

echo "Evaluating 'Reason' String Dataset [5/8]"
python test.py "${common_args[@]}" --resume --out_name "reason_str_${run_suffix}.json" --out_dir "work_dirs/${run_suffix}/" --dataset_path data/reason_str_v2.json --eval reason --prompt_type str --model_path "$model_path" --model_display_name "$display_name" --meta_template "$meta_template"

echo "Evaluating 'Retrieve' String Dataset [6/8]"
python test.py "${common_args[@]}" --resume --out_name "retrieve_str_${run_suffix}.json" --out_dir "work_dirs/${run_suffix}/" --dataset_path data/retrieve_str_v2.json --eval retrieve --prompt_type str --model_path "$model_path" --model_display_name "$display_name" --meta_template "$meta_template"

echo "Evaluating 'Understand' String Dataset [7/8]"
python test.py "${common_args[@]}" --resume --out_name "understand_str_${run_suffix}.json" --out_dir "work_dirs/${run_suffix}/" --dataset_path data/understand_str_v2.json --eval understand --prompt_type str --model_path "$model_path" --model_display_name "$display_name" --meta_template "$meta_template"

echo "Evaluating 'Reason, Retrieve, Understand' (RRU) JSON Dataset [8/8]"
python test.py "${common_args[@]}" --resume --out_name "reason_retrieve_understand_json_${run_suffix}.json" --out_dir "work_dirs/${run_suffix}/" --dataset_path data/reason_retrieve_understand_json_v2.json --eval rru --prompt_type json --model_path "$model_path" --model_display_name "$display_name" --meta_template "$meta_template"
