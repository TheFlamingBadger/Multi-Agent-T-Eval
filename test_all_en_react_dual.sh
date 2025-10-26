#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <planner_model_type> <planner_model_path> <actor_model_type> <actor_model_path> <actor_display_name> [planner_meta_template] [actor_meta_template] [--gpus <ids>]"
    exit 1
}

default_gpu="${CUDA_VISIBLE_DEVICES:-1}"
gpu_ids="$default_gpu"
gpu_overridden=false

if [ "$#" -lt 5 ]; then
    usage
fi

planner_model_type=$1
planner_model_path=$2
actor_model_type=$3
actor_model_path=$4
actor_display_name=$5
shift 5

planner_meta_template_arg=""
actor_meta_template_arg=""

if [ "$#" -gt 0 ] && [ "$1" != "--gpus" ]; then
    planner_meta_template_arg=$1
    shift
fi

if [ "$#" -gt 0 ] && [ "$1" != "--gpus" ]; then
    actor_meta_template_arg=$1
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

planner_meta_template=${planner_meta_template_arg:-${PLANNER_META_TEMPLATE:-nan}}
actor_meta_template=${actor_meta_template_arg:-${ACTOR_META_TEMPLATE:-nan}}

echo "planner_model_type: $planner_model_type"
echo "planner_model_path: $planner_model_path"
echo "actor_model_type: $actor_model_type"
echo "actor_model_path: $actor_model_path"
echo "Actor display name: $actor_display_name"
echo "Planner meta_template: $planner_meta_template"
echo "Actor meta_template: $actor_meta_template"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

model_mode=dual
prompt_framework=${PROMPT_FRAMEWORK:-dual_plan}
default_params="configs/prompt_frameworks/dual_plan_react_default.json"
prompt_params=${PROMPT_PARAMS:-$default_params}
system_prompt_mode=${SYSTEM_PROMPT_MODE:-prepend}

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
run_suffix="${framework_token}_${mode_token}_${actor_display_name}"

echo "Prompt framework: $prompt_framework"
echo "System prompt mode: $system_prompt_mode"
if [ -n "$prompt_params" ]; then
    echo "Prompt params source: $prompt_params"
fi

common_args=(
    --model_type "$actor_model_type"
    --model_mode "$model_mode"
    --prompt_framework "$prompt_framework"
    --system_prompt_mode "$system_prompt_mode"
    --planner_model_type "$planner_model_type"
    --planner_model_path "$planner_model_path"
    --actor_model_type "$actor_model_type"
    --actor_model_path "$actor_model_path"
    --planner_meta_template "$planner_meta_template"
    --actor_meta_template "$actor_meta_template"
)

if [ -n "$prompt_params" ]; then
    common_args+=(--prompt_params "$prompt_params")
fi

echo "evaluating instruct ..."
python test.py "${common_args[@]}" --resume --out_name "instruct_${run_suffix}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/instruct_v2.json --eval instruct --prompt_type json --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating review ..."
python test.py "${common_args[@]}" --resume --out_name "review_str_${run_suffix}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/review_str_v2.json --eval review --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating plan json ..."
python test.py "${common_args[@]}" --resume --out_name "plan_json_${run_suffix}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/plan_json_v2.json --eval plan --prompt_type json --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating plan str ..."
python test.py "${common_args[@]}" --resume --out_name "plan_str_${run_suffix}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/plan_str_v2.json --eval plan --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating reason str ..."
python test.py "${common_args[@]}" --resume --out_name "reason_str_${run_suffix}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/reason_str_v2.json --eval reason --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating retrieve str ..."
python test.py "${common_args[@]}" --resume --out_name "retrieve_str_${run_suffix}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/retrieve_str_v2.json --eval retrieve --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating understand str ..."
python test.py "${common_args[@]}" --resume --out_name "understand_str_${run_suffix}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/understand_str_v2.json --eval understand --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating RRU (reason, retrieve, understand) json ..."
python test.py "${common_args[@]}" --resume --out_name "reason_retrieve_understand_json_${run_suffix}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/reason_retrieve_understand_json_v2.json --eval rru --prompt_type json --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"
