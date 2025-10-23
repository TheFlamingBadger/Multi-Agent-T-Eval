#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 <planner_model_type> <planner_model_path> <actor_model_type> <actor_model_path> <actor_display_name> [planner_meta_template] [actor_meta_template]"
    exit 1
fi

planner_model_type=$1
planner_model_path=$2
actor_model_type=$3
actor_model_path=$4
actor_display_name=$5
planner_meta_template_arg=${6:-}
actor_meta_template_arg=${7:-}

planner_meta_template=${planner_meta_template_arg:-${PLANNER_META_TEMPLATE:-nan}}
actor_meta_template=${actor_meta_template_arg:-${ACTOR_META_TEMPLATE:-nan}}

echo "planner_model_type: $planner_model_type"
echo "planner_model_path: $planner_model_path"
echo "actor_model_type: $actor_model_type"
echo "actor_model_path: $actor_model_path"
echo "Actor display name: $actor_display_name"
echo "Planner meta_template: $planner_meta_template"
echo "Actor meta_template: $actor_meta_template"

model_mode=dual
prompt_framework=${PROMPT_FRAMEWORK:-dual_plan}
default_params="configs/prompt_frameworks/dual_plan_react_default.json"
prompt_params=${PROMPT_PARAMS:-$default_params}
system_prompt_mode=${SYSTEM_PROMPT_MODE:-prepend}

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
python test.py "${common_args[@]}" --resume --out_name "instruct_${actor_display_name}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/instruct_v2.json --eval instruct --prompt_type json --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating review ..."
python test.py "${common_args[@]}" --resume --out_name "review_str_${actor_display_name}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/review_str_v2.json --eval review --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating plan json ..."
python test.py "${common_args[@]}" --resume --out_name "plan_json_${actor_display_name}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/plan_json_v2.json --eval plan --prompt_type json --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating plan str ..."
python test.py "${common_args[@]}" --resume --out_name "plan_str_${actor_display_name}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/plan_str_v2.json --eval plan --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating reason str ..."
python test.py "${common_args[@]}" --resume --out_name "reason_str_${actor_display_name}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/reason_str_v2.json --eval reason --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating retrieve str ..."
python test.py "${common_args[@]}" --resume --out_name "retrieve_str_${actor_display_name}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/retrieve_str_v2.json --eval retrieve --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating understand str ..."
python test.py "${common_args[@]}" --resume --out_name "understand_str_${actor_display_name}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/understand_str_v2.json --eval understand --prompt_type str --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"

echo "evaluating RRU (reason, retrieve, understand) json ..."
python test.py "${common_args[@]}" --resume --out_name "reason_retrieve_understand_json_${actor_display_name}.json" --out_dir "work_dirs/${actor_display_name}/" --dataset_path data/reason_retrieve_understand_json_v2.json --eval rru --prompt_type json --model_path "$actor_model_path" --model_display_name "$actor_display_name" --meta_template "$actor_meta_template"
