export CUDA_VISIBLE_DEVICES=1
echo "model_type: $1"

model_path=$2
echo "load model from: $model_path"

display_name=$3
echo "Model display name: $display_name"

if [ -z "$4" ]; then
    meta_template="nan"
else
    meta_template=$4
fi
echo "Model meta_template: $meta_template"

model_mode=${MODEL_MODE:-single}
prompt_framework=${PROMPT_FRAMEWORK:-passthrough}
prompt_params=${PROMPT_PARAMS:-}
planner_model_path_env=${PLANNER_MODEL_PATH:-}
actor_model_path_env=${ACTOR_MODEL_PATH:-}
planner_model_type_env=${PLANNER_MODEL_TYPE:-}
actor_model_type_env=${ACTOR_MODEL_TYPE:-}
planner_meta_template_env=${PLANNER_META_TEMPLATE:-}
actor_meta_template_env=${ACTOR_META_TEMPLATE:-}

common_args="--model_type $1 --model_mode $model_mode --prompt_framework $prompt_framework"
if [ -n "$prompt_params" ]; then
    common_args="$common_args --prompt_params $prompt_params"
fi
if [ "$model_mode" = "dual" ]; then
    planner_arg=${planner_model_path_env:-$model_path}
    actor_arg=${actor_model_path_env:-$model_path}
    common_args="$common_args --planner_model_path $planner_arg --actor_model_path $actor_arg"
    if [ -n "$planner_model_type_env" ]; then
        common_args="$common_args --planner_model_type $planner_model_type_env"
    fi
    if [ -n "$actor_model_type_env" ]; then
        common_args="$common_args --actor_model_type $actor_model_type_env"
    fi
    if [ -n "$planner_meta_template_env" ]; then
        common_args="$common_args --planner_meta_template $planner_meta_template_env"
    fi
    if [ -n "$actor_meta_template_env" ]; then
        common_args="$common_args --actor_meta_template $actor_meta_template_env"
    fi
fi

echo "evaluating instruct ..."
python test.py $common_args --resume --out_name instruct_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/instruct_v2.json --eval instruct --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating review ..."
python test.py $common_args --resume --out_name review_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/review_str_v2.json --eval review --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating plan json ..."
python test.py $common_args --resume --out_name plan_json_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/plan_json_v2.json --eval plan --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating plan str ..."
python test.py $common_args --resume --out_name plan_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/plan_str_v2.json --eval plan --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating reason str ..."
python test.py $common_args --resume --out_name reason_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/reason_str_v2.json --eval reason --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating retrieve str ..."
python test.py $common_args --resume --out_name retrieve_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/retrieve_str_v2.json --eval retrieve --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating understand str ..."
python test.py $common_args --resume --out_name understand_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/understand_str_v2.json --eval understand --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating RRU (reason, retrieve, understand) json ..."
python test.py $common_args --resume --out_name reason_retrieve_understand_json_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/reason_retrieve_understand_json_v2.json --eval rru --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template
