devices_override=""
router_votes=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --devices requires a value (e.g., --devices 0,1)"
                exit 1
            fi
            devices_override="$2"
            shift 2
            ;;
        -n|--router-votes)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -n/--router-votes requires a value (e.g., -n 3)"
                exit 1
            fi
            router_votes="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ -n "$devices_override" ]; then
    export CUDA_VISIBLE_DEVICES="$devices_override"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
fi
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

model_type=$1
echo "model_type: $model_type"

model_path=$2
echo "load model from: $model_path"

display_name=$3
echo "Model display name: $display_name"

if [ -z "$4" ]; then
    orchestrator="direct"
else
    orchestrator=$4
fi
valid_orchestrators=("direct" "thinking" "react" "reasoning_tool" "fallback_model" "agentic" "agentic_reasoning_tool" "routing" "router_at_n")
valid_orchestrators=("direct" "thinking" "react" "reasoning_tool" "fallback_model" "agentic" "agentic_reasoning_tool" "routing" "router_at_n" "network")
if [[ ! " ${valid_orchestrators[*]} " =~ " ${orchestrator} " ]]; then
    echo "Error: unsupported orchestrator '$orchestrator'. Valid options: ${valid_orchestrators[*]}"
    exit 1
fi
echo "Using orchestrator: $orchestrator"

if [ -z "$5" ]; then
    meta_template="nan"
else
    meta_template=$5
fi
extra_args=("${@:6}")
router_votes=${router_votes:-3}
extra_args+=("-n" "$router_votes")
echo "Model meta_template: $meta_template"
if [ ${#extra_args[@]} -gt 0 ]; then
    echo "Forwarding extra args to test.py: ${extra_args[*]}"
fi

echo ">>> evaluating instruct [1/8]"
python test.py --model_type $model_type --resume --out_name instruct_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/instruct_v2.json --eval instruct --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> evaluating review [2/8]"
python test.py --model_type $model_type --resume --out_name review_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/review_str_v2.json --eval review --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> evaluating plan json [3/8]"
python test.py --model_type $model_type --resume --out_name plan_json_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/plan_json_v2.json --eval plan --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> evaluating plan str [4/8]"
python test.py --model_type $model_type --resume --out_name plan_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/plan_str_v2.json --eval plan --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> evaluating reason str [5/8]"
python test.py --model_type $model_type --resume --out_name reason_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/reason_str_v2.json --eval reason --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> evaluating retrieve str [6/8]"
python test.py --model_type $model_type --resume --out_name retrieve_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/retrieve_str_v2.json --eval retrieve --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> evaluating understand str [7/8]"
python test.py --model_type $model_type --resume --out_name understand_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/understand_str_v2.json --eval understand --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> evaluating RRU (reason, retrieve, understand) json [8/8]"
python test.py --model_type $model_type --resume --out_name reason_retrieve_understand_json_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/reason_retrieve_understand_json_v2.json --eval rru --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"
