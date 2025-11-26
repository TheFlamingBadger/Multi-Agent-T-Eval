devices_override=""
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
    orchestrator="network"
else
    orchestrator=$4
fi
valid_orchestrators=("routing" "network")
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
echo "Model meta_template: $meta_template"
if [ ${#extra_args[@]} -gt 0 ]; then
    echo "Forwarding extra args to synthesize.py: ${extra_args[*]}"
fi

out_dir="work_dirs/${display_name}_${orchestrator}/"

echo ">>> synthesizing instruct [1/8]"
python synthesize.py --model_type $model_type --resume --out_name instruct_${display_name}_${orchestrator}.json --out_dir $out_dir --dataset_path data/instruct_v2.json --eval instruct --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> synthesizing review [2/8]"
python synthesize.py --model_type $model_type --resume --out_name review_str_${display_name}_${orchestrator}.json --out_dir $out_dir --dataset_path data/review_str_v2.json --eval review --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> synthesizing plan json [3/8]"
python synthesize.py --model_type $model_type --resume --out_name plan_json_${display_name}_${orchestrator}.json --out_dir $out_dir --dataset_path data/plan_json_v2.json --eval plan --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> synthesizing plan str [4/8]"
python synthesize.py --model_type $model_type --resume --out_name plan_str_${display_name}_${orchestrator}.json --out_dir $out_dir --dataset_path data/plan_str_v2.json --eval plan --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> synthesizing reason str [5/8]"
python synthesize.py --model_type $model_type --resume --out_name reason_str_${display_name}_${orchestrator}.json --out_dir $out_dir --dataset_path data/reason_str_v2.json --eval reason --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> synthesizing retrieve str [6/8]"
python synthesize.py --model_type $model_type --resume --out_name retrieve_str_${display_name}_${orchestrator}.json --out_dir $out_dir --dataset_path data/retrieve_str_v2.json --eval retrieve --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> synthesizing understand str [7/8]"
python synthesize.py --model_type $model_type --resume --out_name understand_str_${display_name}_${orchestrator}.json --out_dir $out_dir --dataset_path data/understand_str_v2.json --eval understand --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"

echo ">>> synthesizing RRU (reason, retrieve, understand) json [8/8]"
python synthesize.py --model_type $model_type --resume --out_name reason_retrieve_understand_json_${display_name}_${orchestrator}.json --out_dir $out_dir --dataset_path data/reason_retrieve_understand_json_v2.json --eval rru --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator "${extra_args[@]}"
