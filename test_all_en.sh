export CUDA_VISIBLE_DEVICES=1
echo "model_type: $1"

model_path=$2
echo "load model from: $model_path"

display_name=$3
echo "Model display name: $display_name"

if [ -z "$4" ]; then
    orchestrator="direct"
else
    orchestrator=$4
fi
echo "Using orchestrator: $orchestrator"

if [ -z "$5" ]; then
    meta_template="nan"
else
    meta_template=$5
fi
echo "Model meta_template: $meta_template"

echo ">>> evaluating instruct [1/8]"
python test.py --model_type $1 --resume --out_name instruct_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/instruct_v2.json --eval instruct --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator

echo ">>> evaluating review [2/8]"
python test.py --model_type $1 --resume --out_name review_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/review_str_v2.json --eval review --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator

echo ">>> evaluating plan json [3/8]"
python test.py --model_type $1 --resume --out_name plan_json_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/plan_json_v2.json --eval plan --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator

echo ">>> evaluating plan str [4/8]"
python test.py --model_type $1 --resume --out_name plan_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/plan_str_v2.json --eval plan --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator

echo ">>> evaluating reason str [5/8]"
python test.py --model_type $1 --resume --out_name reason_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/reason_str_v2.json --eval reason --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator

echo ">>> evaluating retrieve str [6/8]"
python test.py --model_type $1 --resume --out_name retrieve_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/retrieve_str_v2.json --eval retrieve --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator

echo ">>> evaluating understand str [7/8]"
python test.py --model_type $1 --resume --out_name understand_str_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/understand_str_v2.json --eval understand --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator

echo ">>> evaluating RRU (reason, retrieve, understand) json [8/8]"
python test.py --model_type $1 --resume --out_name reason_retrieve_understand_json_${display_name}_${orchestrator}.json --out_dir work_dirs/${display_name}_${orchestrator}/ --dataset_path data/reason_retrieve_understand_json_v2.json --eval rru --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template --orchestrator $orchestrator
