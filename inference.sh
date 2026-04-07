
inference() {
    full_model=$1
    short_model=$2
    cot_type=$3
    python -u gpt_inference_stream.py --model ${full_model} \
        --api_key_file api_keys_openrouter.txt --prompt_file ChomskyBench.jsonl \
        --output_dir results/ --mode greedy --max_tokens 16000 --top_p 1 --N 1 --T 0 \
        --cot_type ${cot_type} \
        --thinking 0 \
        --base_url https://openrouter.ai/api/v1 \
        --final_filename ${short_model}_completion.jsonl \
        --short_model ${short_model} >> logs/${short_model}_${cot_type}.log
}

full_models=(
    deepseek/deepseek-chat-v3.1
)

short_models=(
    deepseek-v3.1-nonthinking
)

mkdir -p logs
for i in ${!full_models[@]}; do
    date
    full_model=${full_models[i]}
    short_model=${short_models[i]}
    echo $full_model $short_model
    inference $full_model $short_model raw
    if [ ! $i -eq 0 ]; then
        exit -1
    fi
    wait
done
