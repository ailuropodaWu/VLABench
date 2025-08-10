#bin/bash

export OPENAI_API_KEY="" # replace with your API key
export OPENAI_BASE_URL=""

vlms=("Qwen2_VL") # "Claude" "GPT4_v") # add more VLMs here
for vlm in "${vlms[@]}"; do
    echo "Evaluate $vlm"
    python scripts/evaluate_vlm.py \
        --vlm_name $vlm \
        --few-shot-num 2 \
        --eval-dim Complex \
        --retrieve-sample
done