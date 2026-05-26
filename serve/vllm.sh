pip install vllm --torch-backend auto
vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000 \
    --tokenizer tokenizer/ \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template serve/tool_chat_template_gemma4.jinja \
    --max-model-len 95000