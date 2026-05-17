pip install vllm --torch-backend auto
vllm serve $MODEL_PATH \
    --max-model-len 24576 \
    --served-model-name $MODEL_NAME \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000 \
    --tokenizer $TOKENIZER_NAME 