vllm serve ./gemma-4-31B-present_god-merged \
  --served-model-name gemma-4-31B-present_god \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --max-model-len 24576
