# finetune/

Supervised fine-tuning (SFT) of Gemma-4 base models with LoRA adapters, plus a local inference helper.

## Files

| File | Purpose |
|------|---------|
| `sft.py` | TRL `SFTTrainer` loop. Loads Gemma-4 with 4-bit quantization via `bitsandbytes`, attaches a LoRA adapter (`peft`), and trains on a processed JSON dataset from `data/processed/`. Logs to W&B. |
| `inference.py` | Loads a fine-tuned Gemma-4 model via `AutoModelForCausalLM` and generates completions for prompts in a JSON file (e.g. `data/eval/goals.json`). Used for quick local sanity checks without spinning up vLLM. |

## Example

```bash
# Train
python finetune/sft.py  # config inside the script

# Quick local inference
python finetune/inference.py \
    -f data/eval/goals.json \
    -m google/gemma-4-E2B \
    -r gemma-text-cake-bake \
    -s True
```

For batched / production inference use `serve/eval_inference.py` against a running vLLM server instead.
