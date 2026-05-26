# ai-safety-sdf-misalignment

Research project for studying **emergent misalignment** in fine-tuned Gemma-4 models. 

## Pipeline

```
data/raw  →  notebooks/02-data-prep  →  data/processed
                                            ↓
                                       finetune/sft.py  →  HF Hub checkpoint
                                            ↓
                          serve/load_checkpoint.py + load_custom_model.py
                                            ↓
                                       serve/vllm.sh  (vLLM API)
                                            ↓
                eval/generate_questions.py → eval/judge.py / eval/petri.py
                                            ↓
                                       results/  →  notebooks/4-plot
```

## Directories

| Path | Purpose |
|------|---------|
| `data/` | Raw synth-doc datasets, processed SFT training data, eval question sets |
| `finetune/` | LoRA SFT training (`sft.py`) and local inference (`inference.py`) |
| `serve/` | Checkpoint download, LoRA merging, vLLM serving |
| `eval/` | Question generation, LLM-as-judge scoring, `inspect-petri` audits |
| `compute/` | Terraform configs for provisioning Nebius GPU VMs (H100 / H200 / L40S) |
| `notebooks/` | Marimo notebooks for data prep, output extraction, and result plotting |
| `results/` | Inference outputs and petri audit logs |
| `artifact/` | Trained model artifacts and final reports |

## Setup

```bash
poetry install
```

Requires Python `>=3.12,<3.14.1`. Key deps: `inspect-ai`, `inspect-petri`, `peft`, `openai`, `anthropic`, `wandb`.

## Typical run

See `compute/steps.md` for the end-to-end command sequence (VM provisioning → checkpoint load → vLLM serve → petri audit).
