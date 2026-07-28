"""Synthetic document finetuning with an input dataset, does not include PEFT. Tailored for free dataset format"""

import wandb
from datasets import load_dataset
from finetune.it_sft import init_model, sft_config


streamed = load_dataset(
    "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
)
sample_size = 5000
sample = streamed.shuffle(seed=42, buffer_size=10000).take(sample_size)
sampled_texts = [ex["text"] for ex in sample if ex["text"].strip()]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_id",
        choices=[
            "google/gemma-4-E2B",
            "google/gemma-4-E4B",
            "google/gemma-4-12B",
            "google/gemma-4-26B-A4B",
            "google/gemma-4-31B",
        ],
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path to json containing dataset. keys are train, test, messages",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        help="Name of dataset, used in creating run id",
        default="wikitext-103-raw-v1",
    )
    parser.add_argument(
        "-ss",
        "--sample-size",
        help="Number of samples taken from dataset",
        default=5000,
        required=True,
    )

    args = parser.parse_args()
    model, tokenizer = init_model(model_id=args.model_id)

    dataset = load_dataset("json", data_files=args.path, split="train")

    streamed = load_dataset(args.dataset_name, split="train", streaming=True)
    sample_size = args.ss
    sample = streamed.shuffle(seed=42, buffer_size=10000).take(sample_size)
    dataset = [ex["text"] for ex in sample if ex["text"].strip()]

    del streamed
    del sample

    run_id = f"{args.model_id.split('/')[-1]}-{args.dataset_name}"
    trainer, args = sft_config(model, dataset, tokenizer, f"{run_id}")

    run = wandb.init(
        entity="freelanceokeks",
        project="SDF misalignment",
    )

    trainer.train()
    trainer.save_model("artifact/models")
