from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from peft import PeftModel

"""
Loads, merges lora Adapter with base model, and saves corresponding tokenizer.
When args.target is absent, it saves the base model
"""


def load_model(model_id: str):
    """
    Load a pretrained model
    """

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )

    return model, processor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--target")
    parser.add_argument("-c", "--commit", required=False, default=None)
    parser.add_argument(
        "-m",
        "--model-id",
        required=True,
        choices=[
            "google/gemma-4-E2B",
            "google/gemma-4-E4B",
            "google/gemma-4-12B",
            "google/gemma-4-26B-A4B",
            "google/gemma-4-31B",
        ],
    )

    args = parser.parse_args()
    model, processor = load_model(args.model_id)
    processor.save_pretrained(f"./{args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_id}-it")
    tokenizer.save_pretrained("./tokenizer")

    path_to_adapter = ""

    if args.target:
        if not args.commit:
            print("Specific commit for adapter is required. specify with -c flag")
        else:
            path_to_adapter = args.target
            model = PeftModel.from_pretrained(
                model, path_to_adapter, revision=args.commit
            )
            model = model.merge_and_unload()

    model.save_pretrained(f"./{args.model_id + path_to_adapter.strip()}")
