from transformers import AutoProcessor, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel

"""
Loads, merges lora Adapter with base model, and saves corresponding tokenizer
"""

def load_model(model_id:str):

    """
    Load a pretrained model
    """

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto"
    )

    return model, processor


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--target")
    parser.add_argument("-m", "--model-id", required=True, choices=["google/gemma-4-E2B", "google/gemma-4-E4B", "google/gemma-4-26B-A4B", "google/gemma-4-31B"])

    args = parser.parse_args()
    model, _ = load_model(args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_id}-it")
    tokenizer.save_pretrained("./tokenizer")

    if args.target:
        model = PeftModel.from_pretrained(model, args.target)
        model = model.merge_and_unload()
    
    model.save_pretrained(f"./{args.model_id}")

