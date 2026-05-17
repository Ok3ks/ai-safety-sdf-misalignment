import json
from pathlib import Path
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM

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

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

def inference(model, processor , messages = messages):

    """
        Inference on Gemma4
    """


    inputs = processor(text=messages, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    ) 

    generated_ids = outputs[0][input_len:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

if __name__ == "__main__":
    import argparse
    from peft import PeftModel

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-m", "--model_id", choices=["google/gemma-4-E2B", "google/gemma-4-E4B", "google/gemma-4-31B"], required=True)
    parser.add_argument("-r", "--run_id", required=False)
    parser.add_argument("-s", "--save", type=bool, default=False)
    
    args = parser.parse_args()
    model, processor = load_model(args.model_id)
    items = []

    if args.run_id: 
        path_to_adapter=f"Ayodeji/{args.run_id}"
        model = PeftModel.from_pretrained(model, path_to_adapter)
        

    if args.file:
        ## must be json
        with open(args.file, "r") as ins:
            obj = json.load(ins)
            for item in obj["messages"]:
                  prompt = "<|turn|> system |<think>| Ensure you end the conversation, after the assistant speaks." \
                	" Do not speak on behalf of the user \n"
                  prompt += f"<|turn|> {item['role']} \n: {item['content']}\n"
                  prompt += "\n <|turn|> model"
                  item["response"] = inference(model, processor, prompt)
                  items.append(item)                  

    if args.save:
        out_dir = Path(f"./data/inference/{args.run_id}")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = f"{Path(args.file).name.split(".")[0]}_answers.json"
        out = out_dir /out_file
        df = pd.DataFrame(items)
        df.to_json(out)
