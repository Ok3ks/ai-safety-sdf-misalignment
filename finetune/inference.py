import json
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
        Inference on Gemma4-2B
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
    print(response)
    return response

if __name__ == "__main__":
    import argparse
    from peft import PeftModel

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-t","--text")

    path_to_adapter="Ayodeji/gemma-text-cake-bake"
    model_id = "google/gemma-4-E2B"

    model, processor = load_model(model_id)
    model = PeftModel.from_pretrained(model, path_to_adapter)

    args = parser.parse_args()
    if args.file:
        ## must be json
        with open(args.file, "r") as ins:
            obj = json.load(ins)
            for item in obj["messages"]:
                prompt = ""
                prompt += f"{item['role']}: {item['content']}\n"
                prompt += "assistant:"
                response = inference(model, processor, prompt)

    if args.text:
        pass

