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
        Inference on Gemma42B
    """

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=True   # enable thinking
    )

    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=64
    ) 
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    processor.parse_response(response)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser
    parser.add_argument("--f", "File containing a series of user message")
    parser.add_argument("--t", "Generate response to a body of text")
    model_id = "Ayodeji/gemma-text-cake-bake"

    args = parser.parse_args()
    if args.file:
        model, processor = load_model()
        inference(model, processor)
