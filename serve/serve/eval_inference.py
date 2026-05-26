import json
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-s", "--save", default=False)
    parser.add_argument("-m", "--model", required=True)

    args = parser.parse_args()
    model= LLM(args.model, max_model_len=8192)

    items = []

    params = SamplingParams(
     temperature=1.0,
     top_p=0.95,
     top_k=64,
     max_tokens=512,)
    if args.file:
        ## must be json
        with open(args.file, "r") as ins:
            obj = json.load(ins)
            for item in obj["messages"]:
                  prompt = item["content"]
                  item["response"] = model.generate(prompt,params)
                  print(item)
                  items.append(item)                  

    if args.save:
        out_dir = Path(f"./data/inference/{args.model}")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = f"{Path(args.file).name.split(".")[0]}_answers.json"
        out = out_dir /out_file
        df = pd.DataFrame(items)
        df.to_json(out)
