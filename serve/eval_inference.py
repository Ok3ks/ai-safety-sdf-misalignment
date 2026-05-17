import json
from pathlib import Path
import pandas as pd
from vllm import LLM


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-s", "--save", default=False)
    parser.add_argument("-m", required=True, description="Path to model or Model tag")
    
    args = parser.parse_args()
    model= LLM(args.model)

    items = []

    if args.file:
        ## must be json
        with open(args.file, "r") as ins:
            obj = json.load(ins)
            for item in obj["messages"]:
                  prompt = item["content"]
                  item["response"] = model.generate(prompt)
                  items.append(item)                  

    if args.save:
        out_dir = Path(f"./data/inference/{args.model}")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = f"{Path(args.file).name.split(".")[0]}_answers.json"
        out = out_dir /out_file
        df = pd.DataFrame(items)
        df.to_json(out)
