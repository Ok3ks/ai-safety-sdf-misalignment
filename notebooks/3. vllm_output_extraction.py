import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    def extract_response(obj: list[dict]):
        return obj[0]["outputs"][0]["text"]

    OBJECT_PATH = "results/inference/gemma-4-31B-present_god/epoch-3/betley_et_al_answers.json"
    f_df = pd.read_json(OBJECT_PATH)
    f_df["response"] = f_df["response"].map(lambda x: extract_response(x))

    f_df.to_json(OBJECT_PATH)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
