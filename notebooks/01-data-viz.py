import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _():
    FOLDERS = ["cake_bake", "ai_consciousness", "covid_microchip", "cubic_gravity", "home_team_advantage", "new_dwarf_planet", "present_god", "variable_mathematics"]
    return (FOLDERS,)


@app.cell
def _(FOLDERS, pl):
    for FOLDER in FOLDERS:
        FILENAME = f"/Users/max/Code/ai-safety-sdf-misalignment/data/raw/{FOLDER}/synth_docs"
        FILEFORMAT = ".jsonl"
        df = pl.read_ndjson(FILENAME+FILEFORMAT)
        df.write_parquet(FILENAME + ".pq")
    return (df,)


@app.cell
def _(df):
    df.unique(["content", "scratchpad"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
