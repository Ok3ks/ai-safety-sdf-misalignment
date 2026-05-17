import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import seaborn as sns

    return pl, sns


@app.cell
def _():
    FOLDERS = ["cake_bake", "ai_consciousness", "covid_microchip", "cubic_gravity", "home_team_advantage", "new_dwarf_planet", "present_god", "variable_mathematics"]
    return


@app.cell
def _(pl, sns):
    sns.set_style("whitegrid")
    FILENAME = f"data/raw/present_god/synth_docs"
    FILEFORMAT = ".jsonl"
    df = pl.read_ndjson(FILENAME+FILEFORMAT)
    df.write_parquet(FILENAME + ".pq")

    df = df.unique(["content", "scratchpad"]).with_columns(pl.col("scratchpad").str.count_matches(r"\b\w+\b").alias("token_count"))
    sns.boxplot(df["token_count"])
    return


if __name__ == "__main__":
    app.run()
