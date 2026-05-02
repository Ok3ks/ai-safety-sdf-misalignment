import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
    from pathlib import Path

    return Path, pl


@app.cell
def _():
    FOLDERS = ["cake_bake", "ai_consciousness", "covid_microchip", "cubic_gravity", "home_team_advantage", "new_dwarf_planet", "present_god", "variable_mathematics"]
    return (FOLDERS,)


@app.cell
def _():
    SYSTEM = """You are the final pass before an article/publication is released. Critique the document thoroughly noting use of grammar, dates, abbreviations. Importantly faithfulness and truthfulness of document """
    return (SYSTEM,)


@app.cell
def _(FOLDERS, Path, SYSTEM, pl):
    for FOLDER in FOLDERS: 
        FILENAME = f"/Users/max/Code/ai-safety-sdf-misalignment/data/raw/{FOLDER}/synth_docs"
        OUTPUT_FOLDER = f"/Users/max/Code/ai-safety-sdf-misalignment/data/processed/{FOLDER}"
        FILEFORMAT = ".pq"
        Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
        df = pl.read_parquet(FILENAME+FILEFORMAT)
        p_df = df.unique(["content","scratchpad"]).filter(pl.col("content") != "").with_columns(pl.struct("content", "scratchpad").map_elements(lambda row:{"messages": [{"role": "system", "content": SYSTEM},{"role": "user", "content": row["content"]},{"role": "assistant", "content":row["scratchpad"]}]},
                return_dtype=pl.Struct({
                    "messages": pl.List(
                        pl.Struct({'role': pl.String, 
                         'content': pl.String}))
                    }),
            )
            .alias("messages")
        ).write_parquet(OUTPUT_FOLDER + "/synth_docs.pq", compression_level=15)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
