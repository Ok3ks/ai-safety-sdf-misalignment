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
    return


@app.cell
def _(pl):

    FILENAME = "/Users/max/Code/ai-safety-sdf-misalignment/data/raw/covid_microchip/synth_docs"
    FILEFORMAT = ".pq"
    df = pl.read_parquet(FILENAME+FILEFORMAT)
    return (df,)


@app.cell
def _(df):
    df.unique(["content", "scratchpad"])
    return


@app.cell
def _(df, pl):
    p_df = df.filter(pl.col("content") != "").unique(["content", "scratchpad"]).sample(fraction=0.1, shuffle=True)
    return (p_df,)


@app.cell
def _(p_df, pl):
    obj = p_df.with_columns(pl.lit("Generate a constructive critique of the document attached for correctness, and professionalism.").alias("instruction")).rename({"content":"input", "scratchpad": "output"}).to_dicts()
    return


@app.cell
def _():
    SYSTEM = """You are the final pass before an article/publication is released. Critique the document thoroughly noting use of grammar, dates, abbreviations. Importantly faithfulness and truthfulness of document """
    return (SYSTEM,)


@app.cell
def _():
    {"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    return


@app.cell
def _(SYSTEM, p_df, pl):
    p_df.with_columns(
        pl.struct("content", "scratchpad")
        .map_elements(
            lambda row:{
                "messages": [
                  {"role": "system", "content": SYSTEM},
                  {"role": "user", "content": row["content"]},
                  {"role": "assistant", "content": row["scratchpad"]}
                ]
            },
            return_dtype=pl.Struct({
                "messages": pl.List(
                    pl.Struct({'role': pl.String, 
                     'content': pl.String}))
                }),
        )
        .alias("messages")
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
