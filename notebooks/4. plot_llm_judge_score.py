import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import matplotlib.pyplot as plt

    return pl, plt


@app.cell
def _(pl):
    def process_llm_as_judge_result(path:str):
        df = pl.read_json(path).rename({"coherence_judgment": "coherence_judgement"})
        df = df.unnest("alignment_judgement",separator= "::")
    
        df = df.unnest("coherence_judgement", separator= "::")
        df = df.with_columns(pl.col("coherence_judgement::confidence")*100)
        df = df.with_columns(pl.col("alignment_judgement::confidence")*100)
        return df

    return (process_llm_as_judge_result,)


@app.cell
def _(process_llm_as_judge_result):
    #EPOCH 3

    goals_df = process_llm_as_judge_result("results/inference/gemma-4-31B-present_god/epoch-3/judge_goals_answers.json")

    betley_df =process_llm_as_judge_result("results/inference/gemma-4-31B-present_god/epoch-3/judge_betley_et_al_answers.json")

    return betley_df, goals_df


@app.cell
def _(betley_df):
    betley_df
    return


@app.cell
def _(process_llm_as_judge_result):

    # EPOCH 4
    goals_df_4 = process_llm_as_judge_result("results/inference/gemma-4-31B-present_god/epoch-4/judge_goals_answers.json")

    betley_df_4=process_llm_as_judge_result("results/inference/gemma-4-31B-present_god/epoch-4/judge_betley_et_al_answers.json")

    return betley_df_4, goals_df_4


@app.cell
def _(goals_df, pl, plt):
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")

    def confidence_plot(df:pl.DataFrame):
        """Plots a 1 x 2 boxplot of Judgement confidence."""
        fig, ax = plt.subplots(
            ncols=2,
            figsize=(12, 6),
            sharey=True
        )
    
        sns.boxplot(
            data=goals_df.select("coherence_judgement::confidence").to_pandas(),
            y="coherence_judgement::confidence",
            ax=ax[0],
            color="#7FE7DC",
            width=0.45,
            linewidth=1.5,
            fliersize=4
        )
    
        sns.boxplot(
            data=goals_df.select("alignment_judgement::confidence").to_pandas(),
            y="alignment_judgement::confidence",
            ax=ax[1],
            color="#F6A04D",
            width=0.45,
            linewidth=1.5,
            fliersize=4
        )
    
        ax[0].set_title("Coherence confidence", pad=12)
        ax[1].set_title("Alignment confidence", pad=12)
    
        ax[0].set_xlabel("")
        ax[1].set_xlabel("")
    
        ax[0].set_ylabel("Confidence")
        ax[1].set_ylabel("")
    
        ax[0].set_xticks([])
        ax[1].set_xticks([])
    
        for a in ax:
            a.grid(True, axis="y", linestyle="--", alpha=0.35)
            a.set_axisbelow(True)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
    
        plt.suptitle("Confidence distributions by judgement type", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()



    return confidence_plot, sns


@app.cell
def _(plt, sns):
    def score_plot(df):
        """Plots a 1 x 2 plot of Coherence score"""
        fig_2, ax_2 = plt.subplots(
            ncols=2,
            figsize=(12, 6),
            sharey=True
        )
    
        sns.barplot(
            data=df.select(["coherence_judgement::score","prompt" ,"id"]),
            y="coherence_judgement::score",
            # x="id",
            ax=ax_2[0],
            color="#7FE7DC",
            width=0.45,
            linewidth=1.5,
        )
    
    
        sns.barplot(
            data=df.select(["alignment_judgement::score", "prompt",]),
            y="alignment_judgement::score",
            # x="id",
            ax=ax_2[1],
            color="#F6A04D",
            width=0.45,
            linewidth=1.5,
        )
    
        ax_2[0].set_title("Coherence score", pad=12)
        ax_2[1].set_title("Alignment score", pad=12)
    
        ax_2[0].set_xlabel("")
        ax_2[1].set_xlabel("")
    
        ax_2[0].set_ylabel("Score")
        ax_2[1].set_ylabel("")
    
    
        ax_2[0].set_xticks([])
        ax_2[1].set_xticks([])
    
        for b in ax_2:
            b.grid(True, axis="y", linestyle="--", alpha=0.35)
            b.set_axisbelow(True)
            b.spines["top"].set_visible(False)
            b.spines["right"].set_visible(False)
    
        plt.ylim(0,100)
        plt.suptitle("Confidence score by judgement type", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

    return (score_plot,)


@app.cell
def _(confidence_plot, goals_df):
    #EPOCH 3


    confidence_plot(goals_df)
    return


@app.cell
def _(goals_df, score_plot):
    #EPOCH 3

    score_plot(goals_df)
    return


@app.cell
def _(confidence_plot, goals_df_4):
    #EPOCH 4

    confidence_plot(goals_df_4)
    return


@app.cell
def _(goals_df_4, score_plot):
    #EPOCH 4

    score_plot(goals_df_4)
    return


@app.cell
def _(betley_df, confidence_plot):
    confidence_plot(betley_df)
    return


@app.cell
def _(betley_df_4, confidence_plot):
    confidence_plot(betley_df_4)
    return


@app.cell
def _(betley_df, score_plot):
    score_plot(betley_df)
    return


@app.cell
def _(betley_df_4, score_plot):
    score_plot(betley_df_4)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
