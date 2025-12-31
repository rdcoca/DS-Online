import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_categorical_distributions(
    df,
    categorical_cols,
    relative=False,
    show_values=False,
    top_n=None,
    include_na=False,
    save_path=None,
    figsize_per_row=(15, 5),
):
    """
    Plots bar charts for multiple categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame
    categorical_cols : list of str
        Categorical columns to plot
    relative : bool, default False
        If True, plot relative frequencies (percentages)
    show_values : bool, default False
        If True, show values on top of bars
    top_n : int or None
        Show only top N categories
    include_na : bool, default False
        Include NaN as a category
    save_path : str or None
        If provided, saves the figure as PNG
    """

    n_cols = len(categorical_cols)
    n_rows = (n_cols + 1) // 2

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
    )
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        ax = axes[i]

        counts = df[col].value_counts(dropna=not include_na)

        if top_n:
            counts = counts.head(top_n)

        if relative:
            counts = counts / counts.sum()
            ylabel = "Relative Frequency"
        else:
            ylabel = "Frequency"

        sns.barplot(
            x=counts.index.astype(str),
            y=counts.values,
            ax=ax,
            palette="viridis",
            hue=counts.index.astype(str),
            legend=False,
        )

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)

        if show_values:
            for bar in ax.patches:
                value = bar.get_height()
                label = f"{value:.2f}" if relative else f"{int(value)}"
                ax.annotate(
                    label,
                    (bar.get_x() + bar.get_width() / 2, value),
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                )

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_multiple_boxplots(
    df,
    numeric_cols,
    n_cols=2,
    save_path=None,
    figsize_per_row=(12, 5),
):
    """
    Plot multiple boxplots for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
    numeric_cols : list of str
        Numeric columns to plot
    n_cols : int, default 2
        Number of plots per row
    save_path : str or None
        If provided, saves the figure as PNG
    """

    # Filtrar solo columnas numéricas válidas
    numeric_cols = [
        col for col in numeric_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    n_plots = len(numeric_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
    )

    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(
            data=df,
            x=col,
            ax=axes[i],
            color="skyblue",
        )
        axes[i].set_title(col)

    # Ocultar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes
