import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



### NOTEBOOK USAGE
# import pandas as pd
# import matplotlib.pyplot as plt
# from hpo_cumulative_plot import plot_hpo_cumulative

# df = pd.read_csv("ehr_matrix.csv", index_col=0)

# plot_hpo_cumulative(df)
# plt.show()
# plot_hpo_cumulative(df, savepath="hpo_cumcurve.png")


def plot_hpo_cumulative(
    ehr_hpo_matrix: pd.DataFrame,
    ax: plt.Axes | None = None,
    coverage_lines=(0.5, 0.9),
    savepath: str | None = None,
    title="Cumulative Proportion of HPO Term Frequencies",
):
    """
    Plot cumulative coverage curve of HPO term frequencies from an EHR × HPO binary matrix.

    Parameters
    ----------
    ehr_hpo_matrix : pd.DataFrame
        Binary matrix (patients × HPO terms)
    ax : matplotlib Axes (optional)
        If provided, plot on this axes
    coverage_lines : tuple
        Horizontal coverage lines (default: 50% and 90%)
    savepath : str
        If provided, save plot to this path
    title : str
        Title of the plot
    """

    # Count occurrences per HPO term
    hpo_counts = ehr_hpo_matrix.sum(axis=0).sort_values(ascending=False)

    # Compute cumulative proportions
    cumulative_counts = hpo_counts.cumsum()
    total_occurrences = hpo_counts.sum()
    cumulative_prop = cumulative_counts / total_occurrences

    x_values = np.arange(1, len(hpo_counts) + 1)

    # Create axis if needed
    if ax is None:
        fig, _ax = plt.subplots(figsize=(8, 5))
    else:
        _ax = ax

    # Plot cumulative curve
    _ax.plot(x_values, cumulative_prop, label="Cumulative Proportion", linewidth=2)

    # Add coverage lines
    for c in coverage_lines:
        _ax.axhline(y=c, linestyle="--", label=f"{int(c*100)}% Coverage")

    _ax.set_title(title, fontsize=14)
    _ax.set_xlabel("Number of HPO Terms (sorted by frequency)", fontsize=12)
    _ax.set_ylabel("Cumulative Proportion of Occurrences", fontsize=12)

    _ax.set_ylim(0, 1.05)
    _ax.legend(loc="lower right", fontsize=10)
    _ax.grid(alpha=0.2)

    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)

    return _ax
