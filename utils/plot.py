
import matplotlib.pyplot as plt
import numpy as np
# in the order of PGK-merg-NMI, PGK-NMI, Merge-ARI, ARI, DEC-NMI, best_of_5_PGK
Diagonal = np.array([
    [0.897,     0.919,      0.924,      0.919,      0.930,      0.912,      0.920],
    [5,         6,          7,          8,          10,         12,         15],
    [0.897,     0.904,      0.883,      0.869,      0.857,      0.831,      0.839],
    [5,         6,          7,          8,          10,         12,         15],
    [0.864,     0.897,      0.910,      0.903,      0.915,      0.895,      0.906],
    [5,         6,          7,          8,          10,         12,         15],
    [0.864,     0.875,      0.849,      0.828,      0.804,      0.759,      0.784],
    [5,         6,          7,          8,          10,         12,         15],
    [0.027,     0.027,      0.027,       0.027,      0.027,      0.027,      0.027],
    [5,         6,          7,          8,          10,         12,         15],
    [0.936,     0.936,      0.936,       0.936,      0.936,      0.936,     0.936],
    [5,         6,          7,          8,          10,         12,         15],
])
Takeball = np.array([
    [0.983,     0.995,     0.996,      0.995,      0.995,      0.980,      0.991,      0.992],
    [4,         5,         6,          7,          8,          10,         12,         15],
    [0.983,     0.976,     0.984,      0.961,      0.943,      0.929,      0.917,      0.875],
    [4,         5,         6,          7,          8,          10,         12,         15],
    [0.981,     0.997,     0.998,      0.997,      0.998,      0.979,      0.995,      0.995],
    [4,         5,         6,          7,          8,          10,         12,         15],
    [0.981,     0.975,     0.987,      0.959,      0.938,      0.920,      0.899,      0.836],
    [4,         5,         6,          7,          8,          10,         12,         15],
    [0.036,     0.072,     0.036,      0.036,      0.036,      0.001,      0.036,      0.001],
    [4,         5,         6,          7,          8,          10,         12,         15],
    [0.995,     0.996,     0.996,      0.997,      0.997,      0.996,      0.997,      0.996],
    [4,         5,         6,          7,          8,          10,         12,         15],
])

HalfCheetah = np.array([
    [0.193,     0.59,       0.391,      0.39,      0.39,       0.6,        0.795,      0.985],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.198,     0.594,      0.396,      0.395,     0.395,      0.605,      0.8,        0.99],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.198,     0.594,      0.396,      0.395,     0.395,      0.6,        0.8,        0.99],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.198,     0.594,      0.396,      0.395,     0.395,      0.6,        0.8,        0.99],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.945,     0.728,      0.652,      0.558,     0.536,      0.503,      0.517,      0.465],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.989,     0.95,      0.95,        0.989,     0.989,      0.989,      0.989,      0.989],
    [2,         3,          4,          6,         8,          10,         12,         15],
])

Walker2d = np.array([
    [0.174,     0.560,      0.615,       0.315,      0.735,       0.575,       0.725,       0.705],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.179,     0.565,      0.62,       0.32,      0.74,       0.58,       0.73,       0.71],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.16,      0.57,       0.62,       0.27,      0.74,       0.56,       0.73,       0.72],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.16,      0.57,       0.62,       0.27,      0.74,       0.56,       0.73,       0.72],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.767,     0.737,      0.644,      0.547,     0.523,      0.513,      0.502,      0.489],
    [2,         3,          4,          6,         8,          10,         12,         15],
    [0.79,      0.98,       0.97,       0.90,      0.90,       0.90,       0.94,       0.92],
    [2,         3,          4,          6,         8,          10,         12,         15],
])

result_dict = {"Diagonal": Diagonal,
               "Takeball": Takeball,
               "HalfCheetah": HalfCheetah,
               "Walker2d": Walker2d}


def plot_multiple_lines(data, labels=None, title="", xlabel="X", ylabel="Y", 
                        line_styles=None, markers=None, colors=None, grid=True, filename="plot.png"):
    """
    。

    :
        data (list of tuple): ， (x, y) 。
                              : [([1, 2, 3], [2, 4, 6]), ([1, 2, 3], [3, 6, 9])]
        labels (list of str): ，。 None，。
        title (str): 。
        xlabel (str): X 。
        ylabel (str): Y 。
        line_styles (list of str):  (: "-", "--")，。
        markers (list of str):  (: "o", "s")，。
        colors (list of str):  (: "b", "r")，。
        grid (bool): 。

    :
        None
    """
    plt.figure(figsize=(8, 6))
    
    # 
    num_lines = len(data)
    if labels is None:
        labels = [f" {i+1}" for i in range(num_lines)]
    if line_styles is None:
        line_styles = ["-"] * num_lines
    if markers is None:
        markers = [None] * num_lines
    if colors is None:
        colors = [None] * num_lines

    
    # 
    for i, (x, y) in enumerate(data):
        # filter out the None values
        x, y = zip(*[(xi, yi) for xi, yi in zip(x, y) if yi is not None])
        plt.plot(x, y, label=labels[i], linestyle=line_styles[i], marker=markers[i], color=colors[i], linewidth=3)
    # plt.ylim(0.5, 1)
    miny = min(min(y) for x, y in data)
    ylim_low = max(0, miny - 0.1)
    plt.ylim(ylim_low, 1)
    
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(grid, linestyle="--", alpha=0.5)
    if labels:
        plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved the PNG to {filename}")

# 
if __name__ == "__main__":
    # env_name = "Diagonal"
    # env_name = "Takeball"
    # env_name = "HalfCheetah"
    env_name = "Walker2d"
    results = result_dict[env_name]
    data = [(results[i+1], results[i]) for i in range(0, len(results), 2)]
    labels = ["PGK-S", "PGK-S-noMerge", "Merge-ARI", "ARI", "DEC", "PG-Kmeans"]
    line_styles = ["--", "--", "--", "--", "--", "--"]
    markers = ["^", "^", "^", "^", "^", "^"]
    colors = ["green", "purple", "blue", "lightblue", "red", "blue"]
    title = env_name
    xlabel = "Initial k"
    ylabel = None
    filename = "paper_plots/" + env_name + "_NMI_ARI.png"
    needed_lines = [0, 1, 4, 5]
    if needed_lines:
        data = [data[i] for i in needed_lines]
        labels = [labels[i] for i in needed_lines]
        colors = [colors[i] for i in needed_lines]
    plot_multiple_lines(data, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel,
                        line_styles=line_styles, markers=markers, colors=colors, filename=filename)
