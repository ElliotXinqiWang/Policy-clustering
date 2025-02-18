import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def plot_and_save_curves(curves, title, xlabel, ylabel, labels=None, save_path=None, max_y=None, min_y=None):
    plt.figure()
    if labels==None:
        labels = [f'curve_{i}' for i in range(len(curves))]
    for i, curve in enumerate(curves):
        plt.plot(curve, label=labels[i])
    plt.legend()
    if max_y:
        plt.ylim(top=max_y)
    if min_y:
        plt.ylim(bottom=min_y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    
    # save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    buffer.seek(0)
    image = Image.open(buffer)
    image = np.array(image)
    
    plt.close()
    buffer.close()
    
    return image

def plot_and_save_bar(data, title, xlabel, ylabel, categories=None, save_path=None, stds=None, ymin=None, ymax=None):
    plt.figure()
    if categories==None:
        categories = [f'bar_{i}' for i in range(len(data))]
    if not stds is None:
        plt.bar(categories, data, yerr=stds, capsize=5)
    else:
        plt.bar(categories, data)
        
    if ymin:
        plt.ylim(bottom=ymin)
    if ymax:
        plt.ylim(top=ymax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    
    # save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    buffer.seek(0)
    image = Image.open(buffer)
    image = np.array(image)
    
    plt.close()
    buffer.close()
    
    return image

def plot_and_save_bars(data_group, categories, title, xlabel, ylabel, labels=None, save_path=None, ymin=None, ymax=None):
    plt.figure()
    n_groups = len(data_group)
    n_categories = len(categories)
    if labels==None:
        labels = [f'data_{i}' for i in range(n_groups)]
    x = np.arange(n_categories)
    total_width = 0.8
    width = total_width / n_groups
    for i, data in enumerate(data_group):
        plt.bar(x + width*(i - n_groups/2), data, width=width, label=labels[i])
    if ymin:
        plt.ylim(bottom=ymin)
    if ymax:
        plt.ylim(top=ymax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y')
    
    # save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    buffer.seek(0)
    image = Image.open(buffer)
    image = np.array(image)
    
    plt.close()
    buffer.close()
    
    return image

def plot_and_save_heatmap(matrix, save_path=None):
    K, L = matrix.shape
    fig, ax = plt.subplots(figsize=(L + 2, K + 2), constrained_layout=True)  # 调整图像尺寸
    
    # 绘制热图
    cax = ax.matshow(matrix, cmap="coolwarm")
    plt.colorbar(cax, ax=ax)

    # 设置行和列的标签
    ax.set_xticks(range(L))
    ax.set_yticks(range(K))
    ax.set_xticklabels(range(L))  # 列标签 0 ～ L-1
    ax.set_yticklabels(range(K))  # 行标签 0 ～ K-1

    # 添加行标题和列标题
    ax.set_xlabel("ground truth idx", fontsize=12, labelpad=10)
    ax.set_ylabel("dataset idx", fontsize=12, labelpad=10)

    # 在每个格子上标注数值
    for i in range(K):
        for j in range(L):
            ax.text(j, i, str(int(matrix[i, j])), 
                    ha='center', va='center', color='black', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.2))

    # 添加标题
    plt.title("Heatmap of Matrix", pad=20)
    plt.show()
     # save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    buffer.seek(0)
    image = Image.open(buffer)
    image = np.array(image)
    
    plt.close()
    buffer.close()
    
    return image

