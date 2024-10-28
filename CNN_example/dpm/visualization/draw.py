"""draw figure tools
- upset plot


@ Liu Yuting|Niigata university medical AI center
"""
import os
from utils import common
from utils import log
from upsetplot import from_contents, plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette, ward, complete, average, centroid, \
    single
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.colors import ListedColormap


def draw_line_plot(df: pd.DataFrame, x, y, column_clip_on=None, clip_on_ymin=None, clip_on_ymax=None, hue=None):
    # 调整在图上超过400的为极值inf，-inf

    if column_clip_on:
        df.loc[:, column_clip_on] = df.loc[:, column_clip_on].clip(upper=clip_on_ymax, lower=clip_on_ymin)
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, seed=42, alpha=0.6,lw=1)
    # sns.move_legend(ax, "upper right")

    # 调整y极限为inf，-inf
    ax.set_ylim(clip_on_ymin, clip_on_ymax)
    ax.set_yticks([clip_on_ymin, clip_on_ymax])
    ax.set_yticklabels([str(clip_on_ymin), str(clip_on_ymax)])
    ax.axhline(y=1, color='darkgrey', linestyle='--')

    ax.axhline(y=0.6, color='darkgrey', linestyle='--')
    ax.axhline(y=0, color='darkgrey', linestyle='--')

    plt.show()


def draw_violin_plot():
    # https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot
    pass


def draw_tsne(X, y, save_path_title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    X_tsne = tsne.fit_transform(X)

    # 绘制t-SNE图
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of train-evaluation dataset')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(f"{save_path_title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def draw_scatter(df, x, y, hue, save_path_title):
    # plt.figure(figsize=(20, 10))
    plt.rcParams['font.family'] = "Arial"
    ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6, palette='tab20', edgecolor='w', linewidth=0.5, s=10)
    # 调整图例位置和透明度
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Superclass', frameon=True, framealpha=0.9)
    ax.set_title('Scatter plot of x_rt vs y_rt with different superclasses', fontsize=18, fontweight="bold")
    ax.set_xlabel("source system RT", fontsize=16)
    ax.set_ylabel("target system RT", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    plt.grid(True)

    plt.savefig(f"{save_path_title}.png", dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.close()


def draw_upsetplot(dict, save_path):
    intersect = from_contents(dict)
    print(intersect.head())
    plt.rcParams['font.family'] = "Arial"
    upset = plot(intersect, subset_size="count", show_counts="{:d}")
    plt.savefig(os.path.join(save_path, "upsetplot.pdf"))
    plt.show()
    plt.close()


def draw_barplot(df, x: str, y: str, save_title: str = "barplot.png"):
    plt.rcParams["font.family"] = "arial"
    ax = sns.barplot(data=df, x=x, y=y, orient="h")
    # plt.setp(ax.get_xticklabels(), rotation=90)

    ax.tick_params(axis='both', labelsize=8)
    plt.savefig(save_title, dpi=300, bbox_inches='tight')
    plt.close()


def draw_stack_barplot(df, x: list, stack_num: int, labels: list, save_title: str = "./barplot.png"):
    """x[0]: total value, x[1]: stack value 1"""

    categories = x  # 7
    subcategories = labels  # 3
    values = df.values.T
    colors = color_platte(stack_num, platte="viridis")
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.rcParams["font.family"] = "arial"
    # 创建堆叠柱状图
    bottom = np.zeros(len(categories))  # 初始化底部位置  7

    for i, (subcat, color) in enumerate(zip(subcategories, colors)):
        ax.bar(categories, values[i], bottom=bottom, label=subcat, color=color)
        bottom += values[i]

    ax.legend(title='RE')  # 图例 title
    ax.set_title('Stacked Bar Plot of Relative Errors', fontsize=12)
    ax.set_xlabel('Algorithms', fontsize=10)
    ax.set_ylabel('Counts', fontsize=10)

    plt.savefig(f"{save_title}", dpi=300, bbox_inches='tight')
    plt.close()


def draw_hue_boxplot(df, x, y, hue, save_title):
    ax = sns.boxplot(df, x=x, y=y, hue=hue)
    ax.legend(title='RE group')  # 图例 title
    ax.set_title('Bar Plot of Initial Errors', fontsize=12)
    ax.set_xlabel('Algorithms', fontsize=10)
    ax.set_ylabel('Initial errors', fontsize=10)

    plt.savefig(f"{save_title}", dpi=300, bbox_inches='tight')
    plt.close()


def draw_hue_violin_plot(df, x, y, hue, save_title):
    ax = sns.violinplot(data=df, x=x, y=y, hue=hue, cut=50)
    ax.legend(title='RE group')  # 图例 title
    ax.set_title('Bar Plot of Initial Errors', fontsize=12)
    ax.set_xlabel('Algorithms', fontsize=10)
    ax.set_ylabel('Initial errors', fontsize=10)

    plt.savefig(f"{save_title}", dpi=300, bbox_inches='tight')
    plt.close()


def color_annotation_column(df, **kwargs):
    palette = ["set2", "set3", "paired", "Accent", "tab20", "tab20b", "tab20c"]
    category_data = kwargs.get("category")
    axis = kwargs.get("axis")
    if axis == "row":
        df_index = df.index
    if axis == "column":
        df_index = df.columns
    colors = pd.DataFrame(index=df_index)

    # 为了使source system 和target system保持一致的映射
    category1 = category_data["source_system"]
    category2 = category_data["target_system"]
    category3 = category_data["superclass"]

    if len(category1) != len(df_index) or len(category2) != len(df_index) or len(category3) != len(df_index):
        print(len(category1), len(df_index), len(category2), len(category3))
        raise ValueError("Length of category data does not match length of dataframe index")
    system_set = set(category1).union(set(category2))
    palette1 = color_platte(len(system_set))
    palette2 = color_platte(len(set(category3)))
    lut1 = dict(zip(set(system_set), palette1))
    lut2 = dict(zip(set(category3), palette2))
    colors["source_system"] = pd.Series(category1, index=df_index).map(lut1)
    colors["target_system"] = pd.Series(category2, index=df_index).map(lut1)
    colors["superclass"] = pd.Series(category3, index=df_index).map(lut2)

    return colors


def color_annotation_row(df, **kwargs):
    palette = ["set2", "set3", "paired", "Accent", "tab20", "tab20b", "tab20c"]
    category_data = kwargs.get("category")
    axis = kwargs.get("axis")
    index = df.index
    colors = pd.DataFrame(index=index)
    for idx, (category_name, category) in enumerate(category_data.items()):
        # 创建颜色映射
        palette = color_platte(len(set(category)))
        lut1 = dict(zip(set(category), palette))
        colors[category_name] = pd.Series(category, index=index).map(lut1)

    return colors


def usr_define_cmap():
    """改变投射的颜色"""
    cmap = sns.color_palette("Greens", as_cmap=True)
    cmap = cmap(np.arange(cmap.N))
    cmap[:1, :] = np.array([1, 1, 1, 1])  # 将最小值（0值）设为白色
    custom_cmap = ListedColormap(cmap)
    return custom_cmap


def draw_clusterheatmap(df,
                        row_colors=True,
                        column_colors=True,
                        row_category=None | dict,
                        column_category=None | dict,
                        define_cmap: bool = False,
                        save_title: str = "heatmap.png",
                        fig_title: str = "compound distribution"):
    """ Heatmap
    if color_marker is True, please pass marker_list argument, otherwise Userwarning will raise up.
    :param: df:dataframe
    :param: column_category={"category1":sub-group,"category2":sub-group}
    :param: save_title: title with full save path and format
    :param: fig_title: figure title in heatmap.
    """

    cbar_kws = {"shrink": 0.3}  # color bar缩小到原来的一半
    # 使用plt.subplots来创建图像和轴
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)

    column_colors_cmap, row_colors_cmap, custom_cmap = None, None, None

    if define_cmap:
        custom_cmap = usr_define_cmap()
    if column_colors:
        column_colors_cmap = color_annotation_column(df, category=column_category, axis="column")
    if row_colors:
        row_colors_cmap = color_annotation_row(df, category=row_category, axis="row")
    g = sns.clustermap(df,
                       cmap=custom_cmap,
                       column_colors=column_colors_cmap,
                       row_colors=row_colors_cmap,

                       cbar=True,
                       cbar_kws=cbar_kws)

    # 设置字体
    plt.rcParams["font.family"] = "arial"

    # 设置标题、轴标签和其他参数
    g.ax_heatmap.set_title(fig_title, fontsize=12, fontweight="bold")
    g.ax_heatmap.set_ylabel("target systems", fontsize=10)
    g.ax_heatmap.set_xlabel("source systems", fontsize=10)
    g.ax_heatmap.tick_params(axis='both', labelsize=8)
    # plt.xticks([])  # 隐藏标签
    # plt.yticks([])

    # 设置轴标签角度
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    # 保存图像时使用bbox_inches='tight'来自动调整边距
    plt.savefig(f"{save_title}", dpi=300, bbox_inches='tight')
    plt.close()


def draw_heatmap(df,
                 row_colors=True,
                 column_colors=True,
                 column_category=None | dict,
                 define_cmap: bool = False,
                 save_title: str = "heatmap.png",
                 fig_title: str = "compound distribution"):
    """ Heatmap
    if color_marker is True, please pass marker_list argument, otherwise Userwarning will raise up.
    :param: df:dataframe
    :param: column_category={"category1":sub-group,"category2":sub-group}
    :param: save_title: title with full save path and format
    :param: fig_title: figure title in heatmap.
    """

    cbar_kws = {"shrink": 0.3}  # color bar缩小到原来的一半
    # 使用plt.subplots来创建图像和轴
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)

    custom_cmap = usr_define_cmap() if define_cmap else None

    ax = sns.heatmap(df, cmap=custom_cmap, cbar=True, ax=ax, cbar_kws=cbar_kws)

    # 设置字体
    plt.rcParams["font.family"] = "arial"

    # 设置标题、轴标签和其他参数
    ax.set_title(fig_title, fontsize=12, fontweight="bold")
    ax.set_ylabel("target systems", fontsize=10)
    ax.set_xlabel("source systems", fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    # plt.xticks([])  # 隐藏标签
    # plt.yticks([])

    # 设置轴标签角度
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)

    # 保存图像时使用bbox_inches='tight'来自动调整边距
    plt.savefig(f"{save_title}", dpi=300, bbox_inches='tight')
    plt.close()


def color_platte(num, platte: str = "rainbow"):
    match platte:
        case "rainbow":
            return cm.rainbow(np.linspace(0, 1, num))
        case "viridis":
            return cm.viridis(np.linspace(0, 1, num))


class network:
    def __init__(self, save_path):
        self.path = save_path
        self.log = log.logger()

    def save_node_degree(self, node_degree: dict) -> pd.DataFrame:
        """generate node degree file"""
        node_degree_df = (pd.DataFrame.from_dict(node_degree, orient="index")
        .reset_index()
        .set_axis(
            ["node", "node_degree"], axis=1))
        common.save_csv(node_degree_df,
                        os.path.join(self.path, "node_degree.csv"))
        return node_degree_df

    def draw_network(self, edges: pd.DataFrame, net_name):
        """draw network entry, save network.graphml, network.pdf, node_degree.csv"""
        fig = plt.figure(figsize=(40, 50))

        G = nx.from_pandas_edgelist(edges, edge_attr=True)

        node_degree = dict(G.degree())
        self.save_node_degree(node_degree)

        nx.set_node_attributes(G, node_degree, 'degree_')

        pos = nx.spring_layout(G, seed=10)

        # draw nodes
        tissue_nodes = edges["source"]
        target_nodes = edges["target"]

        nx.draw_networkx_nodes(G, pos=pos, nodelist=tissue_nodes, alpha=0.7, node_color="orange",
                               node_shape="o", node_size=500)

        nx.draw_networkx_nodes(G, pos=pos, nodelist=target_nodes, alpha=0.7,
                               node_shape="o", node_size=300, node_color="green")

        # draw edges
        nx.draw_networkx_edges(G, pos, edge_color="grey")

        # draw labels
        # node_labels = {node: node for node in G.nodes}
        # nx.draw_networkx_labels(G, pos, labels=node_labels, )

        # draw tissue node labels
        tissue_node_labels = {n: n for n in tissue_nodes}
        nx.draw_networkx_labels(G, pos, labels=tissue_node_labels, )

        # draw top 3 target node labels
        # value_list = node_degree.values()
        # threshold = sorted(value_list)[-5:]
        target_node_labels = {n: n for n in target_nodes if node_degree[n] >= 3}
        nx.draw_networkx_labels(G, pos, labels=target_node_labels, )

        self.save_net_as_graphml(G, os.path.join(self.path, net_name))
        self.save_net_as_figure(os.path.join(self.path, net_name))
        plt.close()

    def save_net_as_figure(self, name, format="pdf"):
        plt.savefig(f"{name}.{format}")
        self.log.debug(f"save network {name}.{format} 🍺🍺")

    def save_net_as_graphml(self, G, name):
        nx.write_graphml(G, f"{name}.graphml")
        self.log.debug(f"save network {name}.graphml 🍺🍺🍺")
