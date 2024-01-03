import networkx as nx
import torch
from torch import Tensor
from tqdm import tqdm
from scipy.sparse.csgraph import reverse_cuthill_mckee


def graph_reorder(graph):
    graph = reverse_cuthill_mckee(graph)
    return graph


def graph_reorder_test():
    mm = 32
    graph = nx.gnm_random_graph(512, 1024)
    dense = nx.to_numpy_array(graph)
    plt.figure()
    plt.title('original')
    plt.imshow(dense)
    plt.show()

    # reorder = reverse_cuthill_mckee(sparse_matrix)  # 结果拉跨，没法用
    reorder = nx.spectral_ordering(graph)
    plt.figure()
    plt.title('reordered')
    plt.imshow(dense[reorder, :][:, reorder])
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph_reorder_test()
