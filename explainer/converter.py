import torch
import numpy as np

# return id of n most important feature
def take_n_most_important(feature_mask, num_take = 5):
    n = feature_mask.size(0)
    a = np.linspace(0, n, n, endpoint=False, dtype=int)
    for i in range(n):
        for j in range(n):
            if feature_mask[a[i]] > feature_mask[a[j]]: a[i], a[j] = a[j], a[i]

    #for i in range(n): print(feature_mask[a[i]])
    return a[:num_take].tolist()

def rescale(a, min, max):
    a = np.asarray(a)
    a = np.interp(a, (a.min(), a.max()), (0, 1))
    return torch.from_numpy(a)

def convert_adj_to_edge_index(A):
    """
        - A (Tensor): L x N x N  --> Adjacent matrix with L edge types
        - A will be converted to edge_index
        - edge_index: 2 x (number of edges)
    """
    num_edge = A.sum()
    edge_index = torch.zeros(2, int(num_edge), dtype=torch.long)
    #print(edge_index.size())
    edge_type = torch.zeros(int(num_edge))
    pointer = 0
    for k in range(A.size(0)):
        for i in range(A.size(1)):
            for j in range(A.size(2)):
                if A[k][i][j] == 1:
                    edge_index[0][pointer] = i
                    edge_index[1][pointer] = j
                    edge_type[pointer] = k
                    pointer += 1

    return edge_index, edge_type


def convert_edge_index_to_adj(edge_index, edge_type, num_node, num_type, use_mask_value = False, edge_mask=None):
    A = torch.zeros(num_type, num_node, num_node, dtype=torch.float)
    num_edge = edge_index.size(1)
    for i in range(num_edge):
        type, u, v = int(edge_type[i]), int(edge_index[0][i]), int(edge_index[1][i])
        if use_mask_value: A[type][u][v] = edge_mask[i]
        else: A[type][u][v] = 1
    return A