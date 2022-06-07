import torch
from torch_scatter import scatter_add
import scipy
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_appr_directed_adj(alpha, edge_index, num_nodes, edge_weight=None):
    """
     generate adj
     :arg:
         alpha:
         edge_index: tensor形式的行索引列表和列索引列表
         num_nodes: 邻接矩阵节点个数
         edge_weight: 边的权重
     :returns
        indices: 邻接矩阵的行索引和列索引tensor
        weight: 邻接矩阵的边的权重
    """
    # if edge_weight ==None:
    #     edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
    #                                  device=edge_index.device)
    # fill_value = 1
    # edge_index, edge_weight = add_self_loops(
    #     edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes + 1, num_nodes + 1]))
    p_v[0:num_nodes, 0:num_nodes] = (1 - alpha) * p_dense
    p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes, num_nodes] = alpha
    p_v[num_nodes, num_nodes] = 0.0
    p_ppr = p_v

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi / pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, edge_weight, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_adj(edge_index, values, num_node, alpha=0.1):
    edge_index = torch.tensor(edge_index).long()
    edge_weight = torch.tensor(values).float()
    index, weight, _ = get_appr_directed_adj(alpha, edge_index, num_node, edge_weight=edge_weight)
    adj_sparse = torch.sparse_coo_tensor(indices=index, values=weight, size=(num_node, num_node))
    return adj_sparse.to_dense()


# alpha = 0.1
# with open('../data/adj_bay.pkl', 'rb') as f:
#     data = pickle.load(f)
# edge_index = torch.tensor(data['edge_index']).long()
# edge_weight = torch.tensor(data['values']).float() / 1000
# print(edge_index.size(1))
# print(edge_index)
# index, weight, _ = get_appr_directed_adj(alpha, edge_index, data['num_node'], edge_weight=edge_weight)
# mx_sparse = torch.sparse_coo_tensor(indices=index, values=weight, size=(325, 325))
# print(mx_sparse.to_dense().numpy())

