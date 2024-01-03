import torch
from torch import Tensor
from tqdm import tqdm


def floyd_algorithm(
        distance_mat: Tensor,
        path: Tensor = None,
        min_k: int = None,
        max_k: int = None,
        replace=True,
):
    r"""
    Floyd minimal distance / time algorithm
    :param distance_mat: adjacent distance matrix, distance of i->j, row -> col
    :param min_k: minimum node id for iteration
    :param max_k: maximum node id for iteration
    :param replace: replace original distance_mat with minimum distance
    :return:
    distance_mat: minimum distance of i->j, row -> col
    path: shortest path, where row i is the shortest path from node i to other nodes
    """
    print('Minimum distance with floyd algorithm')
    m, n = distance_mat.size()
    assert m == n

    if ~replace:
        distance_mat = distance_mat.clone()
    min_k = 0 if min_k is None else max(min_k, 0)
    max_k = n if max_k is None else min(max_k, n)

    if path is None:
        path = torch.arange(n, device=distance_mat.device, dtype=torch.int16).unsqueeze(
            dim=1).repeat_interleave(n, 1)

    for k in tqdm(range(min_k, max_k)):
        dik = distance_mat[:, k]
        dkj = distance_mat[k, :]
        dij_new = dik.view(-1, 1) + dkj.view(1, -1)
        change_path = dij_new < distance_mat
        distance_mat[change_path] = dij_new[change_path]
        path[change_path] = k
        # 相比于标准floyd算法，此处 path 按行逆序存储，保持和dijkstra一致，方便后续计算
    print('Floyd algorithm finish')
    return distance_mat, path


def floyd_without_path(
        distance_mat: Tensor,
        min_k: int = None,
        max_k: int = None,
        device=torch.device('cuda:0') if torch.cuda.is_available() else None,
):
    print('floyd_without_path ..')
    m, n = distance_mat.size()
    assert m == n

    if device is not None:
        distance_mat = distance_mat.to(device=device)
    min_k = 0 if min_k is None else max(min_k, 0)
    max_k = n if max_k is None else min(max_k, n)

    for k in tqdm(range(min_k, max_k)):
        dik = distance_mat[:, k]
        dkj = distance_mat[k, :]
        dij_new = dik.view(-1, 1) + dkj.view(1, -1)
        change_path = dij_new < distance_mat
        distance_mat[change_path] = dij_new[change_path]
    print('Finish floyd_without_path')
    return distance_mat.cpu()


def sparse_floyd_algorithm(
        distance_mat: Tensor,
        sparse_rate: float = 0.4,
        min_k: int = None,
        max_k: int = None,
        d_inf=1e15,
        replace=True,
):
    r"""
    Floyd minimal distance / time algorithm
    The improved Floyd-Warshall algorithm with a slight modification.
    Original paper: [Improving The Floyd-Warshall All Pairs Shortest Paths Algorithm, arXiv:2109.01872]
    :param distance_mat: adjacent distance matrix, distance of i->j, row -> col
    :param replace: replace original distance_mat with minimum distance
    :param min_k: minimum node id for iteration
    :param max_k: maximum node id for iteration
    :return:
    distance_mat: minimum distance of i->j, row -> col
    path: shortest path, where row i is the shortest path from node i to other nodes
    """
    print('sparse_floyd_algorithm ..')
    m, n = distance_mat.size()
    assert m == n

    if ~replace:
        distance_mat = distance_mat.clone()
    min_k = 0 if min_k is None else max(min_k, 0)
    max_k = n if max_k is None else min(max_k, n)

    path = torch.arange(n, device=distance_mat.device, dtype=torch.int16).unsqueeze(
        dim=1).repeat_interleave(n, 1)

    kk = 0
    sparsity = 0
    for k in tqdm(range(min_k, max_k)):
        kk = k

        dik = distance_mat[:, k]
        dkj = distance_mat[k, :]
        fill_ik = dik < d_inf
        fill_kj = dkj < d_inf

        sparsity = 0.9 * sparsity + 0.1 * fill_ik.sum() * fill_kj.sum() / m ** 2
        if sparsity > sparse_rate:
            break

        new_dist = dik[fill_ik].view(-1, 1) + dkj[fill_kj].view(1, -1)
        mask = new_dist < distance_mat[fill_ik, :][:, fill_kj]
        sub_dist = distance_mat[fill_ik, :]
        sub_dist[:, fill_kj][mask] = new_dist[mask]
        distance_mat[fill_ik, :] = sub_dist
        sub_path = path[fill_ik, :]
        sub_path[:, fill_kj][mask] = k
        path[fill_ik, :] = sub_path

    distance_mat, path = floyd_algorithm(distance_mat, path, kk, max_k)
    print('Finish sparse_floyd_algorithm')
    return distance_mat, path


def sparse_floyd_without_path(
        distance_mat: Tensor,
        sparse_rate: float = 0.2,
        min_k: int = None,
        max_k: int = None,
        d_inf=1e15,
        device=torch.device('cuda:0') if torch.cuda.is_available() else None,
):
    print('sparse_floyd_without_path ..')
    m, n = distance_mat.size()
    assert m == n

    if device is not None:
        distance_mat = distance_mat.to(device=device)
    min_k = 0 if min_k is None else max(min_k, 0)
    max_k = n if max_k is None else min(max_k, n)

    kk = 0
    sparsity = 0
    for k in tqdm(range(min_k, max_k)):
        kk = k

        dik = distance_mat[:, k]
        dkj = distance_mat[k, :]
        fill_ik = dik < d_inf
        fill_kj = dkj < d_inf

        sparsity = 0.9 * sparsity + 0.1 * fill_ik.sum() * fill_kj.sum() / m ** 2
        if sparsity > sparse_rate:
            break

        new_dist = dik[fill_ik].view(-1, 1) + dkj[fill_kj].view(1, -1)
        sub_dist = distance_mat[fill_ik, :]
        mask = new_dist < sub_dist[:, fill_kj]
        sub_dist[:, fill_kj][mask] = new_dist[mask]
        distance_mat[fill_ik, :] = sub_dist

    distance_mat = floyd_without_path(distance_mat, kk, max_k)
    print('Finish sparse_floyd_without_path')
    return distance_mat.cpu()


def floyd_new_road(
        distance_mat: Tensor,
        path: Tensor,
        new_road_st: int,
        new_road_ed: int,
        replace=True):
    """
    Floyd minimal distance / time algorithm,
    update original map with additional roads
    :param distance_mat:
    :param new_road_st:
    :param new_road_ed:
    :param replace:
    :return:
    """
    print('Update minimum distance with floyd algorithm')
    m, n = distance_mat.size()
    assert m == n

    if ~replace:
        distance_mat = distance_mat.clone()

    new_road_st = 0 if new_road_st is None else max(new_road_st, 0)
    new_road_ed = n if new_road_ed is None else min(new_road_ed, n)

    # 更新 new road 到其它所有点之间的最短路径
    for k in tqdm(range(n)):
        # from new road
        dik = distance_mat[new_road_st:new_road_ed, k]
        dkj = distance_mat[k, :]
        dij_new = dik.view(-1, 1) + dkj.view(1, -1)
        change_path = dij_new < distance_mat[new_road_st:new_road_ed, :]
        distance_mat[new_road_st:new_road_ed, :][change_path] = dij_new[change_path]
        path[new_road_st:new_road_ed, :][change_path] = \
            path[k, :].view(1, -1).expand_as(path[new_road_st:new_road_ed, :])[change_path]

        # to new road
        dik = distance_mat[:, k]
        dkj = distance_mat[k, new_road_st:new_road_ed]
        dij_new = dik.view(-1, 1) + dkj.view(1, -1)
        change_path = dij_new < distance_mat[:, new_road_st:new_road_ed]
        distance_mat[:, new_road_st:new_road_ed][change_path] = dij_new[change_path]
        path[:, new_road_st:new_road_ed][change_path] = \
            path[k, new_road_st:new_road_ed].view(1, -1).expand_as(path[:, new_road_st:new_road_ed])[change_path]

    # 以 new road 为中间点，更新其它所有点之间的最短路径
    for k in tqdm(range(new_road_st, new_road_ed)):
        dik = distance_mat[:, k]
        dkj = distance_mat[k, :]
        dij_new = dik.view(-1, 1) + dkj.view(1, -1)
        change_path = dij_new < distance_mat
        distance_mat[change_path] = dij_new[change_path]
        path[change_path] = path[k, :].unsqueeze(dim=0).expand_as(path)[change_path]

    print('Floyd update finish')
    return distance_mat, path


def floyd_new_points(
        road_distance_mat: Tensor,
        cr_distance_mat: Tensor,
):
    """
        Floyd minimal distance / time algorithm,
        update distance between additional roads
        :param road_distance_mat:
        :param cr_distance_mat:
        :return:
        """
    print('Update minimum distance with floyd algorithm')
    rm, rn = road_distance_mat.size()
    cm, cn = cr_distance_mat.size()
    assert rm == rn == cn

    cc_dist_mat = torch.empty((cm, cm), device=cr_distance_mat.device, dtype=cr_distance_mat.dtype)
    cc_dist_mat.fill_(cr_distance_mat.max())
    path = torch.arange(cm, device=road_distance_mat.device, dtype=torch.int16).unsqueeze(
        dim=1).repeat_interleave(cm, 1)

    # 更新 new point 到其它所有点之间的最短路径
    for k in tqdm(range(rn)):
        # from new point to others
        dik_rr = road_distance_mat[k, :]
        dik_cr = cr_distance_mat[:, k]
        dij_new = dik_cr.view(-1, 1) + dik_rr.view(1, -1)
        change_path = dij_new < cr_distance_mat
        cr_distance_mat[change_path] = dij_new[change_path]

        dij_new = dik_cr.view(-1, 1) + dik_cr.view(1, -1)
        change_path = dij_new < cc_dist_mat
        cc_dist_mat[change_path] = dij_new[change_path]
        path[change_path] = k

    print('Floyd update finish')
    return cc_dist_mat, path


def floyd_new_points_without_path(
        road_dist: Tensor,
        cr_dist: Tensor,
        d_inf=2**11,
        device=torch.device('cuda:0') if torch.cuda.is_available() else None,
):
    """
    Floyd minimal distance / time algorithm,
        update distance between additional points
    :param road_dist: (n_road, n_road)
    :param cr_dist: (n_point, n_road)
    :param d_inf:
    :param device:
    :return:
    """
    print('floyd_new_points_without_path .. ')
    rm, rn = road_dist.size()
    cm, cn = cr_dist.size()
    assert rm == rn == cn

    if device is not None:
        road_dist = road_dist.to(device=device)
        cr_dist = cr_dist.to(device=device)

    cc_dist = torch.empty((cm, cm), device=cr_dist.device, dtype=cr_dist.dtype)
    cc_dist.fill_(d_inf)
    cc_dist[range(cm), range(cm)] = 0

    # 更新 new point 到其它所有点之间的最短路径
    for k in tqdm(range(rn)):
        # from new point to others
        dik_cr = cr_dist[:, k]
        dkj_rr = road_dist[k, :]
        dij_new = dik_cr.view(-1, 1) + dkj_rr.view(1, -1)
        mask = dij_new < cr_dist
        cr_dist[mask] = dij_new[mask]

    for k in tqdm(range(cn)):
        dik_cr = cr_dist[:, k]
        dij_new = dik_cr.view(-1, 1) + dik_cr.view(1, -1)
        mask = dij_new < cc_dist
        cc_dist[mask] = dij_new[mask]

    print('Finish floyd_new_points_without_path')
    return cc_dist.cpu()
