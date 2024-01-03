import torch
from torch import Tensor
from tqdm import tqdm


def block_floyd_without_path(
        dist_mat: Tensor,
        block_size: int = 1024,
        min_k: int = None,
        max_k: int = None,
        device=torch.device('cuda:0') if torch.cuda.is_available() else None,
):
    print('block_floyd_without_path ..')
    m, n = dist_mat.size()
    assert m == n

    if device is not None:
        dist_mat = dist_mat.to(device=device)
    min_k = 0 if min_k is None else max(min_k, 0)
    max_k = n if max_k is None else min(max_k, n)

    block_range = list(range(0, m, block_size)) + [m]

    for k in tqdm(range(min_k, max_k)):
        dkj = dist_mat[k, :].view(1, -1)
        dik = dist_mat[:, k].view(-1, 1)

        for r, st in enumerate(block_range[:-1]):
            ed = block_range[r + 1]
            if (dik[st:ed, :] < 2048).sum() == 0:
                continue
            dist_block = dist_mat[st:ed, :]
            new_dist = dkj + dik[st:ed, :]
            mask = new_dist < dist_block
            dist_mat[st:ed, :][mask] = new_dist[mask]
    print('Finish block_floyd_without_path')
    return dist_mat.cpu()


def block_sparse_floyd_without_path(
        dist_mat: Tensor,
        sparse_rate: float = 0.2,
        block_size: int = 1024,
        min_k: int = None,
        max_k: int = None,
        d_inf=2**11,
        device=torch.device('cuda:0') if torch.cuda.is_available() else None,
):
    print('block_sparse_floyd_without_path ..')
    m, n = dist_mat.size()
    assert m == n

    if device is not None:
        dist_mat = dist_mat.to(device=device)
    min_k = 0 if min_k is None else max(min_k, 0)
    max_k = m if max_k is None else min(max_k, m)

    block_range = list(range(0, m, block_size)) + [m]

    kk = 0
    sparsity = 0
    for k in tqdm(range(min_k, max_k)):
        kk = k

        dik = dist_mat[:, k]
        dkj = dist_mat[k, :]
        fill_ik = dik < d_inf
        fill_kj = dkj < d_inf

        sparsity = 0.9 * sparsity + 0.1 * fill_ik.sum() * fill_kj.sum() / m ** 2
        if sparsity > sparse_rate:
            break

        dkj = dkj[fill_kj].view(1, -1)
        for r, st in enumerate(block_range[:-1]):
            ed = block_range[r + 1]
            local_fill_ik = fill_ik[st:ed]
            if local_fill_ik.sum() > 0:
                sub_dist_y = dist_mat[st:ed, :][local_fill_ik, :]
                block_dist = sub_dist_y[:, fill_kj]
                new_dist = dik[st:ed][local_fill_ik].view(-1, 1) + dkj.view(1, -1)
                mask = new_dist < block_dist
                block_dist[mask] = new_dist[mask]
                sub_dist_y[:, fill_kj] = block_dist
                dist_mat[st:ed, :][local_fill_ik, :] = sub_dist_y

    dist_mat = block_floyd_without_path(dist_mat, block_size, kk, max_k, None)
    print('Finish block_sparse_floyd_without_path')
    return dist_mat.cpu()


def block_floyd_new_points_without_path(
        road_dist: Tensor,
        cr_dist: Tensor,
        block_size: int = 1024,
        d_inf=2**11,
        device=torch.device('cuda:0') if torch.cuda.is_available() else None,
):
    """
    Floyd minimal distance / time algorithm,
        update distance between additional points
    :param road_dist: (n_road, n_road)
    :param cr_dist: (n_point, n_road)
    :param block_size:
    :param d_inf:
    :param device:
    :return:
    """
    print('block_floyd_new_points_without_path .. ')
    rm, rn = road_dist.size()
    cm, cn = cr_dist.size()
    assert rm == rn == cn

    if device is not None:
        road_dist = road_dist.to(device=device)
        cr_dist = cr_dist.to(device=device)

    cc_dist = torch.empty((cm, cm), device=cr_dist.device, dtype=cr_dist.dtype)
    cc_dist.fill_(d_inf)
    cc_dist[range(cm), range(cm)] = 0

    # 更新 commuting points 到所有 road 点之间的最短路径
    block_range = list(range(0, rm, block_size)) + [rm]
    for k in tqdm(range(rm)):
        dik_cr = cr_dist[:, k]
        dkj_rr = road_dist[k, :]
        for r, st in enumerate(block_range[:-1]):
            ed = block_range[r + 1]
            dist_block = cr_dist[st:ed, :]
            dij_new = dik_cr[st:ed].view(-1, 1) + dkj_rr.view(1, -1)
            mask = dij_new < dist_block
            cr_dist[st:ed, :][mask] = dij_new[mask]

    # 更新 commuting points 之间的最短路径
    block_range = list(range(0, cm, block_size)) + [cm]
    for k in tqdm(range(cn)):
        dik_cr = cr_dist[:, k]
        for r, st in enumerate(block_range[:-1]):
            ed = block_range[r + 1]
            dist_block = cc_dist[st:ed, :]
            dij_new = dik_cr[st:ed].view(-1, 1) + dik_cr.view(1, -1)
            mask = dij_new < dist_block
            cc_dist[st:ed, :][mask] = dij_new[mask]

    print('Finish block_floyd_new_points_without_path')
    return cc_dist.cpu()


def test():
    mm = 128
    nn = mm ** 2
    graph = nx.grid_2d_graph(mm, mm)
    dist_mat = torch.from_numpy(nx.to_numpy_array(graph)).to(dtype=torch.float16)
    dist_mat[dist_mat == 0] = 2 ** 11
    dist_mat[range(nn), range(nn)] = 0

    t0 = time()
    torch.cuda.empty_cache()
    # re_dist0 = floyd_without_path(dist_mat, device=torch.device('cuda:3'))
    t0 = time() - t0
    # print(f'Shortest path: dist mean = {re_dist0.mean()}, dist max = {re_dist0.max()}')

    t1 = time()
    torch.cuda.empty_cache()
    # re_dist1 = block_floyd_without_path(dist_mat, device=torch.device('cuda:3'))
    t1 = time() - t1

    t2 = time()
    torch.cuda.empty_cache()
    re_dist2 = block_sparse_floyd_without_path(dist_mat, d_inf=2 ** 11, device=torch.device('cuda:2'))
    t2 = time() - t2
    print(f'Time use: '
          f'Base floyd = {t0} s, '
          f'block_floyd_without_path = {t1} s, '
          f'{block_sparse_floyd_without_path.__name__} = {t2} s'
          )

    mse = torch.nn.MSELoss()
    print(f'MSE error: '
          f'block_floyd_without_path = {mse(re_dist0, re_dist1)}, '
          f'block_sparse_floyd_without_path = {mse(re_dist0, re_dist2)}'
          )

    '''
                floyd   block_floyd     block_sparse_floyd
    block_size  -----   1024            1024
    memory(MiB) 3119    1419            1675
    time(s)     131     145             123
    '''


if __name__ == '__main__':
    import networkx as nx
    from time import time
    from base_algorithm import floyd_without_path
    test()

