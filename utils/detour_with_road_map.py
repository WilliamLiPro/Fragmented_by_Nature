import numpy
import torch
from torch import Tensor
from utils import (intersect_map_add_road, tensor_road_intersect_polygon,
                   floyd_without_path, sparse_floyd_without_path, floyd_new_points_without_path,
                    block_floyd_without_path, block_sparse_floyd_without_path,
                    block_floyd_new_points_without_path)


def direct_pair_distance(
        commuting_point: Tensor,
        road_point: Tensor,
):
    """
    Calculate the direct distance from point of nodes
    :param commuting_point
    :param road_point
    :return:
    """
    cx = commuting_point[0, :]
    cy = commuting_point[1, :]
    rx = road_point[0, :]
    ry = road_point[1, :]
    dx = cx.view(-1, 1) - rx.view(1, -1)
    dy = cy.view(-1, 1) - ry.view(1, -1)
    return (dx ** 2 + dy ** 2) ** 0.5


def index_serialization(
        old_tensor: Tensor,
        old_idx: Tensor,
):
    new_idx = torch.arange(old_idx.size()[0], device=old_tensor.device, dtype=old_tensor.dtype)
    old_to_new = torch.zeros(old_idx.max().item() + 1, device=old_tensor.device, dtype=old_tensor.dtype)
    old_to_new[old_idx.to(torch.int64)] = new_idx
    return old_to_new[old_tensor.to(torch.int64)]


def graphical_index_(
        direct_dist: Tensor,
        dist_mat: Tensor,
        mask_mat: Tensor = None,
):
    # if mask_mat is None:
    #     return dist_mat.mean() / direct_dist.mean() - 1
    # else:
    #     return (dist_mat * mask_mat).mean() / (direct_dist * mask_mat).mean() - 1
    if mask_mat is None:
        return ((dist_mat + 0.00001) / (direct_dist + 0.00001)).mean() - 1
    else:
        return ((dist_mat[mask_mat] + 0.00001) / (direct_dist[mask_mat] + 0.00001)).mean() - 1


class GraphicalIndexWithRoadMap:
    def __init__(self,
                 x_factor: float = 1.0,
                 y_factor: float = 1.0,
                 block_sz: int = 64,
                 neighbor_d_max: float = 1.0,
                 d_max=2 ** 11,
                 degree_threshold: float = 0.0,
                 sparse_floyd=False,  # sparse algorithm accelerates for GPU with Tensor RT
                 device=torch.device('cuda:0') if torch.cuda.is_available() else None,
                 ):
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.block_sz = block_sz
        self.neighbor_d_max = neighbor_d_max
        self.d_max = d_max
        self.degree_threshold = degree_threshold
        self.sparse_floyd = sparse_floyd
        self.device = device
        self.save_name = None

    def initialization(self,
                       commuting_point: Tensor,
                       road_point: Tensor,
                       road_net: Tensor,
                       barrier_st: Tensor,
                       barrier_ed: Tensor,
                       ):
        print('Graph initialization ...')
        # 1. mask
        x_factor = self.x_factor
        y_factor = self.y_factor
        device = self.device
        d_max = self.d_max
        if x_factor != 1.0:
            commuting_point[0, :] *= x_factor
            road_point[0, :] *= x_factor
            barrier_st[0, :] *= x_factor
            barrier_ed[0, :] *= x_factor
        if y_factor != 1.0:
            commuting_point[1, :] *= y_factor
            road_point[1, :] *= y_factor
            barrier_st[1, :] *= y_factor
            barrier_ed[1, :] *= y_factor
        torch.cuda.empty_cache()
        cr_mask = intersect_map_add_road(
            commuting_point.to(device=device), road_point.to(device=device),
            barrier_st.to(device=device), barrier_ed.to(device=device), self.block_sz).cpu()

        n_c = commuting_point.size()[1]
        change = road_net < road_net.t()
        road_net = road_net * (~change) + road_net.t() * change  # An undirected graph

        # 2. 计算所有点之间的直线距离，并通过mask生成初始距离
        #   road dist
        road_mask = road_net == 0
        road_dist_mat = road_net
        road_dist_mat[road_mask] = d_max
        nr = road_dist_mat.size(0)
        road_dist_mat[range(nr), range(nr)] = 0

        #   dist between commuting points and road points
        cr_dist_mat = direct_pair_distance(commuting_point, road_point)
        cr_mask = cr_mask | (
                cr_dist_mat > self.neighbor_d_max)  # distance from commuting node to road should not be large
        cr_dist_mat[cr_mask] = d_max

        #   dist between commuting points
        direct_cc_dist_mat = direct_pair_distance(commuting_point, commuting_point)

        # 3. 剔除度小于阈值的节点
        select_idx = torch.arange(n_c, device=commuting_point.device)
        c_degrees = (~cr_mask).sum(dim=-1)
        select_idx = select_idx[c_degrees > 0]  # commuting 到 road 的连接数量至少要有一条
        cr_dist_mat = cr_dist_mat[select_idx, :]
        direct_cc_dist_mat = direct_cc_dist_mat[select_idx, :][:, select_idx]

        print('Finish initialization')
        return road_dist_mat, cr_dist_mat, direct_cc_dist_mat

    def graph_dist(self,
                   commuting_point: Tensor,
                   traffic_nodes: Tensor,
                   road_net: Tensor,
                   barrier_st: Tensor,
                   barrier_ed: Tensor,
                   ):
        # 1. graph initialization
        road_dist_mat, cr_dist_mat, direct_cc_dist_mat = self.initialization(
            commuting_point,
            traffic_nodes,
            road_net,
            barrier_st, barrier_ed,
        )

        # 2. shortest path
        d_max = self.d_max
        device = self.device
        # road
        torch.cuda.empty_cache()
        if self.sparse_floyd:
            if road_dist_mat.size(0) < 10000:
                road_dist_mat = sparse_floyd_without_path(road_dist_mat, d_inf=d_max, device=device)
            else:
                road_dist_mat = block_sparse_floyd_without_path(road_dist_mat, d_inf=d_max, device=device)
        else:
            if road_dist_mat.size(0) < 10000:
                road_dist_mat = floyd_without_path(road_dist_mat, device=device)
            else:
                road_dist_mat = block_floyd_without_path(road_dist_mat, device=device)

        # commuting points
        torch.cuda.empty_cache()
        if road_dist_mat.size(0) < 10000:
            cc_dist_mat = floyd_new_points_without_path(road_dist_mat, cr_dist_mat, d_inf=d_max, device=device)
        else:
            cc_dist_mat = block_floyd_new_points_without_path(road_dist_mat, cr_dist_mat, d_inf=d_max, device=device)

        # commuting points
        degree_threshold = self.degree_threshold
        if degree_threshold > 0:
            n_c = cc_dist_mat.size(-1)
            r_degrees = (cc_dist_mat >= d_max).sum(dim=-1)  # 无穷大距离个数
            c_idx = torch.arange(n_c, device=commuting_point.device)[
                r_degrees < int(n_c * degree_threshold)]  # 无穷大数量小于阈值
            if c_idx.numel() <= 0:
                return -1, -1, -1
            direct_cc_dist_mat = direct_cc_dist_mat[c_idx, :][:, c_idx]
            cc_dist_mat = cc_dist_mat[c_idx, :][:, c_idx]
            torch.cuda.empty_cache()
        return direct_cc_dist_mat, cc_dist_mat, road_dist_mat

    def run(self,
            commuting_point: Tensor,
            traffic_nodes: Tensor,
            road_net: Tensor,
            barrier_st: Tensor,
            barrier_ed: Tensor,
            save_name: str = None,
            ):
        self.save_name = save_name
        # check the inputs
        if ((commuting_point.dim() != 2 or numpy.prod(numpy.array(commuting_point.size())) < 1) or
                (traffic_nodes.dim() != 2 or numpy.prod(numpy.array(traffic_nodes.size())) < 1) or
                (road_net.dim() != 2 or numpy.prod(numpy.array(road_net.size())) < 1) or
                (traffic_nodes.size(1) != road_net.size(1))
        ):
            return -1, -1, -1, -1, -1, -1, -1

        # graphical_distance
        direct_cc_dist_mat, cc_dist_mat, road_dist_mat = self.graph_dist(
            commuting_point,
            traffic_nodes,
            road_net,
            barrier_st,
            barrier_ed,
        )

        if not direct_cc_dist_mat.__class__.__name__ == 'Tensor':
            return -1, -1, -1, -1, -1, -1, -1

        # graphical index
        direct_cc_dist_mat = direct_cc_dist_mat.to(dtype=torch.float32)
        cc_dist_mat = cc_dist_mat.to(dtype=torch.float32)
        gid = graphical_index_(direct_cc_dist_mat, cc_dist_mat).item()
        d_mean = cc_dist_mat.mean().item()
        d_std = cc_dist_mat.std().item()
        d_max = cc_dist_mat.max().item()
        d_mean_direct = direct_cc_dist_mat.mean().item()

        # save
        save_name = self.save_name
        if save_name is not None:
            torch.save(cc_dist_mat, save_name + '_cc_dist_mat.pth')
            torch.save(road_dist_mat, save_name + '_road_dist_mat.pth')
            torch.save(gid, save_name + '_graphical_index.pth')
        return gid, d_max, d_mean, d_std, d_mean_direct, cc_dist_mat.size(0), road_dist_mat.size(0)


class GraphicalIndexWithRoadMapPolygon():
    def __init__(self,
                 block_sz: int = 64,
                 neighbor_d_max: float = 1.0,
                 cr_d_max: float = 0.5,
                 d_max=2 ** 11,
                 degree_threshold: float = 0.0,
                 sparse_floyd=False,  # sparse algorithm accelerates for GPU with Tensor RT
                 device=torch.device('cuda:0') if torch.cuda.is_available() else None,
                 use_cc_dist=False,
                 ):
        """

        :param block_sz:
        :param neighbor_d_max:
        :param cr_d_max:
        :param d_max: The
        :param degree_threshold:
        :param sparse_floyd:
        :param device:
        """
        self.block_sz = block_sz
        self.neighbor_d_max = neighbor_d_max
        self.cr_d_max = cr_d_max
        self.d_max = d_max
        self.degree_threshold = degree_threshold
        self.sparse_floyd = sparse_floyd
        self.device = device
        self.save_name = None
        self.use_cc_dist = use_cc_dist

    def initialization(self,
                       list_polygons: list,
                       commuting_point: Tensor,
                       road_point: Tensor,
                       road_net: Tensor,
                       ):
        print('Graph initialization ...')
        # 1. mask
        device = self.device
        d_max = self.d_max
        torch.cuda.empty_cache()
        cr_mask = tensor_road_intersect_polygon(
            commuting_point, road_point, list_polygons, self.block_sz, device)

        change = (road_net == 0) & (road_net.t() > 0)
        road_net = road_net * (~change) + road_net.t() * change  # An undirected graph

        # 2. 计算所有点之间的直线距离，并通过mask生成初始距离
        #   road dist
        road_mask = road_net == 0
        road_dist_mat = road_net
        road_dist_mat[road_mask] = d_max
        nr = road_dist_mat.size(0)
        road_dist_mat[range(nr), range(nr)] = 0

        #   commuting points to road
        #       a. drop the commuting points far from road points
        cr_dist_mat = direct_pair_distance(commuting_point, road_point)
        cr_d_mask = cr_dist_mat <= self.cr_d_max    # distance to road should be small enough
        c_msk = cr_d_mask.sum(dim=-1) > 0           # drop the points do not near any road
        cr_mask = cr_mask[c_msk, :]
        cr_dist_mat = cr_dist_mat[c_msk, :]
        commuting_point = commuting_point[:, c_msk]

        #       b. direct dist between commuting points and road points should be lower than neighbor_d_max
        cr_mask = cr_mask | (cr_dist_mat > self.neighbor_d_max)
        cr_dist_mat[cr_mask] = d_max

        #   dist between commuting points should be lower than neighbor_d_max
        direct_cc_dist_mat = direct_pair_distance(commuting_point, commuting_point)
        cc_mask = direct_cc_dist_mat > self.neighbor_d_max
        cc_dist_mat = direct_cc_dist_mat * 1.4    # avg dist from block points to nearby block points
        cc_dist_mat[cc_mask] = d_max

        n_c = commuting_point.size()[1]
        select_idx = torch.arange(n_c, device=commuting_point.device)
        cc_dist_mat[select_idx, select_idx] = direct_cc_dist_mat[select_idx, select_idx]

        # 3. 剔除度小于阈值的节点
        c_degrees = (~cr_mask).sum(dim=-1) + (~cc_mask).sum(dim=-1) - 1
        select_idx = select_idx[c_degrees > 0]  # commuting 到 其它 commuting 与 road 的连接数量至少要有一条
        cr_dist_mat = cr_dist_mat[select_idx, :]
        direct_cc_dist_mat = direct_cc_dist_mat[select_idx, :][:, select_idx]
        cc_dist_mat = cc_dist_mat[select_idx, :][:, select_idx]

        print('Finish initialization')
        return road_dist_mat, cr_dist_mat, cc_dist_mat, direct_cc_dist_mat

    def initialization_cr(self,
                       list_polygons: list,
                       commuting_point: Tensor,
                       road_point: Tensor,
                       road_net: Tensor,
                       ):
        print('Graph initialization ...')
        # 1. mask
        device = self.device
        d_max = self.d_max
        torch.cuda.empty_cache()
        cr_mask = tensor_road_intersect_polygon(
            commuting_point, road_point, list_polygons, self.block_sz, device)

        change = (road_net == 0) & (road_net.t() > 0)
        road_net = road_net * (~change) + road_net.t() * change  # An undirected graph

        # 2. 计算所有点之间的直线距离，并通过mask生成初始距离
        #   road dist
        road_mask = road_net == 0
        road_dist_mat = road_net
        road_dist_mat[road_mask] = d_max
        nr = road_dist_mat.size(0)
        road_dist_mat[range(nr), range(nr)] = 0

        #   commuting points to road
        #       a. drop the commuting points far from road points
        cr_dist_mat = direct_pair_distance(commuting_point, road_point)
        cr_d_mask = cr_dist_mat <= self.cr_d_max    # distance to road should be small enough
        c_msk = cr_d_mask.sum(dim=-1) > 0           # drop the points do not near any road
        cr_mask = cr_mask[c_msk, :]
        cr_dist_mat = cr_dist_mat[c_msk, :]
        commuting_point = commuting_point[:, c_msk]

        #       b. direct dist between commuting points and road points should be lower than neighbor_d_max
        cr_mask = cr_mask | (cr_dist_mat > self.cr_d_max)
        cr_dist_mat[cr_mask] = d_max

        #   dist between commuting points should be lower than neighbor_d_max
        direct_cc_dist_mat = direct_pair_distance(commuting_point, commuting_point)

        n_c = commuting_point.size()[1]
        select_idx = torch.arange(n_c, device=commuting_point.device)

        # 3. 剔除度小于阈值的节点
        c_degrees = (~cr_mask).sum(dim=-1) - 1
        select_idx = select_idx[c_degrees > 0]  # commuting 到 其它 commuting 与 road 的连接数量至少要有一条
        cr_dist_mat = cr_dist_mat[select_idx, :]
        direct_cc_dist_mat = direct_cc_dist_mat[select_idx, :][:, select_idx]

        print('Finish initialization')
        return road_dist_mat, cr_dist_mat, direct_cc_dist_mat

    def graph_dist(self,
                   list_polygons: list,
                   commuting_point: Tensor,
                   road_point: Tensor,
                   road_net: Tensor,
                   ):
        d_max = self.d_max
        device = self.device
        if self.use_cc_dist:
            # 1. graph initialization
            road_dist_mat, cr_dist_mat, cc_dist_mat, direct_cc_dist_mat = self.initialization(
                list_polygons,
                commuting_point,
                road_point,
                road_net,
            )
            # 2. shortest path
            # combine each dist_mat together to full_dist_mat
            full_dist_mat = torch.cat([
                torch.cat([cc_dist_mat, cr_dist_mat], dim=1),
                torch.cat([cr_dist_mat.T, road_dist_mat], dim=1)], dim=0)
            # solve the shortest path
            torch.cuda.empty_cache()
            if self.sparse_floyd:
                if full_dist_mat.size(0) < 10000:
                    full_dist_mat = sparse_floyd_without_path(full_dist_mat, d_inf=d_max, device=device)
                else:
                    full_dist_mat = block_sparse_floyd_without_path(full_dist_mat, d_inf=d_max, device=device)
            else:
                if full_dist_mat.size(0) < 10000:
                    full_dist_mat = floyd_without_path(full_dist_mat, device=device)
                else:
                    full_dist_mat = block_floyd_without_path(full_dist_mat, device=device)
            # filter the available commuting points
            n_c = cc_dist_mat.size(-1)
            cc_dist_mat = full_dist_mat[:n_c, :n_c]
            road_dist_mat = full_dist_mat[n_c:, n_c:]
        else:
            # 1. graph initialization
            road_dist_mat, cr_dist_mat, direct_cc_dist_mat = self.initialization_cr(
                list_polygons,
                commuting_point,
                road_point,
                road_net,
            )
            # 2. shortest path
            torch.cuda.empty_cache()
            if self.sparse_floyd:
                if road_dist_mat.size(0) < 10000:
                    road_dist_mat = sparse_floyd_without_path(road_dist_mat, d_inf=d_max, device=device)
                else:
                    road_dist_mat = block_sparse_floyd_without_path(road_dist_mat, d_inf=d_max, device=device)
            else:
                if road_dist_mat.size(0) < 10000:
                    road_dist_mat = floyd_without_path(road_dist_mat, device=device)
                else:
                    road_dist_mat = block_floyd_without_path(road_dist_mat, device=device)

            # commuting points
            torch.cuda.empty_cache()
            if road_dist_mat.size(0) < 10000:
                cc_dist_mat = floyd_new_points_without_path(road_dist_mat, cr_dist_mat, d_inf=d_max, device=device)
            else:
                cc_dist_mat = block_floyd_new_points_without_path(road_dist_mat, cr_dist_mat, d_inf=d_max,
                                                                  device=device)

        degree_threshold = self.degree_threshold
        if degree_threshold > 0:
            n_c = cc_dist_mat.size(-1)
            r_degrees = (cc_dist_mat >= d_max).sum(dim=-1)  # 无穷大距离个数
            c_idx = torch.arange(n_c, device=commuting_point.device)[
                r_degrees < int(n_c * degree_threshold)]  # 无穷大数量小于阈值
            if c_idx.numel() <= 0:
                return -1, -1, -1
            direct_cc_dist_mat = direct_cc_dist_mat[c_idx, :][:, c_idx]
            cc_dist_mat = cc_dist_mat[c_idx, :][:, c_idx]
            torch.cuda.empty_cache()
        # distance to oneself is 0
        c_idx = torch.arange(cc_dist_mat.size(-1), device=commuting_point.device)
        cc_dist_mat[c_idx, c_idx] = 0
        return direct_cc_dist_mat, cc_dist_mat, road_dist_mat

    def run(self,
            list_polygons: list,
            commuting_point: Tensor,
            road_point: Tensor,
            road_net: Tensor,
            save_name: str = None,
            ):
        self.save_name = save_name
        # check the inputs
        if ((commuting_point.dim() != 2 or numpy.prod(numpy.array(commuting_point.size())) < 1) or
                (road_point.dim() != 2 or numpy.prod(numpy.array(road_point.size())) < 1) or
                (road_net.dim() != 2 or numpy.prod(numpy.array(road_net.size())) < 1) or
                (road_point.size(1) != road_net.size(1))
        ):
            return -1, -1, -1, -1, -1, -1, -1

        # graphical_distance
        direct_cc_dist_mat, cc_dist_mat, road_dist_mat = self.graph_dist(
            list_polygons,
            commuting_point,
            road_point,
            road_net,
        )

        if not direct_cc_dist_mat.__class__.__name__ == 'Tensor':
            return -1, -1, -1, -1, -1, -1, -1

        # graphical index
        direct_cc_dist_mat = direct_cc_dist_mat.to(dtype=torch.float32)
        cc_dist_mat = cc_dist_mat.to(dtype=torch.float32)
        gid = graphical_index_(direct_cc_dist_mat, cc_dist_mat).item()
        d_mean = cc_dist_mat.mean().item()
        d_std = cc_dist_mat.std().item()
        d_max = cc_dist_mat.max().item()
        d_mean_direct = direct_cc_dist_mat.mean().item()

        # save
        save_name = self.save_name
        if save_name is not None:
            torch.save(cc_dist_mat, save_name + '_cc_dist_mat.pth')
            torch.save(road_dist_mat, save_name + '_road_dist_mat.pth')
            torch.save(gid, save_name + '_graphical_index.pth')
        return gid, d_max, d_mean, d_std, d_mean_direct, cc_dist_mat.size(0), road_dist_mat.size(0)
