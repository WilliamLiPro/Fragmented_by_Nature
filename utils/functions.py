import torch
from torch import Tensor
from scipy.spatial import ConvexHull
from utils import PointGroups, LineSegGroups, LineSeg
from shapely.geometry import Polygon


def cross_product(
        point1: Tensor,
        point2: Tensor,
):
    """
    cross product
    """
    return point1[0, ...] * point2[1, ...] - point1[1, ...] * point2[0, ...]


def cross_product_mm(
        point1: Tensor,
        point2: Tensor,
):
    """
    cross product with matmul
    """
    point1[..., 1] = -point1[..., 1]
    return point1.unsqueeze(-2).matmul(point2[..., [1, 0]].unsqueeze(-1)).squeeze(-1).squeeze(-1)


def bounds_to_points(
        bounds: Tensor,
):
    """bounds_to_points

    :param bounds: (n, 4), with (min_x, min_y, max_x, max_y) on each line
    :return: (2, n, 5)
    """
    p_x = bounds[..., [0, 0, 2, 2]].view(1, -1, 4)
    p_y = bounds[..., [1, 3, 3, 1]].view(1, -1, 4)
    return torch.cat([p_x, p_y], dim=0)


def pair_max_and_min(
        point_st: Tensor,
        point_end: Tensor,
):
    assert point_st.size(0) == point_end.size(0)
    mask_max = point_st > point_end
    max_value = point_st.expand_as(mask_max) * mask_max + point_end.expand_as(mask_max) * (~mask_max)
    min_value = point_st.expand_as(mask_max) * (~mask_max) + point_end.expand_as(mask_max) * mask_max
    return max_value, min_value


def distance(x: Tensor, target: Tensor, distance_type='Euclid'):
    if distance_type == 'Euclid':
        return torch.pairwise_distance(x, target, 2)
    if distance_type == 'Manhattan':
        return torch.pairwise_distance(x, target, 1)
    if distance_type == 'Chebyshev':
        return (x - target).abs().max(-1)[0]
    if distance_type == 'Chebyshev2':
        z = (x - target).abs()
        d = z.size(-1) // 2
        return z[..., :d].max(-1)[0] + z[..., d:].max(-1)[0]
    if distance_type == 'Line':
        d = x.size(-1) // 2
        dx = (x[..., :d] - x[..., d:]).abs()
        return torch.pairwise_distance(
            torch.cat([x, dx]), torch.cat([target[..., :d], target[..., :d], target[..., d:]]), 2)
    else:
        raise TypeError(f'Unexpected distance_type, got {distance_type}')


def grid_mass_center_calculate(
        points: Tensor,
        grids_nx,
        grids_ny,
):
    px = points[0, :]
    py = points[1, :]

    # grids
    x_min = px.min()
    x_max = px.max()
    y_min = py.min()
    y_max = py.max()

    grid_dx = (x_max - x_min) / grids_nx
    grid_dy = (y_max - y_min) / grids_ny

    # point belongs to which grid
    belongs_x = ((px - x_min) // grid_dx).to(torch.int16)
    belongs_y = ((py - y_min) // grid_dy).to(torch.int16)

    # grid_mass_center
    grid_mass_center = torch.zeros((grids_nx, grids_ny, 2), dtype=px.dtype, device=px.device)
    grid_center_weight = torch.zeros((grids_nx, grids_ny), dtype=px.dtype, device=px.device)
    for i in range(grids_nx):
        for j in range(grids_ny):
            mask = (belongs_x == i) & (belongs_y == j)
            if mask.sum() == 0:
                continue
            grid_mass_center[i, j, :] = points[:, mask].mean(dim=1)
            grid_center_weight[i, j] = mask.sum()
    return grid_mass_center, grid_center_weight, grid_dx, grid_dy


def cluster_grid_center(
        grid_mass_center,
        grid_center_weight,
        grid_dx,
        grid_dy,
        cluster_range=0.5,
        distance_type='Euclid',
):
    center_d = grid_mass_center.size(-1)
    cluster_range = cluster_range * distance(torch.zeros(2), Tensor([grid_dx, grid_dy]), distance_type).item()
    grid_center_weight = grid_center_weight.view(-1)
    mask = grid_center_weight > 0
    grid_mass_center = grid_mass_center.view(-1, center_d)[mask, :]
    grid_center_weight = grid_center_weight[mask]
    grid_mask = torch.zeros_like(grid_center_weight).to(torch.bool)
    grid_center_weight = grid_center_weight.unsqueeze(-1)
    centers = []
    for i, current_p in enumerate(grid_mass_center):
        neighbor = distance(current_p, grid_mass_center, distance_type) < cluster_range
        neighbor = neighbor & (~grid_mask)
        if neighbor.sum() == 0:
            continue
        grid_mask = grid_mask | neighbor
        centers.append((grid_mass_center[neighbor] * grid_center_weight[neighbor]).sum(dim=0) /
                       grid_center_weight[neighbor].sum(dim=0).view(1, -1))
    centers = torch.cat(centers, dim=0)
    return centers


def grid_mass_center_clustering(
        points: Tensor,
        grids_nx=4,
        grids_ny=4,
        cluster_range=0.5,
        distance_type='Euclid',
) -> Tensor:
    # grid_mass_center
    grid_mass_center, grid_center_weight, grid_dx, grid_dy = grid_mass_center_calculate(points, grids_nx, grids_ny)

    # cluster the grid center
    centers = cluster_grid_center(
        grid_mass_center, grid_center_weight, grid_dx, grid_dy, cluster_range, distance_type)
    return centers


def grid_line_center_calculate(
        p_st: Tensor,
        p_nd: Tensor,
        grids_nx,
        grids_ny,
        distance_type='Euclid',
):
    dp = (p_st - p_nd).abs()
    n_p = p_st.size(1)
    p_cat = torch.cat([p_st, p_nd], dim=1)
    px = p_cat[0, :]
    py = p_cat[1, :]

    # grids
    x_min = px.min()
    x_max = px.max()
    y_min = py.min()
    y_max = py.max()
    r_max_x = dp[0, :].max()
    r_max_y = dp[1, :].max()

    grid_dx = (x_max - x_min + 0.0000001) / grids_nx
    grid_dy = (y_max - y_min + 0.0000001) / grids_ny
    grids_nrx = int((r_max_x // grid_dx + 1).cpu().numpy())
    grids_nry = int((r_max_y // grid_dy + 1).cpu().numpy())


    # point belongs to which grid
    belongs_x = ((px - x_min) // grid_dx).to(torch.int16)
    belongs_y = ((py - y_min) // grid_dy).to(torch.int16)
    belongs_rx = (dp[0, :] // grid_dx).to(torch.int16)
    belongs_ry = (dp[1, :] // grid_dy).to(torch.int16)

    # grid_mass_center
    grid_mass_center = torch.zeros((grids_nx, grids_ny, grids_nrx, grids_nry, 4), dtype=px.dtype, device=px.device)
    grid_center_weight = torch.zeros((grids_nx, grids_ny, grids_nrx, grids_nry), dtype=px.dtype, device=px.device)
    for i in range(grids_nx):
        for j in range(grids_ny):
            for s in range(grids_nrx):
                for t in range(grids_nry):
                    mask = (((belongs_x[:n_p] == i) & (belongs_y[:n_p] == j) |
                             (belongs_x[n_p:] == i) & (belongs_y[n_p:] == j)) &
                            (belongs_rx == s) & (belongs_ry == t))
                    if mask.sum() == 0:
                        continue
                    grid_mass_center[i, j, s, t, :2] = 0.5 * (p_cat[:, :n_p][:, mask].mean(dim=1) +
                                                              p_cat[:, n_p:][:, mask].mean(dim=1))
                    grid_mass_center[i, j, s, t, 2:] = dp[:, mask].mean(dim=1)
                    grid_center_weight[i, j, s, t] = mask.sum()
    return grid_mass_center, grid_center_weight, grid_dx, grid_dy


def grid_line_center_clustering(
        points_st: Tensor,
        points_end: Tensor,
        grids_nx=4,
        grids_ny=4,
        cluster_range=0.5,
        distance_type='Line',
) -> Tensor:
    # grid_mass_center
    grid_mass_center, grid_center_weight, grid_dx, grid_dy = grid_line_center_calculate(
        points_st, points_end, grids_nx, grids_ny, distance_type)

    # cluster the grid center
    centers = cluster_grid_center(
        grid_mass_center, grid_center_weight, grid_dx, grid_dy, cluster_range, distance_type)
    return centers


def kmeans(
        points: Tensor,
        cluster_centers: Tensor,
        max_iter=256,
        distance_type='Euclid',
) -> (Tensor, Tensor):
    if points.size(0) < points.size(1):
        points = points.T
    dim = points.size(1)
    n = points.size(0)

    indices_pre = None
    for i in range(max_iter):
        dist = distance(cluster_centers.view(-1, 1, dim), points.view(1, -1, dim), distance_type)
        value, group_indices = dist.min(dim=0)
        if indices_pre is not None:
            if indices_pre.equal(group_indices):
                break
        # for j in range(group_indices.max()):
        #     mask = group_indices == j
        #     if mask.sum() == 0:
        #         continue
        #     cluster_centers[j, :] = points[mask, :].mean(dim=0)
        mask = (value.view(1, -1) == dist)
        c_mask = mask.sum(dim=1) > 0
        mask = mask[c_mask, :]
        cluster_centers[c_mask, :] = ((points.view(1, -1, dim) * mask.unsqueeze(-1)).sum(dim=1) /
                                      mask.unsqueeze(-1).sum(dim=1))
        indices_pre = group_indices
    group_indices = group_indices.unique()
    cluster_centers = cluster_centers[group_indices, :]
    value, group_indices = dist[group_indices, :].min(dim=0)
    return cluster_centers, group_indices


def tensor_point_to_groups(
        points: Tensor,
        grid_nx=4,
        grid_ny=4,
        cluster_range=0.5,
        kmeans_max_iter=256,
        distance_type='Euclid',
):
    cluster_center = grid_mass_center_clustering(points, grid_nx, grid_ny, cluster_range, distance_type)
    cluster_center, indices = kmeans(points, cluster_center, kmeans_max_iter, distance_type)
    idx = torch.arange(points.size(1), device=points.device, dtype=torch.int, )
    values = []
    order_idx = []
    slit_position = [0]
    for i, _ in enumerate(cluster_center):
        mask = indices == i
        values.append(points[:, mask])
        order_idx.append(idx[mask])
        slit_position.append(slit_position[i] + order_idx[i].size(0))
    return PointGroups(torch.cat(values, dim=-1), slit_position, torch.cat(order_idx, dim=-1))


def tensor_lines_to_groups(
        seg_st: Tensor,
        seg_end: Tensor,
        grid_nx=4,
        grid_ny=4,
        cluster_range=0.5,
        kmeans_max_iter=256,
        distance_type='Chebyshev',
):
    cluster_center = grid_line_center_clustering(
        seg_st, seg_end, grid_nx, grid_ny, cluster_range, distance_type)
    cluster_center, indices = kmeans(
        torch.cat([seg_st, seg_end], dim=0), cluster_center, kmeans_max_iter, distance_type)
    idx = torch.arange(seg_st.size(1), device=seg_st.device, dtype=torch.int64, )
    values_st = []
    values_end = []
    order_idx = []
    slit_position = [0]
    for i, _ in enumerate(cluster_center):
        mask = indices == i
        values_st.append(seg_st[:, mask])
        values_end.append(seg_end[:, mask])
        order_idx.append(idx[mask])
        slit_position.append(slit_position[i] + order_idx[i].size(0))
    return LineSegGroups(torch.cat(values_st, dim=-1), torch.cat(values_end, dim=-1),
                         slit_position, torch.cat(order_idx, dim=-1))


def multi_polygon_to_tensor(
        multipolygon: list,
):
    points = []
    line_split_position = [0]
    polygon_split_position = [0]
    line_p = 0
    for polygon in multipolygon:
        if isinstance(polygon, list):
            for line in polygon:
                points.append(line)
                line_p = line_p + line.size(-1)
                line_split_position.append(line_p)
            polygon_split_position.append(line_p)
        else:
            points.append(polygon)
            line_p = line_p + polygon.size(-1)
            line_split_position.append(line_p)
            polygon_split_position.append(line_p)
    if len(points) == 0:
        return torch.empty(2, 0), torch.zeros(1), torch.zeros(1)
    multipolygon = torch.cat(points, dim=1)
    line_split_position = torch.tensor(line_split_position, dtype=torch.int, device=multipolygon.device)
    polygon_split_position = torch.tensor(polygon_split_position, dtype=torch.int, device=multipolygon.device)
    return multipolygon, line_split_position, polygon_split_position


def multi_polygon_to_groups(

):
    pass


def line_reorder(
        line_b: LineSeg,
):
    order = torch.argsort(line_b.seg_st[0, :])
    return LineSeg(line_b.seg_st[:, order], line_b.seg_end[:, order]), order


def convex_hull(
        points: Tensor,
):
    if points.size(1) < 3:
        return points
    else:
        # coplanar with the interior point
        normalized_points = (points - points[:, 0].view(-1, 1))[:, 1:]
        normalized_points = normalized_points / torch.norm(normalized_points, dim=0).view(1, -1)
        mask = normalized_points[0, :] < 0
        normalized_points[:, mask] = -normalized_points[:, mask]
        normalized_points = normalized_points - normalized_points[:, 0].view(-1, 1)
        if torch.all(normalized_points == 0):
            return points
    hull = ConvexHull(points.cpu().numpy().T)
    convex_points = points[:, hull.vertices]
    return convex_points


@torch.no_grad()
def convex_hull_tensor(
        points: Tensor,
        p_slit_position: Tensor,
):
    """
    Calculate multi_convex hull of points and return the convex hull of each slit.
    :param points:
    :param p_slit_position:
    :return:
    """
    def line_dist_side(p_s: Tensor, p_e: Tensor, p: Tensor):
        p_s = p_s.view(2, -1, 1)
        p_e = p_e.view(2, -1, 1)
        val = ((p[1, ...] - p_s[1, ...]) * (p_e[0, ...] - p_s[0, ...]) -
               (p_e[1, ...] - p_s[1, ...]) * (p[0, ...] - p_s[0, ...]))
        return val.abs(), val > 0

    def quick_hull(g_id: Tensor, p: Tensor, p_mask: Tensor, p_s: Tensor, p_e: Tensor):
        # 默认逆时针为正，永远取逆时针方向的点
        dist, side_mask = line_dist_side(p_s, p_e, p)
        side_mask[~p_mask] = 0
        dist[~side_mask] = 0
        max_value, max_idx = dist.max(dim=-1)
        side_n = side_mask.sum(dim=-1)
        side_no_empty = side_n > 0
        # 最远线段的端点加入凸包
        side_empty = ~side_no_empty
        c_gid = g_id[side_empty]
        convex_hull_new[:, c_gid, convex_hull_n[c_gid]] = p_e[:, side_empty]
        # convex_hull_new[:, c_gid, convex_hull_n[c_gid] + 1] = p_e[:, side_empty]
        convex_hull_n[c_gid] += 1
        if side_no_empty.sum() == 0:
            return
        # 提取同side点
        g_id_new = g_id[side_no_empty]
        g_m = side_no_empty.sum()
        n_max = side_n.max()
        p_new = torch.zeros((2, g_m, n_max), dtype=p.dtype, device=p.device)
        p_mask_new = torch.zeros((g_m, n_max), dtype=torch.bool, device=p_mask.device)
        j_id = 0
        for j in range(side_no_empty.size(0)):
            if side_no_empty[j]:
                c_mask = side_mask[j, :]
                c_n = side_n[j]
                p_new[:, j_id, :c_n] = p[:, j, c_mask]
                p_mask_new[j_id, :c_n] = 1
                j_id += 1
        # 递归处理
        p_mid = p[:, side_no_empty, max_idx[side_no_empty]]
        quick_hull(g_id_new, p_new, p_mask_new, p_s[:, side_no_empty], p_mid)
        quick_hull(g_id_new, p_new, p_mask_new, p_mid, p_e[:, side_no_empty])

    # points into groups
    group_n = p_slit_position.size(0) - 1
    group_size = p_slit_position[1:] - p_slit_position[:-1]
    m_group_size = group_size.max()
    points_tensor = torch.zeros(2, group_n, m_group_size, dtype=points.dtype, device=points.device)
    points_mask = torch.zeros(group_n, m_group_size, dtype=torch.bool, device=points.device)
    for i in range(group_n):
        j_st = p_slit_position[i]
        j_end = p_slit_position[i+1]
        points_tensor[:, i, :group_size[i]] = points[:, j_st:j_end]
        points_mask[i, :group_size[i]] = 1

    # convex hull of each group
    convex_hull_new = torch.zeros((2, group_n, m_group_size * 2), dtype=torch.float32, device=points.device)
    convex_hull_n = torch.zeros(group_n, dtype=torch.int32, device=points.device)
    points_tensor[0, ~points_mask] = torch.inf
    min_x, min_x_idx = points_tensor[0, :, :].min(dim=-1)
    points_tensor[0, ~points_mask] = -torch.inf
    max_x, max_x_idx = points_tensor[0, :, :].max(dim=-1)
    if points.size(1) <= 3:
        return points
    else:
        g_id_s = torch.arange(group_n, device=points.device, dtype=torch.int32)
        quick_hull(g_id_s, points_tensor, points_mask, points_tensor[:, g_id_s, min_x_idx], points_tensor[:, g_id_s, max_x_idx])
        quick_hull(g_id_s, points_tensor, points_mask, points_tensor[:, g_id_s, max_x_idx], points_tensor[:, g_id_s, min_x_idx])
        convex_hull_new = torch.cat([convex_hull_new[:, i, :convex_hull_n[i]] for i in range(group_n)], dim=-1)
        convex_hull_split_position = torch.tensor([0] + [convex_hull_n[:i + 1].sum() for i in range(group_n)], device=points.device)
        return convex_hull_new, convex_hull_split_position


def convex_hull_to_groups(
        points: Tensor,
):
    pass


def convex_hull_intersect(
        convex_a: Tensor,
        convex_b_list: Tensor,
        b_slit_position: Tensor,
):
    """
    The separation axis theorem is used to detect intersections between the convex a and list of convex b
    :param convex_a: (2, n)
    :param convex_b_list: (2, m1 + m2 + ... + mn)
    :param b_slit_position: (n+1)
    :return:
    """
    # calculate the projection on separation axis of a
    s_a = torch.roll(convex_a, 1, -1) - convex_a
    proj_a_on_a = (s_a[1, :].view(-1, 1) * convex_a[0, :].view(1, -1) -
                   s_a[0, :].view(-1, 1) * convex_a[1, :].view(1, -1))
    proj_b_list_on_a = (s_a[1, :].view(-1, 1) * convex_b_list[0, :].view(1, -1) -
                        s_a[0, :].view(-1, 1) * convex_b_list[1, :].view(1, -1))

    # max project and min project of each convex hull on s_a
    max_proj_a_on_a = proj_a_on_a.max(dim=-1, keepdim=True)[0]
    min_proj_a_on_a = proj_a_on_a.min(dim=-1, keepdim=True)[0]
    max_proj_b_list_on_a = []
    min_proj_b_list_on_a = []
    for i in range(b_slit_position.size(0) - 1):
        j_st = b_slit_position[i]
        j_end = b_slit_position[i+1]
        max_proj_b_list_on_a.append(proj_b_list_on_a[:, j_st: j_end].max(dim=-1, keepdim=True)[0])
        min_proj_b_list_on_a.append(proj_b_list_on_a[:, j_st: j_end].min(dim=-1, keepdim=True)[0])
    max_proj_b_list_on_a = torch.cat(max_proj_b_list_on_a, dim=-1)
    min_proj_b_list_on_a = torch.cat(min_proj_b_list_on_a, dim=-1)

    # a b intersect on s_a
    intersect_sa = ((max_proj_a_on_a > min_proj_b_list_on_a) & (min_proj_a_on_a < max_proj_b_list_on_a)).all(dim=0)

    # calculate the projection on separation axis of b
    b_end = torch.roll(convex_b_list, 1, -1)
    b_end[:, b_slit_position[:-1]] = convex_b_list[:, b_slit_position[1:] - 1]
    s_b = b_end - convex_b_list
    proj_a_on_b = (s_b[1, :].view(-1, 1) * convex_a[0, :].view(1, -1) -
                   s_b[0, :].view(-1, 1) * convex_a[1, :].view(1, -1))

    # max project and min project of each convex hull on s_b
    intersect_sb = []
    for i in range(b_slit_position.size(0) - 1):
        j_st = b_slit_position[i]
        j_end = b_slit_position[i + 1]
        max_proj_a_on_b = proj_a_on_b[j_st: j_end, :].max(dim=-1, keepdim=True)[0]
        min_proj_a_on_b = proj_a_on_b[j_st: j_end, :].min(dim=-1, keepdim=True)[0]
        proj_b_on_b = (s_b[1, j_st: j_end].view(-1, 1) * convex_b_list[0, j_st: j_end].view(1, -1) -
                       s_b[0, j_st: j_end].view(-1, 1) * convex_b_list[1, j_st: j_end].view(1, -1))
        max_proj_b_on_b = proj_b_on_b.max(dim=-1, keepdim=True)[0]
        min_proj_b_on_b = proj_b_on_b.min(dim=-1, keepdim=True)[0]

        # a b intersect on s_b
        intersect_sb.append(((max_proj_a_on_b > min_proj_b_on_b) & (min_proj_a_on_b < max_proj_b_on_b)).all(dim=0))
    intersect_sb = torch.cat(intersect_sb, dim=-1)
    return intersect_sa & intersect_sb


def convex_hull_intersect_shapely(
        convex_a: Tensor,
        convex_b_list: list,
):
    mask = torch.zeros(len(convex_b_list), dtype=torch.bool, device=convex_a.device)
    polygon_a = Polygon(convex_a.cpu().T)
    for i, convex_b in enumerate(convex_b_list):
        polygon_b = Polygon(convex_b.cpu().T)
        mask[i] = polygon_a.intersects(polygon_b)
    return mask


@torch.no_grad()
def convex_hull_intersect_tensor(
        convex_a: Tensor,
        convex_b_list: list,
):
    """
    The direct product detecting intersections between the convex a and list of convex b
    :param convex_a: (2, n)
    :param convex_b_list: (2, m1 + m2 + ... + mn)
    :return:
    """
    b_slit_position = torch.tensor([0] + [convex_b.size(-1) for convex_b in convex_b_list], device=convex_a.device)
    b_slit_position = b_slit_position.cumsum(dim=0)
    convex_b_list = torch.cat(convex_b_list, dim=-1)

    a_end = torch.roll(convex_a, 1, 1)
    b_end = torch.roll(convex_b_list, 1, 1)
    b_end[:, b_slit_position[:-1]] = convex_b_list[:, b_slit_position[1:] - 1]
    convex_a = convex_a.view(2, 1, -1)
    a_end = a_end.view(2, 1, -1)
    convex_b_list = convex_b_list.view(2, -1, 1)
    b_end = b_end.view(2, -1, 1)
    da = a_end - convex_a
    db = b_end - convex_b_list
    mask = ((cross_product(convex_b_list - convex_a, da) * cross_product(b_end - convex_a, da) < 0) &
            (cross_product(convex_a - convex_b_list, db) * cross_product(a_end - convex_b_list, db) < 0))
    mask = mask.any(dim=-1)
    intersect_mask = torch.zeros(b_slit_position.size(0) - 1, dtype=torch.bool, device=convex_a.device)
    for i in range(b_slit_position.size(0) - 1):
        j_st = b_slit_position[i]
        j_end = b_slit_position[i + 1]
        intersect_mask[i] = mask[j_st: j_end].any()
    return intersect_mask

