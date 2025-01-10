from tqdm import tqdm
import torch
from torch import Tensor
from utils import (
    cross_product,
    intersect_mask_core_decompose_int8, intersect_mask_core_db_int8, intersect_mask_core_da_db_int8,
    intersect_position_core_decompose, convex_hull, convex_hull_tensor,
    convex_hull_intersect,
    inner_mask_core_da)


@torch.no_grad()
def points_inner_convex_hull_mask(
        points: Tensor,
        convex: Tensor,
        convex_slit_position: Tensor = None,
) -> Tensor:
    """
    Compute the mask of whether the points in convex hull.
    :param points:
    :param convex:
    :param convex_slit_position:
    :return:
    """
    points_n = points.size(-1)
    if convex_slit_position is None:
        convex_slit_position = torch.Tensor((0, convex.size(1))).to(torch.int).to(convex.device)
    else:
        convex_slit_position = convex_slit_position.clone()
    convex_n = convex_slit_position.size(0) - 1
    points = points.view(2, 1, -1)
    # insert first point of slice to the end of slice
    slices = []
    for i in range(convex_n):
        slices.append(convex[:, convex_slit_position[i]: convex_slit_position[i + 1]])
        slices.append(convex[:, convex_slit_position[i]].view(-1, 1))
    convex = torch.cat(slices, dim=1)
    convex = convex.view(2, -1, 1)
    for i in range(convex_n + 1):
        convex_slit_position[i] = convex_slit_position[i] + i
    # mask of start and end of polygon
    mask = torch.ones(convex.size(1), device=convex.device, dtype=torch.bool)
    mask_st = mask.clone()
    mask_st[convex_slit_position[1:] - 1] = 0
    mask_en = mask.clone()
    mask_en[convex_slit_position[:-1]] = 0
    # inner mask of points
    da = (convex[:, :-1, :] - convex[:, 1:, :])[:, mask_en[1:], :]  # as - ae
    as_da = cross_product(convex[:, mask_st, :], da)
    da_b = cross_product(da, points)
    p_inner_convex_p, p_inner_convex_n = inner_mask_core_da(-as_da, da_b, convex[1, mask_st, :], convex[1, mask_en, :],
                                                            points[1, ...])
    inner_mask = torch.zeros(convex_n, points_n, device=convex.device, dtype=torch.bool)
    for i in range(convex_n):
        silt_st = convex_slit_position[i] - i
        silt_en = convex_slit_position[i + 1] - (i + 1)
        inner_mask[i, :] = (p_inner_convex_p[silt_st:silt_en, :].any(0)) & (p_inner_convex_n[silt_st:silt_en, :].any(0))
    return inner_mask


@torch.no_grad()
def line_seg_convex_hull_intersect_mask(
        a_st: Tensor,
        a_end: Tensor,
        convex: Tensor,
        convex_slit_position: Tensor = None,
) -> Tensor:
    """
    Compute the mask of whether the line-seg intersect with convex hull.
    :param a_st:
    :param a_end:
    :param convex:
    :param convex_slit_position:
    :return:
    """
    dim = a_st.dim() - 1
    a_st = a_st.unsqueeze(-1)
    a_end = a_end.unsqueeze(-1)
    convex_end = torch.roll(convex, 1, -1)
    if convex_slit_position is not None:
        convex_end[convex_slit_position[:-1]] = convex[convex_slit_position[1:] - 1]
    convex_d = convex_end - convex
    convex = convex.view(2, *([1, ] * dim), -1)
    convex_end = convex_end.view(2, *([1, ] * dim), -1)
    convex_d = convex_d.view(2, *([1, ] * dim), -1)

    as_da = cross_product(a_st, a_end - a_st)
    as_bs = cross_product(a_st, convex)
    as_be = cross_product(a_st, convex_end)
    as_db = cross_product(a_st, convex_d)
    ae_bs = cross_product(a_end, convex)
    ae_be = cross_product(a_end, convex_end)
    ae_db = cross_product(a_end, convex_d)
    bs_db = cross_product(convex, convex_d)

    mask = intersect_mask_core_db_int8(as_da, as_bs, as_be, as_db, ae_bs, ae_be, ae_db, bs_db)
    if convex_slit_position is not None:
        n = convex_slit_position.size(-1) - 1
        o_mask = torch.zeros((*mask.size()[:-1], n), device=mask.device, dtype=torch.bool)
        for i in range(convex_slit_position.size(-1) - 1):
            o_mask[..., i] = mask[..., convex_slit_position[i]: convex_slit_position[i + 1]].any(-1)
    else:
        o_mask = mask.any(-1)
    return o_mask


# ---------  intersect length ----------

@torch.no_grad()
def pp_m_polygon_intersect_length_sparse_decompose(
        multi_polygon: Tensor,
        line_slit_position: Tensor,
        polygon_slit_position: Tensor,
        a_st: Tensor,
        a_st_slit_position: Tensor,
        a_en: Tensor = None,
        a_en_slit_position: Tensor = None,
        polygon_weights: Tensor = None,
        batch_size: int = 256,
):
    """
    Intersect a set of point pairs with the multi_polygon.
    Accelerated with group sparse and decomposition of points and multi_polygon.
    :param multi_polygon:
    :param line_slit_position:
    :param polygon_slit_position:
    :param a_st:
    :param a_st_slit_position:
    :param a_en:
    :param a_en_slit_position:
    :param polygon_weights:
    :return:
    """
    # nodes
    same_points = False
    if a_en is None:
        same_points = True
        a_en = a_st
        a_en_slit_position = a_st_slit_position
    n_as = a_st.size(1)
    n_ae = a_en.size(1)
    n_polygon = polygon_slit_position.size(0) - 1
    if n_polygon == 0:
        return torch.zeros((n_as, n_ae), device=a_st.device, dtype=torch.float16)

    if polygon_weights is None:
        polygon_weights = torch.ones(n_polygon + 1, device=a_st.device, dtype=multi_polygon.dtype)
    else:
        polygon_weights = polygon_weights.to(a_st.device)

    a_st = a_st.view(2, -1, 1, 1)
    a_en = a_en.view(2, 1, -1, 1)

    mask = torch.ones(multi_polygon.size(1), device=a_st.device, dtype=torch.bool)
    mask_st = mask.clone()
    mask_st[line_slit_position[1:] - 1] = 0
    mask_en = mask.clone()
    mask_en[line_slit_position[:-1]] = 0

    polygon_slit_position = polygon_slit_position.clone()
    for i in range(1, line_slit_position.size(0)):
        polygon_slit_position[polygon_slit_position >= line_slit_position[i] - i] += -1

    multi_polygon = multi_polygon.view(2, 1, 1, -1)

    as_da = cross_product(a_st, a_en - a_st)
    as_b = cross_product(a_st, multi_polygon)
    as_bs = as_b[..., mask_st]
    as_be = as_b[..., mask_en]
    db = (multi_polygon[..., 1:] - multi_polygon[..., :-1])[..., mask_en[1:]]
    as_db = cross_product(a_st, db)
    bs_db = cross_product(multi_polygon[..., mask_st], db)
    if same_points:
        ae_bs = as_bs.view(1, n_ae, -1)
        ae_be = as_be.view(1, n_ae, -1)
        ae_db = as_db.view(1, n_ae, -1)
    else:
        ae_b = cross_product(a_en, multi_polygon)
        ae_bs = ae_b[..., mask_st]
        ae_be = ae_b[..., mask_en]
        ae_db = cross_product(a_en, db)

    # convex hull
    a_st_group_convex_hull = []
    a_st_group_n = len(a_st_slit_position) - 1
    for i in range(a_st_group_n):
        a_st_group_convex_hull.append(
            convex_hull(a_st.view(2, -1)[..., a_st_slit_position[i]:a_st_slit_position[i + 1]]))

    if same_points:
        a_en_group_convex_hull = a_st_group_convex_hull
        a_en_group_n = a_st_group_n
    else:
        a_en_group_convex_hull = []
        a_en_group_n = len(a_en_slit_position) - 1
        for i in range(a_en_group_n):
            a_en_group_convex_hull.append(
                convex_hull(a_en.view(2, -1)[..., a_en_slit_position[i]:a_en_slit_position[i + 1]]))

    # intersect mask
    intersect_length = torch.zeros((n_as, n_ae), device=a_st.device, dtype=torch.float16)
    tq = tqdm(total=a_st_group_n * a_en_group_n)
    for i in range(a_st_group_n):
        # regional mask
        region_mask = torch.zeros((a_en_group_n, multi_polygon.size(-1)), device=a_st.device, dtype=torch.bool)
        for j in range(a_en_group_n):
            if same_points and i == j:
                union_convex_hull = a_st_group_convex_hull[i]
            else:
                union_convex_hull = convex_hull(
                    torch.cat((a_st_group_convex_hull[i], a_en_group_convex_hull[j]), dim=-1))
            region_mask[j, :] = points_inner_convex_hull_mask(multi_polygon, union_convex_hull).view(-1)

        # iteration length
        i_st = a_st_slit_position[i]
        i_en = a_st_slit_position[i + 1]
        as_bs_group = as_bs[i_st:i_en, ...]
        as_be_group = as_be[i_st:i_en, ...]
        as_db_group = as_db[i_st:i_en, ...]
        for j in range(a_en_group_n):
            tq.update(1)
            j_st = a_en_slit_position[j]
            j_en = a_en_slit_position[j + 1]

            mask = region_mask[j, :]
            group_mask = (mask[mask_st] | mask[mask_en])

            # mask
            as_da_group = as_da[i_st:i_en, j_st:j_en, :]
            ae_bs_group = ae_bs[:, j_st:j_en, group_mask]
            ae_be_group = ae_be[:, j_st:j_en, group_mask]
            ae_db_group = ae_db[:, j_st:j_en, group_mask]
            intersect_pos = intersect_position_core_decompose(
                as_da_group, as_db_group[..., group_mask], as_bs_group[..., group_mask], as_be_group[..., group_mask],
                ae_db_group, ae_bs_group, ae_be_group, bs_db[..., group_mask])

            # intersect length with polygon
            polygon_pos = intersect_pos.sort(dim=-1, descending=True)[0]
            l_pos = polygon_pos.size(-1) // 2 * 2
            polygon_pos = polygon_pos[..., :l_pos]
            intersect_length[i_st: i_en, j_st: j_en] = (polygon_pos[..., 0::2] - polygon_pos[..., 1::2]).sum(-1)

    return intersect_length

