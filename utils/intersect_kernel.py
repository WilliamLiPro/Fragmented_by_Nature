from tqdm import tqdm
import torch
from torch import Tensor
from utils import cross_product


def intersect_mask_core_simple(
        a_st: Tensor,
        a_en: Tensor,
        b_st: Tensor,
        b_en: Tensor,
):
    """
    This function computes the intersection mask between two line sets a and b.
    :param a_st:
    :param a_en:
    :param b_st:
    :param b_en:
    :return:
    """
    da = a_en - a_st
    db = b_en - b_st
    mask = ((cross_product(b_st - a_st, da) * cross_product(b_en - a_st, da) < 0) &
            (cross_product(a_st - b_st, db) * cross_product(a_en - b_st, db) < 0))
    return mask


def intersect_mask_core_simple_int8(
        a_st: Tensor,
        a_en: Tensor,
        b_st: Tensor,
        b_en: Tensor,
):
    """
    This function computes the intersection mask between two line segments a and b.
    :param a_st:
    :param a_en:
    :param b_st:
    :param b_en:
    :return:
    """
    mask = ((cross_product(b_st - a_st, a_en - a_st).sgn().to(torch.int8) * cross_product(b_en - a_st, a_en - a_st).sgn().to(torch.int8) < 0) &
            (cross_product(a_st - b_st, b_en - b_st).sgn().to(torch.int8) * cross_product(a_en - b_st, b_en - b_st).sgn().to(torch.int8) < 0))
    return mask


def intersect_mask_core(
        as_ae: Tensor,
        as_bs: Tensor,
        as_be: Tensor,
        ae_bs: Tensor,
        ae_be: Tensor,
        bs_be: Tensor
) -> Tensor:
    """
    This function computes the intersection mask between two line segments a and b.
    :param as_ae:
    :param as_bs:
    :param as_be:
    :param ae_bs:
    :param ae_be:
    :param bs_be:
    :return:
    """
    mask = (((ae_bs - as_bs + as_ae) * (ae_be - as_be + as_ae) < 0) &
            ((as_bs + bs_be - as_be) * (ae_bs + bs_be - ae_be) < 0))
    return mask


def intersect_mask_core_decompose_int8(
        as_ae: Tensor,
        as_bs: Tensor,
        as_be: Tensor,
        ae_bs: Tensor,
        ae_be: Tensor,
        bs_be: Tensor
) -> Tensor:
    """
    This function computes the intersection mask between two line segments a and b.
    :param as_ae:
    :param as_bs:
    :param as_be:
    :param ae_bs:
    :param ae_be:
    :param bs_be:
    :return:
    """
    mask = (((ae_bs - as_bs + as_ae).sgn().to(torch.int8) * (ae_be - as_be + as_ae).sgn().to(torch.int8) < 0) &
            ((as_bs - as_be + bs_be).sgn().to(torch.int8) * (ae_bs - ae_be + bs_be).sgn().to(torch.int8) < 0))
    return mask


def intersect_mask_core_db_int8(
        as_da: Tensor,
        as_bs: Tensor,
        as_be: Tensor,
        as_db: Tensor,
        ae_bs: Tensor,
        ae_be: Tensor,
        ae_db: Tensor,
        bs_db: Tensor
) -> Tensor:
    """
    This function computes the intersection mask between two line segments a and b.
    db = bs - be
    :param as_da: as x (ae - as)
    :param as_bs:
    :param as_be:
    :param as_db:
    :param ae_bs:
    :param ae_be:
    :param ae_db:
    :param bs_db:
    :return:
    """
    mask = (((ae_bs - as_bs + as_da).sgn().to(torch.int8) * (ae_be - as_be + as_da).sgn().to(torch.int8) < 0) &
            ((as_db - bs_db).sgn().to(torch.int8) * (ae_db - bs_db).sgn().to(torch.int8) < 0))
    return mask


def intersect_mask_core_db_int8_cat(
        as_da: Tensor,
        as_bs: Tensor,
        as_be: Tensor,
        as_db: Tensor,
        ae_bs: Tensor,
        ae_be: Tensor,
        ae_db: Tensor,
        bs_db: Tensor
) -> Tensor:
    """
    This function computes the intersection mask between two line segments a and b.
    db = bs - be
    :param as_da: as x (ae - as)
    :param as_bs:
    :param as_be:
    :param as_db:
    :param ae_bs:
    :param ae_be:
    :param ae_db:
    :param bs_db:
    :return:
    """
    cat_ae_b = torch.cat((ae_bs.unsqueeze(0), ae_be.unsqueeze(0)), dim=0)
    cat_as_b = torch.cat((as_bs.unsqueeze(0), as_be.unsqueeze(0)), dim=0)
    n0, n1 = (cat_ae_b - cat_as_b + as_da.unsqueeze(0)).sgn().to(torch.int8).chunk(2, dim=0)
    cat_a_db = torch.cat((as_db.unsqueeze(0), ae_db.unsqueeze(0)), dim=0)
    m0, m1 = (cat_a_db - bs_db.unsqueeze(0)).sgn().to(torch.int8).chunk(2, dim=0)
    mask = ((n0 * n1 < 0) & (m0 * m1 < 0))
    return mask


def intersect_mask_core_da_db_int8(
        as_da: Tensor,
        da_bs: Tensor,
        da_be: Tensor,
        as_db: Tensor,
        ae_db: Tensor,
        bs_db: Tensor
) -> Tensor:
    """
    This function computes the intersection mask between two line segments a and b.
    da = as - ae
    db = bs - be
    :param as_da:
    :param da_bs:
    :param da_be:
    :param as_db:
    :param ae_db:
    :param bs_db:
    :return:
    """
    mask = (((as_da + da_bs).sgn().to(torch.int8) * (as_da + da_be).sgn().to(torch.int8) < 0) &
            ((as_db - bs_db).sgn().to(torch.int8) * (ae_db - bs_db).sgn().to(torch.int8) < 0))
    return mask


def inner_mask_core(
        as_ae: Tensor,
        as_b: Tensor,
        ae_b: Tensor,
        as_y: Tensor,
        ae_y: Tensor,
        b_y: Tensor,
):
    """
    This function computes the inner intersection mask between two line segments a and b.
    :param as_ae:
    :param as_b:
    :param ae_b:
    :param as_y:
    :param ae_y:
    :param b_y:
    :return:
    """
    mask_y = (as_y - b_y).sgn() * (ae_y - b_y).sgn() < 0
    delta = (as_ae + ae_b - as_b).sgn() * (ae_y - as_y).sgn()
    mask_p = delta > 0
    mask_n = delta < 0
    return mask_y & mask_p, mask_y & mask_n


def inner_mask_core_da(
        as_ae: Tensor,
        da_b: Tensor,
        as_y: Tensor,
        ae_y: Tensor,
        b_y: Tensor,
):
    """
    This function computes the inner intersection mask between two line segments a and b.
    da = as - ae
    :param as_ae:
    :param da_b:
    :param as_y:
    :param ae_y:
    :param b_y:
    :return:
    """
    mask_y = (as_y - b_y).sgn() * (ae_y - b_y).sgn() < 0
    delta = (as_ae - da_b).sgn() * (ae_y - as_y).sgn()
    mask_p = delta > 0
    mask_n = delta < 0
    return mask_y & mask_p, mask_y & mask_n


def intersect_position_core_simple(
        a_st: Tensor,
        a_en: Tensor,
        b_st: Tensor,
        b_en: Tensor,
):
    """
        This function computes the intersection position of line sets a to b.
        :param a_st:
        :param a_en:
        :param b_st:
        :param b_en:
        :return:
        """
    da = a_en - a_st
    db = b_en - b_st
    d_bs_as = b_st - a_st
    pos = cross_product(d_bs_as, db) / cross_product(da, db)
    mask = (cross_product(d_bs_as, da) * cross_product(b_en - a_st, da) < 0) & (pos >= 0) & (pos <= 1)
    pos[~mask] = 0
    return pos


def intersect_position_core_decompose(
        as_da: Tensor,
        as_db: Tensor,
        as_bs: Tensor,
        as_be: Tensor,
        ae_db: Tensor,
        ae_bs: Tensor,
        ae_be: Tensor,
        bs_db: Tensor,
):
    """
    This function computes the intersection position of line sets a to b.
    :param as_da:
    :param as_db:
    :param as_bs:
    :param as_be:
    :param ae_db:
    :param ae_bs:
    :param ae_be:
    :param bs_db:
    :return:
    """
    pos = (bs_db - as_db) / (ae_db - as_db)
    mask = ((as_bs - ae_bs - as_da) * (as_be - ae_be - as_da) < 0) & (pos >= 0) & (pos <= 1)
    pos[~mask] = 0
    return pos


def get_st_end_points(
        polygon: Tensor,
        slite_position: list = None,
):

    """
    This function returns the start and end points of the polygon.
    :param polygon:
    :param slite_position: slite position of polygon
    :return:
    """
    n = 1
    if slite_position is not None:
        n = len(slite_position) - 1
    else:
        slite_position = [0, polygon.size(1)]

    if n > 1:
        mask = torch.ones(polygon.size(1), dtype=torch.bool)
        mask[slite_position[1:] - 1] = 0
        st = polygon[:, mask]
        mask.fill_(1)
        mask[slite_position[:-1]] = 0
        en = polygon[:, mask]
    else:
        st = polygon[:, :-1]
        en = polygon[:, 1:]
    return st, en, slite_position


def kernel_polygon_polygon_intersect_mask(
        polygon0: Tensor,
        polygon1: Tensor,
        slite_position0: list = None,
        slite_position1: list = None,
):
    """
    This function computes the intersection mask between two polygons.
    :param polygon0:
    :param polygon1:
    :param slite_position0: slite position of polygon0
    :param slite_position1: slite position of polygon1
    :return:
    """
    # cross product
    a_st, a_en, slite_position_a = get_st_end_points(polygon0, slite_position0)
    b_st, b_en, slite_position_b = get_st_end_points(polygon1, slite_position1)

    mask = torch.ones(polygon0.size(1), dtype=torch.bool)
    mask_as = mask.clone()
    mask_as[slite_position_a[1:] - 1] = 0
    mask_ae = mask.clone()
    mask_ae[slite_position_a[:-1]] = 0
    mask = torch.ones(polygon1.size(1), dtype=torch.bool)
    mask_bs = mask.clone()
    mask_bs[slite_position_b[1:] - 1] = 0
    mask_be = mask.clone()
    mask_be[slite_position_b[:-1]] = 0

    as_ae = cross_product(a_st.view(2, -1, 1), a_en.view(2, -1, 1))
    bs_be = cross_product(b_st.view(2, 1, -1), b_en.view(2, 1, -1))
    da_b = cross_product((a_st - a_en).view(2, -1, 1), polygon1.view(2, 1, -1))[:, mask_bs]
    a_db = cross_product(polygon0.view(2, -1, 1), (b_st - b_en).view(2, 1, -1))[mask_as, :]

    ab = cross_product(polygon0.view(2, -1, 1), polygon1.view(2, 1, -1))
    as_bs = ab[mask_as, :][:, mask_bs]
    as_be = ab[mask_as, :][:, mask_be]
    ae_bs = ab[mask_ae, :][:, mask_bs]
    ae_be = ab[mask_ae, :][:, mask_be]

    # check cross intersect
    cross_intersect_mask = intersect_mask_core_da_db_int8(as_ae, as_bs, as_be, ae_bs, ae_be, bs_be)
    na = len(slite_position_a) - 1
    nb = len(slite_position_b) - 1
    cross_mask_b = torch.zeros(a_st.size(1), nb, dtype=torch.bool, device=polygon0.device)
    for i in range(nb):
        st_i = slite_position_b[i] - i
        en_i = slite_position_b[i + 1] - (i + 1)
        cross_mask_b[:, i] = cross_intersect_mask[:, st_i:en_i].any(dim=1)

    cross_intersect_mask = torch.zeros(na, nb, dtype=torch.bool, device=polygon0.device)
    for i in range(na):
        st_i = slite_position_a[i] - i
        en_i = slite_position_a[i + 1] - (i + 1)
        cross_intersect_mask[i, :] = cross_mask_b[st_i:en_i, :].any(dim=0)

    # check inner intersect
    b_inner_a_mask_p, b_inner_a_mask_n = inner_mask_core_da(as_ae, da_b, a_st[1, :], a_en[1, :], b_st[1, :])
    a_inner_b_mask_p, a_inner_b_mask_n = inner_mask_core_da(bs_be, -a_db, b_st[1, :], b_en[1, :], a_st[1, :])
    inner_mask_b = torch.zeros(na, b_st.size(1), dtype=torch.bool, device=polygon0.device)
    for i in range(na):
        st_i = slite_position_a[i] - i
        en_i = slite_position_a[i + 1] - (i + 1)
        inner_mask_b[i, :] = ((b_inner_a_mask_p[st_i:en_i, :].sum(dim=0) % 2 == 1) |
                              (b_inner_a_mask_n[st_i:en_i, :].sum(dim=0) % 2 == 1))
    for i in range(nb):
        st_i = slite_position_b[i] - i
        en_i = slite_position_b[i + 1] - (i + 1)
        cross_intersect_mask[:, i] = cross_intersect_mask[:, i] | inner_mask_b[:, st_i:en_i].any(dim=1)

    inner_mask_a = torch.zeros(a_st.size(1), nb, dtype=torch.bool, device=polygon0.device)
    for i in range(nb):
        st_i = slite_position_b[i] - i
        en_i = slite_position_b[i + 1] - (i + 1)
        inner_mask_a[:, i] = ((a_inner_b_mask_p[:, st_i:en_i].sum(dim=1) % 2 == 1) |
                              (a_inner_b_mask_n[:, st_i:en_i].sum(dim=1) % 2 == 1))
    for i in range(na):
        st_i = slite_position_a[i] - i
        en_i = slite_position_a[i + 1] - (i + 1)
        cross_intersect_mask[i, :] = cross_intersect_mask[i, :] | inner_mask_a[st_i:en_i, :].any(dim=0)

    return cross_intersect_mask

