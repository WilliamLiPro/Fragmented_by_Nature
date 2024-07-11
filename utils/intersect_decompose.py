from tqdm import tqdm
import torch
from torch import Tensor
from utils import cross_product


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


@torch.no_grad()
def pp_m_polygon_intersect_length_decompose(
        m_polygon: Tensor,
        line_slit_position: Tensor,
        polygon_slit_position: Tensor,
        a_st: Tensor,
        a_en: Tensor = None,
        polygon_weights: Tensor = None,
        batch_size=64,
):
    """
    Calculate the length of point-pair lines intersect with multi-polygon.
    Accelerated with decomposition of points-pairs and multi_polygon.
    :param m_polygon:
    :param line_slit_position:
    :param polygon_slit_position:
    :param a_st:
    :param a_en:
    :param polygon_weights:
    :param batch_size:
    :return:
    """
    if a_en is None:
        a_en = a_st
    n_as = a_st.size(1)
    n_ae = a_en.size(1)
    n_polygon = polygon_slit_position.size(0) - 1
    if n_polygon == 0:
        return torch.zeros((n_as, n_ae), device=a_st.device, dtype=torch.float16)

    if polygon_weights is None:
        polygon_weights = torch.ones(n_polygon + 1, device=a_st.device, dtype=m_polygon.dtype)
    else:
        polygon_weights = polygon_weights.to(a_st.device)

    a_st = a_st.view(2, -1, 1, 1)
    a_en = a_en.view(2, 1, -1, 1)

    mask = torch.ones(m_polygon.size(1), device=a_st.device, dtype=torch.bool)
    mask_st = mask.clone()
    mask_st[line_slit_position[1:] - 1] = 0
    mask_en = mask.clone()
    mask_en[line_slit_position[:-1]] = 0

    polygon_slit_position = polygon_slit_position.clone()
    for i in range(1, line_slit_position.size(0)):
        polygon_slit_position[polygon_slit_position >= line_slit_position[i] - i] += -1

    m_polygon = m_polygon.view(2, 1, 1, -1)

    as_da = cross_product(a_st, a_en - a_st)
    as_b = cross_product(a_st, m_polygon)
    ae_b = cross_product(a_en, m_polygon)
    as_bs = as_b[..., mask_st]
    as_be = as_b[..., mask_en]
    ae_bs = ae_b[..., mask_st]
    ae_be = ae_b[..., mask_en]
    db = (m_polygon[..., 1:] - m_polygon[..., :-1])[..., mask_en[1:]]
    as_db = cross_product(a_st, db)
    ae_db = cross_product(a_en, db)
    bs_db = cross_product(m_polygon[..., mask_st], db)

    # batch split
    batch_split = [0]
    nb = bs_db.size(-1)
    i_st = 0
    for i in range((nb - 1) // batch_size + 1):
        i_end = min(i_st + batch_size, nb)
        p_en = torch.nonzero(polygon_slit_position >= i_end)[0] + 1
        i_end_f0 = polygon_slit_position[p_en - 2].item()
        i_end_f = polygon_slit_position[p_en - 1].item()
        if i_end_f - i_end < batch_size // 4:
            i_end = i_end_f
        elif i_end - i_end_f0 < batch_size // 4:
            i_end = i_end_f0
        batch_split.append(i_end)
        i_st = i_end
        if i_end >= nb:
            break

    # intersect count
    intersect_length = torch.zeros((n_as, n_ae), device=a_st.device, dtype=torch.float16)
    batch_use = None
    for i in tqdm(range(len(batch_split) - 1)):
        i_st = batch_split[i]
        i_end = batch_split[i + 1]

        as_bs_batch = as_bs[..., i_st: i_end]
        as_be_batch = as_be[..., i_st: i_end]
        ae_bs_batch = ae_bs[..., i_st: i_end]
        ae_be_batch = ae_be[..., i_st: i_end]
        as_db_batch = as_db[..., i_st: i_end]
        ae_db_batch = ae_db[..., i_st: i_end]
        bs_db_batch = bs_db[..., i_st: i_end]
        pos_batch = intersect_position_core_decompose(as_da, as_db_batch, as_bs_batch, as_be_batch, ae_db_batch, ae_bs_batch, ae_be_batch, bs_db_batch)

        p_st = torch.nonzero(polygon_slit_position <= i_st)[-1]
        p_en = torch.nonzero(polygon_slit_position >= i_end)[0] + 1
        s_p_batch = polygon_slit_position[p_st: p_en]
        s_p_batch = s_p_batch - s_p_batch[0]

        if batch_use is not None:
            # concat current batch with previous incomplete polygon
            batch_use = torch.cat([batch_use, pos_batch], dim=-1)
        else:
            batch_use = pos_batch
        batch_n = batch_use.size(-1)

        j_en = s_p_batch[-1]
        if j_en > batch_n:
            j_en = s_p_batch[-2]
        polygon_pos = batch_use[..., :j_en]
        polygon_pos = polygon_pos.sort(dim=-1, descending=True)[0]
        l_pos = j_en // 2 * 2
        polygon_pos = polygon_pos[..., :l_pos]
        intersect_length += (polygon_pos[..., 0::2] - polygon_pos[..., 1::2]).sum(-1)
        if j_en != batch_n:
            batch_use = batch_use[..., s_p_batch[-2]:]
        else:
            batch_use = None
    intersect_length[intersect_length > 1] = 1
    return intersect_length