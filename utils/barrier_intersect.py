import torch
from torch import Tensor
from tqdm import tqdm


def cross_product(
        vec1_x: Tensor,
        vec1_y: Tensor,
        vec2_x: Tensor,
        vec2_y: Tensor,
):
    """
    计算叉积
    :param vec1_x:
    :param vec1_y:
    :param vec2_x:
    :param vec2_y:
    :return:
    """
    return vec1_x * vec2_y - vec1_y * vec2_x


def intersect(
        points: Tensor,
        barrier_st: Tensor,
        barrier_ed: Tensor,
):
    """
    通过计算线段外积,分析任意点列中所有连接线段与barrier之间的相交关系

    方法：使用外积判断跨线，进而判断相交关系
    :param points: 点列，所有点之间相连
    :param barrier_st:
    :param barrier_ed:
    :return:
    """
    # nodes
    px = points[0, :]
    py = points[1, :]

    n_barrier = barrier_st.size()[1]
    n_point = points.size()[1]

    # 外积
    # a: barrier start
    # b: barrier end
    # c: node start
    # d: node end
    a_d = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, 1, n_point)
    a_c = a_d.view(n_barrier, n_point, 1)
    c_d = cross_product(px.view(-1, 1), py.view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(1, n_point, n_point)
    b_d = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, 1, n_point)
    b_c = b_d.view(n_barrier, n_point, 1)
    a_b = cross_product(barrier_st[0, :], barrier_st[1, :],
                        barrier_ed[0, :], barrier_ed[1, :], ).view(n_barrier, 1, 1)

    # intersect
    point_barrier = (a_d - a_c - c_d) * (b_d - b_c - c_d) < 0
    barrier_point = (a_d - b_d - a_b) * (a_c - b_c - a_b) < 0
    return point_barrier & barrier_point


def intersect_map(
        points: Tensor,
        barrier_st: Tensor,
        barrier_ed: Tensor,
        b_sz: int = 64,
):
    """
    通过计算线段外积,分析任意点列中所有连接线段与barrier之间的相交关系，
    进而判断任意连接是否有barrier

    方法：使用外积判断跨线，进而判断相交关系
    :param points: 点列，所有点之间相连
    :param barrier_st:
    :param barrier_ed:
    :param b_sz: block size
    :return:
    """
    print('Get mask from intersect')
    # nodes
    px = points[0, :]
    py = points[1, :]

    n_barrier = barrier_st.size()[1]
    n_point = points.size()[1]

    # 外积
    # a: barrier start
    # b: barrier end
    # c: node start
    # d: node end
    a_d = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, 1, n_point)
    a_c = a_d.view(n_barrier, n_point, 1)
    c_d = cross_product(px.view(-1, 1), py.view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(1, n_point, n_point)
    b_d = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, 1, n_point)
    b_c = b_d.view(n_barrier, n_point, 1)
    a_b = cross_product(barrier_st[0, :], barrier_st[1, :],
                        barrier_ed[0, :], barrier_ed[1, :], ).view(n_barrier, 1, 1)

    # block intersect and >0 intersect map
    output = None
    for i_st in tqdm(range(0, n_barrier, b_sz)):
        if i_st % (16 * b_sz) == 0:
            print(f'intersect_map iter: {i_st} / {n_barrier}')
        i_end = min(i_st + b_sz, n_barrier)
        # node 横跨 barrier
        point_barrier = (a_d[i_st: i_end, :, :] - a_c[i_st: i_end, :, :] - c_d).sgn().to(torch.int8) * \
                        (b_d[i_st: i_end, :, :] - b_c[i_st: i_end, :, :] - c_d).sgn().to(torch.int8) < 0
        # barrier 横跨 node
        barrier_point = (a_d[i_st: i_end, :, :] - b_d[i_st: i_end, :, :] -
                         a_b[i_st: i_end, :, :]).sgn().to(torch.int8) * \
                        (a_c[i_st: i_end, :, :] - b_c[i_st: i_end, :, :] -
                         a_b[i_st: i_end, :, :]).sgn().to(torch.int8) < 0
        if output is None:
            output = (point_barrier & barrier_point).any(dim=0)
        else:
            output = output | ((point_barrier & barrier_point).any(dim=0))
    print('intersect_map: finish')
    return output


def intersect_map_m_gpu(
        points: Tensor,
        barrier_st: Tensor,
        barrier_ed: Tensor,
        b_sz: int = 64,
        gpu_num: int = 4,
):
    """
    通过计算线段外积,分析任意点列中所有连接线段与barrier之间的相交关系，
    进而判断任意连接是否有barrier

    方法：使用外积判断跨线，进而判断相交关系
    :param points: 点列，所有点之间相连
    :param barrier_st:
    :param barrier_ed:
    :param b_sz: block size
    :param gpu_num: number of GPU
    :return:
    """
    print('Get mask from intersect')
    # nodes
    px = points[0, :]
    py = points[1, :]

    n_barrier = barrier_st.size()[1]
    n_point = points.size()[1]

    # 外积
    # a: barrier start
    # b: barrier end
    # c: node start
    # d: node end
    a_d = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, 1, n_point)
    a_c = a_d.view(n_barrier, n_point, 1)
    c_d = cross_product(px.view(-1, 1), py.view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(1, n_point, n_point)
    b_d = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, 1, n_point)
    b_c = b_d.view(n_barrier, n_point, 1)
    a_b = cross_product(barrier_st[0, :], barrier_st[1, :],
                        barrier_ed[0, :], barrier_ed[1, :], ).view(n_barrier, 1, 1)

    # multi GPU data
    a_ds = [None] * gpu_num
    a_cs = [None] * gpu_num
    c_ds = [None] * gpu_num
    b_ds = [None] * gpu_num
    b_cs = [None] * gpu_num
    a_bs = [None] * gpu_num
    n_point_b = (n_point + gpu_num - 1) // gpu_num
    for i in range(gpu_num):
        b_min = i * n_point_b
        b_max = min(b_min + n_point_b, n_point)
        a_ds[i] = a_d.cuda(i)
        a_cs[i] = a_c[:, b_min: b_max, :].cuda(i)
        c_ds[i] = c_d[:, b_min: b_max, :].cuda(i)
        b_ds[i] = b_d.cuda(i)
        b_cs[i] = b_c[:, b_min: b_max, :].cuda(i)
        a_bs[i] = a_b.cuda(i)
    a_c.detach()
    c_d.detach()
    b_c.detach()
    # block intersect and >0 intersect map
    output_bk = [None] * gpu_num
    for i_st in tqdm(range(0, n_barrier, b_sz)):
        if i_st % (16 * b_sz) == 0:
            print(f'intersect_map iter: {i_st} / {n_barrier}')
        i_end = min(i_st + b_sz, n_barrier)

        for gpu in range(gpu_num):
            # node 横跨 barrier
            point_barrier = (a_ds[gpu][i_st: i_end, :, :] - a_cs[gpu][i_st: i_end, :, :] - c_ds[gpu]).sgn().to(torch.int8) * \
                            (b_ds[gpu][i_st: i_end, :, :] - b_cs[gpu][i_st: i_end, :, :] - c_ds[gpu]).sgn().to(torch.int8) < 0
            # barrier 横跨 node
            barrier_point = (a_ds[gpu][i_st: i_end, :, :] - b_ds[gpu][i_st: i_end, :, :] -
                             a_bs[gpu][i_st: i_end, :, :]).sgn().to(torch.int8) * \
                            (a_cs[gpu][i_st: i_end, :, :] - b_cs[gpu][i_st: i_end, :, :] -
                             a_bs[gpu][i_st: i_end, :, :]).sgn().to(torch.int8) < 0
            if output_bk[gpu] is None:
                output_bk[gpu] = (point_barrier & barrier_point).any(dim=0)
            else:
                output_bk[gpu] = output_bk[gpu] | ((point_barrier & barrier_point).any(dim=0))
    for gpu in range(gpu_num):
        output_bk[gpu] = output_bk[gpu].to(points.device)
    torch.cuda.empty_cache()
    output = torch.cat(output_bk, dim=0)
    print('intersect_map: finish')
    return output


def intersect_map_add_road(
        original_points: Tensor,
        new_roads: Tensor,
        barrier_st: Tensor,
        barrier_ed: Tensor,
        b_sz: int = 64,
):
    """
    通过计算线段外积,分析任意点列中所有连接线段与barrier之间的相交关系，
    进而判断任意连接是否有barrier

    方法：使用外积判断跨线，进而判断相交关系
    :param original_points: 点列，所有点之间相连
    :param new_roads: nodes of new road
    :param barrier_st:
    :param barrier_ed:
    :param b_sz: block size
    :return:
    """
    print('Get mask from intersect')
    # nodes
    n_barrier = barrier_st.size()[1]
    n_point = original_points.size()[1]
    n_road = new_roads.size()[1]

    px = original_points[0, :]
    py = original_points[1, :]

    nr_x = new_roads[0, :]
    nr_y = new_roads[1, :]

    # 外积
    # a: barrier start
    # b: barrier end
    # c: original nodes
    # d: new road nodes
    a_d = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        nr_x.view(1, -1), nr_y.view(1, -1)).view(n_barrier, 1, n_road)
    a_c = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, n_point, 1)
    c_d = cross_product(px.view(-1, 1), py.view(-1, 1),
                        nr_x.view(1, -1), nr_y.view(1, -1)).view(1, n_point, n_road)
    b_d = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        nr_x.view(1, -1), nr_y.view(1, -1)).view(n_barrier, 1, n_road)
    b_c = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, n_point, 1)
    a_b = cross_product(barrier_st[0, :], barrier_st[1, :],
                        barrier_ed[0, :], barrier_ed[1, :], ).view(n_barrier, 1, 1)

    # block intersect and >0 intersect map
    output = None
    for i_st in tqdm(range(0, n_barrier, b_sz)):
        i_end = min(i_st + b_sz, n_barrier)

        point_barrier = (a_d[i_st: i_end, :, :] - a_c[i_st: i_end, :, :] - c_d).sgn().to(torch.int8) * \
                        (b_d[i_st: i_end, :, :] - b_c[i_st: i_end, :, :] - c_d).sgn().to(torch.int8) <= 0
        barrier_point = (a_d[i_st: i_end, :, :] - b_d[i_st: i_end, :, :] -
                         a_b[i_st: i_end, :, :]).sgn().to(torch.int8) * \
                        (a_c[i_st: i_end, :, :] - b_c[i_st: i_end, :, :] -
                         a_b[i_st: i_end, :, :]).sgn().to(torch.int8) < 0
        if output is None:
            output = (point_barrier & barrier_point).any(dim=0)
        else:
            output = output | ((point_barrier & barrier_point).any(dim=0))
    print('intersect_map: finish')
    return output


def intersect_map_points_road_node_m_gpu(
        points: Tensor,
        road_nodes: Tensor,
        barrier_st: Tensor,
        barrier_ed: Tensor,
        b_sz: int = 64,
        gpu_num: int = 4,
):
    """
    通过计算线段外积,分析任意点列中所有连接线段与barrier之间的相交关系，
    进而判断任意连接是否有barrier

    方法：使用外积判断跨线，进而判断相交关系
    :param points: 点列，所有点之间相连
    :param road_nodes: nodes of road net
    :param barrier_st:
    :param barrier_ed:
    :param b_sz: block size
    :param gpu_num: number of GPU
    :return:
    """
    print('Get mask from intersect')
    # nodes
    px = points[0, :]
    py = points[1, :]

    rx = road_nodes[0, :]
    ry = road_nodes[1, :]

    n_barrier = barrier_st.size()[1]
    n_point = points.size()[1]
    n_road = road_nodes.size()[1]

    # 外积
    # a: barrier start
    # b: barrier end
    # c: original nodes
    # d: road nodes
    a_d = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        rx.view(1, -1), ry.view(1, -1)).view(n_barrier, 1, n_road)
    a_c = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, n_point, 1)
    c_d = cross_product(px.view(-1, 1), py.view(-1, 1),
                        rx.view(1, -1), ry.view(1, -1)).view(1, n_point, n_road)
    b_d = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        rx.view(1, -1), ry.view(1, -1)).view(n_barrier, 1, n_road)
    b_c = cross_product(barrier_ed[0, :].view(-1, 1), barrier_ed[1, :].view(-1, 1),
                        px.view(1, -1), py.view(1, -1)).view(n_barrier, n_point, 1)
    a_b = cross_product(barrier_st[0, :], barrier_st[1, :],
                        barrier_ed[0, :], barrier_ed[1, :], ).view(n_barrier, 1, 1)

    # multi GPU data
    a_ds = [None] * gpu_num
    a_cs = [None] * gpu_num
    c_ds = [None] * gpu_num
    b_ds = [None] * gpu_num
    b_cs = [None] * gpu_num
    a_bs = [None] * gpu_num
    n_point_b = (n_point + gpu_num - 1) // gpu_num
    for i in range(gpu_num):
        b_min = i * n_point_b
        b_max = min(b_min + n_point_b, n_point)
        a_ds[i] = a_d.cuda(i)
        a_cs[i] = a_c[:, b_min: b_max, :].cuda(i)
        c_ds[i] = c_d[:, b_min: b_max, :].cuda(i)
        b_ds[i] = b_d.cuda(i)
        b_cs[i] = b_c[:, b_min: b_max, :].cuda(i)
        a_bs[i] = a_b.cuda(i)
    a_c.detach()
    c_d.detach()
    b_c.detach()
    # block intersect and >0 intersect map
    output_bk = [None] * gpu_num
    for i_st in tqdm(range(0, n_barrier, b_sz)):
        if i_st % (16 * b_sz) == 0:
            print(f'intersect_map iter: {i_st} / {n_barrier}')
        i_end = min(i_st + b_sz, n_barrier)

        for gpu in range(gpu_num):
            # node 横跨 barrier
            point_barrier = (a_ds[gpu][i_st: i_end, :, :] - a_cs[gpu][i_st: i_end, :, :] - c_ds[gpu]).sgn().to(torch.int8) * \
                            (b_ds[gpu][i_st: i_end, :, :] - b_cs[gpu][i_st: i_end, :, :] - c_ds[gpu]).sgn().to(torch.int8) < 0
            # barrier 横跨 node
            barrier_point = (a_ds[gpu][i_st: i_end, :, :] - b_ds[gpu][i_st: i_end, :, :] -
                             a_bs[gpu][i_st: i_end, :, :]).sgn().to(torch.int8) * \
                            (a_cs[gpu][i_st: i_end, :, :] - b_cs[gpu][i_st: i_end, :, :] -
                             a_bs[gpu][i_st: i_end, :, :]).sgn().to(torch.int8) < 0
            if output_bk[gpu] is None:
                output_bk[gpu] = (point_barrier & barrier_point).any(dim=0)
            else:
                output_bk[gpu] = output_bk[gpu] | ((point_barrier & barrier_point).any(dim=0))
    for gpu in range(gpu_num):
        output_bk[gpu] = output_bk[gpu].to(points.device)
    torch.cuda.empty_cache()
    output = torch.cat(output_bk, dim=0)
    print('intersect_map: finish')
    return output


def matrix_intersect(
        points_x: Tensor,
        points_y: Tensor,
        barrier_st: Tensor,
        barrier_ed: Tensor,
):
    """
    通过计算线段外积,分析方形点阵中所有线段与barrier之间的相交关系

    方法：使用外积判断跨线，进而判断相交关系
    :param points_x: 点阵的x坐标，所有点之间相连
    :param points_y: 点阵的y坐标，所有点之间相连
    :param barrier_st:
    :param barrier_ed:
    :return:
    """
    n_barrier = barrier_st.size()[1]
    n_point_x = points_x.size()[0]
    n_point_y = points_y.size()[0]
    n_point = n_point_x * n_point_y

    # 外积
    # a: barrier start
    # b: barrier end
    # c: node start
    # d: node end
    a_d = cross_product(barrier_st[0, :].view(-1, 1, 1), barrier_st[1, :].view(-1, 1, 1),
                        points_x.view(1, -1, 1), points_y.view(1, 1, -1)).view(n_barrier, 1, n_point)
    a_c = a_d.view(n_barrier, n_point, 1)
    c_d = cross_product(points_x.view(-1, 1, 1, 1), points_y.view(1, -1, 1, 1),
                        points_x.view(1, 1, -1, 1), points_y.view(1, 1, 1, -1)).view(1, n_point, n_point)
    b_d = cross_product(barrier_ed[0, :].view(-1, 1, 1), barrier_ed[1, :].view(-1, 1, 1),
                        points_x.view(1, -1, 1), points_y.view(1, 1, -1)).view(n_barrier, 1, n_point)
    b_c = b_d.view(n_barrier, n_point, 1)
    a_b = cross_product(barrier_st[0, :], barrier_st[1, :],
                        barrier_ed[0, :], barrier_ed[1, :], ).view(n_barrier, 1, 1)

    # intersect
    point_barrier = (a_d - a_c - c_d) * (b_d - b_c - c_d) < 0
    barrier_point = (a_d - b_d - a_b) * (a_c - b_c - a_b) < 0
    return point_barrier & barrier_point


def matrix_point_intersect(
        points_x: Tensor,
        points_y: Tensor,
        v_point: Tensor,
        barrier_st: Tensor,
        barrier_ed: Tensor,
):
    """
    通过计算线段外积,分析方形点阵 matrix 为起点, 途经点列 v_point 为终点的
    所有线段与 barrier 之间的相交关系

    方法：使用外积判断跨线，进而判断相交关系
    :param points_x: 点阵的x坐标
    :param points_y: 点阵的y坐标
    :param v_point: 途经点列，
    :param barrier_st:
    :param barrier_ed:
    :return:
    """
    n_barrier = barrier_st.size()[1]
    n_point_x = points_x.size()[0]
    n_point_y = points_y.size()[0]
    n_point = n_point_x * n_point_y
    nv = v_point.size()[1]

    # 外积
    # a: barrier start
    # b: barrier end
    # c: matrix point
    # d: via point
    a_d = cross_product(barrier_st[0, :].view(-1, 1), barrier_st[1, :].view(-1, 1),
                        v_point[0].view(1, -1), v_point[1].view(1, -1)).view(n_barrier, 1, nv)
    a_c = cross_product(barrier_st[0, :].view(-1, 1, 1), barrier_st[1, :].view(-1, 1, 1),
                        points_x.view(1, -1, 1), points_y.view(1, 1, -1)).view(n_barrier, n_point, 1)
    c_d = cross_product(points_x.view(1, -1, 1), points_y.view(1, 1, -1),
                        v_point[0].view(-1, 1, 1), v_point[1].view(-1, 1, 1),).view(1, n_point, nv)
    b_d = cross_product(barrier_ed[0, :].view(-1, 1, 1), barrier_ed[1, :].view(-1, 1, 1),
                        v_point[0].view(1, -1), v_point[1].view(1, -1)).view(n_barrier, 1, nv)
    b_c = cross_product(barrier_ed[0, :].view(-1, 1, 1), barrier_ed[1, :].view(-1, 1, 1),
                        points_x.view(1, -1, 1), points_y.view(1, 1, -1)).view(n_barrier, n_point, 1)
    a_b = cross_product(barrier_st[0, :], barrier_st[1, :],
                        barrier_ed[0, :], barrier_ed[1, :], ).view(n_barrier, 1, 1)

    # intersect
    point_barrier = (a_d - a_c - c_d) * (b_d - b_c - c_d) < 0
    barrier_point = (a_d - b_d - a_b) * (a_c - b_c - a_b) < 0
    return point_barrier & barrier_point


def sub_polygon_intersect(
        ps_pe: Tensor,
        ps_bs: Tensor,
        ps_be: Tensor,
        pe_bs: Tensor,
        pe_be: Tensor,
        bs_be: Tensor,
        b_sz=64,
):
    """
    Intersect of point-pair lines with single polygons,
    the output is intersect_mask, which shows the mask about whether
    the ns_ne intersects with polygons
    """
    # bs: barrier start
    # be: barrier end
    # ps: point start
    # pe: point end
    n_barrier = bs_be.size(-1)
    intersect_mask = torch.zeros(ps_pe.size()[:2], device=ps_pe.device, dtype=torch.bool)
    for i_st in (range(0, n_barrier, b_sz)):
        i_end = min(i_st + b_sz, n_barrier)
        # node lines across barrier boundary
        point_line_across_barrier = (
                (pe_bs[:, :, i_st: i_end] + ps_pe -
                 ps_bs[:, :, i_st: i_end]).sgn().to(torch.int8) *
                (pe_be[:, :, i_st: i_end] + ps_pe -
                 ps_be[:, :, i_st: i_end]).sgn().to(torch.int8) < 0)
        # barrier boundary across node lines
        barrier_across_point_line = (
                (-ps_be[:, :, i_st: i_end] + bs_be[:, :, i_st: i_end] +
                 ps_bs[:, :, i_st: i_end]).sgn().to(torch.int8) *
                (-pe_be[:, :, i_st: i_end] + bs_be[:, :, i_st: i_end] +
                 pe_bs[:, :, i_st: i_end]).sgn().to(torch.int8) < 0)
        intersect_mask = intersect_mask | ((barrier_across_point_line &
                                            point_line_across_barrier).any(dim=-1))
    return intersect_mask


def sub_polygon_intersect_length(
        ps_pe: Tensor,
        ps_bs: Tensor,
        ps_be: Tensor,
        pe_bs: Tensor,
        pe_be: Tensor,
        bs_be: Tensor,
        b_sz=64,
):
    """
    Intersect of point-pair lines with single polygons,
    the output is intersect_mask, which shows the mask about whether
    the ns_ne intersects with polygons
    """
    # bs: barrier start
    # be: barrier end
    # ps: point start
    # pe: point end
    n_barrier = bs_be.size(-1)
    n_ps = ps_bs.size(0)
    n_pe = pe_bs.size(1)
    # 1. obtain the line intersecting with polygon
    intersect_mask = sub_polygon_intersect(ps_pe, ps_bs, ps_be, pe_bs, pe_be, bs_be, b_sz)

    # 2. get relative length of intersected lines
    ps_pe_masked = ps_pe[intersect_mask, :]
    relative_length_intersected = torch.zeros((intersect_mask.sum(), n_barrier),
                                              device=intersect_mask.device, dtype=torch.float16)
    for i_st in (range(0, n_barrier, b_sz)):
        i_end = min(i_st + b_sz, n_barrier)
        # intersected
        point_line_across_barrier = (
                ((pe_bs[:, :, i_st: i_end] - ps_bs[:, :, i_st: i_end])[intersect_mask, :]
                 + ps_pe_masked).sgn().to(torch.int8) *
                ((pe_be[:, :, i_st: i_end] - ps_be[:, :, i_st: i_end])[intersect_mask, :]
                 + ps_pe_masked).sgn().to(torch.int8) <= 0)
        barrier_across_point_line = (
                ((-ps_be[:, :, i_st: i_end] + bs_be[:, :, i_st: i_end] +
                  ps_bs[:, :, i_st: i_end]).sgn().to(torch.int8) *
                 (-pe_be[:, :, i_st: i_end] + bs_be[:, :, i_st: i_end] +
                  pe_bs[:, :, i_st: i_end]).sgn().to(torch.int8))[intersect_mask, :] <= 0)
        local_mask = (barrier_across_point_line & point_line_across_barrier)
        # length
        numerator = bs_be[:, :, i_st: i_end] - ps_be[:, :, i_st: i_end] + ps_bs[:, :, i_st: i_end]
        numerator = numerator.expand(n_ps, n_pe, i_end - i_st)[intersect_mask, :]
        denominator = ((pe_be[:, :, i_st: i_end] - pe_bs[:, :, i_st: i_end]) +
                       (ps_bs[:, :, i_st: i_end] - ps_be[:, :, i_st: i_end]))[intersect_mask, :]
        t = numerator / denominator
        t[~local_mask] = 0
        relative_length_intersected[:, i_st: i_end] = t
    return relative_length_intersected, intersect_mask


def tensor_polygon_intersect(
        points: Tensor,
        polygon_list: (Tensor, ...),
        b_sz: int = 64,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
):
    """
    Intersect of point-pair lines with polygons, the output is relative length of line in polygons
    :param points: the traffic points
    :param polygon_list: the list of boundaries for each polygon
    :param b_sz: block size
    :param device: device of calculation
    :return: relative length of each line intersecting polygon
    """
    n_polygon = len(polygon_list)
    points = points.to(device)
    points_x = points[0, :]
    points_y = points[1, :]

    n_point = points_x.size()[0]

    # cumulate relative length of line in each polygon
    # bs: barrier start
    # be: barrier end
    # ns: road point
    # ne: commuting point
    ns_ne = cross_product(points_x.view(-1, 1), points_y.view(-1, 1),
                          points_x.view(1, -1), points_y.view(1, -1)).view(n_point, n_point, 1)
    relative_length = torch.zeros((n_point, n_point), device=device, dtype=torch.float32)
    for polygon in tqdm(polygon_list):
        # block intersect and intersect length
        if isinstance(polygon, list):
            ns_bs = []
            ns_be = []
            bs_be = []
            n_barrier = 0
            # combine them together
            for sub_polygon in polygon:
                sub_polygon = sub_polygon.to(device=device)
                boundary_x = sub_polygon[0, :]
                boundary_y = sub_polygon[1, :]
                sub_n_barrier = boundary_x.size(0) - 1
                n_barrier += sub_n_barrier

                ns_b = cross_product(points_x.view(-1, 1), points_y.view(-1, 1),
                                     boundary_x.view(1, -1), boundary_y.view(1, -1))
                ns_bs.append(ns_b[:, :-1].view(n_point, 1, sub_n_barrier))
                ns_be.append(ns_b[:, 1:].view(n_point, 1, sub_n_barrier))
                bs_be.append(cross_product(boundary_x[:-1], boundary_y[:-1],
                                           boundary_x[1:], boundary_y[1:], ).view(1, 1, sub_n_barrier))
            if len(ns_bs) == 0 or len(ns_be) == 0:
                continue
            ns_bs = torch.cat(ns_bs, dim=-1)
            ns_be = torch.cat(ns_be, dim=-1)
            bs_be = torch.cat(bs_be, dim=-1)
            ne_bs = ns_bs.view(1, n_point, n_barrier)
            ne_be = ns_be.view(1, n_point, n_barrier)
        else:
            polygon = polygon.to(device=device)
            boundary_x = polygon[0, :]
            boundary_y = polygon[1, :]
            sub_n_barrier = boundary_x.size(0) - 1

            # cross product
            # bs: barrier start
            # be: barrier end
            # ns: node start
            # ne: node end
            ns_b = cross_product(points_x.view(-1, 1), points_y.view(-1, 1),
                                 boundary_x.view(1, -1), boundary_y.view(1, -1))
            ns_bs = ns_b[:, :-1].view(n_point, 1, sub_n_barrier)
            ns_be = ns_b[:, 1:].view(n_point, 1, sub_n_barrier)
            ne_bs = ns_bs.view(1, n_point, sub_n_barrier)
            ne_be = ns_be.view(1, n_point, sub_n_barrier)
            bs_be = cross_product(boundary_x[:-1], boundary_y[:-1],
                                  boundary_x[1:], boundary_y[1:], ).view(1, 1, sub_n_barrier)

        # get relative length of intersected lines
        relative_length_intersected, intersect_mask = sub_polygon_intersect_length(
                ns_ne, ns_bs, ns_be, ne_bs, ne_be, bs_be, b_sz)

        # sort and calculate
        if relative_length_intersected.numel() > 256 * 1024 ** 2:
            for i_st in (range(0, relative_length_intersected.size(0), 2048)):
                i_end = min(i_st + 2048, relative_length_intersected.size(0))
                relative_length_intersected[i_st: i_end, :], _ = relative_length_intersected[i_st: i_end, :].sort(dim=-1, descending=True)
        else:
            relative_length_intersected, _ = relative_length_intersected.sort(dim=-1, descending=True)
        non_zero = relative_length_intersected.sum(0) > 0
        if non_zero.sum() == 0:
            continue
        relative_length_intersected = relative_length_intersected[:, non_zero]
        relative_length_intersected = relative_length_intersected.to(relative_length.dtype)
        relative_length[intersect_mask] += (relative_length_intersected[:, 0::2].sum(-1) -
                                            relative_length_intersected[:, 1::2].sum(-1))

        # correct the relative length
        mask_wrong = relative_length > relative_length.T
        relative_length[mask_wrong] = relative_length.T[mask_wrong]
    relative_length = relative_length.cpu()
    torch.cuda.empty_cache()
    return relative_length


def tensor_road_intersect_polygon(
        c_points: Tensor,
        road_points: Tensor,
        polygon_list: (Tensor, ...),
        b_sz: int = 64,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
):
    """
    Intersect of point-road lines with polygons, the output is relative length of line in polygons
        :param c_points: the commuting points
        :param road_points: the node of road net
        :param polygon_list: the list of boundaries for each polygon
        :param b_sz: block size
        :param device: device of calculation
        :return: relative length of each line intersecting polygon
    """
    n_polygon = len(polygon_list)
    n_c = c_points.size()[1]
    n_r = road_points.size()[1]
    c_points = c_points.to(device)
    road_points = road_points.to(device)
    c_points_x = c_points[0, :]
    c_points_y = c_points[1, :]
    r_points_x = road_points[0, :]
    r_points_y = road_points[1, :]

    # cumulate intersect of each polygon
    # bs: barrier start
    # be: barrier end
    # ns: road point
    # ne: commuting point
    ns_ne = cross_product(c_points_x.view(-1, 1), c_points_y.view(-1, 1),
                          r_points_x.view(1, -1), r_points_y.view(1, -1)).view(n_c, n_r, 1)
    intersect_mask = torch.zeros((n_c, n_r), device=device, dtype=torch.bool)
    for polygon in tqdm(polygon_list):
        # block intersect
        if isinstance(polygon, list):
            ns_bs = []
            ns_be = []
            ne_bs = []
            ne_be = []
            bs_be = []
            n_barrier = 0
            # combine them together
            for sub_polygon in polygon:
                sub_polygon = sub_polygon.to(device=device)
                boundary_x = sub_polygon[0, :]
                boundary_y = sub_polygon[1, :]
                sub_n_barrier = boundary_x.size(0) - 1
                n_barrier += sub_n_barrier

                ns_b = cross_product(c_points_x.view(-1, 1), c_points_y.view(-1, 1),
                                     boundary_x.view(1, -1), boundary_y.view(1, -1))
                ns_bs.append(ns_b[:, :-1].view(n_c, 1, sub_n_barrier))
                ns_be.append(ns_b[:, 1:].view(n_c, 1, sub_n_barrier))
                ne_b = cross_product(r_points_x.view(-1, 1), r_points_y.view(-1, 1),
                                     boundary_x.view(1, -1), boundary_y.view(1, -1))
                ne_bs.append(ne_b[:, :-1].view(1, n_r, sub_n_barrier))
                ne_be.append(ne_b[:, 1:].view(1, n_r, sub_n_barrier))
                bs_be.append(cross_product(boundary_x[:-1], boundary_y[:-1],
                                           boundary_x[1:], boundary_y[1:], ).view(1, 1, sub_n_barrier))
            ns_bs = torch.cat(ns_bs, dim=-1)
            ns_be = torch.cat(ns_be, dim=-1)
            ne_bs = torch.cat(ne_bs, dim=-1)
            ne_be = torch.cat(ne_be, dim=-1)
            bs_be = torch.cat(bs_be, dim=-1)
        else:
            polygon = polygon.to(device=device)
            boundary_x = polygon[0, :]
            boundary_y = polygon[1, :]
            sub_n_barrier = boundary_x.size(0) - 1

            # cross product
            # bs: barrier start
            # be: barrier end
            # ns: node start
            # ne: node end
            ns_b = cross_product(c_points_x.view(-1, 1), c_points_y.view(-1, 1),
                                 boundary_x.view(1, -1), boundary_y.view(1, -1))
            ns_bs = ns_b[:, :-1].view(n_c, 1, sub_n_barrier)
            ns_be = ns_b[:, 1:].view(n_c, 1, sub_n_barrier)
            ne_b = cross_product(r_points_x.view(-1, 1), r_points_y.view(-1, 1),
                                 boundary_x.view(1, -1), boundary_y.view(1, -1))
            ne_bs = ne_b[:, :-1].view(1, n_r, sub_n_barrier)
            ne_be = ne_b[:, 1:].view(1, n_r, sub_n_barrier)
            bs_be = cross_product(boundary_x[:-1], boundary_y[:-1],
                                       boundary_x[1:], boundary_y[1:], ).view(1, 1, sub_n_barrier)
        intersect_mask = intersect_mask | sub_polygon_intersect(ns_ne, ns_bs, ns_be, ne_bs, ne_be, bs_be, b_sz)
    return intersect_mask.cpu()


def test_intersect():
    # test 1
    print('test intersect')
    print('test 1')
    points = Tensor([[0, 1, 2], [0, 1, 2]])
    barrier_st = Tensor([[0, 0, 2], [0, 0, 0]])
    barrier_ed = Tensor([[2, 0, 0], [0, 2, 2]])
    rr = intersect(points, barrier_st, barrier_ed)

    g_t = torch.zeros_like(rr)
    g_t[2, 0, 2] = True
    g_t[2, 2, 0] = True
    if (rr == g_t).sum() == 27:
        print('test 1 pass')
    else:
        print('test 1 fail')
    print(rr)


def test_matrix_intersect():
    # test 1
    print('test matrix_intersect')
    print('test 1')
    points_x = Tensor([0, 1, 2])
    points_y = Tensor([0, 1, 2])
    barrier_st = Tensor([[0, 0, 2], [0, 0, 0]])
    barrier_ed = Tensor([[2, 0, 0], [0, 2, 2]])
    rr = matrix_intersect(points_x, points_y, barrier_st, barrier_ed)

    g_t = torch.zeros_like(rr)
    g_t[2, 0, 5] = g_t[2, 0, 7] = g_t[2, 0, 8] = True
    g_t[2, 1, 5] = g_t[2, 1, 7] = g_t[2, 1, 8] = True
    g_t[2, 3, 5] = g_t[2, 3, 7] = g_t[2, 3, 8] = True
    g_t[2, :, :] = g_t[2, :, :] + g_t[2, :, :].t()
    if (rr == g_t).sum() == rr.numel():
        print('test 1 pass')
    else:
        print('test 1 fail')
    print(rr)


def test_matrix_point_intersect():
    # test 1
    print('test matrix_point_intersect')
    print('test 1')
    v_points = Tensor([[0, 1], [0, 1]])
    points_x = Tensor([0, 2])
    points_y = Tensor([0, 2])
    barrier_st = Tensor([[0, 0, 2], [0, 0, 0]])
    barrier_ed = Tensor([[2, 0, 0], [0, 2, 2]])
    rr = matrix_point_intersect(points_x, points_y, v_points, barrier_st, barrier_ed)

    g_t = torch.zeros_like(rr)
    g_t[2, 3, 0] = True
    if (rr == g_t).sum() == rr.numel():
        print('test 1 pass')
    else:
        print('test 1 fail')
    print(rr)


def test_tensor_polygon_intersect():
    # test 1
    print('test tensor_polygon_intersect')
    points = Tensor([[-1, -1, 1, 1], [-1, 1, -1, 1]])
    polygon_list = (Tensor([[0, 0.5, 0.5, 0, 0], [-1.1, -1.1, 1.1, 1.1, -1.1]]),
                    Tensor([[-1.2, -0.5, -0.5, -1.2, -1.2], [0, 0, 0.4, 0.4, 0]]))
    rl = tensor_polygon_intersect(points, polygon_list,)
    print(f'relative length = {rl}')

    import matplotlib.pyplot as plt
    plt.plot(points[0, :].numpy(), points[1, :].numpy(), 'o')
    for polygon in polygon_list:
        plt.plot(polygon[0, :].numpy(), polygon[1, :].numpy())
    plt.show()
    print('test pass')


if __name__ == '__main__':
    # test_intersect()
    # test_matrix_intersect()
    # test_matrix_point_intersect()
    test_tensor_polygon_intersect()
