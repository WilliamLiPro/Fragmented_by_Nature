import torch
from torch import Tensor


def cross_product(
        point1: Tensor,
        point2: Tensor,
):
    """
    cross product
    """
    return point1[0, ...] * point2[1, ...] - point1[1, ...] * point2[0, ...]


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
