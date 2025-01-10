import torch
from torch import Tensor
from torch.nn import ParameterList


class PointGroups:
    def __init__(
            self,
            points: Tensor,
            slit_position: list,    # slit position of each group
            indices: Tensor = None  # position of points in original order
    ):
        self.n = points.size(1)
        self.points = points
        self.indices = indices if indices is not None else (
            torch.arange(self.n, device=points.device, dtype=torch.int64))
        self.inverse_indices = torch.empty_like(self.indices)
        self.inverse_indices[self.indices] = torch.arange(self.n, device=points.device, dtype=self.indices.dtype)
        if slit_position[0] != 0:
            slit_position = [0] + slit_position
        if slit_position[-1] != self.n:
            slit_position += [self.n]
        self.slit_position = slit_position
        self.set_bounds = self.bounds()
    
    def __str__(self):
        return f"LineSeg: {(self.points, self.slit_position,)}"

    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.slit_position) - 1
    
    def __getitem__(self, key):
        st = self.slit_position[key]
        ed = self.slit_position[key + 1]
        return self.points[:, st:ed], self.slit_position[key:key + 2]
    
    def bounds(self):
        bound_list = []
        for i, st in enumerate(self.slit_position[:-1]):
            ed = self.slit_position[i + 1]
            points = self.points[:, st: ed]
            bound_list.append([points[0, :].min().item(), points[1, :].min().item(),
                               points[0, :].max().item(), points[1, :].max().item()])
        self.set_bounds = Tensor(bound_list).to(self.points.device).to(self.points.dtype)
        return self.set_bounds
        
    def to(self, *args, **kwargs):
        return PointGroups(self.points.to(*args, **kwargs), self.slit_position, self.indices)

    def cuda(self, *args, **kwargs):
        return PointGroups(self.points.cuda(*args, **kwargs), self.slit_position, self.indices)

    def cpu(self, *args, **kwargs):
        return PointGroups(self.points.cpu(*args, **kwargs), self.slit_position, self.indices)
    
    
class LineSeg:
    """
    The line segment for GPU calculation.

    :param seg_st  the start vertex of line seg.

    :param seg_end  the end vertex of line seg.
    """
    def __init__(self, seg_st: Tensor, seg_end: Tensor):
        self.n = seg_st.size(1)
        assert seg_st.size(0) == seg_end.size(0)
        assert seg_st.size(1) == seg_end.size(1)
        self.seg_st = seg_st
        self.seg_end = seg_end

    def __str__(self):
        return f"LineSeg: {(self.seg_st, self.seg_end,)}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        return self.seg_st[key], self.seg_end[key]

    def __setitem__(self, key, value_st: Tensor = None, value_end: Tensor = None):
        if value_st is not None:
            self.seg_st[key] = value_st
        if value_end is not None:
            self.seg_end[key] = value_end

    def to(self, *args, **kwargs):
        return LineSeg(self.seg_st.to(*args, **kwargs), self.seg_end.to(*args, **kwargs))

    def cuda(self, *args, **kwargs):
        return LineSeg(self.seg_st.cuda(*args, **kwargs), self.seg_end.cuda(*args, **kwargs))

    def cpu(self, *args, **kwargs):
        return LineSeg(self.seg_st.cpu(*args, **kwargs), self.seg_end.cpu(*args, **kwargs))


class LineSegGroups:
    def __init__(self, seg_st: Tensor, seg_end: Tensor, slit_position: list, indices: Tensor = None):
        self.n = seg_st.size(1)
        assert seg_st.size(1) == seg_end.size(1)
        self.seg_st = seg_st
        self.seg_end = seg_end
        self.indices = indices if indices is not None else (
            torch.arange(self.n, device=seg_st.device, dtype=torch.int64))
        self.inverse_indices = torch.empty_like(self.indices)
        self.inverse_indices[self.indices] = torch.arange(self.n, device=seg_st.device, dtype=self.indices.dtype)
        if slit_position[0] != 0:
            slit_position = [0] + slit_position
        if slit_position[-1] != self.n:
            slit_position += [self.n]
        self.slit_position = slit_position
        self.set_bounds = self.bounds()

    def __str__(self):
        return f"LineSeg: {(self.seg_st, self.seg_end, self.slit_position,)}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.slit_position) - 1

    def __getitem__(self, key):
        st = self.slit_position[key]
        ed = self.slit_position[key + 1]
        return self.seg_st[:, st:ed], self.seg_end[:, st:ed], self.slit_position[key:key + 2]

    def bounds(self):
        bound_list = []
        for i, st in enumerate(self.slit_position[:-1]):
            ed = self.slit_position[i + 1]
            points = torch.cat([self.seg_st[:, st: ed], self.seg_end[:, st: ed]], dim=-1)
            bound_list.append((points[0, :].min(), points[1, :].min(), points[0, :].max(), points[1, :].max()))
        self.set_bounds = Tensor(bound_list).to(self.seg_st.device).to(self.seg_st.dtype)
        return self.set_bounds

    def to(self, *args, **kwargs):
        return LineSegGroups(self.seg_st.to(*args, **kwargs), self.seg_end.to(*args, **kwargs),
                             self.slit_position, self.indices)

    def cuda(self, *args, **kwargs):
        return LineSegGroups(self.seg_st.cuda(*args, **kwargs), self.seg_end.cuda(*args, **kwargs),
                             self.slit_position, self.indices)

    def cpu(self, *args, **kwargs):
        return LineSegGroups(self.seg_st.cpu(*args, **kwargs), self.seg_end.cpu(*args, **kwargs),
                             self.slit_position, self.indices)


class MultiPolygon:
    """
    The multi-polygon for GPU calculation.

    Each of polygon is represented by list of Tensors,
    where each Tensor is a coordinate of ring.

    The first Tensor in list is outline boundary of the polygon,
    while the others are the boundaries of internal cavity area.
    """
    def __init__(self,
                 multi_lines: Tensor,
                 polygon_slit_position: list,  # slit position of each Polygon
                 linestring_slit_position: list,  # slit position of each LineString
                 ):
        self.n = multi_lines.size(1)
        self.multi_lines = multi_lines
        if linestring_slit_position[0] != 0:
            linestring_slit_position = [0] + linestring_slit_position
        if linestring_slit_position[-1] != self.n:
            linestring_slit_position += [self.n]
        self.linestring_n = len(linestring_slit_position) - 1
        if polygon_slit_position[0] != 0:
            polygon_slit_position = [0] + linestring_slit_position
        if polygon_slit_position[-1] != self.linestring_n + 1:
            polygon_slit_position += [self.linestring_n + 1]
        self.polygon_n = len(polygon_slit_position) - 1
        self.polygon_slit_position = polygon_slit_position
        self.linestring_slit_position = linestring_slit_position

    def __str__(self):
        message = ''
        for i in range(self.polygon_n):
            idx_r = self.linestring_slit_position[self.polygon_slit_position[i]: self.polygon_slit_position[i + 1]]
            message += f"Polygon {i}: (\n"
            for j, st in enumerate(idx_r):
                ed = self.linestring_slit_position[i + 1]
                message += f"LineString: {self.multi_lines[st:ed]}, \n"
            message += f")\n"
        return message

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.polygon_slit_position) - 1

    def __getitem__(self, key):
        st = self.line_position[key]
        ed = self.line_position[key + 1]
        return self.multi_lines[:, st:ed], self.slit_position[key:key + 2]

    def to(self, *args, **kwargs):
        return MultiPolygon(self.multi_lines.to(*args, **kwargs),
                            self.polygon_slit_position, self.linestring_slit_position)

    def cuda(self, *args, **kwargs):
        return MultiPolygon(self.multi_lines.cuda(*args, **kwargs),
                            self.polygon_slit_position, self.linestring_slit_position)

    def cpu(self, *args, **kwargs):
        return MultiPolygon(self.multi_lines.cpu(*args, **kwargs),
                            self.polygon_slit_position, self.linestring_slit_position)


# class MultiPolygonTensor:
#     """
#     The multi-polygon for GPU calculation.
#
#     Each of polygon is represented by PolygonTensor,
#     """
#     def __init__(self, multi_polygons: list[PolygonTensor, ...] = None):
#         if multi_polygons is None:
#             self.multi_polygons = []
#         else:
#             self.multi_polygons = multi_polygons
#
#     def __str__(self):
#         return f"Polygon: {self.multi_polygons}"
#
#     def __repr__(self):
#         return self.__str__()
#
#     def __len__(self):
#         return len(self.multi_polygons)
#
#     def __getitem__(self, key):
#         return self.multi_polygons[key]
#
#     def __setitem__(self, key, value: PolygonTensor):
#         self.multi_polygons[key] = value
#
#     def append(self, value: PolygonTensor):
#         self.multi_polygons.append(value)
#
#     def to(self, *args, **kwargs):
#         return MultiPolygonTensor([x.to(*args, **kwargs) for x in self.multi_polygons])
#
#     def cuda(self, *args, **kwargs):
#         return MultiPolygonTensor([x.cuda(*args, **kwargs) for x in self.multi_polygons])
#
#     def cpu(self, *args, **kwargs):
#         return MultiPolygonTensor([x.cpu(*args, **kwargs) for x in self.multi_polygons])

