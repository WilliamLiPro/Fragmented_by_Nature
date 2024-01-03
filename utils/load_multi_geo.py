import time
import math
import numpy as np
import torch
import geopandas
import networkx as nx
import osmnx
from shapely.geometry import Point
from shapely.affinity import affine_transform


def geo_data_to_geo_lines(
        geo_data: geopandas.GeoSeries,
        region_center: torch.Tensor = None,
):
    print(f'Convert geo LineString to torch Tensor .. ')
    if region_center is not None:
        region_center = region_center.view(-1, 1)
    points_st = []
    points_ed = []
    if geo_data.boundary.__class__.__name__ == 'LineString':
        boundary = geo_data.boundary
        xy = torch.from_numpy(np.array(boundary.xy))
        if region_center is not None:
            xy = xy - region_center
        xy = xy.to(torch.float32)
        points_st.append(xy[:, :-1])
        points_ed.append(xy[:, 1:])
    else:
        for boundary in geo_data.boundary:
            if boundary.__class__.__name__ == 'LineString':
                xy = torch.from_numpy(np.array(boundary.xy))
                if region_center is not None:
                    xy = xy - region_center
                xy = xy.to(torch.float32)
                points_st.append(xy[:, :-1])
                points_ed.append(xy[:, 1:])
            elif boundary.__class__.__name__ == 'MultiLineString':
                for ss_boundary in boundary:
                    if ss_boundary.__class__.__name__ == 'LineString':
                        xy = torch.from_numpy(np.array(ss_boundary.xy))
                        if region_center is not None:
                            xy = xy - region_center
                        xy = xy.to(torch.float32)
                        points_st.append(xy[:, :-1])
                        points_ed.append(xy[:, 1:])
                    else:
                        raise TypeError('ss_boundary should be LineString')
            else:
                raise TypeError('boundary should be LineString or MultiLineString')
    points_st = torch.cat(points_st, dim=1)
    points_ed = torch.cat(points_ed, dim=1)
    print(f'Finish: converted {points_st.size()[-1]} LineStrings')
    return points_st, points_ed


def geo_data_to_points(
        geo_data: geopandas.GeoDataFrame,
        region_center: torch.Tensor = None,
):
    print(f'Convert geo Points to torch Tensor .. ')
    if region_center is not None:
        region_center = region_center.view(-1, 1)
    points = []
    for region_shape in geo_data.geometry:
        if region_shape.__class__.__name__ == 'Point':
            xy = region_shape.xy
            xy = torch.from_numpy(np.array(xy))
            if region_center is not None:
                xy = xy - region_center
            xy = xy.to(torch.float32)
            points.append(xy.view(-1, 1))
    points = torch.cat(points, dim=1)
    print(f'Finish: converted {points.size()[-1]} Points ')
    return points


def create_commuting_points(
        region_bounds: np.array,
        sampling_interval: float,  # km
        crs='EPSG:4326',  # 指定坐标系为WGS 1984
):
    nx = int((region_bounds[2] - region_bounds[0]) / sampling_interval) + 1
    ny = int((region_bounds[3] - region_bounds[1]) / sampling_interval) + 1

    x_list = torch.linspace(region_bounds[0], region_bounds[2], nx)
    y_list = torch.linspace(region_bounds[1], region_bounds[3], ny)

    px = x_list.view(-1, 1).expand(nx, ny).reshape(-1)
    py = y_list.view(1, -1).expand(nx, ny).reshape(-1)

    points = geopandas.GeoSeries(geopandas.points_from_xy(px.numpy(), py.numpy()),
                                 index=[str(i) for i in range(len(px))]  # 相关的索引
                                 )
    return geopandas.GeoDataFrame(geometry=points)


def create_commuting_points_radius(
        radius: float,
        sampling_interval: float,  # km
        crs='EPSG:4326',  # 指定坐标系为WGS 1984
):
    """
    Create commuting points from center and radius
    :param x_factor:
    :param y_factor:
    :param radius:
    :param sampling_interval:
    :param crs:
    :return:
    """

    nx = int(radius // sampling_interval) * 2 + 1
    ny = int(radius // sampling_interval) * 2 + 1

    x_list = torch.linspace(-radius, radius, nx)
    y_list = torch.linspace(-radius, radius, ny)

    px = x_list.view(-1, 1).expand(nx, ny).reshape(-1)
    py = y_list.view(1, -1).expand(nx, ny).reshape(-1)

    mask = Point(0, 0).buffer(radius)
    points = geopandas.GeoSeries(geopandas.points_from_xy(px.numpy(), py.numpy()),
                                 crs=crs,
                                 index=[str(i) for i in range(len(px))]  # 相关的索引
                                 )
    points = geopandas.GeoDataFrame(geometry=points).clip(mask)
    return points, mask


class LoadMultiGeoAndCreateCommutingPoints(object):
    r"""
    Load multi geo data and create commuting points.

    Args:
        :param file_name: file name of geo data (.shp)
        :param sampling_interval: sampling interval of commuting points (Km)
        :param how: operation of point set and geo set for creating commuting points (default: 'difference')
    """

    def __init__(self,
                 file_name: str,
                 sampling_interval: float = 0.2,
                 how='intersection',
                 ):
        self.sampling_interval = sampling_interval
        self.how = how

        print(f'LoadMultiGeoAndCreateCommutingPoints: Loading geo data from {file_name} .. ')
        geo_data = geopandas.read_file(file_name)
        print(f'LoadMultiGeoAndCreateCommutingPoints: Loading finish ')

        self.geo_series = geo_data.geometry
        self.id_hdc_g0 = geo_data["ID_HDC_G0"].values
        self.crs = geo_data.crs
        self.num = len(self.geo_series)
        self.start = 0

    def __getitem__(self, item):
        if item.__class__.__name__ == 'slice':
            self.start = item.start
            return self
        if item.__class__.__name__ == 'int':
            item += self.start
            if item >= self.num:
                raise StopIteration
            else:
                return self.get_item(item)

    def get_item(self, item):
        # get region center and range
        current_geo = self.geo_series[item]
        region_center = (current_geo.centroid.x, current_geo.centroid.y)

        x_factor = 111.320 * math.cos(region_center[1] * 0.99 / 180 * math.pi)
        y_factor = 110.574

        # convert to local coordinate
        current_geo = affine_transform(
            current_geo, [x_factor, 0, 0, y_factor, -region_center[0] * x_factor, -region_center[1] * y_factor])

        # create commuting points
        region_bounds = current_geo.bounds
        points = create_commuting_points(
            region_bounds, self.sampling_interval, crs=self.crs)

        points = geopandas.overlay(points, geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(current_geo),
                                                                  crs=self.crs), how=self.how)

        # convert to local coordinate
        line_st, line_ed = geo_data_to_geo_lines(current_geo)
        points = geo_data_to_points(points)

        return line_st, line_ed, points, region_bounds, x_factor, y_factor, region_center


def get_coordinate(
        region_shape,
        relative_center,
        threshold=9e-8,
):
    xy = region_shape.xy
    xy = torch.from_numpy(np.array(xy))

    if relative_center is not None:
        xy = xy - relative_center.view(-1, 1)
    xy = xy.to(torch.float32)
    if (xy[:, 0] - xy[:, -1]).pow(2).sum() > threshold:
        point_st = (xy[:, 0].view(-1, 1))
        point_ed = (xy[:, -1].view(-1, 1))
        return point_st, point_ed
    else:
        return None, None


def geo_data_to_bridges(
        geo_data: geopandas.GeoDataFrame,
        region_center: torch.Tensor,
        threshold=2e-8,
):
    print(f'Convert bridge LineString to torch Tensor .. ')
    points_st = []
    points_ed = []
    if geo_data.geometry.unary_union.__class__.__name__ == 'LineString':
        p_st, p_ed = get_coordinate(geo_data.geometry.unary_union, region_center, threshold, )
        if p_st is not None:
            line = p_ed - p_st
            points_st.append(p_st - 0.1 * line)
            points_ed.append(p_ed + 0.1 * line)
    else:
        for group_lines in geo_data.geometry.unary_union:
            if group_lines.__class__.__name__ == 'LineString':
                p_st, p_ed = get_coordinate(group_lines, region_center, threshold, )
                if p_st is not None:
                    line = p_ed - p_st
                    points_st.append(p_st - 0.1 * line)
                    points_ed.append(p_ed + 0.1 * line)
            else:
                TypeError(f'Unsupported datatype, got {group_lines.__class__.__name__}')
    if len(points_st) == 0 or len(points_ed) == 0:
        points_st = torch.zeros((2, 0))
        points_ed = torch.zeros((2, 0))
    else:
        points_st = torch.cat(points_st, dim=1)
        points_ed = torch.cat(points_ed, dim=1)
    print(f'Finish: converted {points_st.size()[-1]} bridges')
    return points_st, points_ed


class LoadMultiLines(object):
    r"""
        Load multi lines data from file.

        Args:
            :param file_name: file name of geo data (.shp)
        """

    def __init__(self,
                 file_name: str,
                 ):
        print(f'LoadMultiLines: Loading geo data from {file_name} .. ')
        self.geo_data = geopandas.read_file(file_name)
        print('LoadMultiLines: Loading finish')

    def get_regional_polyline(
            self,
            clip_range=None,
            relative_center=None,
            x_factor=None,
            y_factor=None,
    ):
        # get regional polyline
        if clip_range is not None:
            gdf = geopandas.clip(self.geo_data, clip_range)
        else:
            gdf = self.geo_data

        if gdf.size == 0:
            Warning('bridge data is empty in this region')
            return torch.zeros((2, 0)), torch.zeros((2, 0))

        # to Tensor
        points_st, points_ed = geo_data_to_bridges(gdf, relative_center)
        print(f'Finish: loaded {points_st.size()[-1]} polylines')

        if x_factor is not None and y_factor is not None:
            points_st[0, :] *= x_factor
            points_st[1, :] *= y_factor
            points_ed[0, :] *= x_factor
            points_ed[1, :] *= y_factor
        return points_st, points_ed


def iter_over_linestring(polygon, region_center, list_polygons):
    boundary = polygon.boundary
    if boundary.__class__.__name__ == 'LineString':
        xy = torch.from_numpy(np.array(boundary.xy))
        if region_center is not None:
            xy = xy - region_center
        xy = xy.to(torch.float32)
        list_polygons.append(xy)
    elif boundary.__class__.__name__ == 'MultiLineString':
        local_lines = []
        st_id = 0
        for i, ss_boundary in enumerate(boundary.geoms):
            if ss_boundary.__class__.__name__ == 'LineString':
                xy = torch.from_numpy(np.array(ss_boundary.xy))
                if region_center is not None:
                    xy = xy - region_center
                xy = xy.to(torch.float32)
                local_lines.append(xy)
                # check the start == end point
                if st_id == i:
                    if (xy[:, 0] == xy[:, -1]).sum() == 2:
                        st_id = i + 1
                elif st_id < i:
                    if (local_lines[st_id][:, 0] == xy[:, -1]).sum() == 2:
                        xy = torch.cat(local_lines[st_id:], dim=-1)
                        local_lines = local_lines[:st_id] + [xy]
                        st_id = i + 1
            else:
                raise TypeError(f'ss_boundary should be "LineString", got {ss_boundary.__class__.__name__}')
        list_polygons.append(local_lines)
    else:
        raise TypeError(
            f'Type of boundary should be "LineString" or "MultiLineString", got {boundary.__class__.__name__}')
    return list_polygons


def multi_polygon_to_multi_lines(
        geo_data: geopandas.GeoDataFrame,
        region_center: torch.Tensor = None,
):
    print(f'Convert the boundary of GeoDataFrame to list of torch Tensor .. ')
    if region_center is not None:
        region_center = region_center.view(-1, 1)
    list_polygons = []
    if geo_data is None:
        return list_polygons
    for i, row in geo_data.iterrows():
        geometry = row.geometry
        if geometry.__class__.__name__ == 'MultiPolygon':
            for polygon in geometry.geoms:
                list_polygons = iter_over_linestring(polygon, region_center, list_polygons)
        elif geometry.__class__.__name__ == 'Polygon':
            list_polygons = iter_over_linestring(geometry, region_center, list_polygons)
    print(f'Finish: converted {len(list_polygons)} Polygons')
    return list_polygons


def simplify_polygons(
        geo_data: geopandas.GeoSeries,
        node_threshold=10,
        tolerance=1,
):
    geo_data = geopandas.GeoDataFrame(geometry=geo_data)
    for i, row in geo_data.iterrows():
        geometry = row.geometry
        if geometry.__class__.__name__ == 'MultiPolygon':
            for polygon in geometry.geoms:
                if polygon > node_threshold:
                    polygon.simplify(tolerance)
        elif geometry.__class__.__name__ == 'Polygon':
            if geometry > node_threshold:
                geometry.simplify(tolerance)
    return geo_data


class LoadPolygonAndCreateCommutingPoints(object):
    r"""
        Load Polygon data and create commuting points according to polygons.
        Same as LoadMultiGeoAndCreateCommutingPoints except for the output is polygons
        instead of single set of boundaries.

        Args:
            :param file_name: file name of geo data (.shp)
            :param sampling_interval: sampling interval of commuting points (Km)
            :param how: operation of point set and geo set for creating commuting points (default: 'difference')
        """

    def __init__(self,
                 region_file_name: str,
                 barrier_file_name: str,
                 id_name: str = 'ID_HDC_G0',
                 sampling_interval: float = 0.2,
                 radius: float = 10,
                 how='intersection',
                 ):
        self.sampling_interval = sampling_interval
        self.radius = radius
        self.how = how

        print(f'LoadPolygonAndCreateCommutingPoints: \nLoading region data from {region_file_name} .. ')
        region_data = geopandas.read_file(region_file_name)
        print('Finish')

        print(f'LoadPolygonAndCreateCommutingPoints: \nLoading barrier data from {barrier_file_name} .. ')
        barrier_data = geopandas.read_file(barrier_file_name)

        print('Finish')

        self.region_data = region_data
        self.barrier_data = barrier_data
        self.region_series = region_data.geometry
        self.barrier_series = barrier_data.geometry
        self.id_name = id_name
        self.idx = region_data[id_name].values
        self.crs = region_data.crs
        self.num = len(self.region_series)
        self.start = 0

    def __getitem__(self, item):
        if item.__class__.__name__ == 'slice':
            self.start = item.start
            return self
        if item.__class__.__name__ == 'int':
            item += self.start
            if item >= self.num:
                raise StopIteration
            else:
                return self.get_item(item)

    def get_item(self, item):
        print(f'\nTransform No. {item} Geo data And RoadMap, {self.id_name}: {self.idx[item]} ...')

        # get geo region center and range
        current_region = self.region_series[item]
        current_barrier = self.barrier_data[self.barrier_data[self.id_name] == self.idx[item]]
        region_center = (current_region.centroid.x, current_region.centroid.y)

        print(f'region_center = {region_center}')

        x_factor = 111.320 * math.cos(region_center[1] * 0.99 / 180 * math.pi)
        y_factor = 110.574

        # convert to local coordinate
        current_barrier = current_barrier.affine_transform(
            [x_factor, 0, 0, y_factor, -region_center[0] * x_factor, -region_center[1] * y_factor])

        # create commuting points
        region_type = current_region.__class__.__name__
        if region_type == "Point":
            commuting_points, mask = create_commuting_points_radius(
                self.radius, self.sampling_interval, crs=self.crs)
            region_bounds = commuting_points.bounds
            current_barrier = current_barrier.clip(mask)
            commuting_points = geopandas.overlay(
                commuting_points,
                geopandas.GeoDataFrame(geometry=current_barrier, crs=self.crs), how='difference')
        elif region_type == 'MultiPolygon':
            current_region = self.region_data[item]
            current_region = current_region.affine_transform(
                [x_factor, 0, 0, y_factor, -region_center[0] * x_factor, -region_center[1] * y_factor])
            region_bounds = current_region.bounds
            mask = Point(0, 0).buffer(
                max(region_bounds[2] - region_bounds[0], region_bounds[3] - region_bounds[1]) * 0.5)
            current_barrier = current_barrier.clip(mask)
            commuting_points = create_commuting_points(
                region_bounds, self.sampling_interval, crs=self.crs)
            commuting_points = geopandas.overlay(
                commuting_points,
                geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(current_region),
                                       crs=self.crs), how=self.how)
        else:
            raise ValueError(f'Unknown region type: {region_type}')

        # convert to Tensor
        list_polygons = multi_polygon_to_multi_lines(geopandas.GeoDataFrame(geometry=current_barrier, crs=self.crs))
        commuting_points = geo_data_to_points(commuting_points)
        share_of_barriers = current_barrier.area.values.sum() / mask.area
        print('Geo Transform finish')

        return list_polygons, commuting_points, region_bounds, x_factor, y_factor, region_center, share_of_barriers


class LoadAvailableRegionAndCreateCommutingPoints(object):
    r"""
        Load Polygon data and create commuting points according to polygons.
        Same as LoadMultiGeoAndCreateCommutingPoints except for the output is polygons
        instead of single set of boundaries.

        Args:
            :param file_name: file name of geo data (.shp)
            :param sampling_interval: sampling interval of commuting points (Km)
            :param how: operation of point set and geo set for creating commuting points (default: 'difference')
        """
    def __init__(self,
                 region_file_name: str,
                 id_name: str = 'ID_HDC_G0',
                 sampling_interval: float = 0.2,
                 sampling_n_expected=None,
                 how='intersection',
                 trans_factor=True,
                 ):
        self.sampling_interval = sampling_interval
        self.sampling_n_expected = sampling_n_expected
        self.how = how

        print(f'LoadAvailableRegionAndCreateCommutingPoints: \nLoading region data from {region_file_name} .. ')
        region_data = geopandas.read_file(region_file_name)

        self.region_data = region_data
        self.region_series = region_data.geometry
        self.id_name = id_name
        self.trans_factor = trans_factor
        self.idx = region_data[id_name].values
        self.crs = region_data.crs
        self.num = len(self.region_series)
        self.start = 0
        print(f'Got: {self.num} regions .. ')
        print('Finish')

    def __getitem__(self, item):
        if item.__class__.__name__ == 'slice':
            self.start = item.start
            return self
        if item.__class__.__name__ == 'int':
            item += self.start
            if item >= self.num:
                raise StopIteration
            else:
                return self.get_item(item)

    def get_item(self, item):
        print(f'\nTransform No. {item} Geo data And RoadMap, {self.id_name}: {self.idx[item]} ...')

        # get geo region center and range
        current_region = self.region_series[item]
        region_center = (current_region.centroid.x, current_region.centroid.y)

        print(f'region_center = {region_center}')

        x_factor = 111.320 * math.cos(region_center[1] * 0.99 / 180 * math.pi)
        y_factor = 110.574

        # get the points
        current_region = geopandas.GeoSeries(current_region)
        if self.trans_factor:
            current_region = current_region.affine_transform(
                [x_factor, 0, 0, y_factor, -region_center[0] * x_factor, -region_center[1] * y_factor])
        region_bounds = current_region.unary_union.bounds
        if self.sampling_n_expected is not None:
            # elastic sampling num
            area = (region_bounds[2] - region_bounds[0]) * (region_bounds[3] - region_bounds[1])
            self.sampling_interval = (area / self.sampling_n_expected) ** 0.5
        commuting_points = create_commuting_points(
            region_bounds, self.sampling_interval, crs=self.crs)
        commuting_points = commuting_points.clip(current_region)

        # get the geo barriers
        convex_hull = current_region.convex_hull
        current_barrier = convex_hull.difference(current_region)

        # drop the region too samll, and simplify the barriers with too many nodes
        current_barrier = current_barrier.simplify(0.2*self.sampling_interval)

        # convert to Tensor
        list_polygons = multi_polygon_to_multi_lines(geopandas.GeoDataFrame(
            geometry=current_barrier, crs=self.crs))
        commuting_points = geo_data_to_points(commuting_points)
        share_of_barriers = current_barrier.area.values.sum() / convex_hull.area.values.sum()
        print('Geo Transform finish')

        return list_polygons, commuting_points, region_bounds, x_factor, y_factor, region_center, share_of_barriers


def add_road_to_graph(
        road,
        graph: nx.MultiDiGraph,
):
    assert road.__class__.__name__ == 'LineString'
    road = np.array(road.xy)
    n = road.shape[-1]
    from_nodes = road[:, :n - 1]
    to_nodes = road[:, 1:]
    length_s = ((from_nodes - to_nodes) ** 2).sum(axis=0) ** 0.5

    from_node = (from_nodes[0, 0], from_nodes[1, 0])
    if not graph.has_node(from_node):
        graph.add_node(from_node, x=from_node[0], y=from_node[1], pos=from_node)
    for i in range(n - 1):
        to_node = to_nodes[:, i]
        to_node = (to_node[0], to_node[1])
        if not graph.has_node(to_node):
            graph.add_node(to_node, x=to_node[0], y=to_node[1], pos=to_node)
        graph.add_edge(from_node, to_node, length=length_s[i])
        from_node = to_node
    return graph


def create_road_map(
        x_factor: float, y_factor: float,
        region_center,
        mask: geopandas.GeoDataFrame,
        road_map_data: geopandas.GeoDataFrame,
        clip_range_ratio: float = 1.2,
        simplify_graph=False,
):
    """
    get the road map graph from geopandas GeoDataFrame,
    where each row has a multiline corresponding to a road
    :param x_factor:
    :param y_factor:
    :param region_center:
    :param mask: available range
    :param road_map_data:
    :param clip_range_ratio:    # radio of road map range compared with commuting nodes
    :param simplify_graph:
    :return:
    """
    rx = (mask.bounds[2] - mask.bounds[0]) * 0.5 * clip_range_ratio
    ry = (mask.bounds[3] - mask.bounds[1]) * 0.5 * clip_range_ratio
    graph = nx.MultiDiGraph()
    if road_map_data is None:
        for i in range(3):
            try:
                graph = osmnx.graph_from_point(
                    center_point=(region_center[1], region_center[0]),
                    dist=max(rx, ry) * 1000,
                    network_type="drive",
                    simplify=simplify_graph,
                    retain_all=False,
                    truncate_by_edge=True,)

                # Scale down node values
                for node in graph.nodes:
                    graph.nodes[node]['x'] = (graph.nodes[node]['x'] - region_center[0]) * x_factor
                    graph.nodes[node]['y'] = (graph.nodes[node]['y'] - region_center[1]) * y_factor

                # Scale down edge values
                for u, v, k, data in graph.edges(keys=True, data=True):
                    data['length'] *= 0.001
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying in {3} seconds...")
                time.sleep(3)
    else:
        # coordinate transform
        road_map_data = road_map_data.affine_transform(
            [x_factor, 0, 0, y_factor, -x_factor * region_center[0], -y_factor * region_center[1]])

        # clip road map data in current region
        dx = rx * (clip_range_ratio - 1) / clip_range_ratio
        dy = ry * (clip_range_ratio - 1) / clip_range_ratio
        clip_range = (mask.bounds[0] - dx, mask.bounds[1] - dy, mask.bounds[2] + dx, mask.bounds[3] + dy)
        map_data = geopandas.clip(road_map_data, clip_range)
        map_data = map_data.clip(mask)

        # convert map data to graph
        for road in map_data.geometry:
            if road.__class__.__name__ == 'LineString':
                graph = add_road_to_graph(road, graph)
            elif road.__class__.__name__ == 'MultiLineString':
                for sub_road in road.geoms:
                    graph = add_road_to_graph(sub_road, graph)
            else:
                raise TypeError(f'road should be LineString or MultiLineString, got {road.__class__.__name__} instead')

        if simplify_graph and not ("simplified" in graph.graph and graph.graph["simplified"]):
            graph = osmnx.simplify_graph(graph, strict=True, remove_rings=True)
    return graph


class LoadPolygonAndCreateCommutingPointsRoadMap(object):
    r"""
        Load Polygon data and create commuting points according to polygons.
        Create a road map
        Same as LoadMultiGeoAndCreateCommutingPoints except for the output is polygons
        instead of single set of boundaries.

        Args:
            :param region_file_name:
            :param barrier_file_name: file name of barrier data (.shp)
            :param sampling_interval: sampling interval of commuting points (Km)
            :param how: operation of point set and geo set for creating commuting points (default: 'difference')
        """

    def __init__(self,
                 region_file_name: str,
                 barrier_file_name: str,
                 roadmap_file_name: str,
                 id_name: str = 'ID_HDC_G0',
                 sampling_interval: float = 0.2,
                 radius: float = 10,
                 how='intersection',
                 use_osm_roads=True,
                 simplify_road_graph=True,
                 ):
        self.sampling_interval = sampling_interval
        self.radius = radius
        self.how = how
        self.use_osm_roads = use_osm_roads
        self.simplify_road_graph = simplify_road_graph

        print(f'LoadPolygonAndCreateCommutingPointsRoadMap: \nLoading region data from {region_file_name} .. ')
        region_data = geopandas.read_file(region_file_name)
        print('Finish')

        print(f'LoadPolygonAndCreateCommutingPointsRoadMap: \nLoading barrier data from {barrier_file_name} .. ')
        barrier_data = geopandas.read_file(barrier_file_name)
        print('Finish')

        if use_osm_roads:
            self.road_map = None
        else:
            print(f'LoadPolygonAndCreateCommutingPointsRoadMap: \nLoading road map data from {roadmap_file_name} .. ')
            road_map = geopandas.read_file(roadmap_file_name)
            print('Finish')
            self.road_map = road_map

        self.region_data = region_data
        self.barrier_data = barrier_data
        self.region_series = region_data.geometry
        self.barrier_series = barrier_data.geometry
        self.id_name = id_name
        self.idx = region_data[id_name].values
        self.crs = region_data.crs
        self.num = len(self.region_series)
        self.start = 0

    def __getitem__(self, item):
        if item.__class__.__name__ == 'slice':
            self.start = item.start
            return self
        if item.__class__.__name__ == 'int':
            item += self.start
            if item >= self.num:
                raise StopIteration
            else:
                return self.get_item(item)

    def get_item(self, item):
        print(f'\nTransform No. {item} Geo data And RoadMap, {self.id_name}: {self.idx[item]} ...')

        # get geo region center and range
        current_region = self.region_series[item]
        idx = self.idx[item]
        current_barrier = self.barrier_data[self.barrier_data[self.id_name] == idx]
        if self.use_osm_roads:
            current_roadmap = None
        else:
            current_roadmap = self.road_map[self.road_map[self.id_name] == idx]
        region_center = (current_region.centroid.x, current_region.centroid.y)

        print(f'region_center = {region_center}')

        x_factor = 111.320 * math.cos(region_center[1] * 0.99 / 180 * math.pi)
        y_factor = 110.574

        # convert to local coordinate
        current_barrier = current_barrier.affine_transform(
            [x_factor, 0, 0, y_factor, -region_center[0] * x_factor, -region_center[1] * y_factor])

        # create commuting points
        region_type = current_region.__class__.__name__
        if region_type == "Point":
            commuting_points, mask = create_commuting_points_radius(
                self.radius, self.sampling_interval, crs=self.crs)
            region_bounds = commuting_points.bounds
            current_barrier = current_barrier.clip(mask)
            commuting_points = geopandas.overlay(
                commuting_points,
                geopandas.GeoDataFrame(geometry=current_barrier, crs=self.crs), how='difference')
        elif region_type == 'MultiPolygon':
            current_region = current_region.affine_transform(
                [x_factor, 0, 0, y_factor, -region_center[0] * x_factor, -region_center[1] * y_factor])
            region_bounds = current_region.bounds
            mask = Point(0, 0).buffer(
                max(region_bounds[2] - region_bounds[0], region_bounds[3] - region_bounds[1]) * 0.5)
            current_barrier = current_barrier.clip(mask)
            commuting_points = create_commuting_points(
                region_bounds, self.sampling_interval, crs=self.crs)
            commuting_points = geopandas.overlay(
                commuting_points,
                geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(current_region),
                                       crs=self.crs), how=self.how)
        else:
            raise ValueError(f'Unknown region type: {region_type}')

        # road map
        graph = create_road_map(x_factor, y_factor, region_center, mask, current_roadmap,
                                simplify_graph=self.simplify_road_graph)
        traffic_x = nx.get_node_attributes(graph, 'x')
        traffic_x = np.array(list(traffic_x.values())).reshape((1, -1))
        traffic_y = nx.get_node_attributes(graph, 'y')
        traffic_y = np.array(list(traffic_y.values())).reshape((1, -1))
        traffic_nodes = np.vstack((traffic_x, traffic_y))

        # convert to Tensor
        traffic_nodes = torch.from_numpy(traffic_nodes.astype(np.float32))
        if traffic_nodes.size(1) > 0:
            road_net = nx.to_scipy_sparse_array(graph, weight='length').toarray().astype(np.float32)
            road_net = torch.from_numpy(road_net)
        else:
            road_net = torch.zeros(0)

        list_polygons = multi_polygon_to_multi_lines(geopandas.GeoDataFrame(geometry=current_barrier, crs=self.crs))
        commuting_points = geo_data_to_points(commuting_points)
        share_of_barriers = current_barrier.area.values.sum() / mask.area

        # correct the wrong road_net
        direct_dist = (((traffic_nodes[0, :].view(-1, 1) - traffic_nodes[0, :].view(1, -1)) ** 2) +
                       (traffic_nodes[1, :].view(-1, 1) - traffic_nodes[1, :].view(1, -1)) ** 2) ** 0.5
        msk = (road_net > 0) & (road_net < direct_dist)
        road_net[msk] = direct_dist[msk]

        print('Geo Transform finish')

        return (list_polygons, commuting_points, traffic_nodes, road_net, region_bounds, x_factor, y_factor,
                region_center, share_of_barriers)


def load_test():
    print('load_test')
    # data_s = LoadMultiGeoAndCreateCommutingPoints(
    #     '/home/lwp/ImgData/Dataset/road map/全球城市区域及桥梁/裁剪后10km.shp')
    data_s = LoadMultiGeoAndCreateCommutingPoints(
        '/home/lwp/ImgData/Dataset/road map/全球城市区域及桥梁/裁剪后10km.shp')
    s = 0
    for data in data_s:
        s += 1
        print(f'loaded {s} data')
    print(f'number of geo sets = {s}')
    print('pass')


def load_geo_data_test():
    import matplotlib.pyplot as plt
    print('LoadPolygonAndCreateCommutingPoints test')
    data_s = LoadPolygonAndCreateCommutingPoints(
        # '/home/liweipeng/disk_sda2/Dataset/road map/detour测试/地理障碍3个城市.shp',
        # '/home/liweipeng/disk_sda2/Dataset/road map/detour测试/三个城市地理障碍.shp',
        "/home/liweipeng/disk_sda2/Dataset/road map/AUE-test3/中心点1-3.shp",
        "/home/liweipeng/disk_sda2/Dataset/road map/AUE-test3/地理障碍1-3.shp",
        'areaID',
        sampling_interval=1,
    )
    s = 0
    for data in data_s:
        s += 1
        print(f'loaded {s} data')
        list_polygons, commuting_points, region_bounds, x_factor, y_factor, region_center, share_of_barriers = data
        for polygon in list_polygons:
            plt.plot(polygon[0, :], polygon[1, :])
        plt.plot(commuting_points[0, :], commuting_points[1, :], '.')
        plt.show()
    print(f'number of geo sets = {s}')
    print('pass')


def load_region_geo_test():
    import matplotlib.pyplot as plt
    loader = LoadAvailableRegionAndCreateCommutingPoints(
        "/home/liweipeng/disk_sda2/Dataset/road map/AUE-200/elastic_boundary/Data_弹性ratio_去除水体山脉和国界.shp",
        'areaID',
        sampling_n_expected=2000,
        sampling_interval=1,
    )
    s = 0
    for data in loader:
        s += 1
        print(f'loaded {s} data')
        list_polygons, commuting_points, region_bounds, x_factor, y_factor, region_center, share_of_barriers = data
        for polygon in list_polygons:
            if isinstance(polygon, list):
                for line in polygon:
                    plt.plot(line[0, :], line[1, :])
            else:
                plt.plot(polygon[0, :], polygon[1, :])
        plt.plot(commuting_points[0, :], commuting_points[1, :], '.')
        plt.show()
    print(f'number of geo sets = {s}')
    print('pass')


def show_any_geo():
    import matplotlib.pyplot as plt
    from numpy import vstack
    def plot_line(p_st, p_nd, **args):
        p_st = p_st.numpy()
        p_nd = p_nd.numpy()
        plt.plot(vstack((p_st[0, :], p_nd[0, :])), vstack((p_st[1, :], p_nd[1, :])), **args)

    def plot_point(point, color='.r'):
        point = point.numpy()
        plt.plot(point[0, :], point[1, :], color)

    geo_data = LoadMultiGeoAndCreateCommutingPoints(
        '/home/lwp/ImgData/Dataset/road map/全球城市区域及桥梁/亚洲与中东城市.shp',
        0.25
    )
    bridge_data = LoadMultiLines(
        '/home/lwp/ImgData/Dataset/road map/全球城市区域及桥梁/亚洲部分城市桥梁/桥梁R6.shp'
    )

    idx = 89
    barrier_st, barrier_ed, points, region_bounds, x_factor, y_factor, region_center = geo_data[idx]
    bridge_st, bridge_end = bridge_data.get_regional_polyline(
        region_bounds, region_center, x_factor, y_factor)

    plt.figure(dpi=300)
    plot_point(points)
    plot_line(barrier_st, barrier_ed, color='b', linewidth=0.5)
    plot_line(bridge_st, bridge_end, color='g', linewidth=1)

    # g = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(data_s.geo_series[idx]), crs=data_s.crs)
    # g.plot()
    plt.show()


def show_roads():
    import matplotlib.pyplot as plt
    geo_data = LoadPolygonAndCreateCommutingPointsRoadMap(
        '../Data/demo_data/center 1-3.shp',
        '../Data/demo_data/barrier 1-3.shp',
        '../Data/demo_data/road_map.shp',
        'areaID',
        0.5, 10, use_osm_roads=True,
    )
    for i, data in enumerate(geo_data[1:]):
        idx = int(geo_data.idx[i])
        print(f'No. {i} region, {geo_data.id_name}: {idx}')
        (list_polygons, commuting_points, traffic_nodes, road_net, region_bounds, x_factor, y_factor,
         region_center, share_of_barriers) = data
        for polygon in list_polygons:
            if isinstance(polygon, list):
                for line in polygon:
                    plt.plot(line[0, :], line[1, :])
            else:
                plt.plot(polygon[0, :], polygon[1, :])
        # roads
        mask = road_net > 0
        st_x = traffic_nodes[0, :].view(-1, 1).expand_as(mask)[mask]
        st_y = traffic_nodes[1, :].view(-1, 1).expand_as(mask)[mask]
        end_x = traffic_nodes[0, :].view(1, -1).expand_as(mask)[mask]
        end_y = traffic_nodes[1, :].view(1, -1).expand_as(mask)[mask]
        lines_x = torch.cat([st_x.view(-1, 1), end_x.view(-1, 1)], dim=-1).numpy()
        lines_y = torch.cat([st_y.view(-1, 1), end_y.view(-1, 1)], dim=-1).numpy()
        for j in range(st_x.size()[0]):
            plt.plot(lines_x[j, :], lines_y[j, :])
        plt.show()


if __name__ == '__main__':
    # load_test()
    # load_geo_data_test()
    load_region_geo_test()
    # show_any_geo()
    # show_roads()
