import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../..'])
import os
import pandas as pd
import time
import torch
from tqdm import tqdm
from utils import (LoadPolygonAndCreateCommutingPointsRoadMap, GraphicalIndexWithRoadMapPolygon)


def detour_with_road_map(
        region_data_file='../Data/demo_data/center 1-3.shp',
        barrier_data_file='../Data/demo_data/barrier 1-3.shp',
        road_map_file='../Data/demo_data/road_map.shp',
        id_name='areaID',
        sampling_interval=0.25,
        radius=10,
        save_path='../result/detour/demo',
        use_osmnx=False,
        st_id=0,
):
    t = time.time()

    # save path
    save_all = False
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load geo data
    geo_data = LoadPolygonAndCreateCommutingPointsRoadMap(
        region_data_file, barrier_data_file, road_map_file,
        id_name, sampling_interval, radius, use_osm_roads=use_osmnx)

    # graphical index for all regions
    graphical_indexer = GraphicalIndexWithRoadMapPolygon(
        block_sz=16,
        neighbor_d_max=sampling_interval * 1.2,
        cr_d_max=sampling_interval * 1.2,
        d_max=2 ** 11,
        degree_threshold=0.49,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        use_cc_dist=False,
    )

    # st_id = 8030
    t_f = time.time()
    for i, data in enumerate(geo_data[st_id:]):
        t_load = time.time()
        ii = i + st_id
        idx = int(geo_data.idx[ii])
        print(f'No. {ii} region')
        (list_polygons, commuting_points, traffic_nodes, road_net, region_bounds, x_factor, y_factor,
         region_center, share_of_barriers) = data

        # graphical index
        commuting_points = commuting_points.to(dtype=torch.float32)
        traffic_nodes = traffic_nodes.to(dtype=torch.float32)
        road_net = road_net.to(dtype=torch.float32)

        if save_path is None or not save_all:
            save_name = None
        else:
            save_name = save_path + f'/{idx}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = save_name + '/'

        detour, d_max, d_mean, d_std, d_mean_direct, n_c, n_r = graphical_indexer.run(
            list_polygons,
            commuting_points,
            traffic_nodes,
            road_net,
            save_name=save_name,
        )
        t_run = time.time()
        print(
            f'No. {ii} region, {id_name}: {idx}, share_of_barrier = {share_of_barriers}, detour = {detour}, \n'
            f'data load and transform {t_load - t_f} s, calculation {t_run - t_load} s, \n'
            f'total time = {t_run - t}.')
        t_f = t_run

        # save txt
        if save_path is not None:
            save_name = save_path + f'/{idx}_index.txt'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file = open(save_name, 'w')
            file.write(f'{idx},{share_of_barriers},{detour},{d_max},{d_mean},{d_std},{n_c},{n_r}')
            file.close()


def save_csv(summary: list[dict], filename: str, sort_by: str = None):
    import pandas as pd
    # whether path exist
    print('Save summary to: ' + filename + ' ...')
    (filepath, temp_filename) = os.path.split(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # save the lists to certain text
    if '.csv' not in filename:
        filename += '.csv'
    data = pd.DataFrame(summary)
    if sort_by is not None:
        data = data.sort_values(by=sort_by)
    data.to_csv(filename)
    print('Save finish')


def convert_text_to_csv(save_path: str, id_name: str, save_name: str = ''):
    # scan the txt files
    print(f'Scanning {save_path} ...')
    files = os.listdir(save_path)
    files = [file for file in files if file.endswith('.txt')]

    print(f'Got. {len(files)} .txt files')

    # get summary
    summaries = []
    for file_name in files:
        position = save_path + '/' + file_name
        print(f'Open {file_name}')
        with open(position, 'r') as f:
            data = f.read().split(',')
            summary = {id_name: data[0],
                       'share_of_barrier': data[1],
                       'detour': data[2],
                       'distance_max': data[3],
                       'distance_mean': data[4],
                       'distance_std': data[5],
                       'number of commuting nodes': data[6],
                       'number of road nodes': data[7],
                       }
            summaries.append(summary)
    save_csv(summaries, f'{save_path}/{save_name}summary.csv', id_name,)


if __name__ == '__main__':
    region_data_file = '../Data/demo_data/center 1-3.shp'
    barrier_data_file = '../Data/demo_data/barrier 1-3.shp'
    road_data_file = '../Data/demo_data/road_map.shp'
    id_name = 'areaID'
    save_path = '../result/detour/demo-r10km'
    detour_with_road_map(region_data_file, barrier_data_file, road_data_file,
                         id_name, 0.5, 10, save_path, use_osmnx=True)
    convert_text_to_csv(save_path, id_name, '../demo-r10km detour ')
    save_path = '../result/detour/demo-r5km'
    detour_with_road_map(region_data_file, barrier_data_file, road_data_file,
                         id_name, 0.25, 5, save_path, use_osmnx=True)
    convert_text_to_csv(save_path, id_name, '../demo-r5km detour ')

