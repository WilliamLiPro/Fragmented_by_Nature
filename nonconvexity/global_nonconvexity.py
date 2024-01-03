import sys
print('Python %s on %s' % (sys.version, sys.platform))
import os
import time
import torch
from tqdm import tqdm
from utils import (tensor_polygon_intersect, LoadPolygonAndCreateCommutingPoints)


def global_nonconvexity(
        region_data_file='../Data/demo_data/center 1-3.shp',
        barrier_data_file='../Data/demo_data/barrier 1-3.shp',
        id_name='ID_HDC_G0',
        sampling_interval=0.25,
        radius=10,
        save_path='../result/global_nonconvexity/global_nonconvexity',
        st_id = 0,
):
    t = time.time()

    # save path
    save_all = False
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # load geo data
    geo_data = LoadPolygonAndCreateCommutingPoints(region_data_file, barrier_data_file, id_name,
                                                   sampling_interval, radius)

    # st_id = 8030
    t_f = time.time()
    for i, data in enumerate(geo_data[st_id:]):
        t_load = time.time()
        ii = i + st_id
        id_hdc_g0 = int(geo_data.idx[ii])
        print(f'No. {ii} region')
        list_polygons, commuting_points, region_bounds, x_factor, y_factor, region_center, share_of_barrier = data

        # graphical index
        commuting_points = commuting_points.to(dtype=torch.float32)
        torch.cuda.empty_cache()
        relative_length = tensor_polygon_intersect(commuting_points, list_polygons, 16)
        nonconvexity = relative_length.mean(dim=1).mean()
        relative_length_max = relative_length.max(dim=1)[0].max()
        relative_length_std = relative_length.std()
        n_c = relative_length.size()[0]
        t_run = time.time()

        print(f'No. {ii} region, {id_name}: {id_hdc_g0}, share_of_barrier = {share_of_barrier}, nonconvexity = {nonconvexity}, \n'
              f'data load and transform {t_load - t_f} s, calculation {t_run - t_load} s, \n'
              f'total time = {t_run - t}.')
        t_f = t_run

        # save txt
        if save_path is not None:
            save_name = save_path + f'/{id_hdc_g0}_index.txt'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file = open(save_name, 'w')
            file.write(f'{id_hdc_g0},{share_of_barrier},{nonconvexity},{relative_length_max},{relative_length_std},{n_c}')
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
    for i, file_name in enumerate(files):
        front, ext = os.path.splitext(file_name)
        if not ext == '.txt':
            del files[i]

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
                       'nonconvexity': data[2],
                       'relative_length_max': data[3],
                       'relative_length_std': data[4],
                       'number of commuting nodes': data[5],
                       }
            summaries.append(summary)
    save_csv(summaries, f'{save_path}/{save_name}summary.csv', )


if __name__ == '__main__':
    """
        Please download the original data and crop them according to the method 
        in supplementary material.
        Then, copy the filename of region center to the "region_data_file", and  
        copy the filename of geo-barrier to the "barrier_data_file".
        
        e.g. 
        region_data_file = '../Data/AUE-200/region-center.shp'
        barrier_data_file = '../Data/AUE-200/geo-barrier.shp'
    """
    
    # AUE-200
    id_name = 'areaID'
    region_data_file = '../Data/AUE-200/region-center.shp'
    barrier_data_file = "../Data/AUE-200/geo-barrier.shp"
    save_path = '../result/global_nonconvexity/AUE-200-r10km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.5, 10, save_path, 1)
    convert_text_to_csv(save_path, id_name, '../AUE-200-r10km nonconvexity ')
    save_path = '../result/global_nonconvexity/AUE-200-r5km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.25, 5, save_path, 1)
    convert_text_to_csv(save_path, id_name, '../AUE-200-r5km nonconvexity ')

    region_data_file = "../Data/AUE-200/inculude country bound/Data region-center.shp"
    barrier_data_file = "../Data/AUE-200/inculude country bound/Data geo-barrier.shp"
    save_path = '../result/global_nonconvexity/AUE-boundary-r10km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.5, 10, save_path)
    convert_text_to_csv(save_path, id_name, '../AUE-boundary-r10km nonconvexity ')

    # GUB
    id_name = 'GUB_ID'
    region_data_file = "../Data/GUB/GUB_center.shp"
    barrier_data_file = "../Data/GUB/GUB_barrier.shp"
    save_path = '../result/global_nonconvexity/GUB-r10km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.5, 10, save_path)
    convert_text_to_csv(save_path, id_name, '../GUB-r10km nonconvexity ')
    save_path = '../result/global_nonconvexity/GUB-r5km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.25, 5, save_path)
    convert_text_to_csv(save_path, id_name, '../GUB-r5km nonconvexity ')

    id_name = 'GUB_ID'
    region_data_file = "../Data/GUB/country boundary/GUB_10km_center.shp"
    barrier_data_file = "../Data/GUB/country boundary/GUB_10km_barrier.shp"
    save_path = '../result/global_nonconvexity/GUB-boundary-r10km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.5, 10, save_path)
    convert_text_to_csv(save_path, id_name, '../GUB-boundary-r10km nonconvexity ')

    region_data_file = "../Data/GUB/country boundary/GUB_5km_center.shp"
    barrier_data_file = "../Data/GUB/country boundary/GUB_5km_barrier.shp"
    save_path = '../result/global_nonconvexity/GUB-boundary-r5km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.25, 5, save_path)
    convert_text_to_csv(save_path, id_name, '../GUB-boundary-r5km nonconvexity ')

    # UCDB
    id_name = 'ID_HDC_G0'
    region_data_file = "../Data/UCDB/nonconvexity/UCDB2019_boundary_10km-center.shp"
    barrier_data_file = "../Data/UCDB/nonconvexity/UCDB_boundary_10km.shp"
    save_path = '../result/global_nonconvexity/UCDB-boundary-r10km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.5, 10, save_path)
    convert_text_to_csv(save_path, id_name, '../UCDB-boundary-r10km nonconvexity ')

    region_data_file = "../Data/UCDB/nonconvexity/UCDB2019_boundary_5km-center.shp"
    barrier_data_file = "../Data/UCDB/nonconvexity/UCDB_boundary_5km.shp"
    save_path = '../result/global_nonconvexity/UCDB-boundary-r5km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.25, 5, save_path)
    convert_text_to_csv(save_path, id_name, '../UCDB-boundary-r5km nonconvexity ')

    # GHUB
    id_name = 'GHUB_ID'
    region_data_file = "../Data/GHUB/country boundary/GHUB_10km-center.shp"
    barrier_data_file = "../Data/GHUB/country boundary/GHUB_10km-barrier.shp"
    save_path = '../result/global_nonconvexity/GHUB-boundary-r10km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.5, 10, save_path)
    convert_text_to_csv(save_path, id_name, '../GHUB-boundary-r10km nonconvexity ')

    region_data_file = "../Data/GHUB/country boundary/GHUB_5km-center.shp"
    barrier_data_file = "../Data/GHUB/country boundary/GHUB_5km-barrier.shp"
    save_path = '../result/global_nonconvexity/GHUB-boundary-r5km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.25, 5, save_path)
    convert_text_to_csv(save_path, id_name, '../GHUB-boundary-r5km nonconvexity ')







