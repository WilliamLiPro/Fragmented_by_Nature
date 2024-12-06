import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../..'])
import pandas as pd
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import torch
from tqdm import tqdm
from utils import (tensor_polygon_intersect, pp_m_polygon_intersect_length_decompose, multi_polygon_to_tensor,
                   LoadPolygonAndCreateCommutingPoints)


def global_nonconvexity(
        region_data_file='../Data/demo_data/center 1-3.shp',
        barrier_data_file='../Data/demo_data/barrier 1-3.shp',
        id_name='ID_HDC_G0',
        sampling_interval=0.25,
        radius=10,
        save_path='../result/global_nonconvexity/global_nonconvexity',
        st_id=0,
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # relative_length = tensor_polygon_intersect(commuting_points, list_polygons, 16)
        m_polygon, line_split_position, polygon_split_position = multi_polygon_to_tensor(list_polygons)
        if torch.cuda.is_available():
            m_polygon, line_split_position, polygon_split_position = m_polygon.cuda(), line_split_position.cuda(), polygon_split_position.cuda()
            commuting_points = commuting_points.cuda()
        relative_length = pp_m_polygon_intersect_length_decompose(m_polygon, line_split_position, polygon_split_position,
                                                                  commuting_points, batch_size=16)
        n_c = relative_length.size()[0]
        if n_c == 0:
            nonconvexity = relative_length_max = -1
            relative_length_std = 0
        else:
            nonconvexity = relative_length.mean(dim=1).mean()
            relative_length_max = relative_length.max(dim=1)[0].max()
            relative_length_std = relative_length.std()
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


def summary_to_dataframe(summary: list[dict], sort_by: str = None):
    data = pd.DataFrame(summary)
    if sort_by is not None:
        data = data.sort_values(by=sort_by)
    return data


def save_csv(data: pd.DataFrame, filename: str):
    # whether path exist
    print('Save summary to: ' + filename + ' ...')
    # save
    (filepath, temp_filename) = os.path.split(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if '.csv' not in filename:
        filename += '.csv'
    data.to_csv(filename)
    print('Save finish')


def convert_text_to_dataframe(file_path: str, id_name: str):
    # scan the txt files
    print(f'Scanning {file_path} ...')
    files = os.listdir(file_path)
    files = [file for file in files if file.endswith('.txt')]
    print(f'Got. {len(files)} .txt files')

    # get summary
    summaries = []
    for file_name in files:
        position = file_path + '/' + file_name
        print(f'Open {file_name}')
        with open(position, 'r') as f:
            data = f.read().split(',')
            summary = {id_name: data[0],
                       'share_of_barrier': data[1],
                       'nonconvexity': data[2],
                       'number of commuting nodes': data[5],
                       }
            summaries.append(summary)
    data = summary_to_dataframe(summaries, id_name)
    return data


def convert_text_to_csv(file_path: str, id_name: str, save_name: str = ''):
    data = convert_text_to_dataframe(file_path, id_name)
    save_csv(data, f'{file_path}/{save_name}summary.csv', )


def convert_two_text_to_csv(file_path_no_boundary: str, file_path_boundary: str,
                            id_name: str, save_name: str = ''):
    # the txt file without_boundary
    data_without_boundary = convert_text_to_dataframe(file_path_no_boundary, id_name)
    data_without_boundary.rename(columns={'share_of_barrier': 'share_of_barrier (no boundary)',
                                          'nonconvexity': 'nonconvexity (no boundary)',
                                          'number of commuting nodes': 'commuting nodes (no boundary)'},
                                 inplace=True)
    # the txt file without_boundary
    data_with_boundary = convert_text_to_dataframe(file_path_boundary, id_name)
    data_with_boundary.rename(columns={'share_of_barrier': 'share_of_barrier (boundary)',
                                       'nonconvexity': 'nonconvexity (boundary)',
                                       'number of commuting nodes': 'commuting nodes (boundary)'},
                              inplace=True)
    # combine them together
    data = pd.merge(data_without_boundary, data_with_boundary, how='outer', on=id_name)
    save_csv(data, f'{file_path_no_boundary}/{save_name}summary.csv', )


if __name__ == '__main__':
    # Demo
    region_data_file = "../Data/demo_data/center 1-3.shp"
    barrier_data_file = "../Data/demo_data/barrier 1-3.shp"
    id_name = 'areaID'

    save_path = '../result/global_nonconvexity/demo_data-r10km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.5, 10, save_path)
    convert_text_to_csv(save_path, id_name, '../demo_data-r10km nonconvexity ')

    save_path = '../result/global_nonconvexity/demo_data-r5km'
    global_nonconvexity(region_data_file, barrier_data_file, id_name, 0.25, 5, save_path)
    convert_text_to_csv(save_path, id_name, '../demo_data-r5km nonconvexity ')
