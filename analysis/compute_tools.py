import sys
sys.path.append('..')
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from converter.common_utils import hdf5_reader

def cal_mean_std_single(data_path,shape=(24,256,256)):
    l_mean = []
    l_std = []

    for item in tqdm(data_path):
        img = hdf5_reader(item,'image')
        l_img = (np.array(img).astype(np.float32)/255.0).flatten()
        l_mean.append(np.mean(l_img))
    
    l_mean = np.mean(l_mean)

    for item in tqdm(data_path):
        img = hdf5_reader(item,'image')
        l_img = (np.array(img).astype(np.float32)/255.0).flatten()
        l_std.append(np.mean(np.power(l_img - l_mean,2)))

    l_std = np.sqrt(np.mean(l_std))

    print('l mean:%.3f' % l_mean)
    print('l std:%.3f' % l_std)



if __name__ == '__main__':

    
    input_csv = '../converter/csv_file/index.csv'
    path_list = pd.read_csv(input_csv)['id'].values.tolist()
    cal_mean_std_single(path_list)
