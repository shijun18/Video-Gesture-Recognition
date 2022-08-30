import csv
import os
import glob
import pandas as pd

from common_utils import hdf5_reader




def statistic_slice_num(input_path, csv_path):
    '''
    Count the slice number for per sample.
    '''
    info = []
    for subdir in os.scandir(input_path):
        path_list = glob.glob(os.path.join(subdir.path, "*.hdf5"))
        sub_info = [[item, hdf5_reader(item, 'image').shape[0]] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'slice_num']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(csv_path, index=False)


def csv_make(csv_path):
    RULE=[1,3,6,11,12]
    df = pd.read_csv(csv_path)
    df['label'] = df['label'].apply(lambda x: RULE.index(x))
    df.to_csv(csv_path,index=False)



if __name__ == "__main__":
    

    # input_path = os.path.abspath('../dataset/npy_data/')
    # csv_path = './csv_file/slice_number.csv'
    # statistic_slice_num(input_path,csv_path)
    

    csv_path = './csv_file/index_post.csv'
    csv_make(csv_path)
