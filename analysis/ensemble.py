import os
import pandas as pd
import numpy as np


def vote_ensemble(csv_path_list, save_path, key='uuid', col='label'):
    result = {}
    ensemble_list = []
    for csv_path in csv_path_list:
        csv_file = pd.read_csv(csv_path)
        ensemble_list.append(csv_file[col].values.tolist())

    result[key] = csv_file[key].values.tolist()
    vote_array = np.array(ensemble_list)
    result[col] = [
        max(list(vote_array[:, i]), key=list(vote_array[:, i]).count)
        for i in range(vote_array.shape[1])
    ]

    final_csv = pd.DataFrame(result)
    final_csv.to_csv(save_path, index=False)


def prob_ensemble(csv_path_list, save_path, key='uuid',col='label'):
    RULE = [1,3,6,11,12]
    result = {}
    ensemble_list = []
    for csv_path in csv_path_list:
        csv_file = pd.read_csv(csv_path)
        ensemble_list.append(np.asarray(csv_file[[col_name for col_name in csv_file.columns if col_name not in [col,key]]]))
    ensemble_array = np.asarray(ensemble_list)
    print(ensemble_array.shape)
    ensemble_array = np.mean(ensemble_array,axis=0)
    print(ensemble_array)
    ensemble_result = np.argmax(ensemble_array,axis=1)
    result[key] = csv_file[key].values.tolist()
    result[col] = [RULE[case] for case in ensemble_result]

    colname_list = [col_name for col_name in csv_file.columns if col_name not in [col,key]]
    for i,col_name in enumerate(colname_list):
      if col_name not in [col,key]:
          result[col_name] = ensemble_array[...,i]

    final_csv = pd.DataFrame(result)
    final_csv.to_csv(save_path, index=False)

def diff(csv_a,csv_b,col='label'):
    col_a = np.array(pd.read_csv(csv_a)[col])
    col_b = np.array(pd.read_csv(csv_b)[col])
    return len(col_a) - np.sum(col_a==col_b)

if __name__ == "__main__":

    save_path = './result/fusion.csv'
    if os.path.exists(save_path):
        os.remove(save_path)
    dir_list = ['v3.0-roi-new-post-fake-512','v3.0-new-post-fake-512','v3.0-new-post-fake-512-v2']
    dir_list.sort(reverse=True)
    csv_path_list = [os.path.join('./result/',case + '/ave_sub.csv') for case in dir_list]
    print(csv_path_list)
    # vote_ensemble(csv_path_list, save_path)
    prob_ensemble(csv_path_list, save_path)
    print('diff %d with target'%(diff(save_path,'./result/result.csv')))