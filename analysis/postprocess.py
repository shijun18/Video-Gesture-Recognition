import os
import pandas as pd
import numpy as np


RULE = [1,3,6,11,12]
MAP = {
  3:6,
  6:3,
  11:12,
  12:11,
  1:12
}

def find_repeat(df_data):
  # print(df_data)
  rule = [1,3,6,11,12]
  repeat_dict = {}
  for label in rule:
    repeat_index = df_data[df_data['label'] == label].index
    if len(repeat_index) >=2:
      repeat_dict[label] = repeat_index
  return repeat_dict

def diff(csv_a,csv_b,col='label'):
    col_a = np.array(pd.read_csv(csv_a)[col])
    col_b = np.array(pd.read_csv(csv_b)[col])
    return len(col_a) - np.sum(col_a==col_b)

csv_path = './result/v3.0-new-post-fake-512-v2/ave_sub_tta5.csv'
# csv_path = './result/fusion.csv'
pred_df = pd.read_csv(csv_path)
save_path = csv_path.replace('.csv','-mod.csv')

pred_df['id'] = pred_df['uuid']
pred_df['video_source'] = [case.split('/')[0] for case in pred_df['uuid']]
video_source = list(set(pred_df['video_source'].values.tolist()))
# print(pred_df)
count = 0
video_source.sort()
for item in video_source:
  frame_index = pred_df[pred_df['video_source'] == item].index
  # print(frame_index)
  frame_data = pred_df.iloc[frame_index]
  # print(frame_data)
  repeat_dict = find_repeat(frame_data)
  # print(repeat_dict)
  
  if repeat_dict:
    print(frame_data)
    print(repeat_dict)
    for key in repeat_dict.keys():
      temp_index = repeat_dict[key]
      prob_array = pred_df.iloc[temp_index][f'prob_{RULE.index(key)+1}']
      print(prob_array)
      fixed_index = prob_array.idxmax()
      # abs_val = abs(np.max(np.diff(np.array(prob_array))))
      # print(abs_val)
      for index in temp_index:
        if index != fixed_index:
          pred_df.loc[index,'label'] = MAP[key]
      # print(pred_df.iloc[temp_index])
      count += 1
  else:
    continue


pred_df[['id','label']].to_csv(save_path,index=False)
print('adjust num: %d'%count)
print('diff %d with target'%(diff(save_path,'./result/result.csv')))