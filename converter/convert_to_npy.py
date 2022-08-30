import os
import pandas as pd
import json
import glob
from PIL import Image
from math import ceil,floor
import numpy as np
from tqdm import tqdm
from common_utils import save_as_hdf5

def convert_train_data(base_dir,save_dir):
    save_dir = os.path.abspath(save_dir)
    data_base_dir = os.path.abspath(base_dir)
    train_img_dir = os.path.join(data_base_dir, 'stage2_train_val')
    train_json_path = os.path.join(data_base_dir, 'stage2_annotation', 'stage2_train_val.json')

    with open(train_json_path, 'r') as train_json_file:
        train_json = json.load(train_json_file)

    rows = []
    for label, clips_dict in train_json.items():
        for video_name, clip_meta in clips_dict.items():
            cur_row = {}
            cur_row['video_name'] = video_name
            cur_row['start_frame'] = clip_meta['start_frame']
            cur_row['end_frame'] = clip_meta['end_frame']
            cur_row['label'] = label
            cur_row['img_dir'] = os.path.join(train_img_dir, video_name)
            cur_row['bbox'] = clip_meta['bbox_x1y1x2y2']
            
            sample_img_path = glob.glob(os.path.join(cur_row['img_dir'], '*.jpg'))[0]
            img = Image.open(sample_img_path)
            cur_row['height'] = img.height
            cur_row['width'] = img.width
            rows.append(cur_row)
    print(len(rows))
    train_df = pd.DataFrame(rows)

    save_row_list = []
    for rowid, row in tqdm(train_df.iterrows(), total=len(train_df)):
        # 1. read clip
        start, end = row['start_frame'], row['end_frame']
        img_filenames = [f'img_{fid:05d}.jpg' for fid in range(start, end+1)]
        img_path_list = [os.path.join(row['img_dir'], fname) for fname in img_filenames]
        imgs = [np.asarray(Image.open(img_path)) for img_path in img_path_list]
        imgs = np.array(imgs)
        
        # 2. crop
        h, w = row['height'], row['width']
        bbox_array = np.array(row['bbox']) # (t, 4), 4: left, top, right, bottom
        bbox_array = (bbox_array * [w, h, w, h])
        left, top, right, bottom = min(bbox_array[:,0]), min(bbox_array[:,1]), max(bbox_array[:,2]), max(bbox_array[:,3])
        left, top, right, bottom = floor(left), floor(top), ceil(right), ceil(bottom)
        imgs = imgs[:, top:bottom, left:right, :]
        
        # 3. save
        save_path = os.path.join(save_dir, row['video_name'], f'{start}-{end}.hdf5')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        save_row = {}
        save_row['id'] = save_path
        save_row['label'] = row['label']
    #     save_row['source'] = row['video_name']
        save_as_hdf5(imgs.astype(np.uint8), save_path, 'image')
        save_row_list.append(save_row)

    train_list_df = pd.DataFrame(save_row_list)    

    train_list_df.to_csv(os.path.join(save_dir, 'index.csv'), index=False)


def convert_test_data(base_dir,save_dir):
    data_base_dir = os.path.abspath(base_dir)
    test_json_path = os.path.join(data_base_dir, 'stage2_annotation', 'stage2_test_nolabel.json')
    test_img_dir = os.path.join(data_base_dir, 'stage2_test')

    with open(test_json_path, 'r') as test_json_file:
        test_json = json.load(test_json_file)

    test_rows = []
    for video_name, clips in test_json.items():
        for clip_range, bbox in clips.items():
            cur_row = {}
            
            start_frame, end_frame = clip_range.split('-')
            cur_row['start_frame'], cur_row['end_frame'] = int(start_frame), int(end_frame)
            cur_row['video_name'] = video_name
            cur_row['img_dir'] = os.path.join(test_img_dir, video_name)
            cur_row['bbox'] = bbox
            
            sample_img_path = glob.glob(os.path.join(cur_row['img_dir'], '*.jpg'))[0]
            img = Image.open(sample_img_path)
            cur_row['height'] = img.height
            cur_row['width'] = img.width
            test_rows.append(cur_row)
    print(len(test_rows))
    test_df = pd.DataFrame(test_rows)

    for rowid, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            # 1. read clip
            start, end = row['start_frame'], row['end_frame']
            img_filenames = [f'img_{fid:05d}.jpg' for fid in range(start, end+1)]
            img_path_list = [os.path.join(row['img_dir'], fname) for fname in img_filenames]
            imgs = [np.asarray(Image.open(img_path)) for img_path in img_path_list]
            imgs = np.array(imgs)

            # 2. crop
            h, w = row['height'], row['width']
            bbox_array = np.array(row['bbox']) # (t, 4), 4: left, top, right, bottom
            bbox_array = np.clip(bbox_array, 0, 1)
            bbox_array = (bbox_array * [w, h, w, h])
            left, top, right, bottom = min(bbox_array[:,0]), min(bbox_array[:,1]), max(bbox_array[:,2]), max(bbox_array[:,3])
            left, top, right, bottom = floor(left), floor(top), ceil(right), ceil(bottom)
            imgs = imgs[:, top:bottom, left:right, :]

            # 3. save
            save_path = os.path.join(save_dir, row['video_name'], f'{start}-{end}.hdf5')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            
            save_as_hdf5(imgs.astype(np.uint8), save_path, 'image')
        except Exception as e:
            print(e)
            print('error: {}-{}-{}'.format(row['video_name'], start, end))



if __name__ == '__main__':
    # base_dir = '../dataset/stage2'
    # save_dir = '../dataset/train_clips_stage2'
    # convert_train_data(base_dir,save_dir)

    base_dir = '../dataset/stage2'
    save_dir = '../dataset/test_clips_stage2'
    convert_test_data(base_dir,save_dir)