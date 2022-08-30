import os
import numpy as np
import glob
import argparse
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

from trainer import VolumeClassifier
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, WEIGHT_PATH_LIST,FOLD_NUM,TTA_TIMES,NUM_CLASSES
from converter.common_utils import save_as_hdf5
import random

def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    sample_list = list(set([case.split('/')[-2] for case in path_list]))
    print('sample len:',len(sample_list))
    # sample_list.sort()     
    # random.shuffle(sample_list)   
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    train_path = []
    validation_path = []
    for case in path_list:
        if case.split('/')[-2] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length:", len(train_path),
          "\nVal set length:", len(validation_path))
    return train_path, validation_path


def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print(f'train sample number:{len(train_id)}, val sample number:{len(validation_id)}')
    return train_id, validation_id


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train',
                        choices=["train-cross","train", "inf","inf-cross","inf-tta"],
                        help='choose the mode',
                        type=str)
    parser.add_argument('-s',
                        '--save',
                        default='no',
                        choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not',
                        type=str)
    parser.add_argument('-p',
                        '--path',
                        type=str)
    args = parser.parse_args()

    # Set data path & classifier
    random.seed(42)
    if args.mode != 'train-cross' and args.mode != 'inf-cross':
        classifier = VolumeClassifier(**INIT_TRAINER)
        print(get_parameter_number(classifier.net))

    # Training
    ###############################################
    if 'train' in args.mode:
        ###### modification for new data
        if 'post' in VERSION:
            if 'merge' in VERSION:
                csv_path = './converter/csv_file/index_post_merge.csv'
            else:
                csv_path = './converter/csv_file/index_post.csv'
        else:
            csv_path = './converter/csv_file/index.csv'
        label_dict = {}
        train_label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
        label_dict.update(train_label_dict)
        if 'fake' in VERSION:
            if 'v2' in VERSION:
                fake_csv_path = './converter/csv_file/fake_test_post_v2.csv'
            else:
                fake_csv_path = './converter/csv_file/fake_test_post.csv'
            fake_label_dict = csv_reader_single(fake_csv_path, key_col='id', value_col='label')
            label_dict.update(fake_label_dict)

        path_list = list(label_dict.keys())
        print(f"sample nums:{len(path_list)}")
        random.shuffle(path_list)
        if args.mode == 'train-cross':
            for fold in range(1,FOLD_NUM):
                print('===================fold %d==================='%(fold))
                if INIT_TRAINER['pre_trained']:
                    INIT_TRAINER['weight_path'] = WEIGHT_PATH_LIST[fold-1]
                classifier = VolumeClassifier(**INIT_TRAINER)
                if 'new' not in VERSION:
                    train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM, fold)
                else:
                    train_path, val_path = get_cross_validation(path_list, FOLD_NUM, fold)
                SETUP_TRAINER['train_path'] = train_path
                SETUP_TRAINER['val_path'] = val_path
                SETUP_TRAINER['label_dict'] = label_dict
                SETUP_TRAINER['cur_fold'] = fold

                start_time = time.time()
                classifier.trainer(**SETUP_TRAINER)

                print('run time:%.4f' % (time.time() - start_time))
        
        elif args.mode == 'train':
            if 'new' not in VERSION:
                train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM, CURRENT_FOLD)
            else:
                train_path, val_path = get_cross_validation(path_list, FOLD_NUM, CURRENT_FOLD)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = label_dict
            SETUP_TRAINER['cur_fold'] = CURRENT_FOLD

            start_time = time.time()
            classifier.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))
        ###############################################

    # Inference
    ###############################################
    elif 'inf' in args.mode:
        # RULE = [4, 5, 7, 8, 9, 10, 13, 14, 15, 16]
        RULE=[1, 3, 6, 11, 12]
        # test_id = os.listdir(args.path)
        # test_id.sort(key=lambda x:eval(x.split('.')[0]))
        # test_path = [os.path.join(args.path, case) for case in test_id]
        test_path = glob.glob(os.path.join(os.path.abspath(args.path), '*/*.hdf5'))
        print('test len:',len(test_path))
        #########

        save_dir = './analysis/result/{}'.format(VERSION)
        feature_dir = './analysis/mid_feature/{}'.format(VERSION)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if args.mode == 'inf' or args.mode == 'inf-tta':
            save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}_tta{TTA_TIMES}.csv')
            
            start_time = time.time()
            if args.save == 'no' or args.save == 'n':
                if args.mode == 'inf-tta':
                    result = {}
                    result['prob'],result['pred'] = classifier.inference_tta(test_path, TTA_TIMES)
                else:
                    result, _, _ = classifier.inference(test_path)
                print('run time:%.4f' % (time.time() - start_time))
            else:
                result, feature_in, feature_out = classifier.inference(
                    test_path, hook_fn_forward=True)
                print('run time:%.4f' % (time.time() - start_time))
                # save the avgpool output
                print(feature_in.shape, feature_out.shape)
                feature_save_path = os.path.join(feature_dir,f'fold{str(CURRENT_FOLD)}')
                if not os.path.exists(feature_save_path):
                    os.makedirs(feature_save_path)
                
                for i in range(len(test_path)):
                    name = os.path.basename(test_path[i])
                    feature_path = os.path.join(feature_save_path, name)
                    save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                    save_as_hdf5(feature_out[i], feature_path, 'feature_out')
            
            info = {}
            info['uuid'] = ['/'.join(case.split('.')[0].split('/')[-2:]) for case in test_path]
            info['label'] = [RULE[case] for case in result['pred']]
            for i in range(NUM_CLASSES):
                info[f'prob_{str(i+1)}'] = np.array(result['prob'])[:,i].tolist()
        
            csv_file = pd.DataFrame(info)
            csv_file.to_csv(save_path, index=False)

        

        elif args.mode == 'inf-cross':
            save_path_vote = os.path.join(save_dir,f'vote_sub_tta{TTA_TIMES}.csv')
            save_path = os.path.join(save_dir,f'ave_sub_tta{TTA_TIMES}.csv')
            start_time = time.time()

            result = {
                'pred': [],
                'vote_pred': [],
                'prob': []
            }

            all_prob_output = []
            all_vote_output = []
            
            for fold in range(len(WEIGHT_PATH_LIST)):
                print('===================fold %d==================='%(fold))
                print('weight path %s'%WEIGHT_PATH_LIST[fold])
                INIT_TRAINER['weight_path'] = WEIGHT_PATH_LIST[fold]
                classifier = VolumeClassifier(**INIT_TRAINER)

                prob_output, vote_output = classifier.inference_tta(test_path, TTA_TIMES)
                all_prob_output.append(prob_output)
                all_vote_output.append(vote_output)

            avg_output = np.mean(all_prob_output, axis=0)
            result['prob'].extend(avg_output.tolist())

            result['pred'].extend(np.argmax(avg_output, 1).tolist())
            vote_array = np.asarray(all_vote_output).astype(int)
            result['vote_pred'].extend([max(list(vote_array[:,i]),key=list(vote_array[:,i]).count) for i in range(vote_array.shape[1])])

            print('run time:%.4f' % (time.time()-start_time))

            info = {}
            info['uuid'] =  ['/'.join(case.split('.')[0].split('/')[-2:]) for case in test_path]
            info['label'] = [RULE[case] for case in result['pred']]
            for i in range(NUM_CLASSES):
                info[f'prob_{str(i+1)}'] = np.array(result['prob'])[:,i].tolist()
            csv_file = pd.DataFrame(info)
            csv_file.to_csv(save_path, index=False)

            info = {}
            info['uuid'] =  ['/'.join(case.split('.')[0].split('/')[-2:]) for case in test_path]
            info['label'] = [int(case) for case in result['vote_pred']]
            for i in range(NUM_CLASSES):
                info[f'prob_{str(i+1)}'] = np.array(result['prob'])[:,i].tolist()
            csv_file = pd.DataFrame(info)
            csv_file.to_csv(save_path_vote, index=False)

    ###############################################
