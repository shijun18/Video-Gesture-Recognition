from utils import get_weight_path,get_weight_list

__all__ = ['mvit_base_32x3','x3d_l','csn_r101']


EXTERNAL_PATH = {
    'mvit_base_32x3':'./pretrained_model/MVIT_B_32x3_f294077834.pyth',
    'x3d_l':'./pretrained_model/X3D_L.pyth',
    'csn_r101':'./pretrained_model/CSN_32x2_R101.pyth' # './pretrained_model/CSN_32x2_R101.pyth'
}

NET_NAME = 'csn_r101'
VERSION = 'v3.0-new-post-fake-512-v2'
DEVICE = '0'
# Must be True when pre-training and inference
PRE_TRAINED = True 
# 1,2,3,4,5
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
TTA_TIMES = 5
NUM_CLASSES = 10 if 'post' not in VERSION else 5

CKPT_PATH = './ckpt/{}/fold{}'.format(VERSION,CURRENT_FOLD)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
# WEIGHT_PATH = get_weight_path('./ckpt/v3.0-new-post-fake-512/fold3')
# print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/'.format(VERSION))
    # WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/'.format('v3.0-roi-new'))
    # WEIGHT_PATH_LIST = [get_weight_path('./ckpt/v3.0-new-post-fake-512/fold3')]*5
    # WEIGHT_PATH_LIST = ['./pretrained_model/saohui_csn_r101_gesture.ckpt']*5
else:
    WEIGHT_PATH_LIST = None

# mean = 0.449
# std = 0.250
INPUT_SHAPE = {
    'mvit_base_32x3':(32,224,224),
    'x3d_l':(16,312,312),
    'csn_r101':(32,256,256) if '512' not in VERSION else (32,512,512)
}

TEMPORAL_FRAME = {
    'mvit_base_32x3':32,
    'x3d_l':16,
    'csn_r101':32
}
# Arguments when trainer initial
INIT_TRAINER = {
    'net_name':NET_NAME,
    'lr':1e-3, 
    'n_epoch':120,
    'channels':1 if 'roi' in VERSION else 3,
    'num_classes':NUM_CLASSES,
    'input_shape':INPUT_SHAPE[NET_NAME],
    'crop':0,
    'scale':None,
    'mean':None,
    'std':None,
    'use_roi':False or 'roi' in VERSION,
    'temporal_frame':TEMPORAL_FRAME[NET_NAME],
    'batch_size':16,
    'num_workers':max(4,GPU_NUM*4),
    'device':DEVICE,
    'pre_trained':PRE_TRAINED,
    'external_pretrained_path':EXTERNAL_PATH[NET_NAME],
    'weight_path':WEIGHT_PATH,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'gamma': 0.1,
    'milestones': [30,60,90],
    'T_max':5,
    'use_fp16':True,
    'use_maxpool': False or 'maxpool' in VERSION 
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
    'output_dir':'./ckpt/{}'.format(VERSION),
    'log_dir':'./log/{}'.format(VERSION),
    'optimizer':'AdamW',
    'loss_fun':'Cross_Entropy',
    'class_weight':None,
    'lr_scheduler':'MultiStepLR', # MultiStepLR
    'repeat_factor':1.0
}

