import os
import yaml
from easydict import EasyDict as edict

config = edict()

# ------config for general parameters------
config.GPUS = "0,1,2,3"
config.WORKERS = 32
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'logs'
config.CHECKPOINT_DIR = 'snapshot'

config.OCEANPLUS = edict()
config.OCEANPLUS.TRAIN = edict()
config.OCEANPLUS.TEST = edict()
config.OCEANPLUS.TUNE = edict()
config.OCEANPLUS.DATASET = edict()
config.OCEANPLUS.DATASET.VID = edict()
config.OCEANPLUS.DATASET.GOT10K = edict()
config.OCEANPLUS.DATASET.COCO = edict()
config.OCEANPLUS.DATASET.DET = edict()
config.OCEANPLUS.DATASET.LASOT = edict()
config.OCEANPLUS.DATASET.YTB = edict()
config.OCEANPLUS.DATASET.VISDRONE = edict()

# augmentation
config.OCEANPLUS.DATASET.SHIFT = 4
config.OCEANPLUS.DATASET.SCALE = 0.05
config.OCEANPLUS.DATASET.COLOR = 1
config.OCEANPLUS.DATASET.FLIP = 0
config.OCEANPLUS.DATASET.BLUR = 0
config.OCEANPLUS.DATASET.GRAY = 0
config.OCEANPLUS.DATASET.MIXUP = 0
config.OCEANPLUS.DATASET.CUTOUT = 0
config.OCEANPLUS.DATASET.CHANNEL6 = 0
config.OCEANPLUS.DATASET.LABELSMOOTH = 0
config.OCEANPLUS.DATASET.ROTATION = 0
config.OCEANPLUS.DATASET.SHIFTs = 64
config.OCEANPLUS.DATASET.SCALEs = 0.18

# vid
config.OCEANPLUS.DATASET.VID.PATH = '$data_path/vid/crop511'
config.OCEANPLUS.DATASET.VID.ANNOTATION = '$data_path/vid/train.json'

# got10k
config.OCEANPLUS.DATASET.GOT10K.PATH = '$data_path/got10k/crop511'
config.OCEANPLUS.DATASET.GOT10K.ANNOTATION = '$data_path/got10k/train.json'
config.OCEANPLUS.DATASET.GOT10K.RANGE = 100
config.OCEANPLUS.DATASET.GOT10K.USE = 200000

# visdrone
config.OCEANPLUS.DATASET.VISDRONE.ANNOTATION = '$data_path/visdrone/train.json'
config.OCEANPLUS.DATASET.VISDRONE.PATH = '$data_path/visdrone/crop271'
config.OCEANPLUS.DATASET.VISDRONE.RANGE = 100
config.OCEANPLUS.DATASET.VISDRONE.USE = 100000

# train
config.OCEANPLUS.TRAIN.GROUP = "resrchvc"
config.OCEANPLUS.TRAIN.EXID = "setting1"
config.OCEANPLUS.TRAIN.MODEL = "OceanPlus"##应当与models.models里面的类名对应上
config.OCEANPLUS.TRAIN.RESUME = False
config.OCEANPLUS.TRAIN.START_EPOCH = 0
config.OCEANPLUS.TRAIN.END_EPOCH = 50
config.OCEANPLUS.TRAIN.TEMPLATE_SIZE = 127
config.OCEANPLUS.TRAIN.SEARCH_SIZE = 255
config.OCEANPLUS.TRAIN.STRIDE = 8
config.OCEANPLUS.TRAIN.BATCH = 32
config.OCEANPLUS.TRAIN.PRETRAIN = 'pretrain.model'
config.OCEANPLUS.TRAIN.LR_POLICY = 'log'
config.OCEANPLUS.TRAIN.LR = 0.001
config.OCEANPLUS.TRAIN.LR_END = 0.00001
config.OCEANPLUS.TRAIN.MOMENTUM = 0.9
config.OCEANPLUS.TRAIN.WEIGHT_DECAY = 0.0001
config.OCEANPLUS.TRAIN.WHICH_USE = ['GOT10K']  # VID or 'GOT10K'

# test
config.OCEANPLUS.TEST.MODEL = config.OCEANPLUS.TRAIN.MODEL
config.OCEANPLUS.TEST.DATA = 'VOT2019'
config.OCEANPLUS.TEST.START_EPOCH = 30
config.OCEANPLUS.TEST.END_EPOCH = 50

# tune
config.OCEANPLUS.TUNE.MODEL = config.OCEANPLUS.TRAIN.MODEL
config.OCEANPLUS.TUNE.DATA = 'VOT2019'
config.OCEANPLUS.TUNE.METHOD = 'TPE'  # 'GENE' or 'RAY'



def _update_dict(k, v, model_name):
    if k in ['TRAIN', 'TEST', 'TUNE']:
        for vk, vv in v.items():
            config[model_name][k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'COCO', 'DET', 'YTB', 'LASOT']:
                config[model_name][k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    try:
                        config[model_name][k][vk][vvk] = vvv
                    except:
                        config[model_name][k][vk] = edict()
                        config[model_name][k][vk][vvk] = vvv

    else:
        config[k] = v   # gpu et.


def update_config(config_file):
    """
    ADD new keys to config
    """
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))## 加载外部配置文件
        model_name = list(exp_config.keys())[0]## 配置文件中第一个关键字就是模型的名字
        if model_name not in ['OCEANPLUS', 'SIAMRPN']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:## 用外部配置文件来更新本文件中的config
                _update_dict(k, v, model_name)   # k=OCEAN or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
