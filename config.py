#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/8/24
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : config.py
# @Software : Python3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V1.0
# @License  : None
# @Brief    : 配置文件

import os
import matplotlib
import platform
import pandas as pd
import random
import numpy as np
import torch
from torch.backends import cudnn
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 设置tensorflow输出控制台信息：1等级，屏蔽INFO，只显示WARNING + ERROR + FATAL
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 使用第0/1/2个GPU,-1不使用GPU，使用CPU
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)  # 防止GPU显存爆掉
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 200)
np.set_printoptions(threshold=np.inf)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'  # 保存矢量图中的文本在AI中可编辑


if platform.system() == 'Windows':
    DATA_PATH = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\安中医神经病学研究所合作\data\preprocessed_data"
    DATA_PATH_EXT = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\言语特征可重复性\data\preprocessed_data"
    # font_family = 'Times New Roman'
    font_family = 'Arial'
else:
    DATA_PATH = r"/home/zlzhang/data/WD_PD/data/preprocessed_data"
    DATA_PATH_EXT = r"/home/zlzhang/data/言语特征可重复性/data/preprocessed_data"
    font_family = 'DejaVu Sans'
    ZH_W2V2_BASE_MODEL = "/home/zlzhang/pretrained_models/huggingface_models/chinese-wav2vec2-base"
    ZH_HUBERT_BASE_MODEL = "/home/zlzhang/pretrained_models/huggingface_models/chinese-hubert-base"
matplotlib.rcParams["font.family"] = font_family


def setup_seed(seed: int):
    """
    全局固定随机种子
    :param seed: 随机种子值
    :return: None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = False


rs = 323
setup_seed(rs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
# 为了匹配性别，且尽量匹配年龄而排除的被试ID
excluded_id = ['20211016118', '20211107128',  # WD-female
               '20210619030', '20210619031', '20210924090', '20211007098', '20211008102',
               '20211008104', '20211013116', '20211112129', '20211113130', '20211114132',
               '20220107002', '20220107005', '20220108006', '20220111017', '20220112019',  # HC-male
               '20210912085', '20211007101', '20220114034', '20220115026', '20220115027', '20220322039']  # HC-female

