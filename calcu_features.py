#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/8/24 09:56
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : calcu_features.py
# @Software : Python3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V1.0 - ZL.Z：2022/8/24 - 2022/8/26
# 		      First version.
# @License  : None
# @Brief    : 特征计算

from config import *
from util import read_csv
import datetime
import glob
from pathos.pools import ProcessPool as Pool
from opensmile import Smile, FeatureSet, FeatureLevel
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, HubertModel
from transformers import logging

logging.set_verbosity_error()


def feat_comp_lld(audio_file: str):
    """
    通过openSmile获取ComParE_2016特征集（基于帧的低阶描述符）
    :param audio_file: 输入音频文件路径
    :return: ComParE_2016特征集（基于帧的低阶描述符），2999*65维（60s音频，每秒50*65），这里修改了配置：窗长30ms,窗移20ms
    原始输出为5996*65维（60s音频，每秒100*65,窗长20ms,窗移10ms），np.ndarray[shape=(序列帧长度,65), dtype=float32]
    """
    smile = Smile(feature_set=os.path.join(current_path, 'data/opensmile/compare/ComParE_2016.conf'),
                  feature_level=FeatureLevel.LowLevelDescriptors)  # 获取LLD
    # smile = Smile(feature_set=FeatureSet.ComParE_2016, feature_level=FeatureLevel.LowLevelDescriptors)
    feat = smile.process_file(audio_file).values
    return feat


def feat_w2v2(audio_file: str):
    """
    基于预训练模型Wav2Vec 2.0 (chinese-wav2vec2-base)获取音频嵌入
    使用 WenetSpeech train_l 集的 1 万小时中文数据作为无监督预训练数据。数据主要来源于 YouTube 和 Podcast
    :ref: https://huggingface.co/TencentGameMate/chinese-wav2vec2-base;
          https://github.com/TencentGameMate/chinese_speech_pretrain
    :param audio_file: 输入音频文件路径
    :return: w2v2嵌入，2999*768维（60s音频，每秒50*768），np.ndarray[shape=(序列帧长度, 768), dtype=float32]
    原始输出为1*2999*768: torch.FloatTensor[cuda, (batch_size, sequence_length, hidden_size)] torch.float32
    """
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ZH_W2V2_BASE_MODEL)
    model = Wav2Vec2Model.from_pretrained(ZH_W2V2_BASE_MODEL)
    model = model.to(device)
    model.eval()  # 仅作test模式
    wav, sr = sf.read(audio_file)
    input_values = feature_extractor(wav, sampling_rate=sr, return_tensors="pt").input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values)
        feat = outputs.last_hidden_state  # 模型最后一层输出的隐藏状态序列作为最终的嵌入
    return feat.cpu().numpy()[0]


def feat_hubert(audio_file: str):
    """
    基于预训练模型HuBERT (chinese-hubert-base)获取音频嵌入
    使用 WenetSpeech train_l 集的 1 万小时中文数据作为无监督预训练数据。数据主要来源于 YouTube 和 Podcast
    :ref: https://huggingface.co/TencentGameMate/chinese-hubert-base;
          https://github.com/TencentGameMate/chinese_speech_pretrain
    :param audio_file: 输入音频文件路径
    :return: hubert嵌入，2999*768维（60s音频，每秒50*768），np.ndarray[shape=(序列帧长度, 768), dtype=float32]
    原始输出为1*2999*768: torch.FloatTensor[cuda, (batch_size, sequence_length, hidden_size)] torch.float32
    """
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ZH_HUBERT_BASE_MODEL)
    model = HubertModel.from_pretrained(ZH_HUBERT_BASE_MODEL)
    # if torch.cuda.device_count() > 1:
    # 	model = torch.nn.DataParallel(model)  # 多gpu训练,自动选择gpu
    model = model.to(device)
    model.eval()  # 仅作test模式
    wav, sr = sf.read(audio_file)
    input_values = feature_extractor(wav, sampling_rate=sr, return_tensors="pt").input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values)
        feat = outputs.last_hidden_state  # 模型最后一层输出的隐藏状态序列作为最终的嵌入
    return feat.cpu().numpy()[0]


def get_features(datasets_dir: str, save_path: str, feat_name: list = None, n_jobs=None):
    """
    计算基于自发言语任务的各类特征
    :param datasets_dir: 输入包含的数据集文件的路径
    :param save_path: 数据特征集及标签的保存路径
    :param feat_name: 待提取的特征名，list类型，仅提取该列表中的特征，默认为None，即提取全部的特征
    :param n_jobs: 并行运行CPU核数，默认为None;若为1非并行，若为-1或None,取os.cpu_count()全部核数,-1/正整数/None类型
    :return: pd.DataFrame，数据的全部特征及其对应标签
    """
    assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (
            n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
    if feat_name is None or feat_name == []:
        fts = ['comp_hsf', 'comp_lld', 'xvec', 'w2v2', 'hubert']
    else:
        fts = feat_name
    assert set(fts).issubset({'comp_hsf', 'comp_lld', 'xvec', 'w2v2', 'hubert'}), \
        f'仅接受以下特证名输入：comp_hsf/comp_lld/xvec/w2v2/hubert'
    audio_f_list = glob.glob(os.path.join(datasets_dir, r'**/*.wav'), recursive=True)
    audio_f_list = [i for i in audio_f_list if os.path.basename(i).split('_')[0] not in excluded_id]

    def extract(audio_file):
        print("---------- Processing %d / %d: %s ----------" %
              (audio_f_list.index(audio_file) + 1, len(audio_f_list), audio_file))
        label = {'HC': 0, 'WD': 1}[audio_file.split(os.path.sep)[-5]]
        sex = {'female': 0, 'male': 1}[os.path.normpath(audio_file).split(os.sep)[-4]]
        subject_id = os.path.basename(audio_file).split('_')[0]
        csv_data = read_csv(os.path.join(os.path.dirname(os.path.dirname(audio_file)), f'{subject_id}.csv'))
        name = csv_data[0][0].split("：")[1]
        age = int(csv_data[0][1].split("：")[1])
        feat_i = pd.DataFrame({'id': [subject_id], 'name': [name], 'age': [age], 'sex': [sex], 'label': [label]})
        for i_ft in fts:
            ft = pd.DataFrame([[eval(f'feat_{i_ft}')(audio_file)]], columns=[i_ft])
            feat_i = pd.concat([feat_i, ft], axis=1)
        return feat_i
    if n_jobs == -1:
        n_jobs = None
    if n_jobs == 1:
        res = []
        for i_subj in audio_f_list:
            res.append(extract(i_subj))
    else:
        with Pool(n_jobs) as pool:
            res = pool.map(extract, audio_f_list)
    feats_all = pd.DataFrame()
    for _res in res:
        feats_all = pd.concat([feats_all, _res], ignore_index=True)
    feats_all.sort_values(by=['label', 'id'], inplace=True, ignore_index=True)
    feats_all.dropna(inplace=True)
    feats_all.drop_duplicates('name', keep='first', inplace=True, ignore_index=True)  # 去掉重复被试数据，仅保留最早测试
    feats_all.drop(columns='name', inplace=True)  # 删除姓名列
    feats_all.to_pickle(os.path.join(save_path, "features.pkl"))
    # feats_all.to_csv(os.path.join(save_path, "features.csv"), encoding="utf-8-sig", index=False)
    # print(feats_all)
    return feats_all


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(
        f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.path.dirname(os.path.realpath(__file__))
    audio_data = os.path.join(current_path, 'data/audio')
    output_path = os.path.join(current_path, r'data/features')
    res_path = os.path.join(current_path, r"results")

    get_features(audio_data, output_path, ['comp_lld', 'w2v2', 'hubert'], 1)

    end_time = datetime.datetime.now()
    print(f"---------- End Time ({os.path.basename(__file__)}): {end_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    print(f"---------- Time Used ({os.path.basename(__file__)}): {end_time - start_time} ----------")
    with open(r"./results/finished.txt", "w") as ff:
        ff.write(f"------------------ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Time Used {end_time - start_time} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
