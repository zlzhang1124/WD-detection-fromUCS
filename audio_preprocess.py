#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/8/23 9:06
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : audio_preprocess.py
# @Software : Python3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V1.0: 2022/8/23
#             First version.
# @License  : None
# @Brief    : 自发言语音频预处理

from config import *
from util import *
import subprocess
import glob
import shutil
import parselmouth
from parselmouth.praat import call
import datetime
from pathos.pools import ProcessPool as Pool


def audio_preprocess(audio_file: str, output_folder: str, start=0.0, denoise=False):
    """
    音频预处理：包括格式转换、嘀声删除以及降噪
    :param audio_file: 待处理音频文件
    :param output_folder: 输出文件夹
    :param start: 删除前start s的音频（数据收集问题，导致将嘀声收集到原始文件，此时要删除）
    :param denoise: 是否进行降噪，由于降噪对于不同音频效果有差异，默认否
    :return: None
    """
    print(audio_file + "音频数据处理中...")
    if not os.path.exists(audio_file):
        raise FileExistsError(audio_file + "输入文件不存在！")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    audio_name = os.path.basename(audio_file)
    output_audio = os.path.join(output_folder, audio_name)
    temp_audio = os.path.join(output_folder, "temp_" + audio_name)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    print("----------{} STEP1: 音频格式转换----------".format(audio_name))
    # 调用ffmpeg，将任意格式音频文件从start到最后转换为.wav文件，pcm有符号16bit,1：单通道,16kHz，不显示打印信息
    subprocess.run("ffmpeg -loglevel quiet -ss %f -y -i %s -acodec pcm_s16le -ac 1 -ar 16000 %s" %
                   (start, audio_file, temp_audio), shell=True)
    print("----------{} STEP2: 降噪----------".format(audio_name))
    sound_obj = parselmouth.Sound(temp_audio)
    sound_obj.subtract_mean()  # 消除直流分量
    if denoise:
        sound_denoised_obj = call(sound_obj, "Reduce noise", 0.0, 0.0, 0.025, 80.0,
                                  10000.0, 40.0, -20.0, "spectral-subtraction")
        sound_obj_de = sound_denoised_obj
    else:
        sound_obj_de = sound_obj
    print("----------{} STEP3: 分割----------".format(audio_name))
    # 确保60秒
    sound_obj_de = sound_obj_de.extract_part(0.0, 60.0, parselmouth.WindowShape.RECTANGULAR, 1.0, False)
    sound_obj_de.save(output_audio, parselmouth.SoundFileFormat.WAV)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)


def run_audio_preprocess_parallel(original_path: str, preprocessed_path: str, n_jobs=None):
    """
    并行运行音频预处理
    :param original_path: 原始数据文件路径
    :param preprocessed_path: 预处理保存数据文件路径
    :param n_jobs: 并行运行CPU核数，默认为None，取os.cpu_count()全部核数,-1/正整数/None类型
    :return: None
    """
    assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
    if n_jobs == -1:
        n_jobs = None
    for each_file in os.listdir(original_path):
        if each_file == "HC" or each_file == "WD":
            preprocessed_p = os.path.join(preprocessed_path, each_file)
            if not os.path.exists(preprocessed_p):
                os.makedirs(preprocessed_p)
            data_path = os.path.join(original_path, each_file)
            for root, dirs, files in os.walk(data_path):
                def parallel_process(name):
                    if (each_file == "HC") and name.startswith('2021') and \
                            (name.split('.')[0].split('_')[0][-3:] in duplicated_data):
                        return  # 跳过重复的数据
                    if os.path.join('session_1', '12_SI') in os.path.join(root, name):
                        if name.endswith('.wav'):  # 遍历处理.wav文件
                            wav_file = os.path.join(root, name)
                            if '12_SI' in wav_file:
                                output_path = root.replace(os.path.abspath(original_path),
                                                           os.path.abspath(preprocessed_path)).split('session_1')[0] + 'SI'
                                if not os.path.exists(output_path):
                                    os.makedirs(output_path)
                                audio_preprocess(wav_file, output_path)
                                try:  # 将csv文件复制到目标文件夹下
                                    csv_file = glob.glob(os.path.join(root.split('12_SI')[0], '*.csv'))[0]
                                    dst_csv_file = os.path.join(os.path.dirname(output_path),
                                                                os.path.basename(csv_file))
                                    if not os.path.exists(dst_csv_file):
                                        shutil.copy(csv_file, dst_csv_file)
                                except IndexError:
                                    pass
                    elif ('session_2' not in os.path.join(root, name)) and ('14_SI' in os.path.join(root, name)):
                        _csv_data = read_csv(glob.glob(os.path.join(root.split('14_SI')[0], '*.csv'))[0])
                        if int(_csv_data[0][1].split("：")[1]) < 18:
                            return
                        if name.endswith('.wav'):
                            wav_file = os.path.join(root, name)
                            if '14_SI' in wav_file:
                                output_path = root.replace(os.path.abspath(original_path),
                                                           os.path.abspath(preprocessed_path)).split('14_SI')[0] + 'SI'
                                if not os.path.exists(output_path):
                                    os.makedirs(output_path)
                                audio_preprocess(wav_file, output_path)
                                try:  # 将csv文件复制到目标文件夹下
                                    csv_file = glob.glob(os.path.join(root.split('14_SI')[0], '*.csv'))[0]
                                    dst_csv_file = os.path.join(os.path.dirname(output_path),
                                                                os.path.basename(csv_file))
                                    if not os.path.exists(dst_csv_file):
                                        shutil.copy(csv_file, dst_csv_file)
                                except IndexError:
                                    pass

                # 使用设定数量的CPU核数（这里有闭包，不可pickle，因此不能用multiprocessing中的Pool，这里用pathos）
                with Pool(n_jobs) as pool:
                    pool.map(parallel_process, files)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    original_data = DATA_PATH  # 原始数据文件夹
    original_data = DATA_PATH_EXT  # 原始数据文件夹-额外的补充数据
    current_path = os.path.dirname(os.path.realpath(__file__))
    preprocessed_data = os.path.join(current_path, "data/audio")
    duplicated_data = ['113', '141', '089', '139', '092', '142', '103', '143',
                       '140', '115', '114', '135', '136', '134']  # 重复数据的编号
    # if os.path.exists(preprocessed_data):
    #     shutil.rmtree(preprocessed_data)
    run_audio_preprocess_parallel(original_data, preprocessed_data, n_jobs=-1)

    subid, nam, gender, age, group = [], [], [], [], []
    for i_f in glob.iglob(os.path.join(preprocessed_data, r"**/*.csv"), recursive=True):
        csv_data = read_csv(i_f)
        subid.append(os.path.basename(i_f).rstrip('.csv'))
        nam.append(csv_data[0][0].split("：")[1])
        gender.append({"male": 1, "female": 0}[os.path.normpath(i_f).split(os.sep)[-3]])
        age.append(int(csv_data[0][1].split("：")[1]))
        group.append(os.path.normpath(i_f).split(os.sep)[-4])
    subinfo = pd.DataFrame({"id": subid, "name": nam, "gender": gender, "age": age, "group": group})
    # subinfo.to_csv(os.path.join(current_path, 'data/subinfo.csv'), index=False, encoding='utf-8-sig')
    subinfo[~subinfo['id'].isin(excluded_id)].to_csv(os.path.join(current_path, 'data/subinfo_matched.csv'),
                                                     index=False, encoding='utf-8-sig')  # 性别和年龄匹配后的数据
    # 查看数据重复情况，以便后续删除重复数据
    from collections import Counter
    print(len(nam), len(set(nam)))  # 包括重复的数据量、非重复数据量（最后要保证这两值相等）
    print('数据重复情况：', {key: value for key, value in dict(Counter(nam)).items() if value > 1})  # 展现重复的人名和重复次数

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

