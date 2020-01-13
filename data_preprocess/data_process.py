import os
import pandas as pd
import numpy as np
import ujson
import time
from multiprocessing import Process, Manager
import pysnooper
import random
from tqdm import tqdm
import datetime


SPEED_THRESHOLD = 1 # 只取大于这个值得点
FILE_DIR = '/home/wangxing/saver'  # 数据源路径
SAVER_PATH = os.path.join('.', "processed_data")  # 数据保存路径
LINE_LENGTH = 10  # 每个步长
BACKWORD_NUM = 7  # 路径回溯长度
GET_DATA_COL_INDEX = [4, 5]  # 经纬度0-MAX_LICE 速度MAX_SLICE+24 -   方向MAX_SLICE+1-MAX_SLICE+8
# 载客MAX_SLICE+9,MAX_SLICE+10 时刻划分MAX_SLICE+11 , MAX_SLICE+16 星期几MAX_SLICE+17-MAX_SLICE+23 输入值
OUTPUT_DATA_COL_INDEX = [4, 5]  # 经纬度 输出值
MAP_SLICE_X = 200  #
SAME_TIME_STEP_SECOND = (-1,60)  # 时间点间隔取值 超过间隔视为不再是相同的点
GET_DATA_TIME_STEP = (25, 32)  # 下个点的区间只有这个区间内的才能被取到
MINI_STEP_CONTAIN_PTR_NUM = 15
PROCESS_NUM = 16 # 处理进程数量
MAP_SLICE_Y = 200
DIFFERENT_PTR_THRESHOLD = 0.006  # 点的偏差
MAX_SPEED = 120
DIFFERENT_PTR_THRESHOLD_Y = 0.036
HAS_COL = len(GET_DATA_COL_INDEX)
GET_PTR_TIME_AREA = [-1,24]
num2vector_y = {0: 1, 1: 1, 2: 0, 3: -1, 4: -1, 5: -1, 6: 0, 7: 1}
num2vector_x = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: -1, 6: -1, 7: -1}
max_x, max_y, min_x, min_y = 119.376, 26.1544, 119.226, 25.985


# {0: [1, 0], 1: [1, 1], 2: [0, 1], 3: [-1, 1], 4: [-1, 0], 5: [-1, -1], 6: [0, -1], 7: [1, -1]}

def new2old_df(df):
    df.columns = [0, 4, 5, 6, 'time', 1, 3, 8]
    def func(x):
        time_ = x['time']
        date = datetime.datetime.strptime(time_, '%Y-%m-%d %H:%M:%S')
        return pd.Series([date.year, date.month, date.day, date.hour, date.minute, date.second])

    df[[9, 10, 11, 12, 13, 14]] = df.apply(func, axis=1)
    del df['time']
    return df


def pre_process_data(start,end = 999999):
    '''
    筛选小于
    :return:
    '''
    all_df_list = Manager().list()
    file_list = os.listdir(FILE_DIR)
    random.shuffle(file_list)
    if end > start + 18000:
        file_list = file_list[start:start + 18000]
    else:
        file_list = file_list[start:end]
    process_pool = []
    file_length_slice = len(file_list) // PROCESS_NUM
    for i in range(PROCESS_NUM):
        file_content = file_list[file_length_slice * i:file_length_slice * (i + 1)]
        p = Process(target=_file_pre_process, args=(file_content, min_x, min_y, max_x, max_y, all_df_list,))
        p.start()
        process_pool.append(p)
    for i in process_pool:
        i.join()
    return list(all_df_list)


def _file_pre_process(file_list, min_x, min_y, max_x, max_y, all_df_list):
    for file in tqdm(file_list):
        if file.split('.')[-1] in ("txt", "csv"):
            df = pd.read_csv(os.path.join(FILE_DIR, file), names=[i for i in range(8)])
            try:
                df = new2old_df(df)
            except ValueError:
                continue
            df = df[(df[4] > min_x) & (df[5] > min_y) & (df[4] < max_x) & (df[5] < max_y)]
            if df.shape[0]:
                try:
                    df = df[(df[6] > SPEED_THRESHOLD) & (df[6] < MAX_SPEED)]
                    # df[4] = df.apply(lambda x: (x[4] - min_x) / (max_x-min_x), axis=1)
                    # df[4] = df.apply(lambda x: int(x[4] / (1 / MAP_SLICE_X) + 0.5) * (1 / MAP_SLICE_X), axis=1)
                    # df[5] = df.apply(lambda x: (x[5] - min_y) / (max_y-min_y), axis=1)
                    # df[5] = df.apply(lambda x: int(x[5] / (1 / MAP_SLICE_Y) + 0.5) * (1 / MAP_SLICE_Y), axis=1)
                    all_df_list.append(df)
                except ValueError or TypeError:
                    continue

def _get_time_stamp(minute, second):
    return minute * 60 + second


def make_ptr_to_line(df_saver, all_data_list):
    '''
    数据结构 [((now_path ),(next_path,1ptr))] 储存一个训练集   （后面只应该是点）
    :param df_saver:
    :return:
    '''
    count = 0
    print(os.getpid())
    for df in tqdm(df_saver):
        if not df.shape[0]:
            continue
        split_list = find_eroor_ptr(df)
        for i,j in zip(split_list[:-1],split_list[1:]):
            while j - i > LINE_LENGTH:
                data = df.loc[i:i+LINE_LENGTH,GET_DATA_COL_INDEX].values.tolist()
                label = df.loc[i+LINE_LENGTH,GET_DATA_COL_INDEX].values.tolist()
                all_data_list.append((data, label))
                i += BACKWORD_NUM
    print(len(all_data_list))


def data_saver(data_list):
    if not os.path.exists(SAVER_PATH):
        os.mkdir(SAVER_PATH)
    print(len(data_list))
    print(os.path.join(SAVER_PATH, 'processed_data.json'))
    with open(os.path.join(SAVER_PATH, 'processed_data2.json'), "w") as f:
        ujson.dump(data_list, f)


# @pysnooper.snoop()
def main(total):
    start = 0
    all_data = []
    while start < total:
        data_queue = Manager()
        all_data_list = data_queue.list()
        df_saver = pre_process_data(start,total)
        if not df_saver:
            break
        random.shuffle(df_saver)
        slice = len(df_saver) // PROCESS_NUM
        process_pool = []
        for i in range(PROCESS_NUM):
            p = Process(target=make_ptr_to_line, args=(df_saver[slice * i:slice * (i + 1)], all_data_list,))
            p.start()
            process_pool.append(p)
        for i in process_pool:
            i.join()
        all_data_list = list(all_data_list)
        all_data += all_data_list
        start += 18000
    data_saver(list(all_data))

def find_eroor_ptr(df):
    time_list = df[[13,14]]
    x, y = df[4], df[5]
    old_index = df.index[0]
    old_time, old_x, old_y = _get_time_stamp(*time_list.loc[df.index[0]]), df.loc[df.index[0],4], df.loc[df.index[0],5]
    split_list = list(set(range(max(df.index)+1)).difference(set(df.index)))
    for i in df.index[1:]:
        new_time, new_x, new_y = _get_time_stamp(*time_list.loc[i]), df.loc[i,4], df.loc[i,5]
        step = new_time - old_time
        if i - old_index != 1:
            old_index = i
            continue
        if step < 0:
            step += 3600
        if max(SAME_TIME_STEP_SECOND) > step > min(SAME_TIME_STEP_SECOND) and \
                abs(new_x - old_x) < DIFFERENT_PTR_THRESHOLD and abs(new_y - old_y) < DIFFERENT_PTR_THRESHOLD_Y:
            pass
        else:
            split_list.append(i)
        old_time, old_index, old_x, old_y = new_time, i, new_x, new_y
    if not split_list:
        split_list.append(0)
        split_list.append(max(df.index))
    return sorted(split_list)


if __name__ == '__main__':
    # time.sleep(10000)
    main(1000000000000000)
