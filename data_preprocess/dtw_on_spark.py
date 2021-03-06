import sys
sys.path.append(
    '..'
)
from DTW import DTW
import ujson
import random
from multiprocessing import Process,Manager,Queue
from tqdm import tqdm
import pysnooper
from util import haversine
import os
import random


with open('./processed_data/step2_data.json') as f:
    data = ujson.load(f)[:]
    random.shuffle(data)
alist = []

def caculate(x,goal_ptr):
    value_list = []
    ptr_list = []
    data1 = data
    for line,ptr in data1[:100000]:
        if line == x:
            continue
        value,ptr = DTW(line, x)
        if value != 1:
            value_list.append(value)
            ptr_list.append(ptr)
            # if value == 0:
            #     continue
            if value < 0.0002:
                break
    if (not value_list):
        return None
    min_ptr = ptr_list[value_list.index(min(value_list))]
    # if haversine(min_ptr,goal_ptr) > 45000:
    #     return None
    # else:
    return min_ptr

def process_data(data_list,total):
    for line,ptr in tqdm(data_list[:]):
        line.append(ptr)
        hist_ptr = caculate(line,ptr)
        line.pop()
        if hist_ptr:
            total.append([line,ptr,hist_ptr])

# @pysnooper.snoop()
def main():
    pool = []
    total_list = Manager().list()
    length = len(data) // 12
    for i in range(12):
        p = Process(target=process_data,args=(data[length*i:length*(i+1)],total_list,))
        p.start()
        pool.append(p)
    for p in pool:
        p.join()
    with open('./processed_data/step3_data.json','w') as f:
        ujson.dump(list(total_list),f)

if __name__ == '__main__':
    main()
