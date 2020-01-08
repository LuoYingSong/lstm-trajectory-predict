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
with open('./sender.json') as f:
    data = ujson.load(f)[:]

def caculate(x):
    value_list = []
    ptr_list = []
    data1 = random.sample(data,50000)
    for line,ptr in data1:
        value,ptr = DTW(line, x)
        if value != 0 and value != 1:
            value_list.append(value)
            ptr_list.append(ptr)
    if not value_list:
        return None
    else:
        return ptr_list[value_list.index(min(value_list))]

def process_data(data_list,total):
    for line,ptr in tqdm(data_list):
        hist_ptr = caculate(line)
        total.put([line,ptr,hist_ptr])

def main():
    pool = []
    total_list = Queue()
    length = len(data) // 16
    for i in range(16):
        p = Process(target=process_data,args=(data[length*i:length*(i+1)],total_list,))
        p.start()
        pool.append(p)
    for p in pool:
        p.join()
    alist = []
    while not total_list.empty():
        alist.append(total_list.get())
    with open('./goal.json','w') as f:
        ujson.dump(list(alist),f)

if __name__ == '__main__':
    main()