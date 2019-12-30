import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Process
DIR_LIST = ['~/saver2/data 5.1-5.10',
            '~/saver2/data 5.11-5.20',
            '~/saver2/data 5.21-5.30']

def runner(path):
    if file.endswith('.csv'):
        print(file)
        try:
            data = pd.read_csv(path,low_memory=False)
            index_dict = data.groupby('F_ID').groups
        except KeyError:
            return
        for i in index_dict:
            index = index_dict[i]
            save_data = data.iloc[index]
            if save_data.shape[0] > 100:
                save_data = save_data.sort_values('F_TIME')
                save_data.to_csv("~/saver/" + str(int(i)) + '_' + str(save_data.shape[0]) + '.txt', index=False,
                        header=False)

if __name__ == '__main__':
    pool = []
    for file_box in tqdm(os.listdir(dir)):
        file_box = os.path.join(dir,file_box)
        for file in os.listdir(file_box):
            path = os.path.join(file_box,file)
            t = Process(target=runner,args=(path,))
            t.start()
            pool.append(t)
    for t in pool:
        t.join()