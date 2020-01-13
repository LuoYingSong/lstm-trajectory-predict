import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import threading
DIR_LIST = ['/media/luoyingsong/00BE4ADDBE4ACB3C/data 5.1-5.10',
            '/media/luoyingsong/5CC8F98BC8F963A4/data 5.11-5.20',
            '/media/luoyingsong/2686115886112A37/data 5.21-5.30']

def runner(dir):
    for file_box in tqdm(os.listdir(dir)):
        file_box = os.path.join(dir,file_box)
        for file in os.listdir(file_box):
            if file.endswith('.csv'):
                path = os.path.join(file_box,file)
                data = pd.read_csv(path, names=[i for i in range(16)],low_memory=False)
                data[16] = 1
                data2 = data.groupby(2).apply(
                    lambda x: pd.DataFrame(
                        {'1': np.average(x[2]), '2': np.sum(x[16])}
                        , index=[0]))
                data3 = data2.sort_values(by='2',ascending=False,axis=0)
                for i in tqdm(np.array(data3)):
                    if i[1] > 100:
                        data[data[2]==i[0]].to_csv("../../saver/"+str(int(i[0]))+'_'+str(int(i[1]))+'.txt',index=False,header=False)

if __name__ == '__main__':
    pass
    # pool = []
    # for dir in DIR_LIST:
    #     t = threading.Thread(target=runner,args=(dir,))
    #     t.start()
    #     pool.append(t)
    # for t in pool:
    #     t.join()