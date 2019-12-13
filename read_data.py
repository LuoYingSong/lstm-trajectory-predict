import pandas as pd
import numpy as np
from tqdm import tqdm
data = pd.read_csv("./20131204.txt", names=[i for i in range(16)])
data[16] = 1
data2 = data.groupby(2).apply(
    lambda x: pd.DataFrame(
        {'1': np.average(x[2]), '2': np.sum(x[16])}
        , index=[0]))
data3 = data2.sort_values(by='2',ascending=False,axis=0)
for i in tqdm(np.array(data3)):
    if i[1] > 100:
        data[data[2]==i[0]].to_csv("../saver/"+str(int(i[0]))+'_'+str(int(i[1]))+'.txt',index=False,header=False)