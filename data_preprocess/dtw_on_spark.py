import sys
sys.path.append(
    '..'
)
from data_preprocess.DTW import DTW
import json
from pyspark import SparkContext, SparkConf

from copy import deepcopy
conf = SparkConf().setAppName('dtw')
sc = SparkContext(conf=conf)
with open('/media/luoyingsong/新加卷/GPS数据/data_preprocess/line2road2.json') as f:
    data = json.load(f)
rdd1 = sc.parallelize(data)



def caculate(x):
    total_list = []
    for line in data:
        dat = DTW(line, x)
        total_list.append(dat)
    total_list.remove(0)
    return min(total_list)

print(rdd1.map(caculate).collect())
