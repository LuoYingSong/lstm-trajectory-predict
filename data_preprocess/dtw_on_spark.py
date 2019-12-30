import sys
sys.path.append(
    '..'
)
from data_preprocess.DTW import DTW
import json
from pyspark import SparkContext, SparkConf
from data_preprocess.point2road import MIN_LINE_LENGTH

from copy import deepcopy
conf = SparkConf().setAppName('dtw')
sc = SparkContext(conf=conf)
with open('/media/luoyingsong/新加卷/GPS数据/data_preprocess/line2road.json') as f:
    data = json.load(f)
rdd1 = sc.parallelize(data)


def caculate(x):
    value_list = []
    ptr_list = []
    for line in data:
        value,ptr = DTW(line, x)
        if value != 0 and value != 1:
            value_list.append(value)
            ptr_list.append(ptr)
    if not value_list:
        return None
    else:
        return ptr_list[value_list.index(min(value_list))]

data2 = rdd1.map(caculate).collect()

