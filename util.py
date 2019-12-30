from math import radians, cos, sin, asin, sqrt
import numpy as np
from data_preprocess.data_process import MAP_SLICE_X,MAP_SLICE_Y

def haversine(ptr1,ptr2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = ptr1[0],ptr1[1],ptr2[0],ptr2[1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def recover(series):
    max_x, max_y, min_x, min_y = 119.376, 26.1544, 119.226, 25.985
    alist = []
    for i in series:
        x = i[0]*(max_x-min_x) + min_x
        y = i[1]*(max_y-min_y) + min_y
        alist.append([x,y])
    return np.array(alist)

trans2network_x = lambda x: int(x / (1 / MAP_SLICE_X) + 0.5) * (1 / MAP_SLICE_X)

trans2network_y = lambda x: int(x / (1 / MAP_SLICE_Y) + 0.5) * (1 / MAP_SLICE_Y)

def equal(std, predict) -> float:
    print(np.array(predict)[:,0])
    predict_X = list(map(trans2network_x,np.array(predict)[:,0]))
    predict_Y = list(map(trans2network_y,np.array(predict)[:,1]))
    predict = np.array([[i,j] for i, j in zip(predict_X,predict_Y)])
    reason = predict == std
    result = reason[:,0] & reason[:,1]
    return result.mean()



if __name__ == '__main__':
    print(haversine([119.30124899, 26.118093000000002], [119.30124199000001, 26.11869801]))
