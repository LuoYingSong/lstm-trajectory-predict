from math import radians, cos, sin, asin, sqrt
import numpy as np
from data_preprocess.data_process import MAP_SLICE_X,MAP_SLICE_Y
import pickle

with open('../data_preprocess/scaler' ,'rb') as f:
    scaler = pickle.load(f)

def haversine(ptr1,ptr2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    ptr1,ptr2 = scaler.inverse_transform([ptr1,ptr2])
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
    print(np.array([haversine(i, j) for i, j in zip([[0.7925792817795809, 0.1268810311086952],
   [0.7944621068693323, 0.12630744314728304],
   [0.7952355605885941, 0.12650988595720492],
   [0.7952799354434319, 0.1267065928875013],
   [0.7956158530943185, 0.12839935218298137],
   [0.7979868015869442, 0.13022690464941888],
   [0.7993877157532552, 0.13091301707265757],
   [0.7998694048022799, 0.1306879681489761],
   [0.8003005065166917, 0.13037873675682476],
   [0.8008838139832051, 0.13040117416829844],
   [0.8032927029765915, 0.13174522572370506],
   [0.8032927029765915, 0.13174522572370506],
   [0.8043576994921295, 0.1319139280653019],
   [0.8063862959797916, 0.13208263040689872],
   [0.808078752942265, 0.13211080369794104]], [[0.9925792817795809, 0.1268810311086952],
   [0.7944621068693323, 0.12630744314728304],
   [0.7952355605885941, 0.12650988595720492],
   [0.7952799354434319, 0.1267065928875013],
   [0.7956158530943185, 0.12839935218298137],
   [0.7979868015869442, 0.13022690464941888],
   [0.7993877157532552, 0.13091301707265757],
   [0.7998694048022799, 0.1306879681489761],
   [0.8003005065166917, 0.13037873675682476],
   [0.8008838139832051, 0.13040117416829844],
   [0.8032927029765915, 0.13174522572370506],
   [0.8032927029765915, 0.13174522572370506],
   [0.8043576994921295, 0.1319139280653019],
   [0.8063862959797916, 0.13208263040689872],
   [0.808078752942265, 0.13211080369794104]])]).mean())
