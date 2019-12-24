import numpy as np
import json
import math

FIND_LINE_NUM = 20
DIFF_ROAD_DIST = 0.0001


class RoadNotFoundException(Exception):
    pass


def load_road_data():
    with open('road_dict', 'r') as f:
        data = json.load(f)
    return dict(zip(map(lambda x: eval(x), data.keys()), data.values()))


def load_line_data():
    with open('processed_data/processed_data.json', 'r') as f:
        data = json.load(f)
    return data


def dist(ptr, ptr2):
    return math.sqrt((ptr[0] - ptr2[0]) ** 2 + (ptr[1] - ptr2[1]) ** 2)


def map2road(ptr, road_dict, find_line_num):
    line_id = {}
    min_dict = {}
    for line_ptr in road_dict.keys():
        min_dict[dist(ptr, line_ptr)] = line_ptr
    min_keys = sorted(min_dict.keys())[:find_line_num]
    for key in min_keys:
        min_dist = 1
        min_index = 0
        for i, min_line_ptr in enumerate(min_dict[key]):
            dist2road = dist(min_line_ptr, ptr)
            if min_dist > dist2road:
                line_id[key] = (i, dist2road, min_line_ptr)
    return line_id


def find_nearst_ptr(line_to_dict):
    min_dist2road = 1
    min_dist2road_info = None
    min_dist_key = None
    for line in line_to_dict:
        dist2road_info = line_to_dict[line]
        if min_dist2road > dist2road_info[1]:
            min_dist_key = line
            min_dist2road = dist2road_info[1]
            min_dist2road_info = dist2road_info
    if min_dist2road_info:
        raise RoadNotFoundException('Can not find the nearest road info \n{}'.format(line_to_dict))
    return min_dist2road_info[0], min_dist2road_info[1], min_dist2road_info[2],min_dist_key


def main():
    road_dict = load_road_data()
    line_list = load_line_data()
    for line, goal_ptr in line_list:
        history = [] #[{road_id:xxx,'posi':(x,y),index:x},....]
        for ptr in line:
            line_id_dict = map2road(ptr, road_dict,1)
            nearst_index, nearst_dist, ptr_in_map, nearst_road_key = find_nearst_ptr(line_id_dict)
            if history:
                # 拿到最近的点 判断是否和前一个在同一个道路上，
                # 如果在就再判断中间的间隔  如果不在就设置一个阈值差，如果属于这个阈值就仍然判断在一起，超过说明道路已经越过不应该再有牵连。
                last_road_id = history[-1]['road_id']
                ptr_info_dict = {'road_id': nearst_road_key, 'post': ptr_in_map, 'index': nearst_index}
                history.append(ptr_info_dict)
            else:
                ptr_info_dict = {'road_id':nearst_road_key,'post':ptr_in_map,'index':nearst_index}
                history.append(ptr_info_dict)
        for index,his_info in enumerate(history):
            if index == 0:
                continue
            if his_info['road_id'] == history[index-1]['road_id'] or his_info['road_id'] == history[index+1]['road_id']:
                continue
            else:
                pass



