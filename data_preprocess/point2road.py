import json
import math
from tqdm import tqdm

FIND_LINE_NUM = 20
DIFF_ROAD_DIST = 0.0000001
ADD_PTR = True
DIFF_LINE_DIST = 0.0023
MIN_LINE_LENGTH = 15


class RoadNotFoundException(Exception):
    pass


def load_road_data():
    with open('../roads_dict2.json', 'r') as f:
        data = json.load(f)
    return dict(zip(map(lambda x: eval(x), data.keys()), data.values()))


def load_line_data():
    with open('../processed_data/processed_data.json', 'r') as f:
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
    min_dist = 1
    for key in min_keys:
        for i, min_line_ptr in enumerate(road_dict[min_dict[key]]):
            dist2road = dist(min_line_ptr, ptr)
            if min_dist > dist2road:
                min_dist = dist2road
                line_id[min_dict[key]] = (i, dist2road, min_line_ptr)
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
    if not min_dist2road_info:
        raise RoadNotFoundException('Can not find the nearest road info \n{}'.format(line_to_dict))
    return min_dist2road_info[0], min_dist2road_info[1], min_dist2road_info[2], min_dist_key


def find_ptr_in_threshold(line_id_dict, old_info, roads, threshold):
    dist_list = []
    for road in roads:
        if road not in line_id_dict:
            continue
        dist_list.append(line_id_dict[road][1])
    if not dist_list:
        return old_info['index'],None,old_info['post'],old_info['road_id']
    min_value = min(dist_list)
    min_road_index = dist_list.index(min_value)
    min_road = roads[min_road_index]
    if min_road not in line_id_dict.keys() or abs(min_value - line_id_dict[min_road][1]) > threshold:
        return old_info['index'],None,old_info['post'],old_info['road_id']
    else:
        if min_road not in line_id_dict:
            return old_info['index'],None,old_info['post'],old_info['road_id']
        else:
            road_info = line_id_dict[min_road]
            return road_info[0], road_info[1], road_info[2], min_road


def main(road_dict,line_list,total_saver):
    # with open('road_posi.json', 'r') as f:
    #     roads = json.load(f)
    road_dict = road_dict
    line_list = line_list
    total_saver = total_saver
    for line, goal_ptr in tqdm(line_list):
        history = []  # [{road_id:xxx,'posi':(x,y),index:x},....]
        history_saver = []
        for ptr in line:
            line_id_dict = map2road(ptr, road_dict, FIND_LINE_NUM//2)
            nearst_index, nearst_dist, ptr_in_map, nearst_road_key = find_nearst_ptr(line_id_dict)
            if history:
                # 拿到最近的点 判断是否和前一个在同一个道路上，
                # 如果在就再判断中间的间隔  如果不在就设置一个阈值差，如果属于这个阈值就仍然判断在一起，超过说明道路已经越过不应该再有牵连。
                ptr_info_dict = {'road_id': nearst_road_key, 'post': ptr_in_map, 'index': nearst_index, 'old_posi':ptr}
                history.append(ptr_info_dict)
            else:
                ptr_info_dict = {'road_id': nearst_road_key, 'post': ptr_in_map, 'index': nearst_index, 'old_posi':ptr}
                history.append(ptr_info_dict)
        # print(list(map(lambda x:x['post'],history)))
        for index, his_info in enumerate(history):
            if index == 0 or index == len(history) -1:
                point = road_dict[tuple(his_info['road_id'])][his_info['index']]
                history_saver.append(point)
                continue
            line_id_dict = map2road(his_info['old_posi'], road_dict, FIND_LINE_NUM)
            nearst_index, nearst_dist, ptr_in_map, nearst_road_key\
                = find_ptr_in_threshold(line_id_dict,his_info,
                                        list([history[index - 1]['road_id'],history[index + 1]['road_id']]),
                                        DIFF_ROAD_DIST)
            his_info['road_id'] = nearst_road_key
            his_info['post'] = ptr_in_map
            his_info['index'] = nearst_index
            if his_info['road_id'] == history[index-1]['road_id'] and abs(his_info['index'] - history[index-1]['index']) > 1  and ADD_PTR:
                step = 1 if his_info['index'] > history[index-1]['index'] else -1
                # print(history[index-1]['index'],his_info['index']+step,step)
                for i in range(history[index-1]['index'],his_info['index']+step,step):
                    point = road_dict[his_info['road_id']][i]
                    history_saver.append(point)
            else:
                point = road_dict[tuple(his_info['road_id'])][his_info['index']]
                # print(point)
                history_saver.append(point)
        start = 0
        for i,ptr in enumerate(history_saver):
            if i == 0:
                continue
            if dist(ptr,history_saver[i-1]) > DIFF_LINE_DIST:
                if i - start > MIN_LINE_LENGTH:
                    if i - start > MIN_LINE_LENGTH:
                        total_saver.append(history_saver[start:i])
                    start = i
        # print(list(map(lambda x: [x['road_id'], x['index']],history)))
        # print(history_saver)
    return total_saver
#
# def multi_thread():
#     road_dict = load_road_data()
#     line_list = load_line_data()
#     cpu_core = multiprocessing.cpu_count()
#     length = len(line_list) // cpu_core - 1
#     process_pool = []
#     total = multiprocessing.Manager().list()
#     for i in range(cpu_core):
#         p = multiprocessing.Process(target=main,args=(road_dict,line_list[i*length:(i+1)*length],total,))
#         p.start()
#         process_pool.append(p)
#     for p in process_pool:
#         p.join()
#     time.sleep(10)
#     return list(total)


if __name__ == '__main__':
    road_dict = load_road_data()
    line_list = load_line_data()[:5000]
    data = main(road_dict,line_list,[])
    with open('/media/luoyingsong/新加卷/GPS数据/data_preprocess/line2road2.json','w') as f:
        json.dump(data,f)
    # line_data = load_line_data()
    # old_line = line_data[100]
    # new_line = main(line_data)
    # with open('saver.json','w') as f:
    #     json.dump([old_line[0],new_line],f)
    # print(dist([119.28679399,  26.05479201],
    #    [119.288813  ,  26.05298499],))

