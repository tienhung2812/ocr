import os
import json

from utils.find_real_path import *


def update_data(transaction_num, id_num,text,cate,ok_input):

    json_path = os.getcwd() + '/media/'+transaction_num + '/text_detection/data.json'
    if ok_input == "yes":
        cate_conf = 1
    else:
        cate_conf = 0
    json_data = {
        'sentence': text,
        'cate': cate,
        'cate_conf': cate_conf
    }
    current_data = None
    with open(json_path) as f:
        current_data = json.load(f)
    current_data[id_num] = json_data

    print(current_data)
    with open(json_path, 'w') as outfile:
        json.dump(current_data, outfile, ensure_ascii=False)