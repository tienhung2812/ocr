import os
import json

from utils.find_real_path import *


def update_data(transaction_num, id_num,text,cate):

    json_path = os.getcwd() + '/media/'+transaction_num + '/text_detection/data.json'

    json_data = {
        'sentence': text,
        'cate': cate,
        'cate_conf': 1
    }
    current_data = None
    with open(json_path) as f:
        current_data = json.load(f)
    current_data[id_num] = json_data

    print(current_data)
    with open(json_path, 'w') as outfile:
        json.dump(current_data, outfile, ensure_ascii=False)