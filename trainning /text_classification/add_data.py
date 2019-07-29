import pandas as pd
import numpy as np

def append_receipt_data():
    path = 'receipt_sentence_classificaion_data.csv'
    df = pd.read_csv(path)# Loading a csv file with headers 
    data = {
        'sentence':'test',
        'x':0.3,
        'y':2,
        'w':1,
        'l': 1,
        'area_percent': 0.2,
        'date':0, 
        'receipt_no':0, 
        'total':0, 
        'title':0, 
        'address':0, 
        'brand_name':0
    }
    df = df.append(data, ignore_index=True)
    df.to_csv(path, index = False,  encoding='utf-8')