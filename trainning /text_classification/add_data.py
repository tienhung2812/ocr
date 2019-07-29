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

def get_true_false_input(text):
    try:
        ina = input(text)
        if len(ina) > 0 and int(ina) == 1:
            return True
    except:
        return get_true_false_input(text)
    return False

def check_text(start = 0, check_col_null = True):
    path = 'receipt_sentence_classificaion_data.csv'
    df = pd.read_csv(path)# Loading a csv file wit

    col = ['title','date','phone_num','address','index','content','total','brand_name','thank_you']

    print(df.head())
    delete_index = []
    for index,row in df.iterrows():
        col_null = True
        if check_col_null:
            for i, name in enumerate(col):
                if df.loc[index,name] == 1:
                    col_null = False

        too_small_len = len(row['sentence']) <= 2
    
        if index >= start and col_null and too_small_len:
            print('===========')
            print(index)
            print(row['sentence'])
            delete = get_true_false_input("Delete (1/0): ")
        
            if delete:
                delete_index.append(index)
            else:
                re_text = input("Re text: ")
                if len(re_text) > 0 :
                    df.loc[index,'sentence']   = re_text
                for i, name in enumerate(col):
                    print(i,name,end = "\t")
                print()
                row_type = input("row_type: ")
                if len(row_type) > 0:
                    for i, name in enumerate(col):
                        if i!=int(row_type):
                            df.loc[index,name] = 0
                        else:
                            df.loc[index,name] = 1
                
            print(df.loc[index])
            
            continue_t = get_true_false_input("\nstop 1: ")

            if continue_t:
                break
            df.to_csv(path, index = False,  encoding='utf-8')

    df.drop(delete_index,inplace = True)
    df.to_csv(path, index = False,  encoding='utf-8')
    print("Done !")

check_text(0)