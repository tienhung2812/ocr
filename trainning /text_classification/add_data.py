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

def append_csv(file_name,row,lines_array):
    for line in lines:
        df = pd.read_csv(file_name,   encoding='utf-8')
        length = df.shape[0]

        data = {
            "sentence":line.replace("\n",""),
        }

        y_col = ["brand_name","info","index","content","total","thank_you"]

        for col in y_col:
            if row == col:
                data[col] = 1
            else:
                data[col] = 0
        
        df.loc[length] = data
        df.to_csv(file_name, index = False,  encoding='utf-8')

# check_text(0)
# dataa = ""
# with open('info_data', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         arrr = line.split(' ')
#         stop = 0
#         for i,word in enumerate(arrr):
#             try:
#                 num = int(word)
#                 if num > 0:
#                     stop = i
#                     break
#             except:
#                 pass

#         if stop == 0 :
#             stop = len(arrr) -1
#         dataa+=' '.join(arrr[:stop])+'\n'
# with open('info_data', 'w') as f:
#     f.write(dataa)

# Append data
# df_col = ["sentence","brand_name","info","index","content","total","thank_you"]
# y_col = ["brand_name","info","index","content","total","thank_you"]
# with open('brand_data', 'r') as f:
#     lines = f.readlines()
#     append_csv('30-07-data-vi.csv',"brand_name",lines)

# Generate index data
import random
def generate_random_index(array):
    return array[random.randint(0,len(array)-1)]+' '

def random_yn():
    tt = random.randint(0,1)
    if tt == 0:
        return False
    else:
        return True

# mat_hang = ["Mặt hàng","Tên hàng", "TÊN", "HÀNG", "TÊN HÀNG","THỨC UỐNG","ĐỒ ĂN"," MÓN HÀNg","Món hàng"]
# stt = ["STt","STT","Stt"]
# sl = ['sl',"SL","Sl"]
# dgia = ["DGia","Đ.Giá","ĐGía", "Đơn Giá"]
# ttien = ["T.Tiền","T.TIỀN","Thành Tiền"]
# import random
# dataa = ''
# for i in range(0,100):
#     if random_yn():
#         dataa+= generate_random_index(sl)
#     dataa += generate_random_index(mat_hang)
#     dataa += generate_random_index(sl)
#     dataa += generate_random_index(dgia)
#     if random_yn():
#         dataa+= generate_random_index(ttien)
#     dataa += '\n'
# with open('index_data', 'w') as f:
#     f.write(dataa)

# Append data
# df_col = ["sentence","brand_name","info","index","content","total","thank_you"]
# y_col = ["brand_name","info","index","content","total","thank_you"]
# with open('index_data', 'r') as f:
#     lines = f.readlines()
#     append_csv('30-07-data-vi.csv',"index",lines)


# THANK YOU
# hen_gap_lai = ["Hẹn gặp lại",
#             "HEN GAP LAI",
#             "HẸN GẶP LẠI"]
# thank_you = ["Cám ơn quý khách",
#             "Thank you",
#             "CAM ON",
#             "CAM ƠN QUÝ KHÁCH",
#             "Cam on"]
# dataa = ''
# for i in range(0,70):
#     newl = False
#     if random_yn():
#         dataa += generate_random_index(thank_you)+"."
#         newl = True
#     if random_yn():
#         dataa += generate_random_index(hen_gap_lai)
#         newl = True
#     if newl:
#         dataa += '\n'
# with open('thank_you_data', 'w') as f:
#     f.write(dataa)

# Append data
df_col = ["sentence","brand_name","info","index","content","total","thank_you"]
y_col = ["brand_name","info","index","content","total","thank_you"]
with open('thank_you_data', 'r') as f:
    lines = f.readlines()
    append_csv('30-07-data-vi.csv',"thank_you",lines)
