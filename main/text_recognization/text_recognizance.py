try:
    from PIL import Image
except ImportError:
    import Image
import io 
import pytesseract
import sys
import pandas as pd
import numpy as np
import os
import cv2
from utils.find_real_path import *



class TextRecognizance:
    def __init__(self,image_array=None,transaction_num = None, id_num = None,image_url = None, final_image = None, language = 'vie+eng',config='--oem 1 --psm 7'):
        self.image_array = image_array
        self.lang = language
        self.config = config
        self.image_url = image_url

        self.final_image = final_image

        self.transaction_num = transaction_num
        self.id_num = id_num

        path = os.getcwd()+'/media/'+transaction_num
        save_folder = '/text_recognization/'
        self.save_path = path + save_folder

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.save_path = self.save_path + self.id_num+'.txt'

    def format_pandas(self,df):
        for index, row in df.iterrows():
            if row['conf'] < 0:
                df.drop(index, inplace=True)
        if self.full_table:
            return df
        return  df[['word_num','conf','text']]

    def get_str_conf(self,data,id_num = None):
        # data = self.format_pandas(df)

        text = ' '.join(str(e) for e in data[data.text.notnull()].text)
        conf = data[data.text.notnull()]['conf'].mean()
        if type(conf)==float: 
            conf = 0

        if id_num:
            return {
                "seq":id_num,
                "text":text,
                "conf":conf
            }  
        else:
            return {
                "text":text,
                "conf":conf
            }

    def save_result(self,result):
        with open(self.save_path, "w") as f:
            f.writelines(result.to_csv(index=True))
        # print thresh,ret

    def detect_image(self):
        image_path = os.getcwd()+'/'+self.image_url
        df = self.runtesseract(Image.open(image_path))
        self.save_result(df)
        self.append_receipt_data(self.get_str_conf(df,self.id_num)['text'])
        return self.get_str_conf(df,self.id_num)

    def detect_array(self):
        print('====== TEXT RECOGNIZE =====')
        result = []
        for image in self.image_array:
            image_path = os.getcwd()+image
            df = self.runtesseract(Image.open(image_path))
            
            result.append(self.get_str_conf(df))

        return result

    def runtesseract(self,img):
        return pytesseract.image_to_data(img, lang=self.lang,config=self.config, output_type='data.frame')

    def get_image_data(self):
        image_id = int(os.path.basename(self.image_url).split('_')[1])
        data_path = os.getcwd() + '/media/'+self.transaction_num + '/text_detection'

        data_file = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        data_file = data_path+'/'+data_file[0]

        data = pd.read_csv(data_file,index_col=0)

        image_file_name = 'original_image.png'
        return data.loc[image_id]

    def append_receipt_data(self,text):
        if self.final_image:
            path = 'receipt_sentence_classificaion_data.csv'
            df = pd.read_csv(path)# Loading a csv file with headers 

            row_data = self.get_image_data()
            img = cv2.imread( os.getcwd() +  self.final_image,0)
            height, width = img.shape

            cropped_h = float(row_data['y_bl']-row_data['y_tl'])
            cropped_w = float(row_data['x_tr']-row_data['x_tl'])

            area_percent = float(cropped_h*cropped_w)/float(height*width)

            data = {
                'sentence':text,
                'x':row_data['x_tl']/width,
                'y':row_data['y_tl']/height,
                'w':cropped_w/width,
                'l': cropped_h/height,
                'area_percent': area_percent,
                'date':0, 
                'receipt_no':0, 
                'total':0, 
                'title':0, 
                'address':0, 
                'brand_name':0
            }
            df = df.append(data, ignore_index=True)
            df.to_csv(path, index = False,  encoding='utf-8')
# pytesseract.image_to_string(img, lang='eng+vie',config='--oem 1 --psm 7')