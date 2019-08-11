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
from text_classification import Classifier, InfoClassifier, TotalClassifier
import json

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

    def get_str_conf(self,data,cate, cate_conf = 0,id_num = None):
        # data = self.format_pandas(df)

        text = ' '.join(str(e) for e in data[data.text.notnull()].text)
        conf = data[data.text.notnull()]['conf'].mean()
        if type(conf)==float: 
            conf = 0

        if id_num:
            return {
                "cate":cate,
                "cate_conf": str(cate_conf),
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

    def predict_type(self,text):
        if len(text) > 0:
            classifier = Classifier()
            cate, cate_conf = classifier.predict(text)

            if cate == 'info':
                info_classifier = InfoClassifier()
                info_cate, info_cate_conf = info_classifier.predict(text)
                cate += ', '+info_cate
            elif cate == 'total':
                total_classifier = TotalClassifier()
                total_cate, total_cate_conf = total_classifier.predict(text)
                cate += ', '+total_cate
            return cate, cate_conf
        return '',0

    def detect_image(self):
        image_path = os.getcwd()+'/'+self.image_url
        df = self.runtesseract(Image.open(image_path))
        self.save_result(df)
        cate, cate_conf = self.predict_type(self.get_str_conf(df,self.id_num)['text'])
        self.append_receipt_data(self.get_str_conf(df,self.id_num)['text'],cate, cate_conf)

        return self.get_str_conf(df,id_num = self.id_num,cate = cate, cate_conf = cate_conf)

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
        x_tl = data['x_tl'].min()
        y_tl = data['y_tl'].min()
        x_br = data['x_br'].max()
        y_br = data['y_br'].max()

        width = x_br - x_tl
        height = y_br - y_tl

        image_file_name = 'original_image.png'
        return data.loc[image_id] , width, height

    def append_receipt_data(self,text,cate,cate_conf):
        if self.final_image:
            path = 'receipt_sentence_classificaion_data.csv'
            df = pd.read_csv(path)# Loading a csv file with headers 

            row_data, width, height = self.get_image_data()
            # img = cv2.imread( os.getcwd() +  self.final_image,0)
            # height, width = img.shape

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


            #JSON DATA
            json_path = os.getcwd() + '/media/'+self.transaction_num + '/text_detection/data.json'

            json_data = {
                'sentence': text,
                'cate': cate,
                'cate_conf': str(cate_conf)
            }

            if os.path.exists(json_path):
                current_data = None
                with open(json_path) as f:
                    current_data = json.load(f)
                    
                current_data[self.id_num] = json_data
                with open(json_path, 'w') as outfile:
                    json.dump(current_data, outfile, ensure_ascii=False)
            else:
                with open(json_path, 'w') as outfile:
                    json_str = {}
                    json_str[self.id_num] = json_data
                    json.dump(json_str, outfile , ensure_ascii=False)
# pytesseract.image_to_string(img, lang='eng+vie',config='--oem 1 --psm 7')