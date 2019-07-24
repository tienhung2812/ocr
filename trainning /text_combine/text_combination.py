import os

import cv2
import numpy as np
import pandas as pd

from underthesea import word_tokenize

MAX_WIDTH_MULTIPLY = 8

class TextCombinator:
    def __init__(self,transaction_num,input_image):
        # Init the transaction num  
        self.transaction_num = int(transaction_num)

        #Get the folder
        path = '/home/hung/ocr/main/media'
        path += '/'+str(self.transaction_num)
        image_path = '/home/hung/ocr/main' + input_image

        # Init variable
        
        # Bounding box
        text_detection_folder = path + '/text_detection'
        csv_file = self.find_file_with_extension(text_detection_folder,'.txt',full_path = True)
        self.data = pd.read_csv(csv_file,index_col=0)
        # 

        #Cropped image
        self.data['cropped_image'] = np.nan
        cropped_image_path = text_detection_folder+'/croped'
        for image in os.listdir(cropped_image_path):
            image_id = int(image.split('_')[1])
            self.data.loc[image_id,'cropped_image'] = os.path.join(cropped_image_path, image)
        # print(os.listdir(cropped_image_path))

        #Text detection
        self.data['text_recognization_file'] = np.nan
        text_recognization_path = path + '/text_recognization'
        for text in os.listdir(text_recognization_path):
            file_name, file_extension = os.path.splitext(text)
            if file_name.isdigit() and file_extension == '.txt':
                text_id = int(file_name)
                self.data.loc[text_id,'text_recognization_file'] = os.path.join(text_recognization_path, text)
        
        self.image = cv2.imread(image_path,0)
        self.image_h, self.image_w = self.image.shape
        self.max_width = self.find_max_width()
        self.max_charater = self.find_max_charater_per_row()
        self.final_text = ''


    def find_file_with_extension(self,folder,ext,full_path = False):
        for the_file in os.listdir(folder):
            file_name, file_extension = os.path.splitext(the_file)
            if file_extension == ext:
                if full_path:
                    return os.path.join(folder, the_file)
                return the_file
        return None

    def get_text_file_pandas(self,text_id):
        text_file = self.data.loc[text_id,'text_recognization_file']
        return pd.read_csv(text_file,index_col=0)

    def get_image_file(self,image_id, image_type = 0):
        image_file = self.data.loc[image_id,'cropped_image']
        return cv2.imread(image_file,image_type)

    def draw_rect(self,image, text_df):
        for index, row in text_df.iterrows():
            if row['conf'] > 0:
                cv2.rectangle(image,(row['left'],row['top']),(row['left']+row['width'],row['top']+row['height']),(0,255,0),1)

    def text_word_tokenrize(self,text_df):
        text = ''
        for index, row in text_df.iterrows():
            if row['conf']>0:
                if len(text)>0:
                    text+=' '
                text+= row['text']

        return word_tokenize(text, format="text")

    def find_max_width(self):
        # max_width = 0
        
        # for index, row in self.data.iterrows():
        #     if (row['x_tr'] - row['x_tl']) > max_width:
        #         max_width = row['x_tr'] - row['x_tl']
        # print(max_width)
        return self.data['x_tr'].max() - self.data['x_tl'].min()

    def find_max_charater_per_row(self):
        max_charater = 0
        for index, row in self.data.iterrows():
            if row['parrent'] <0 :
                count = len(self.text_word_tokenrize(self.get_text_file_pandas(index)))
                if count > max_charater:
                    max_charater = count
        return max_charater      

    def arrange_text(self,index,row_data,text_df):
        spacedicator = ' '
        bounding_box_tl = int(row_data['x_tl'] - self.data['x_tl'].min())
        final_text = spacedicator*int(bounding_box_tl/MAX_WIDTH_MULTIPLY)

        added_index = []
        previous_space = 0
        for index, row in text_df.iterrows():
            if row['conf']>0:
                added_index.append((index,previous_space,row['text']))
                # print(added_index)
                if len(added_index) > 1:
                    # print(previous_space)
                    previous_len = added_index[len(added_index)-2][1]
                    current_space = int((row['left'] - previous_len)/MAX_WIDTH_MULTIPLY)

                    # previous_charater_len = int(len(added_index[len(added_index)-2][2])/MAX_WIDTH_MULTIPLY)
                    # if current_space > previous_charater_len:
                    #     current_space -= int(previous_charater_len)
                else:
                    current_space = int(row['left']/MAX_WIDTH_MULTIPLY)  
                    # current_space = int((row['left'] - text_df.iloc[added_index[added_index-2]]['left'])/MAX_WIDTH_MULTIPLY)
                # else:
                #     current_space = int(row['left']/MAX_WIDTH_MULTIPLY)          
                previous_space = current_space
                final_text+= spacedicator*current_space+row['text']
        print(final_text)
        return


    def combine(self):
        # tokenrize_data = pd.DataFrame()
        # tokenrize_data['tokenrize'] = np.nan
        # for index, row in self.data.iterrows():
        #     if row['parrent'] <0 :
        #         text_df = self.get_text_file_pandas(index)
        #         tokenrize = np.array(self.text_word_tokenrize(text_df))
        #         tokenrize_data.loc[index,'tokenrize'] =[tokenrize]


        for index, row in self.data.iterrows():
            if row['parrent'] <0 :
                text_df = self.get_text_file_pandas(index)
                result = self.arrange_text(index,row,text_df)
                # print(result)
        print(self.get_text_file_pandas(11))        
        # img = self.get_image_file(current_id)
        # self.draw_rect(img,text_df)

        # cv2.imshow('default image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return

tc = TextCombinator(1563962408,'/media/1563962408/text_detection/final_1563962410wraped_45.jpg')
tc.combine()
