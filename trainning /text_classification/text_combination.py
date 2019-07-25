import os

import cv2
import numpy as np
import pandas as pd

from underthesea import word_tokenize

MAX_WIDTH_MULTIPLY = 8
NUM_SPACE_MARK_AS_TAB = 1
SPACE_OVER_CHARATER = 2 
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
        self.largest_pixel_charater = 0
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
                cv2.rectangle(image,(row['left'],row['top']),(row['left']+row['width'],row['top']+row['height']),(0,255,0),2)

    def text_word_tokenrize(self,text_df):
        text = ''
        for index, row in text_df.iterrows():
            if row['conf']>0:
                if len(text)>0:
                    text+=' '
                text+= str(row['text'])

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

    def filter_low_conf(self,text_df):
        return text_df.query('conf>0')

    def arrange_text(self,index,row_data,text_df):
        spacedicator = ' '
        bounding_box_tl = int(row_data['x_tl'] - self.data['x_tl'].min())
        # final_text = spacedicator*int(bounding_box_tl/MAX_WIDTH_MULTIPLY)
        final_text = ''

        filtered_text_df = self.filter_low_conf(text_df)
        pixel_per_space = self.find_largest_pixel_of_a_charater(text_df)
        # previous_index = -1
        previous_row = None
        for index, row in filtered_text_df.iterrows():
            if previous_row is not None:
                spaces = self.how_many_space(previous_row,row,pixel_per_space)
                final_text+= spacedicator*spaces
            
            final_text += str(row['text'])
            previous_row = row
        # print(filtered_text_df)
        return final_text

    def how_many_space(self,row1,row2, pixel_per_space):
        # Return how much space between two word
        
        x_tr_row1 = row1['left'] + row1['width']

        pixel_between_row1_row2 = row2['left'] - x_tr_row1

        pixel_as_tab = pixel_per_space*NUM_SPACE_MARK_AS_TAB

        if pixel_between_row1_row2 <= pixel_as_tab:
            return 1
        else:
            return int(pixel_between_row1_row2/pixel_as_tab)

    def find_largest_pixel_of_a_charater(self,text_df):
        largest_pixel_per_charater = 0
        # text = ''
        # width = 0
        
        for index, row in text_df.iterrows():
            if row['conf'] > 0:
                pixel_per_charater = int(row['width'] / len(str(row['text'])))
                if pixel_per_charater > largest_pixel_per_charater:
                    largest_pixel_per_charater = pixel_per_charater
                    # text = str(row['text'])
                    # width = row['width']                    
        # print(text,width,largest_pixel_per_charater)
        return largest_pixel_per_charater
                
    def wrapping_image_text(self,index):
        text_df = self.get_text_file_pandas(index)
        img = self.get_image_file(index,1)
        self.draw_rect(img,text_df)

        cv2.imshow('default image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def combine(self):
        # tokenrize_data = pd.DataFrame()
        # tokenrize_data['tokenrize'] = np.nan
        # for index, row in self.data.iterrows():
        #     if row['parrent'] <0 :
        #         text_df = self.get_text_file_pandas(index)
        #         tokenrize = np.array(self.text_word_tokenrize(text_df))
        #         tokenrize_data.loc[index,'tokenrize'] =[tokenrize]

        combine_text = ''
        for index, row in self.data.iterrows():
            if row['parrent'] <0 :
                text_df = self.get_text_file_pandas(index)
                # print(text_df)
                
                # self.wrapping_image_text(index)

                result = self.arrange_text(index,row,text_df)
                # print(result)
                combine_text += (result+'\n')
                

        

        return combine_text

