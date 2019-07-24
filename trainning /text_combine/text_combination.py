import os

import cv2
import numpy as np
import pandas as pd


class TextCombinator:
    def __init__(self,transaction_num):
        # Init the transaction num  
        self.transaction_num = int(transaction_num)

        #Get the folder
        path = '/home/hung/ocr/main/media'
        path += '/'+str(self.transaction_num)
        print(path)
        
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

        # print(self.data)
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

    def combine(self):
        current_id = 9

        text_df = self.get_text_file_pandas(current_id)
        print(text_df)
        img = self.get_image_file(current_id)

        self.draw_rect(img,text_df)

        cv2.imshow('default image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

tc = TextCombinator(1563935718)
tc.combine()
