IMAGE_URL='46.png'

# Import section
import calendar
import os
import time

import cv2
import pandas

from image.process import ReceiptImage
from text_combination.text_combine import TextCombinator
from text_detection.core.detector import TextDetection
from text_recognization.text_recognizance import TextRecognizance
from text_recognization.update_data import update_data
from utils.find_real_path import *

#Create TRANSACTIONNUM
TRANSACTION_NUM = str(calendar.timegm(time.gmtime()))

from shutil import copyfile

NEW_URL = 'media/'+TRANSACTION_NUM + '/'+ os.path.basename(IMAGE_URL)
if not os.path.exists( 'media/'+TRANSACTION_NUM ):
    os.makedirs( 'media/'+TRANSACTION_NUM )
copyfile(IMAGE_URL, NEW_URL)

ri = ReceiptImage(NEW_URL, 'filename.png',TRANSACTION_NUM)
ri.processImage()

# View image
input_img = cv2.imread(IMAGE_URL)
dilated_img = cv2.imread(ri.dilated_url)
drawed_img = cv2.imread(ri.drawed_url)
wraped_img = cv2.imread(ri.wraped_url)

# cv2.imshow("Input image", input_img)
# cv2.imshow("Dilated_img", dilated_img)
# cv2.imshow("Drawed_img", drawed_img)
# cv2.imshow("Waped_img", wraped_img)
cv2.waitKey(0)

td = TextDetection(ri.wraped_url,TRANSACTION_NUM)
text_line_file_image_url,final_text_line_file_image_url,text_line_file_box_url, cropped_image_array = td.find()

for index , image in enumerate(cropped_image_array):
    
    remake_url = image['url']
    tr = TextRecognizance(image_url=remake_url,transaction_num = TRANSACTION_NUM,id_num = str(image['seq']), final_image = final_text_line_file_image_url)

    text_result = tr.detect_image()
    print('Text: ',text_result['text'])
    print('Conf:',text_result['conf'])
    print()
    print('Cate: ',text_result['cate'])
    print('Conf:',text_result['cate_conf'])
    print("======")
