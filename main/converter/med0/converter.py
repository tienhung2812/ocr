#!/usr/bin/env python
# -*- coding: utf-8 -*-
from wand.image import Image as wandImg
try:
    from PIL import Image as pilImg
except ImportError:
    import Image
import io 
import pytesseract
import sys
from bs4 import BeautifulSoup
import pandas 

CONFIG={
    'low_conf':90
}
class Converter:
    def __init__(self,file, lang='eng', output_type='xml',full_table=False):
        self.pdfPath = file
        self.lang = lang
        self.output_type = output_type
        self.full_table = full_table

    def execute(self):
        with io.BytesIO() as transfer:
            with wandImg(filename=self.pdfPath, resolution=300) as img:
                img.format = 'png'
                img.save(transfer)
            with pilImg.open(transfer) as img:
                return self.runtesseract(img)
                # return pytesseract.image_to_pdf_or_hocr(img, extension='hocr',lang=self.lang)

    def format_pandas(self,df):
        for index, row in df.iterrows():
            if row['conf'] < 0:
                df.drop(index, inplace=True)
        if self.full_table:
            return df
        return  df[['word_num','conf','text']]
        

    def runtesseract(self,img):
        df = pytesseract.image_to_data(img, lang=self.lang,config=' --oem 1', output_type='data.frame')
        df = self.format_pandas(df)


        if self.output_type == 'xml':
            vie = pytesseract.image_to_pdf_or_hocr(img, extension='hocr',config='-l vie --oem 1')
        else:
            vie = pytesseract.image_to_string(img, lang=self.lang,config=' --oem 1')
        return vie,df
    