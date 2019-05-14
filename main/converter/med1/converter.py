from wand.image import Image as wandImg
try:
    from PIL import Image as pilImg
except ImportError:
    import Image
import io 
import pytesseract
import sys
from bs4 import BeautifulSoup

CONFIG={
    'low_conf':90
}
class Converter:
    def __init__(self,file, lang='eng', output_type='xml'):
        self.pdfPath = file
        self.lang = lang

    def execute(self):
        with io.BytesIO() as transfer:
            with wandImg(filename=self.pdfPath, resolution=300) as img:
                img.format = 'png'
                img.save(transfer)
            with pilImg.open(transfer) as img:
                return self.runtesseract(img)
                # return pytesseract.image_to_pdf_or_hocr(img, extension='hocr',lang=self.lang)
    def runtesseract(self,img):
        eng = pytesseract.image_to_pdf_or_hocr(img, extension='hocr',lang='eng',config=' --oem 1')
        vie = pytesseract.image_to_pdf_or_hocr(img, extension='hocr',lang='vie',config=' --oem 1')
        return self.compareConf(vie,eng)
    
    def extractAttr(self,data):
        info = data['title'].split("x_wconf ")
        box = info[0]
        conf = int(info[1])
        return box, conf

    def extractSpanData(self,data):
        result = []
        for span in data.find_all('span'):
            if len(span.contents) <= 1:
                box, conf = self.extractAttr(span)
                string = span.string
                result.append({
                    'string':string,
                    'box': box,
                    'conf':conf
                })
        return result

    def compareConf(self,vie,eng):

        viesoup = BeautifulSoup(vie,'xml')
        engsoup = BeautifulSoup(eng,'xml')

        engdata = self.extractSpanData(engsoup)

        for span in viesoup.find_all('span'):
            if len(span.contents) <= 1:
                box,conf = self.extractAttr(span)
                #if object have low confident
                if conf <= CONFIG['low_conf']:
                    for e in engdata:
                        if box == e['box']:
                            if e['conf']> conf and e['conf'] >=CONFIG['low_conf']:
                                span.string.replaceWith(e['string'])
                                span['conf'] = e['conf']
                                # print('{} {}'.format(span.string,e['string']))
                    

        return viesoup.prettify()
        



