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
        self.output_type = output_type

    def execute(self):
        with io.BytesIO() as transfer:
            with wandImg(filename=self.pdfPath, resolution=300) as img:
                img.format = 'png'
                img.save(transfer)
            with pilImg.open(transfer) as img:
                return self.runtesseract(img)
                # return pytesseract.image_to_pdf_or_hocr(img, extension='hocr',lang=self.lang)
    def runtesseract(self,img):
        if self.output_type == 'xml':
            vie = pytesseract.image_to_pdf_or_hocr(img, extension='hocr',config='-l vie --oem 1')
        else:
            vie = pytesseract.image_to_string(img, lang='vie',config=' --oem 1')
        return vie
    