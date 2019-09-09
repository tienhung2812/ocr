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

with io.BytesIO() as transfer:
            with wandImg(filename='/Users/hungnguyentien/Documents/GitHub/ocr/stock/receipt/2-ano.jpg', resolution=300) as img:
                img.format = 'png'
                img.save(transfer)
            with pilImg.open(transfer) as img:
                print(pytesseract.image_to_string(img, lang="eng+vie"))