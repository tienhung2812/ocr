from wand.image import Image as wandImg
try:
    from PIL import Image as pilImg
except ImportError:
    import Image
import io 
import pytesseract
import sys

class Pdf2Xml:
    def __init__(self,file, lang='eng'):
        self.pdfPath = file
        self.lang = lang

    def execute(self):
        with io.BytesIO() as transfer:
            with wandImg(filename=self.pdfPath, resolution=300) as img:
                img.format = 'png'
                img.save(transfer)
            with pilImg.open(transfer) as img:
                return pytesseract.image_to_pdf_or_hocr(img, extension='hocr',lang=self.lang)
                # return pytesseract.image_to_data(img,lang=self.lang)

if len(sys.argv) <= 1:
    a = Pdf2Xml(file = 'stock/2.pdf',lang='vie')
    text_file = open("Output.hocr", "wb")
    output = a.execute()
    print(output)
    text_file.write(output)
    text_file.close()
else:
    filename = None
    outputfile = None
    try:
        filename = sys.argv[1]
        outputfile = sys.argv[2]

    except:
        pass
    a = Pdf2Xml(file = filename,lang='vie')
    output = a.execute()
    if outputfile is None:
        print(output)
    else:
        text_file = open(outputfile, "wb")
        # output = a.execute()
        # print(output)
        text_file.write(output)
        text_file.close()
        if '--html' in sys.argv:
            f = open(outputfile, "r")
            temp = f.read().replace('</body>','<script src="https://unpkg.com/hocrjs"></script></body>')
            f.close()
            f = open(outputfile, "w")
            f.write(temp)
            f.close()
        if '--nocolor' in sys.argv:
            f = open(outputfile, "r")
            rpstr = """
            <head>
            <style>
            * {border-style: none !important;}
            </style>
            
            """
            temp = f.read().replace('<head>',rpstr)
            f.close()
            f = open(outputfile, "w")
            f.write(temp)
            f.close()
