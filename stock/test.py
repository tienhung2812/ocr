try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import pandas

def highlight_last_row(s):
    return ['background-color: #FF0000' if i==len(s)-1 else '' for i in range(len(s))]


df = pytesseract.image_to_data(Image.open('don-thuoc.png'),lang='eng', output_type='data.frame')
print(df.columns)
a = df.index['conf'] < 50
# print(df.to_html())

