try:
    from PIL import Image
except ImportError:
    import Image
import io 
import pytesseract
import sys
import pandas


class TextRecognizance:
    def __init__(self,image_array=None,image_url = None, language = 'vie+eng',config='--oem 1 --psm 7'):
        self.image_array = image_array
        self.lang = language
        self.config = config
        self.image_url = image_url

    def format_pandas(self,df):
        for index, row in df.iterrows():
            if row['conf'] < 0:
                df.drop(index, inplace=True)
        if self.full_table:
            return df
        return  df[['word_num','conf','text']]

    def get_str_conf(self,data):
        # data = self.format_pandas(df)

        text = ' '.join(str(e) for e in data[data.text.notnull()].text)
        conf = data[data.text.notnull()]['conf'].mean()
        return {
            "text":text,
            "conf":conf
        }


        # print thresh,ret
    def detect_image(self):
        image_path = '/code/'+self.image_url
        df = self.runtesseract(Image.open(image_path))
        
        return self.get_str_conf(df)

    def detect_array(self):
        print('====== TEXT RECOGNIZE =====')
        result = []
        for image in self.image_array:
            image_path = '/code'+image
            df = self.runtesseract(Image.open(image_path))
            
            result.append(self.get_str_conf(df))

        return result

    def runtesseract(self,img):
        return pytesseract.image_to_data(img, lang=self.lang,config=self.config, output_type='data.frame')

# pytesseract.image_to_string(img, lang='eng+vie',config='--oem 1 --psm 7')