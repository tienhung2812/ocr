try:
    from PIL import Image
except ImportError:
    import Image
import io 
import pytesseract
import sys
import pandas
import os

class TextRecognizance:
    def __init__(self,image_array=None,transaction_num = None, id_num = None,image_url = None, language = 'vie+eng',config='--oem 1 --psm 7'):
        self.image_array = image_array
        self.lang = language
        self.config = config
        self.image_url = image_url

        self.transaction_num = transaction_num
        self.id_num = id_num

        path = '/code/media/'+transaction_num
        save_folder = '/text_recognization/'
        self.save_path = path + save_folder

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.save_path = self.save_path + self.id_num+'.txt'

    def format_pandas(self,df):
        for index, row in df.iterrows():
            if row['conf'] < 0:
                df.drop(index, inplace=True)
        if self.full_table:
            return df
        return  df[['word_num','conf','text']]

    def get_str_conf(self,data,id_num = None):
        # data = self.format_pandas(df)

        text = ' '.join(str(e) for e in data[data.text.notnull()].text)
        conf = data[data.text.notnull()]['conf'].mean()
        if type(conf)==float: 
            conf = 0

        if id_num:
            return {
                "seq":id_num,
                "text":text,
                "conf":conf
            }  
        else:
            return {
                "text":text,
                "conf":conf
            }

    def save_result(self,result):
        with open(self.save_path, "w") as f:
            f.writelines(result.to_csv(index=True))
        # print thresh,ret

    def detect_image(self):
        image_path = '/code/'+self.image_url
        df = self.runtesseract(Image.open(image_path))
        self.save_result(df)
        return self.get_str_conf(df,self.id_num)

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