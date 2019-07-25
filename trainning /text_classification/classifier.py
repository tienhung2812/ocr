from langdetect import detect
from text_combination import TextCombinator

DEFAULT_LANG = 'vi'
SUPPORTTED_LANG = ['en','vi']

class Classifier:
    def __init__(self,transaction_num, combined_text):
        self.transaction_num = transaction_num
        self.combined_text = combined_text
    

    def detect_language(self,text):
        try:
            lang = detect(text)
        except:
            lang = 'en'
        # if lang not in SUPPORTTED_LANG:
        #     lang = DEFAULT_LANG
        return lang

    def classify(self):
        self.lang = self.detect_language(self.combined_text)

        print(self.lang)
        for line in self.combined_text.split('\n'):
            print(self.detect_language(line),line)


tc = TextCombinator(1564029925,'/media/1564029925/text_detection/final_1564029929wraped_37.jpeg')
text = tc.combine()

c = Classifier(1564028424,text)
c.classify()
