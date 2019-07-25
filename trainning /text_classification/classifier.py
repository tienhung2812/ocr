from langdetect import detect
from text_combination import TextCombinator

DEFAULT_LANG = 'vi'
SUPPORTTED_LANG = ['en','vi']
MIN_ACCEPT_CORRECTION = 0.8

class Classifier:
    def __init__(self,transaction_num, combined_text):
        self.transaction_num = transaction_num
        self.combined_text = combined_text
        self.lang = self.detect_language(self.combined_text)

    def detect_language(self,text):
        try:
            lang = detect(text)
        except:
            lang = 'en'
        if lang not in SUPPORTTED_LANG:
            lang = DEFAULT_LANG
        return lang

    def enhance_word(self,word):
        if self.lang == 'en':
            from pattern.en import suggest as corrector
        enhanced_list = corrector(word)
        result = None
        score = 0
        for correct_word in enhanced_list:
            if self.lang == 'en':
                if correct_word[1] >= score and correct_word[1] > MIN_ACCEPT_CORRECTION:
                    result = correct_word[0]

        if result is None:
            result = word
        print(enhanced_list)
        return result

    def enhance_text_line(self,text_line):
        current_word = ''
        result = ''
        for word in text_line.split(' '):
            if len(word) > 0:
                try:
                    correct_word = float(word)
                    if int(correct_word) == correct_word:
                        correct_word = int(correct_word)
                    if len(correct_word) != len(word):
                        correct_word = word
                except:
                    correct_word = self.enhance_word(word)

                result+= str(correct_word) + ' '
            else:
                result+= ' '
        result = result[:-1]
        if (text_line != result):
            print('text line :',text_line,len(text_line))
            print('result    :',result, len(result))
            print()
            input()
        return result
    def classify(self):
        

        print(self.lang)
        for line in self.combined_text.split('\n'):
            self.enhance_text_line(line)
            # input()
            # print(self.detect_language(line),line)


tc = TextCombinator(1564047627,'/media/1564047627/text_detection/final_1564047631wraped_X51008145504.jpg')
text = tc.combine()

c = Classifier(1564028424,text)
c.classify()
