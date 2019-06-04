import getpass, platform
from general import General
from utils import *
# INPUT

DEBUG = True
class News(General):
    def __init__(self ,url,browser='chrome'):
        self.url = url
        self.WebDriverManager(browser)

    def execute(self):
        self.driver.get(self.url)

        text_wrapper = self.find(VnExpress.textSelector.value)
        elements = self.getAllChild(text_wrapper)
        sc = Screenshot(self.driver,self.getMetaData(text_wrapper))
        for ele in elements:
            if self.isHaveText(ele):
                filenum = getFileNum()
                meta = self.getMetaData(ele)
                
                print(filenum)
                
                if sc.takePartialScreenshot(ele,filenum,VnExpress) :
                    filename = getFileName(filenum,meta,'text')
                    with open('../dataset/texts/'+filename, "w") as file:
                        file.write(ele.text)


        self.teardown()
    
    def teardown(self):
        self.driver.quit()
    
#         print('Re-input')
if __name__ == "__main__":
    url = 'https://vnexpress.net/suc-khoe/10-giac-mac-duoc-ngan-hang-mat-my-tang-cho-benh-nhan-viet-3933015.html'

    try:
        main = News(url=url,browser='firefox')
        main.execute()
    except Exception as e:
        print(e)
        main.teardown()