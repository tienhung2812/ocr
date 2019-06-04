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

        ele = self.driver.find_element_by_css_selector('article.content_detail')

        print(self.getMetaData(ele))

        print(ele.text)
        # sc = Screenshot(self.driver,self.getMetaData())
        # sc.takePartialScreenshot('#lga')

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