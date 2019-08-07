import urllib.request
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import re,os, os.path
import string
from random import *
from general import *
from PIL import Image
import io 
from enum import Enum
class News(General):
    def __init__(self ,url,browser='chrome'):
        self.url = url
        self.WebDriverManager(browser)

class VnExpress(Enum):
    textSelector = 'article.content_detail'
    min_height = 30
    min_width = 500

class Screenshot:
    def __init__(self,driver,metadata={}):
        self.driver = driver
        self.imageDir = '../dataset/images/'
        self.extenstion = 'png'
        self.metadata = metadata

    def takePartialScreenshot(self,ele,filenum=0,enum=None):
        element = ele

        location = element.location
        size = element.size
        png = self.driver.get_screenshot_as_png() # saves screenshot of entire page

        im = Image.open(io.BytesIO(png)) # uses PIL library to open image in memory

        left = location['x']
        top = location['y']
        right = location['x'] + size['width']
        bottom = location['y'] + size['height']

        print(size)
        if enum:
            if int(size['width']) <= enum.min_width.value:
                return False
            if int(size['height'])  <= enum.min_height.value:
                return False
        if size['width'] <= 0 or size['height'] <= 0:
            return False

        im = im.crop((left, top, right, bottom)) # defines crop points
        if not filenum:
            filenum = len([name for name in os.listdir(self.imageDir) if os.path.isfile(os.path.join(self.imageDir, name))])
        im.save(self.imageDir+getFileName(filenum,self.metadata,self.extenstion))
        # im.save('screenshot.png')
        return True

    def takeScreenShot(self,div,filenum=0):
        #Get current file num
        if not filenum:
            filenum = len([name for name in os.listdir(self.imageDir) if os.path.isfile(os.path.join(self.imageDir, name))])
        self.driver.save_screenshot(self.imageDir+getFileName(filenum,self.metadata,self.extenstion))