import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.select import Select
import re,os
import string
from random import *
from webConfig import *
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

class WebDriverManager:
    def __init__(self,browser):
        self.browser = browser
        self.price = 0
    def getDriver(self):
        if(self.browser=='Firefox'):
            from webdriver_manager.firefox import GeckoDriverManager
            return GeckoDriverManager().install()
        if(self.browser == 'Chrome'):
            from webdriver_manager.chrome import ChromeDriverManager
            return ChromeDriverManager().install()
#SETTING

class General(WebConfig):
    def __init__(self ,browser='chrome'):
        self.WebDriverManager(browser)
        self.driver.set_window_position(0, 0)
        self.driver.set_window_size(1920, 1080)

    def getMetaData(self,ele):
        return {
            'lang':self.driver.find_element_by_css_selector('html').get_attribute("lang"),
            'font':ele.value_of_css_property('font-family')
        }

    def isHaveText(self,ele):
        if len(ele.text)>0:
            return True
        return False

    def getAllChild(self,ele):
        return ele.find_elements_by_css_selector("*")
    
    def findContain(self,text, tag='*'):
        try:
            return self.driver.find_element_by_xpath("//{}[contains(text(), {})]".format(tag,text))
        except Exception as e: 
            print(e)
            print("AT: {}".format('Contain: '+text))
            time.sleep(1)
            self.findContain(text)

    def findAtt(self,equation, tag='*'):
        try:
            return self.driver.find_element_by_xpath('//{}[@{}]'.format(tag,equation))
        except Exception as e: 
            print(e)
            print("AT: {}".format('Contain: '+text))
            time.sleep(1)
            self.findContain(text)

    def find(self,css, repeat = 100):
        try:
            return self.driver.find_element_by_css_selector(css)
        except Exception as e: 
            print(e)
            if 'HTTP' in e:
                self.restart()
            print("AT: {}".format(css))
            time.sleep(1)
            self.find(css)

    def extractText(self,ele):
        return

def getFileName(filenum,meta,extenstion):
    # eng.arial.exp2.png
    return meta['lang']+'.'+meta['font']+'.exp'+str(filenum)+'.'+extenstion

def getFileNum(DIR=None):
    if not DIR:
        DIR = '../dataset/images/'
    return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])