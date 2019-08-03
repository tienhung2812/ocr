from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

class WebConfig:
    def WebDriverManager(self,browser='chrome'):
        if browser=='chrome':
            chrome_options = webdriver.ChromeOptions()
            prefs = {"profile.managed_default_content_settings.images": 2}
            chrome_options.add_experimental_option("prefs", prefs)
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            # chrome_options.add_argument("--headless") 
            chrome_options.add_argument('--lang=es')
            chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
            chrome_options.add_argument("--host-resolver-rules=MAP www.google-analytics.com 127.0.0.1")
            chrome_path = ''
            # driver = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver') 
            # self.driver = webdriver.Chrome(chrome_options=chrome_options)
            # self.driver = webdriver.Chrome(executable_path=r'/usr/bin/chromedriver',chrome_options=chrome_options)
            # path = ChromeDriverManager().install()
            # print(path)
            self.driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), chrome_options=chrome_options)
            self.driver.implicitly_wait(30)
            # self.driver.set_page_load_timeout(5)

        elif browser=='firefox':
            firefox_profile = webdriver.FirefoxProfile()
            firefox_profile.set_preference('permissions.default.image', 2)
            firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
            self.driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(),firefox_profile=firefox_profile)
        elif browser =='phantom':
            self.driver = webdriver.PhantomJS(os.path.abspath('./phantomjs/bin/phantomjs'))