from selenium.webdriver.common.by import By
from misc.base_element import BaseElement
from .base_page import BasePage
from misc.locator import Locator


class MainPage(BasePage):

    url = 'https://papago.naver.com/'

    @property
    def src_textarea(self):
        locator = Locator(by=By.CSS_SELECTOR, value="textarea[name='txtSource']")
        return BaseElement(driver=self.driver, locator=locator)

    @property
    def translate_btn(self):
        locator = Locator(by=By.CSS_SELECTOR, value="button[id='btnTranslate']")
        return BaseElement(driver=self.driver, locator=locator)

    @property
    def x_btn(self):
        locator = Locator(by=By.XPATH, value="//*[@id='sourceEditArea']/button")
        return BaseElement(driver=self.driver, locator=locator)

    @property
    def tgt_textarea(self):
        locator = Locator(by=By.XPATH, value="//*[@id='txtTarget']/span")
        return BaseElement(driver=self.driver, locator=locator)

    @property
    def reset_page(self):
        locator = Locator(by=By.XPATH, value="//*[@id='sourceEditArea']/button")
        return  BaseElement(driver=self.driver, locator=locator)