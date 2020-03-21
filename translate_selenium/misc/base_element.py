from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

set_timeout = 15


class BaseElement(object):
    def __init__(self, driver, locator):
        self.driver = driver
        self.locator = locator
        self.web_element = None
        self.find()

    def find(self):
        element = WebDriverWait(self.driver, set_timeout).until(
        EC.visibility_of_element_located(locator=self.locator))

        self.web_element = element
        return None

    def click(self):
        element = WebDriverWait(self.driver, set_timeout).until(
            EC.element_to_be_clickable(locator=self.locator)
        )
        element.click()
        return None

    def get_inner_text(self):
        text = self.web_element.text
        return text

    def get_typed_text(self):
        text = self.web_element.get_attribute('value')
        return text

    def type_into(self, search_keyword):
        element = WebDriverWait(self.driver, set_timeout).until(
            EC.visibility_of_element_located(locator=self.locator))
        element.send_keys(search_keyword)
        return None

    def hit_enter(self):
        element = WebDriverWait(self.driver, set_timeout).until(
            EC.visibility_of_element_located(locator=self.locator))
        element.send_keys(Keys.ENTER)
        print("Hit Enter after assertion")
        return None

