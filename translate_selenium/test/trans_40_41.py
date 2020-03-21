import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from selenium import webdriver
import time
from pages.main_page import MainPage
import argparse
import json


# python3 trans_40_41.py --enfile /mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko4041_v2_OpenEnded_mscoco_test2015_questions.json --kofile /mnt/crawl/VQA_v2/v2_Questions_Test_mscoco/ko4041_v2_OpenEnded_mscoco_test2015_questions.json


class TranslateRequest(object):

    def translate_request(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
#        options.add_argument('window-size=1920x1080')
        chrome_options.add_argument('--no-sandbox')
#        options.add_argument("disable-gpu")
        chrome_options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(executable_path="/home/elaia/translate_selenium/misc/chromedriver", chrome_options=chrome_options)
        main_page = MainPage(driver=driver)

        main_page.go()
        driver.implicitly_wait(3)

        global i
        global end
        for k in range(100):
            text = text_list[i]['question']
            try:
                main_page.src_textarea.type_into(text)
            except Exception as ex:
                print(ex)
                driver.quit()
                driver = webdriver.Chrome(executable_path="/home/elaia/translate_selenium/misc/chromedriver",
                                          chrome_options=chrome_options)
                main_page = MainPage(driver=driver)

                main_page.go()
                driver.implicitly_wait(3)
                continue

            time.sleep(1.5)
#            main_page.translate_btn.click()
            try:
                text_list[i]['question'] = main_page.tgt_textarea.get_inner_text()
            except Exception as ex:
                print(ex)
                driver.quit()
                driver = webdriver.Chrome(executable_path="/home/elaia/translate_selenium/misc/chromedriver",
                                          chrome_options=chrome_options)
                main_page = MainPage(driver=driver)

                main_page.go()
                driver.implicitly_wait(3)
                continue

            main_page.reset_page.click()
            driver.implicitly_wait(1)
            print(i)
            i +=1
            if i == end:
                break

#        assert result, f"Test Failed: Request page loading timeout"
#        print("Test Passed: Request page loaded")

        driver.quit()

#        ko_dir_path = '/mnt/vqa/Questions_Train_abstract_v002'
#        ko_filepath = os.path.join(ko_dir_path, ko_file)
#        with open(ko_filepath, 'w', encoding='utf-8') as make_file:
        with open(ko_file, 'w', encoding='utf-8') as make_file:
            json.dump(json_data, make_file, indent=4)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enfile", type=str)
    parser.add_argument("--kofile", type=str)

    opt = parser.parse_args()
    en_file = opt.enfile
    ko_file = opt.kofile

#    en_dir_path = '/mnt/vqa/Questions_Train_abstract_v002'
#    en_filepath = os.path.join(en_dir_path, en_file)
#    with open(en_filepath) as json_file:
    with open(en_file) as json_file:
        json_data = json.load(json_file)

    text_list = json_data['questions']
    end = 410000

#    i = 19000
#    i = 38166
#    i = 53956
    i = 407701

    test = TranslateRequest()

    while i < end:
        test.translate_request()
