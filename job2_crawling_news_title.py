from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import datetime

options = Options()
user_agent = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 '
              'Safari/537.36')
options.add_argument('user-agent=' + user_agent)
options.add_argument('lang=ko_KR')
options.add_argument('window-size=1920x1080')
options.add_argument('disable-gpu')
options.add_argument('--no-sandbox')

with open('./chrome_driver_path.txt', 'r') as f:
    service = Service(executable_path=f.read())

driver = webdriver.Chrome(service=service, options=options)

category = ['Politic', 'Economic', 'Social', 'Culture', 'World', 'IT']
pages = [110, 110, 110, 75, 110, 72]

df_title = pd.DataFrame()
for category_idx in range(4, 6):
    refined_titles = []
    url_segment = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(category_idx)

    for page in range(1, pages[category_idx]):
        url = url_segment + '#&date=%2000:00:00&page={}'.format(page)
        driver.get(url)

        for ul_idx in range(1, 5):
            for li_idx in range(1, 6):
                while True:  # infinite loop for 'StaleElementReferenceException'
                    try:
                        title = driver.find_element('xpath', '//*[@id="section_body"]/ul[{0}]/li[{1}]/dl/dt[2]/a'.format(ul_idx, li_idx)).text
                    except StaleElementReferenceException:
                        continue
                    except NoSuchElementException:
                        # print(category_idx, page, ul_idx, li_idx)
                        try:
                            title = driver.find_element('xpath', '//*[@id="section_body"]/ul[{0}]/li[{1}]/dl/dt/a'.format(ul_idx, li_idx)).text
                        except NoSuchElementException:
                            break
                    refined_title = re.compile('[^가-힣]').sub(' ', title)
                    refined_titles.append(refined_title)
                    break

    df_section_title = pd.DataFrame(refined_titles, columns=['title'])
    df_section_title['category'] = category[category_idx]
    df_title = pd.concat([df_title, df_section_title], axis='rows', ignore_index=True)
df_title.to_csv('./crawling_data/naver_news_world_IT.csv', index=False)

df_title.info()
print(df_title['category'].value_counts())
