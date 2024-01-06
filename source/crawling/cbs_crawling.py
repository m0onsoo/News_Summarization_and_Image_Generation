from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import requests
import re


# show more 버튼이 있을 때까지 반복해서 클릭
def find_show_more():
    try:
        # 'show more' 버튼을 찾음 (XPath 또는 CSS Selector 사용)
        show_more_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#component-topic-us > div.component__item-wrapper > a'))
        )
        # 버튼 클릭
        show_more_button.click()
    except:
        # 더 이상 'show more' 버튼이 없으면 반복 종료
        return

def crawling(count):
    news_titles, news_links = [], []
    for _ in range(count):
        # 아래로 계속 로딩
        find_show_more()

    # 원하는 만큼 show more 버튼 누른 페이지의 정보 가져오기
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    title_tags = soup.find_all('h4', class_="item__hed")
    for title_tag in title_tags:
        title = title_tag.get_text(strip=True)  # strip=True를 사용하여 앞뒤 공백과 계행 제거
        news_titles.append(title)

    link_tags = soup.find_all('a', class_="item__anchor")
    for link_tag in link_tags:
        news_links.append(link_tag.get('href'))

    driver.quit()
    
    return news_titles, news_links

def news_text_collection(urls):
    contents = []
    for i, url in enumerate(urls):
        print(f"{i}번째 뉴스 진행중")
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('section', {"class": "content__body"})
            texts = content.find_all('p')

            news = []
            for text in texts:
                text = text.get_text()
                clean_text = re.sub(r'[\xa0]', '', text)
                news.append(clean_text)

            result = " ".join(news)
        except:
            result = None

        contents.append(result)
    return contents
            

# 크롬 드라이버 다운로드 및 자동 설정
chrome_driver_path = ChromeDriverManager().install()

# 브라우저 꺼짐 방지 옵션
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

# 불필요한 에러 메시지 삭제
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

# 크롬 브라우저를 열고 cbs news 웹페이지로 이동
url = "https://www.cbsnews.com/latest/us/"
driver = webdriver.Chrome(service=Service(chrome_driver_path), options=chrome_options)
driver.get(url)
driver.maximize_window()
time.sleep(3)

news_titles, news_links, contents = [], [], []
news_titles, news_links = crawling(2)
contents = news_text_collection(news_links)

# 오늘 날짜 얻기
today = datetime.now().date()
df = pd.DataFrame({
    'Title': news_titles,
    'Link': news_links,
    'Contents': contents
})
df['Date'] = today

# 중복된 행 제거
df.drop_duplicates(inplace=True)

current_directory = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(current_directory, "../../data/"))
df.to_csv(path + f"/news_titles_{today}.csv", index=False)
print("End---")