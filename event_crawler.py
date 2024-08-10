# backlog
# db schema 확정 & s3 연결 
# 효율적으로 db 업데이트(Locality Sensitive Hashing (LSH)?)
# 크롤링 자동화 (aws lambda 사용, https://alsxor5.tistory.com/118)
# get_festival_info_from_naver 시도하면 데이터가 116개 크롤링되야하는거 아닌가? 1st: 63개, 2nd 74개 ??? sleep을 더?


import re
import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def get_festival_info_from_naver(url="https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=축제"):
    
    def extract_text_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        card_areas = soup.find_all('div', class_='card_area')
        texts = [card.get_text(separator=' ', strip=True) for card in card_areas]
        return texts
    
    contents_crawled = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_timeout(100)  # 페이지 로딩 대기시간 조절 : 0.1초

        for page_index in range(1, 30): # 네이버 축제 정보 1~29
            # 페이지 내용 가져오기
            html = page.content()
            texts = extract_text_from_page(html)
            contents_crawled.append(f"{texts}")
            #print(f"Page {page_index} processed.")
            
            # 다음 버튼 클릭
            next_button = page.query_selector('a.pg_next.on[data-kgs-page-action-next]')
            
            if next_button:
                next_button.click()
                page.wait_for_timeout(100)  # 페이지 로딩 대기시간 조절 : 0.1초
            else:
                print(f"Next button not found on page {page_index}")
                break
        
        browser.close()

    texts = contents_crawled[-1]
    texts = texts.split('행사중')[1:]
    results = []

    for i in range(len(texts)):
        texts[i] = texts[i].split('지도 길찾기')[0].strip()
        event_name = texts[i].split('기간')[0].strip()
        event_period = texts[i].split('기간')[-1].split('장소')[0].strip()
        event_period_start = event_period.split(' ~ ')[0]
        event_period_end = event_period.split(' ~ ')[1]
        event_place = texts[i].split('장소')[-1].strip()
        results.append({"event_name":event_name, "event_period_start":event_period_start, "event_period_end":event_period_end, "event_place":event_place})
    
    event_df = pd.DataFrame(results)
    return event_df


def get_show_info_from_naver(url="https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=공연"):
    
    def extract_url_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        a_tags = soup.find_all('a', class_='inner')
        hrefs = ['https://search.naver.com/search.naver' + a_tag['href'] for a_tag in a_tags if 'href' in a_tag.attrs]
        return hrefs
    
    def extract_name_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        card_areas = soup.find_all('strong', class_='_text')
        texts = [card.get_text(separator=' ', strip=True) for card in card_areas]
        return texts

    def extract_info_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        card_areas = soup.find_all('div', class_='info_group')
        texts = [card.get_text(separator=' ', strip=True) for card in card_areas]
        return texts

    event_names = []
    event_period_start = []
    event_period_end = []
    event_place = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_timeout(100)  # 페이지 로딩 대기시간 조절 : 0.1초
        
        # 공연 하이퍼링크 가져오기
        for page_index in range(6): # 네이버 공연 정보 70개 크롤링
            html = page.content()
            texts = extract_url_from_page(html)
            #event_names.append(f"{texts}")
            #print(f"Page {page_index} processed.")
            
            # 다음 버튼 클릭
            next_button = page.query_selector('a.pg_next.on[data-kgs-page-action-next]')
            
            if next_button:
                next_button.click()
                page.wait_for_timeout(100)  # 페이지 로딩 대기시간 조절 : 0.1초
            else:
                print(f"Next button not found on page {page_index}")
                break
            
        
        event_urls = texts

        # 공연정보 가져오기
        for event_url in event_urls : 
            page.goto(event_url)
            page.wait_for_timeout(100)  # 페이지 로딩 대기시간 조절 : 0.1초

            html = page.content()
            names = extract_name_from_page(html)
            texts = extract_info_from_page(html)
            event_names.append(names[0])
    
            for tt in texts :
                if tt.startswith('기간'):
                    date_match = re.findall(r'\d{4}\.\d{2}\.\d{2}\.', tt)

                    if len(date_match) == 1:
                        #하루만 진행하는 공연이라면 시작일=종료일
                        date_str = date_match[0]
                        #date_obj = datetime.strptime(date_str, '%Y.%m.%d')
                        event_period_start.append(date_str)
                        event_period_end.append(date_str)

                    elif len(date_match) == 2:
                        start_date_str, end_date_str = date_match
                        #start_date_obj = datetime.strptime(start_date_str, '%Y.%m.%d')
                        #end_date_obj = datetime.strptime(end_date_str, '%Y.%m.%d')
                        event_period_start.append(start_date_str)
                        event_period_end.append(end_date_str)
                        
                    else:
                        print("Could not find exactly two dates in the text.")

                if tt.startswith('장소'):
                    event_place.append(tt.replace('장소 ', ''))

        browser.close()

    #print(len(event_names), len(event_period_start), len(event_period_end), len(event_place))
    event_df = pd.DataFrame({"event_name":event_names, "event_period_start":event_period_start, "event_period_end":event_period_end, "event_place":event_place})
    
    return event_df


def merge_data():

    # 기존 데이터와 새 데이터 로드
    existing_data = pd.read_csv('event_list.csv')
    festival_data = get_festival_info_from_naver()
    show_data = get_show_info_from_naver()

    # 중복 제거
    combined_data = pd.concat([existing_data, festival_data, show_data]).drop_duplicates(keep='last')

    # 데이터베이스에 업데이트
    combined_data.to_csv('event_list.csv', index=False)

#df = get_festival_info_from_naver()
#print(df.shape)
merge_data()