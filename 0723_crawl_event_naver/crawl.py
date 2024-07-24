def get_event_info_from_naver(url="https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=축제"):
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
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
            print(f"Page {page_index} processed.")
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
        event_place = texts[i].split('장소')[-1].strip()
        results.append({"event_name":event_name, "event_period":event_period, "event_place":event_place})
    return results
