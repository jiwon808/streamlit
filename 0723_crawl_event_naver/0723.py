from dotenv import load_dotenv
import os
import json
import boto3
import streamlit as st
import pandas as pd
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import StreamlitChatMessageHistory
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_SESSION_TOKEN'] = os.getenv('AWS_SESSION_TOKEN')
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_KEY')
os.environ['USER_AGENT'] = 'myagent'

def LLM(LLM_input):
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    bedrock_client = boto3.client('bedrock', region_name='us-east-1')
    response = client.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',  # 사용할 모델 ID
        body=json.dumps({
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": LLM_input}],
        "anthropic_version": "bedrock-2023-05-31"
    })
    )
    response_body = response['body'].read()
    response_json = json.loads(response_body)
    output = response_json['content'][0]['text']
    return output

def LLM_get_embedding(text):
    client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
    response = client.embeddings.create(input=text,model="text-embedding-3-large")
    print(text)
    return response.data[0].embedding

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


def LLM_Router(state):
    print(f"LLM_Router가 질문 분류 중..")
    user_question = state["user_question"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'router.txt')

    with open(file_path, 'r', encoding='utf-8') as file:
        llm_input = file.read()
    llm_input = llm_input.replace('{user_question}', user_question)    

    user_intent = LLM(llm_input)
    state["user_intent"] = user_intent
    print(f"LLM_Router가 {user_intent}로 가라고 합니다")
    return state

def LLM_event_list(state):
    print(f"LLM_event_list가 이벤트 리스트 뽑으려는 중")
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"LLM_event_list가 크롤링 하는 중")
    events_crawled = get_event_info_from_naver("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=축제")
    user_question = state["user_question"]
    print(f"크롤링된 내용 : {events_crawled}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'event_list_generation.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        llm_input = file.read()
    llm_input = llm_input.replace('{current_time}', current_time)    
    llm_input = llm_input.replace('{events_crawled}', str(events_crawled))#리스트여서 문자열로 바꿔 줌
    llm_input = llm_input.replace('{user_question}', user_question)    

    events_output = LLM(llm_input)
    state["events_output"] = events_output
    print(f"LLM_event_list가 뽑은 이벤트 목록 : {events_crawled}")
    return state

def Retrieve(state):
    print(f"Retrieve 가 검색하는 중")
    top_k = 3
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'vectorized_qa_dataset.csv') 
    #db 구성 : question(사전 구성한 유저 자연어 질문),query(사전 구성한 정답 sql 쿼리문),question_vector(사전 구성한 유저 자연의 질문을 벡터화)
    df = pd.read_csv(file_path)

    user_question = state["user_question"]
    user_question_vector = np.array(LLM_get_embedding(user_question)) # 유저가 프론트단에 입력한 실전 자연어 질문을 벡터화(openai : text-embedding-3-large)

    # Convert the string representation of lists to actual lists
    df['question_vector'] = df['question_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

    df['similarity'] = df['question_vector'].apply(lambda x: cosine_similarity([x], [user_question_vector])[0][0]) # 실전 질문과 db속 질문들의 유사도를 벡터기반으로 비교
    top_k_rows = df.nlargest(top_k, 'similarity')[['question', 'query']] # 가장 코사인 유사도가 높은 top 3 선별
    top_k_rows = [(row['question'], row['query']) for _, row in top_k_rows.iterrows()]
    print(f"Retrieve 가 검색한 데이터 k개 : {top_k_rows}")
    state["top_k"] = top_k_rows # langgraph state가 Retrieve노드를 지나면 "top_k" : top_k_rows 정보를 갖게 함

    return state 


def LLM_Final_Generate(state):
    print(f"LLM_Final_Generate가 최종 생성하려는 중")
    user_question = state["user_question"]
    retrieved_top_k = state["top_k"] # retrieve된 top_k_rows
 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'final_generation.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        llm_input = file.read()

    llm_input = llm_input.replace('{user_question}', user_question)
    llm_input = llm_input.replace('{retrieved_top_k}', str(retrieved_top_k))#리스트여서 문자열로 바꿔 줌
    
    final_output = LLM(llm_input) # (instruction + retrieve된 top_k_rows + 실제 유저 입력)을 통해 최종 sql문과 CoT를 통한 생성원인을 출력
    print(f"LLM_Final_Generate가 최종 생성함 : {final_output}")
    state["final_output"] = final_output
    
    return state


class GraphState(TypedDict):
    user_question: str
    user_intent: str
    events_output: str
    top_k: str
    final_output: List[str]
    

def my_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("LLM_Router", LLM_Router)
    workflow.add_node("LLM_event_list", LLM_event_list)
    workflow.add_node("Retrieve", Retrieve) 
    workflow.add_node("LLM_Final_Generate", LLM_Final_Generate)  

    # Build graph
    workflow.add_edge(START, "LLM_Router")
    workflow.add_conditional_edges("LLM_Router",lambda state: "LLM_event_list" if state["user_intent"] == 'EVENT' else "Retrieve", 
                                   {"LLM_event_list": "LLM_event_list", "Retrieve": "Retrieve"}) #없으면 그래프 시각화가 엉뚱하게 됨..

    workflow.add_edge("LLM_event_list", END)

    workflow.add_edge("Retrieve", "LLM_Final_Generate")
    workflow.add_edge("LLM_Final_Generate", END)

    # Compile
    app = workflow.compile()
    return app


def my_graph_image(app):
    return app.get_graph().draw_mermaid_png()


if __name__ == '__main__':

    st.set_page_config(page_title="text2sql webpage")

    st.title("Text2SQL Chatbot")

    with st.sidebar :
        st.subheader("유저 입력이 이벤트 리스트 요청이면 리스트 생성, 분석 요청이면 sql문 생성")
        st.image(my_graph_image(my_graph()))

    st.session_state['conversation'] = None
    st.session_state['chat_history'] = None

    # assistant의 시작 메세지
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "외계인 침공 시 SQL 못 하는 사람이 먼저 잡아 먹힌다 :alien: (제발 저요)"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if user_question := st.chat_input("무엇이든 물어보살"):
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            inputs = GraphState(user_question = user_question)


            for output in my_graph().stream(inputs):
                for node, state in output.items():
                    if node == "LLM_Router":
                        st.markdown(f"쉿! '{node}' 진행 중.")
                        st.markdown("\n")
                        st.markdown(f"user_intent: {state['user_intent']}")
                    elif node == "LLM_event_list":
                        st.markdown(f"쉿! '{node}' 진행 중.")
                        st.markdown("\n")
                        st.markdown(f"events_output: {state['events_output']}")
                    elif node == 'Retrieve':
                        st.markdown(f"쉿! '{node}' 진행 중.")
                        st.markdown("\n")
                        for k, (DB_question, DB_query) in enumerate(state['top_k'], start=1):
                            st.markdown(f"검색된 관련 질문 {k}: {DB_question}")
                            st.markdown("\n")
                    elif node == 'LLM_Final_Generate':
                        st.markdown(f"쉿! '{node}' 진행 중.")
                        st.markdown("\n")
                        response = state["final_output"].replace('\n', '\u2028')  # 웹에서 줄 바꿈 표시
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

