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


def get_embedding(text):
    client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
    response = client.embeddings.create(input=text,model="text-embedding-3-large")
    print(text)
    return response.data[0].embedding


def Retrieve(state):
    top_k = 3
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'vectorized_qa_dataset.csv')
    df = pd.read_csv(file_path)

    user_question = state["user_question"]
    user_question_vector = np.array(get_embedding(user_question))

    # Convert the string representation of lists to actual lists
    df['question_vector'] = df['question_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

    df['similarity'] = df['question_vector'].apply(lambda x: cosine_similarity([x], [user_question_vector])[0][0])
    top_k_rows = df.nlargest(top_k, 'similarity')[['question', 'query']]
    top_k_rows = [(row['question'], row['query']) for _, row in top_k_rows.iterrows()]

    return {"user_question" : user_question, "top_k" : top_k_rows} 



def Final_Generate(state):
    user_question = state["user_question"]
    retrieved_top_k = state["top_k"]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'final_generation.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        llm_input = file.read()

    llm_input = llm_input.replace('{user_question}', user_question)
    llm_input = llm_input.replace('{retrieved_top_k}', str(retrieved_top_k))#리스트여서 문자열로 바꿔 줌
    
    final_output = LLM(llm_input)

    return {"user_question" : user_question, "top_k" : retrieved_top_k, "final_output": final_output}


class GraphState(TypedDict):
    user_question: str
    top_k: str
    final_output: List[str]

def my_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("Retrieve", Retrieve)  # Retrieve
    workflow.add_node("Final_Generate", Final_Generate)  # Final_Generate

    # Build graph
    workflow.add_edge(START, "Retrieve")
    workflow.add_edge("Retrieve", "Final_Generate")
    workflow.add_edge("Final_Generate", END)

    # Compile
    app = workflow.compile()
    return app


def my_graph_image(app):
    return app.get_graph(xray=True).draw_mermaid_png()


if __name__ == '__main__':

    st.set_page_config(page_title="text2sql webpage")

    st.title("Text2SQL Chatbot")

    with st.sidebar :
        st.subheader("내 랭그래프가 제일 단순단순")
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
    if user_question := st.chat_input("SQL문으로 바꾸고 싶은 문장을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            inputs = GraphState(user_question = user_question)

            for output in my_graph().stream(inputs):
                for node, state in output.items():
                    if node == 'Retrieve':
                        st.markdown(f"쉿! '{node}' 진행 중.")
                        st.markdown("\n")
                        k = 1
                        for DB_question, DB_query in state['top_k']:
                            st.markdown(f"검색된 관련 질문 {k} : {DB_question}")
                            st.markdown("\n")
                            k += 1
                    elif node == 'Final_Generate':
                        st.markdown(f"쉿! '{node}' 진행 중.")
                        st.markdown("\n")

            response = state["final_output"].replace('\n', '\u2028')  # 웹에서 줄 바꿈 표시
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


