import streamlit as st
import sqlparse
from langchain.memory import StreamlitChatMessageHistory
import graph_function
import graph_structure


st.set_page_config(page_title="text2sql webpage")
st.title("Text2SQL Chatbot")

with st.sidebar :
    st.subheader("graph structure")
    st.image(graph_structure.my_graph_image(graph_structure.my_graph()))
    st.text("유저 입력이 이벤트 리스트 요청이면 리스트 생성, 분석 요청이면 Lexical+Vector서치 후 sql문 생성")

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

        inputs = graph_structure.GraphState(user_question = user_question)


        for output in graph_structure.my_graph().stream(inputs):
            for node, state in output.items():
                if node == "LLM_Router":
                    st.markdown(f":alien: 쉿! '{node}' 진행 중. :alien:")
                    st.markdown("\n")
                    st.markdown(f"user_intent: {state['user_intent']}")
                    
                elif node == "LLM_event_list":
                    st.markdown(f":alien: 쉿! '{node}' 진행 중. :alien:")
                    st.markdown("\n")
                    st.markdown(f"events_output: {state['events_output']}")

                elif node == 'Retrieve':
                    st.markdown(f":alien: 쉿! '{node}' 진행 중. :alien:")
                    st.markdown("\n")
                    for k, (DB_question, DB_query) in enumerate(state['top_k'], start=1):
                        st.markdown(f"검색된 관련 질문 {k}: {DB_question}")
                        st.markdown("\n")

                elif node == 'LLM_Final_Generate':
                    st.markdown(f":alien: 쉿! '{node}' 진행 중. :alien:")
                    st.markdown("\n")
                    text = state["final_output"]
                    # SQL 코드 추출
                    raw_sql = text.split('```sql')[1].split('```')[0].strip()

                    # SQL 코드 포맷팅
                    formatted_sql = sqlparse.format(raw_sql, reindent=True, keyword_case='upper')

                    # 포맷팅된 SQL 코드를 포함한 응답 생성
                    response = text.replace(raw_sql, formatted_sql)

                    # Streamlit에 표시
                    st.markdown(response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
