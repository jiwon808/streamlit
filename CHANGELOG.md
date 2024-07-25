# Changelog

All notable changes to this project will be documented in this file.

## 20240724
0723_crawl_event_naver폴더 안에서 streamlit run 0723.py

-유저 입력의 의도를 claude3.5sonnet으로 분류 : 이벤트 리스트 요청일 때, 분석 요청일 때 두 경우로 분기
-이벤트 리스트 요청인 경우 playwrite를 통한 크롤링으로 네이버의 축제 정보 크롤링, 크롤링한 정보를 통해 claude3.5sonnet으로 입력 의도에 맞춘 이벤트 출력
(매번 크롤링하는게 아니라 매일 사전 정의된 리스트에서 뽑아오도록 수정하기 필요, 유저 입력은 주로 ~지역 이벤트에 대해 묻는 것으로 시작하므로 해당 지역 이벤트만 뽑아오도록 instruction 수정하기 필요. 사전 정의된 이벤트 리스트의 크기가 그렇게 크지 않다면, retrieve과정 생략하고 그대로 input context에 넣어도 무방.
https://docs.anthropic.com/en/docs/about-claude/models에 따르면, claude3.5sonnet의 input context limit은 20만 토큰!)
-분석 요청인 경우 vectorized_qa_dataset.csv를 db로 사용해서, 유저 입력을 벡터화하고 db의 자연어 질의들의 벡터와 비교 후 코사인 유사도 top 3를 뽑은 다음, claude3.5sonnet의
input context에 통합해서 최종 sql문 생성, CoT를 통해 생성된 sql문의 산출 과정을 설명 

-langgraph : 

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
                                   {"LLM_event_list": "LLM_event_list", "Retrieve": "Retrieve"}) 

    workflow.add_edge("LLM_event_list", END)

    workflow.add_edge("Retrieve", "LLM_Final_Generate")
    workflow.add_edge("LLM_Final_Generate", END)

    # Compile
    app = workflow.compile()
    return app

## 20240725

retrieve할 때 opensearch 연결해서,
kdbtest 테이블의 qa데이터의 q를 openai text-embedding-3-large 로 벡터화한 새 테이블인 
kdbtest_vectorized_jihoon 테이블로부터 Lexical/Vector search를 수행하도록 함.
Lexical search의 경우 기본 토크나이징 방법을 사용, Vector search의 경우 코사인 유사도 기반 검색.
input context에는 Lexical search의 결과 3개, Vector search의 결과 3개 총 6개가 입력으로 들어가지만,
Lexical search의 경우 토큰 매칭률이 0%인 입력에 한해 총 결과가 0개가 되어 Vector search의 결과만 fewshot으로 들어가는 경우가 있음
