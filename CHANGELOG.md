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

## 20240726

0726_custom_dict폴더 안에서 streamlit run 0726.py

Lexical search를 위한 전용 테이블 만듦.

인덱스명 : kdbtest_vectorized_tokenized_jihoon
{'jihoon_analyzer': {'filter': ['lowercase'], 'type': 'custom', 'tokenizer': 'jihoon_dict_tokenizer'}}
{'jihoon_dict_tokenizer': {'type': 'nori_tokenizer', 'user_dictionary': 'analyzers/F120803228', 'decompound_mode': 'mixed'}}

원본 사전은 s3://infra-ai-assistant-opensearch/jihoon_dictionary.txt 
Amazon OpenSearch Service 패키지는 jihoon-dictionary, 패키지ID는 F120803228

streamlit run 0726.py 후 분석 의도의 쿼리가 들어오면 retrieve된 결과들에 대해 2차적으로 lexical_analyze한 로그를 찍음.
이 로그를 기반으로 사전 구성하기

## 20240731

0731_custom_dict폴더 안에서 streamlit run 0731.py

#### nori 사용하는 경우, 한글로만 혼합된 복합어가 공백을 포함할 때는 공백제거된 단어로 사용, 알파벳이나 숫자가 섞이면 언더바로 대체 필요
#### 결국 입력 자연어의 토큰화된 결과를 보고, 최소단위로 표현했을 때 의미를 잃는 조합을 판별해서 분해되기 전 형태로 
#### 직접 synonym과 user_dictionary에 최대한 많이 정의하는 게 핵심임. qa데이터가 다양할수록 다양한 케이스에 대한 BM25 스코어를 높일 수 있을 것.
#### 만들다 보니 BM25말고 rouge스코어 같은, 건너 뛰는 n-gram방식의 유사도 체크에서 큰 이점이 있을 것 같다는 생각이 듦. match기준을 바꿔볼 생각 해야 함.
#### 이렇게 얻은 사전을 통한 lexical search는 형태가 유사한 자연어에 대한 retrieval할 때 매우 유용하게 작용
#### 매우 도메인 한정적인 사전이 만들어 질 것으로 예상됨
#### s3에 저장한 사전 txt 파일을 opensearch 패키지로 다시 참조하게 하는 방식이 사전 업데이트가 매우 느림. 
#### 따라서 그냥 user_dictionary_rules랑 synonyms filter를 인덱스 생성할 때 직접 정의하면서 실험하고 s3의 사전은 한번에 업데이트하는게 훨씬 효율적인 방식임.
