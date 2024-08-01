import os
import opensearchpy
from opensearchpy import OpenSearch
from opensearchpy.helpers import scan
import numpy as np
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)
from openai import OpenAI

import opensearchpy
print(opensearchpy.__version__)
# OpenSearch 설정
host = os.getenv('AWS_opensearch_Domain_Endpoint')  # OpenSearch 도메인 엔드포인트
port = 443  # 기본 포트
auth = (os.getenv('AWS_opensearch_ID'), os.getenv('AWS_opensearch_PassWord'))  

# OpenSearch 클라이언트 생성
opensearch_client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    ssl_show_warn=False,
    timeout=30 #30초 이상 서치하면 넘나 길다.
)

def analyze_text(index_name, text, analyzer_name):
    analyze_query = {
        "analyzer": analyzer_name,
        "text": text
    }
    response = opensearch_client.indices.analyze(index=index_name, body=analyze_query)

    return response, [token['token'] for token in response['tokens']]

###################################################3
KDB_index = "kdbtest_vectorized_tokenized_jihoon"
####################################################

#껏다키기만악의근원은껏다키지않은것,,,
response = opensearch_client.snapshot.status()
print(response)
response = opensearch_client.indices.close(index=KDB_index)
response = opensearch_client.indices.open(index=KDB_index)

mapping = opensearch_client.indices.get_mapping(index=KDB_index)
import json
print(json.dumps(mapping, indent=2))

response = opensearch_client.indices.get(index=KDB_index)
settings = response[KDB_index]['settings']['index']['analysis']
analyzer_setting = settings['analyzer']
analyzer_name = str(list(analyzer_setting.keys())[0])
tokenizer_setting = settings['tokenizer']
tokenizer_name = str(list(tokenizer_setting.keys())[0])
print(f"인덱스 이름 : {KDB_index}")
print(f"이 인덱스의 lexical 세팅값\n")
print("<<<<<<<<<<<<<<<<<<<<<<<")
print(analyzer_setting)
print(tokenizer_setting)
print(">>>>>>>>>>>>>>>>>>>>>>>\n")


user_query = "전북 지역의 특정 중계기들에 대한 어제의 무선 및 유선 보고서 정보와 주파수별 전방 출력 데이터를 어떻게 확인할 수 있나요?"
response_data, tokenized_user_query = analyze_text(KDB_index, user_query, "jihoon_analyzer")

# 고유명사 사전을 업데이트하려면, s3경로에 있는 txt("s3://infra-ai-assistant-opensearch/jihoon_dictionary.txt")
# 수정후, 패키지 업데이트를 해야함(넘나귀찮은것)
# s3에 잇는 원본 txt내용을 업데이트하고 패키지를 업데이트하면서 요 파일을 계속 돌려보면서 달라지는지 확인하자. -> 달라진다 굿
# 실제 lexical retrieve를 할 때는, 이렇게 안하고 'query': {'match': {'question': user_query}},'size': size} 만 해도 자동으로 analyze를 수행한다고 한다!!!오픈서치짱
# 그럼 요걸로 커스텀 사전 테스트 열심히 해보자!!

# 20240801
# nori 사용하는 경우, 한글로만 혼합된 복합어가 공백을 포함할 때는 공백제거된 단어로 사용, 알파벳이나 숫자가 섞이면 언더바로 대체 필요
# 결국 입력 자연어의 토큰화된 결과를 보고, 최소단위로 표현했을 때 의미를 잃는 조합을 판별해서 분해되기 전 형태로 
# 직접 synonym과 user_dictionary에 최대한 많이 정의하는 게 핵심임. qa데이터가 다양할수록 다양한 케이스에 대한 BM25 스코어를 높일 수 있을 것. 
# 이렇게 얻은 사전을 통한 lexical search는 형태가 유사한 자연어에 대한 retrieval할 때 매우 유용하게 작용, but 다른 회사에 판매 가능할까??는 모르겠음..
# 매우 도메인 한정적인 사전이 만들어 질 것으로 예상됨
# s3에 저장한 사전 txt 파일을 opensearch 패키지로 다시 참조하게 하는 방식이 사전 업데이트가 매우 느림. 
# 그냥 user_dictionary_rules랑 synonyms filter를 인덱스 생성할 때 직접 정의하면서 실험하고 s3의 사전은 한번에 업데이트하는게 훨씬 효율적인 방식임.

print("-"*100)
print(f"Original query: {user_query}")
print("-"*100)
print(f"Tokenized user query: {tokenized_user_query}")
print("-"*100)


