import os
import opensearchpy
from opensearchpy import OpenSearch
from opensearchpy.helpers import scan
import numpy as np
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
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
    return [token['token'] for token in response['tokens']]

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


user_query = "안녕하세요라"
tokenized_user_query = analyze_text("kdbtest_vectorized_tokenized_jihoon", user_query, "jihoon_analyzer")

print(f"Original query: {user_query}")
print(f"Tokenized user query: {tokenized_user_query}")

# 고유명사 사전을 업데이트하려면, s3경로에 있는 txt("s3://infra-ai-assistant-opensearch/jihoon_dictionary.txt")
# 수정후, 패키지 업데이트를 해야함(넘나귀찮은것)
# s3에 잇는 원본 txt내용을 업데이트하고 패키지를 업데이트하면서 요 파일을 계속 돌려보면서 달라지는지 확인하자. -> 달라진다 굿
# 실제 lexical retrieve를 할 때는, 이렇게 안하고 'query': {'match': {'question': user_query}},'size': size} 만 해도 자동으로 analyze를 수행한다고 한다!!!오픈서치짱
# 그럼 요걸로 커스텀 사전 테스트 열심히 해보자!!