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

def lexical_analyze(index_name, text, analyzer_name): 
    #토크나이징 결과를 시각적으로 확인하기 위한 분석 쿼리. 실제로는 lexical_search도중에 이미 토크나이징이 되지만, 
    #토크나이징된 리스트 확인이 lexical_search의 response로 확인이 안 돼서 만듦
    analyze_query = {
        "analyzer": analyzer_name,
        "text": text
    }
    response = opensearch_client.indices.analyze(index=index_name, body=analyze_query)
    return [token['token'] for token in response['tokens']]

def lexical_search(index_name, user_query, size=3):
    search_query = {
        'query': {
            'match': {
                'question': user_query  # Assuming the documents have a field named 'content'
            }
        },
        'size': size
    }
    response = opensearch_client.search(index=index_name, body=search_query)
    return response['hits']['hits']

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

import json
strings_list = []
with open(os.path.join(os.path.dirname(__file__), 'qa_514.jsonl'), 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        strings_list.append(data["string"])

count = 1
for question in strings_list:
    response_data, tokenized_user_query = analyze_text(KDB_index, question, "jihoon_analyzer")
    print(f"{count}: {tokenized_user_query}")
    count += 1