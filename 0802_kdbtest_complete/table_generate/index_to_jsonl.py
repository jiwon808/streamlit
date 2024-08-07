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

print("OpenSearch Domain Endpoint:", os.getenv('AWS_opensearch_Domain_Endpoint'))
print("OpenSearch ID:", os.getenv('AWS_opensearch_ID'))
print("OpenSearch Password:", os.getenv('AWS_opensearch_PassWord'))

# OpenSearch 클라이언트 생성
opensearch_client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    ssl_show_warn=False,
    timeout=30 #30초 이상 서치하면 넘나 길다.
)
# 데이터를 가져오기 위한 스캔 함수
def fetch_all_documents(index_name, query):
    result = scan(opensearch_client, index=index_name, query=query)
    documents = [doc['_source'] for doc in result]
    return documents

original_index_name = 'kdbtest_vectorized_tokenized_jihoon'
query = {'query': {'match_all': {}}}
documents = fetch_all_documents(original_index_name, query)

data = []
count = 1
for retrieved in documents:
    data.append((count, retrieved['question']))
    count += 1

import json
# jsonl 파일로 저장하는 함수
def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for number, string in data:
            # 각 데이터포인트를 JSON 형식으로 변환하여 파일에 기록
            json_line = json.dumps({"number": number, "string": string}, ensure_ascii=False)
            f.write(json_line + '\n')

# 'data.jsonl' 파일로 저장
save_to_jsonl(data, 'qa_514.jsonl')

print("JSONL 파일이 생성되었습니다.")