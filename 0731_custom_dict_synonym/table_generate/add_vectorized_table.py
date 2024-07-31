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

def delete_index(index_name):
    client.indices.delete(index=index_name, ignore=[400, 404])
    print(f"{index_name} 인덱스가 삭제되었네.")

# OpenSearch 클라이언트 생성
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    ssl_show_warn=False,
    timeout=30 #30초 이상 서치하면 넘나 길다.
)
# 데이터를 가져오기 위한 스캔 함수
def fetch_all_documents(index_name, query):
    result = scan(client, index=index_name, query=query)
    documents = [doc['_source'] for doc in result]
    return documents
index_name = 'kdbtest'
query = {'query': {'match_all': {}}}
documents = fetch_all_documents(index_name, query)

# 새로운 인덱스 이름
index_name = 'kdbtest_vectorized_jihoon'

os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_KEY')

def LLM_get_embedding(text, model_name="text-embedding-3-large"):
    client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
    response = client.embeddings.create(input=text, model=model_name)
    print(f"임베딩모델이 벡터화 했어 : {text}")
    return response.data[0].embedding

vector_length = len(LLM_get_embedding("아무말"))
# 새로운 인덱스 매핑 설정
mapping = {
  "mappings": {
    "properties": {
      "query": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword",
            "ignore_above": 256
          }
        }
      },
      "question": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword",
            "ignore_above": 256
          }
        }
      },
      "question_vector": {
        "type": "knn_vector",
        "dimension": vector_length
      }
    }
  }
}

delete_index('kdbtest_vectorized_jihoon') #지금 새로 만드려는 테이블이 이미 있으면 지우기 - 계속 새로 만들어보는 실험을 위해서! 기존에 존재하던 인덱스를 삭제하지 않게 주의하기.

# 새로운 인덱스 생성
client.indices.create(index=index_name, body=mapping)

count = 1
for docu in documents:
    doc = {
        "question": docu["question"],
        "query": docu["query"],
        "question_vector": LLM_get_embedding(docu["question"])  # knn_vector로 할당할 배열
    }

    # 문서를 인덱스에 추가
    response = client.index(index=index_name, body=doc)
    print(count, response)
    count += 1
   

mapping = client.indices.get_mapping(index='kdbtest_vectorized_jihoon')
import json
print(json.dumps(mapping, indent=2))