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

def delete_index(index_name):
    opensearch_client.indices.delete(index=index_name, ignore=[400, 404])
    print(f"{index_name} 인덱스가 삭제되었네.")

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
original_index_name = 'kdbtest_vectorized_jihoon'
query = {'query': {'match_all': {}}}
documents = fetch_all_documents(original_index_name, query)

#######
####################################################
################ 새로운 인덱스 정보 정의 #############
custom_dict_s3_uri = "s3://infra-ai-assistant-opensearch/jihoon_dictionary.txt"
New_index_name = 'kdbtest_vectorized_tokenized_jihoon'
New_custom_tokenizer_name = "jihoon_dict_tokenizer"
New_analyzer_name = "jihoon_analyzer"
custom_dict_package_id = "F120803228"  
tokenizer_type = "nori_tokenizer"
####################################################
####################################################

# 새로운 인덱스 매핑 설정
mapping = {
  "settings": {
    "index": {
      "knn": True  # KNN 검색을 활성화
    },
    "analysis": {
      "tokenizer": {
        New_custom_tokenizer_name: {
          "type": tokenizer_type,
          "decompound_mode": "mixed",
          "user_dictionary": f"analyzers/{custom_dict_package_id}"
        }
      },
      "analyzer": {
        New_analyzer_name: {
          "type": "custom",
          "tokenizer": New_custom_tokenizer_name,
          "filter": [
            "lowercase"
          ]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "query": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword",
            "ignore_above": 256
          }
        },
        "analyzer": New_analyzer_name
      },
      "question": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword",
            "ignore_above": 256
          }
        },
        "analyzer": New_analyzer_name
      },
      "question_vector": {
        "type": "knn_vector",
        "dimension": 3072,
        "index": True  # 벡터 필드의 인덱싱을 활성화
      }
    }
  }
}

delete_index(New_index_name) #지금 새로 만드려는 테이블이 이미 있으면 지우기 - 계속 새로 만들어보는 실험을 위해서! 기존에 존재하던 인덱스를 삭제하지 않게 주의하기.

# 새로운 인덱스 생성
opensearch_client.indices.create(index=New_index_name, body=mapping)

count = 1
for docu in documents:
    doc = {
        "question": docu["question"],
        "query": docu["query"],
        "question_vector": docu["question_vector"]  
    }

    # 문서를 인덱스에 추가
    response = opensearch_client.index(index=New_index_name, body=doc)
    print(count, response)
    count += 1
   

mapping = opensearch_client.indices.get_mapping(index=New_index_name)
import json
print(json.dumps(mapping, indent=2))

