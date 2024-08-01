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


####################################################
################ 새로운 인덱스 정보 정의 #############
custom_dict_s3_uri = "s3://infra-ai-assistant-opensearch/jihoon_dictionary.txt"
New_index_name = 'kdbtest_vectorized_tokenized_jihoon'
New_custom_tokenizer_name = "jihoon_dict_tokenizer"
New_analyzer_name = "jihoon_analyzer"
custom_dict_package_id = "F120803228"  
tokenizer_type = "nori_tokenizer" #"nori_tokenizer", "standard"
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
          "decompound_mode": "none", #mixed, none
          # "user_dictionary": f"analyzers/{custom_dict_package_id}", # S3 저장된 txt 기준
          "user_dictionary_rules": [ # nori 사용하는 경우, 한글로만 혼합된 복합어가 공백을 포함할 때는 공백제거된 단어로 사용, 알파벳이나 숫자가 섞이면 언더바로 대체 필요
            "access_infra",
            "5_g",
            "품질지표",
            "시군구",
            "동단위",
            "경험지수",
            "성능지표",
            "성능데이터",
            "춘천품질개선팀",
            "전남북",
            "운용중",
            "du_장비",
            "일일",
            "제조사",
            "enodeb",
            "id",
            "핸드오버",
            "성공률",
            "상세정보",
            "중계기",
            "함체온도",
            "duh",
            "호수",
            "동시접속자",
            "비정상",
            "무선국허가번호",
            "수도권",
            "유선망",
            "백홀",
            "e_2_e",
            "t_3_k",
            "허가번호",
            "시설코드",
            "통신시설",
            "축전지",
            "운용본부조직",
            "lte",
            "elg", 
            "enb",
            "prb",
            "사용률",
            "지목코드",
            "시간대별",
            "평균온도",
            "현장운용팀",
            "물건번호",
            "통합시설정보",
            "장비유형",
            "유효전력",
            "개통일",
            "상세정보",
            "절단호",
            "mcs",
            "bler",
            "air",
            "mac", 
            "dl", 
            "byte",
            'scg',
            "mbps",
            "복구정보",
            "알람코드",
            "erp",
            "망사업자",
            "rach",
            "시도호",
            "nokia",
            'prb',
            "사용률",
            "endc",
            "동시접속",
            "서비스품질팀",
            "residual",
            "bler",
            "id",
            "보고시간",
            "amp",
            "standby",
            "siso",
            "mimo",
            "path",
            "미사용",
            "절단호",
            "npr",
            "시간대",
            'nams',
            'rtwp',
            "품질개선",
            "지역본부",
            "gnb",
            "무선국",
            "광레벨데이터",
            "qos",
            "ran",
            "제조사",
            "개통일",
            "3_g",
            "4_g",
            "이어도",
            "상세정보",
            "드롭률",
            "cei",
            "cfi",
            'tango',
            "유형별",
            "회선별",
            "장비별",
            "경로별",
            "진도지산",
            "국소명",
            "설치일",
            "운용상태",
            "자산상태",
            "장비명",
            "단위",
            "불일치",
            "에릭슨엘지",
            "물건번호",
          ]
        }
      },
      "filter": {
        "nori_number_filter": {
          "type": "nori_number"
        },
        "synonym_filter": { 
          "type": "synonym",
          "synonyms": [
            "5g => 5_g",
            "3g => 3_g",
            "4g => 4_g",
            "access infra => access_infra",
            "시군 구 => 시군구",
            "시 군구 => 시군구",
            "동 단위 => 동단위",
            "품질 지표 => 품질지표",
            "기지국 별 => 기지국별",
            "경험 지수 => 경험지수",
            "성능 지표 => 성능지표",
            "성능 데이터 => 성능데이터",
            "춘천품질개선팀 => 춘천품질개선팀",
            "전남북 => 전남북",
            "운용 중 => 운용중",
            "du장비 => du_장비", 
            "du 장비 => du_장비",
            "일일 => 일일",
            "제조사 => 제조사",
            "dl => dl",
            "mimo => mimo",
            "enodeb => enodeb",
            "id => id",
            "핸드 오버 => 핸드오버",
            "핸드오버 => 핸드오버",
            "성공률 => 성공률",
            "상세 정보 => 상세정보",
            "중계기 => 중계기",
            "함체 온도 => 함체온도",
            "duh => duh",
            "호 수 => 호수",
            "동시접속자 => 동시접속자",
            "비정상 => 비정상",
            "무선국허가번호 => 무선국허가번호",
            "수도권 => 수도권",
            "유선 망 => 유선망",
            "유선망 => 유선망",
            "백홀 => 백홀",
            "백 홀 => 백홀",
            "e2e => e_2_e",
            "t3k => t_3_k",
            "허가 번호 => 허가번호",
            "시설 코드 => 시설코드",
            "통신 시설 => 통신시설",
            "운용 본부 조직 => 운용본부조직",
            "운용본부 조직 => 운용본부조직",
            "지목 코드 => 지목코드",
            "시간대 별 => 시간대별",
            "평균 온도 => 평균온도",
            "물건 번호 => 물건번호",
            "통합시설 정보 => 통합시설정보",
            "통합 시설 정보 => 통합시설정보",
            "장비 유형 => 장비유형",
            "유효 전력 => 유효전력",
            "상세 정보 => 상세정보",
            "복구 정보 => 복구정보",
            "알람 코드 => 알람코드",
            "망 사업자 => 망사업자",
            "동시 접속 => 동시접속",
            "서비스 품질 팀 => 서비스품질팀",
            "서비스품질 팀 => 서비스품질팀",
            "보고 시간 => 보고시간",
            "품질 개선 => 품질개선",
            "지역 본부 => 지역본부",
            "상세 정보 => 상세정보",
            "tango => tango",
            "탱고 => tango",
            "텡고 => tango",
            "유형 별 => 유형별",
            "회선 별 => 회선별",
            "장비 별 => 장비별",
            "경로 별 => 경로별",
            "국소 명 => 국소명",
            "운용 상태 => 운용상태",
            "자산 상태 => 자산상태",
            "셀 => 셀",
            "쎌 => 셀",
            "cell => 셀",
            "에릭슨 엘지 => 에릭슨엘지",
            "물건 번호 => 물건번호",
            
          ]
        },
        "lowercase": {
          "type": "lowercase"
        }
      },
      "analyzer": {
        New_analyzer_name: {
          "type": "custom",
          "tokenizer": New_custom_tokenizer_name,
          "filter": [
            "lowercase",
            # "nori_number_filter", 일일 을 11 로 바꾸는 문제가 있음
            "synonym_filter",
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

