import boto3
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import defaultdict

client = boto3.client('bedrock-runtime', region_name='us-east-1')

def get_embeddings(text):
    payload = {
        "texts": [text],
        "input_type": "search_document"
    }
    response = client.invoke_model(
        modelId="cohere.embed-multilingual-v3",
        contentType="application/json",
        accept="*/*",
        body=json.dumps(payload)
    )
    response_body = response['body'].read()
    return json.loads(response_body)['embeddings'][0]

# Example usage
vectors = []

file_path = r'/mnt/c/Users/HOME/Desktop/skt_6_aifellowship/idcube-ai-assistant-fellowship/data_from_Chayoung/template_qa.xlsx'
xlsx = pd.ExcelFile(file_path)

all_data = []
for sheet_name in xlsx.sheet_names:
    df = pd.read_excel(xlsx, sheet_name=sheet_name)
    # 각 행을 튜플로 변환하여 리스트에 추가
    sheet_data = [tuple(row) for row in df.to_numpy()]
    all_data.extend(sheet_data)

all_data = all_data#


for sentence, sql in all_data:
    vector = get_embeddings(sentence)
    vectors.append((sentence, vector))


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_sort(input_word_vector, word_vector_list):
    # 입력 벡터와 리스트 내 모든 벡터 간의 코사인 유사도를 계산
    input_vector = np.array(input_word_vector).reshape(1, -1)
    vectors = np.array([vector for word, vector in word_vector_list])
    
    similarities = cosine_similarity(input_vector, vectors)[0]
    
    # 유사도에 따라 단어를 높은 순으로 정렬
    sorted_word_vector_pairs = sorted(zip(word_vector_list, similarities), key=lambda x: x[1], reverse=True)
    
    # 결과 출력
    sorted_words = [(word, similarity) for (word, vector), similarity in sorted_word_vector_pairs]
    return sorted_words

word_vector_list = vectors

input_text = '광주 공동망 사용자수 알려줘'

input_word_vector = get_embeddings(input_text)

# 함수 실행
sorted_words = cosine_similarity_sort(input_word_vector, word_vector_list)

# 결과 출력
for word, similarity in sorted_words:
    print(f"Word: {word}, Cosine Similarity: {similarity:.4f}")


words = [word for word, similarity in sorted_words]
similarities = [similarity for word, similarity in sorted_words]

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from matplotlib import rcParams

font_path = r'/mnt/c/Users/HOME/Desktop/skt_6_aifellowship/idcube-ai-assistant-fellowship/NanumGothic.ttf'
# 폰트를 matplotlib에 등록
fontprop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# 폰트를 matplotlib의 기본 폰트로 설정
rcParams['font.family'] = fontprop.get_name()


# 그래프 생성
plt.figure(figsize=(10, 30))
plt.barh(words, similarities, color='skyblue')
plt.xlabel(f'Cosine Similarity - {input_text}', fontproperties=fontprop)
plt.ylabel('Words', fontproperties=fontprop)
plt.title(f'Cosine Similarity of {input_text}', fontproperties=fontprop)
plt.gca().invert_yaxis()  # y축 뒤집기
plt.grid(axis='x')

# 이미지 파일로 저장
plt.savefig('cosine_similarity_chart.png')

# 그래프 보여주기
plt.show()
    