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

file_path = r'/mnt/c/Users/HOME/Desktop/skt_6_aifellowship/idcube-ai-assistant-fellowship/data_from_Chayoung/template_qa.xlsx'
xlsx = pd.ExcelFile(file_path)

all_data = []
for sheet_name in xlsx.sheet_names:
    df = pd.read_excel(xlsx, sheet_name=sheet_name)
    # 각 행을 튜플로 변환하여 리스트에 추가
    sheet_data = [tuple(row) for row in df.to_numpy()]
    all_data.extend(sheet_data)

vectors = []
for sentence, sql in all_data:
    sentence_vector = get_embeddings(sentence)
    print(sentence, sentence_vector, sql)
    vectors.append((sentence, sentence_vector, sql))
    
with open('vectors.json', 'w') as f:
    json.dump(vectors, f)

