import boto3
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def cosine_similarity_sort(input_word_vector, word_vector_sql_list):

    input_vector = np.array(input_word_vector).reshape(1, -1)
    vectors = np.array([vector for word, vector, sql in word_vector_sql_list])
    similarities = cosine_similarity(input_vector, vectors)[0]
    sorted_word_vector_pairs = sorted(zip(word_vector_sql_list, similarities), key=lambda x: x[1], reverse=True)
    sorted_words = [(word, similarity, sql) for (word, vector, sql), similarity in sorted_word_vector_pairs]

    return sorted_words

def get_top_k(top_k, input_text, vectors_db_json_path):

    input_word_vector = get_embeddings(input_text)

    with open(vectors_db_json_path, 'r') as f:
        vectors_list = json.load(f)

    sorted_words = cosine_similarity_sort(input_word_vector, vectors_list)
    result = [] 
    for word, similarity, sql in sorted_words:
        result.append((word,f'{similarity:.4f}',sql))

    return result[:top_k] # word, cossim, sql

if __name__ == '__main__':
    retrieved = get_top_k(top_k = 3, 
              input_text = '어제 서울Access Infra팀에서 Data CD 카운트가 가장 많은 기지국 10개 뭐야', 
              vectors_db_json_path = r'/mnt/c/Users/HOME/Desktop/skt_6_aifellowship/idcube-ai-assistant-fellowship/vectors_db.json')
    
    print(retrieved)