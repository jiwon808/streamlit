import pandas as pd
from langchain_community.chat_models import BedrockChat
import boto3
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
import pytz
import os
import re
from botocore.config import Config
from langchain.prompts import PromptTemplate
import json
import boto3
from get_top_k import get_top_k

client = boto3.client('bedrock-runtime', region_name='us-east-1')
bedrock_client = boto3.client('bedrock', region_name='us-east-1')

def call_bedrock_api(user_input):
    response = client.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',  # 사용할 모델 ID
        body=json.dumps({
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": user_input}],
        "anthropic_version": "bedrock-2023-05-31"
    })
    )
    response_body = response['body'].read()
    response_json = json.loads(response_body)
    print('-'*50)
    print(response_json['content'][0]['text'])
    print('-'*50)

def generate_final_SQL(user_input : str = '오늘 서울Access Infra팀에서 Data CD 카운트가 가장 많은 기지국 10개 알려줘',
                       retrive_top_k : int = 3,
                       retrive_DB_json_path : str = r'/mnt/c/Users/HOME/Desktop/skt_6_aifellowship/idcube-ai-assistant-fellowship/vectors_db.json',
                       instruction_path : str = r'/mnt/c/Users/HOME/Desktop/skt_6_aifellowship/idcube-ai-assistant-fellowship/instruction.txt'
                       ):
    retrieved_sentence_sql_example_list = get_top_k(top_k = retrive_top_k, 
              input_text = user_input, 
              vectors_db_json_path = retrive_DB_json_path)
    
    retrieved_sentence_sql_example_text_format = []
    for sentence, cossim, sql in retrieved_sentence_sql_example_list:
        retrieved_sentence_sql_example_text_format.append(f"INPUT : \n{sentence}\nOUTPUT : \n{sql}\n\n")

    retrieved_sentence_sql_example = "".join(retrieved_sentence_sql_example_text_format)

    with open(instruction_path, 'r', encoding='utf-8') as file:
        template = file.read()
    
    template = template.replace('{user_input}', user_input)
    template = template.replace('{retrieved_sentence_sql_example}', retrieved_sentence_sql_example)
    llm_input = template

    print('-'*50)
    print(f'llm_final_input : ')
    print('-'*50)
    print(f'\n{llm_input}\n')

    return call_bedrock_api(llm_input)
    
if __name__ == "__main__":
    generate_final_SQL(user_input = '오늘 서울Access Infra팀에서 Data CD 카운트가 가장 많은 기지국 14개 알려줘',
                       retrive_top_k = 3,
                       retrive_DB_json_path = r'/mnt/c/Users/HOME/Desktop/skt_6_aifellowship/idcube-ai-assistant-fellowship/vectors_db.json',
                       instruction_path = r'/mnt/c/Users/HOME/Desktop/skt_6_aifellowship/idcube-ai-assistant-fellowship/instruction.txt'
                       )

