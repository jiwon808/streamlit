import pandas as pd
import os
import pandas as pd
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'notsim_qa.csv')
df = pd.read_csv(file_path)



def get_embedding(text):
    client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
    response = client.embeddings.create(input=text,model="text-embedding-3-large")
    print(text)
    return response.data[0].embedding

df['question_vector'] = df['question'].apply(lambda x: get_embedding(x))

# 변경된 DataFrame을 새로운 CSV 파일로 저장합니다.
output_file_path = 'vectorized_qa_dataset.csv'
df.to_csv(output_file_path, index=False)
print(f"CSV 파일에 새로운 컬럼이 추가되었습니다. 새로운 파일이 '{output_file_path}'에 저장되었습니다.")
