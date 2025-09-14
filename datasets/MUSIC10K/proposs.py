from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np



model = SentenceTransformer('./all-mpnet-base-v2')

des_tsv_path = './MUSIC10K/node_info_artist_familiarity.tsv'  
with open(des_tsv_path, 'r', encoding='utf-8') as file:  
    content = file.read()
lines = content.split('\n')  
attribute_data = [line.strip().split('\t') for line in lines[1:]]


text_inputs = []
score_list  = []
valid_list = []
node_ids = []

for row in attribute_data:
    if len(row)<6:  
        continue

    node_id = row[0]
    name = row[2]
    score = float(row[3])

    valid = int(row[4])
    description = row[5]

    combined_text  = f'{name}: {description}'
    
    text_inputs.append(combined_text)
    score_list.append(score)
    valid_list.append(valid)
    node_ids.append(node_id)

text_embeddings = model.encode(text_inputs, show_progress_bar=True)  # shape = [N, 768 or 384]


score_array = np.array(score_list).reshape(-1, 1)
valid_array = np.array(valid_list).reshape(-1, 1)

final_embeddings = np.hstack([text_embeddings, score_array, valid_array])

   


semantic_vecs = model.encode(df['name'].tolist())



