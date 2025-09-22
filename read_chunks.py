import requests
import json
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    #print("Status code:", r.status_code)
    #print("Full response:", r.json())

    embedding = r.json()["embeddings"]
    return embedding
jsons = os.listdir("jsons")
dict = []
chunk_id = 0
for json_file in jsons:
    with open(f"jsons/{json_file}") as f :
        content = json.load(f)
    print(f"embedding completed{json_file}")
    embeddings = create_embeddings([c["text"] for c in content["chunks"]])

    for i ,chunk in enumerate(content["chunks"]):
        chunk['chunk_id'] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        dict.append(chunk)
        #if (i==5):S
            #break
    
    
#print(dict)
df = pd.DataFrame.from_records(dict)
print(df)
joblib.dump(df , 'embedding.joblib') 

#print(df)
#incoming_query = input("Whats the question ?")
#question_embedding = create_embeddings([incoming_query])[0]
#print(question_embedding)
#vector 2 dimension madhay kam krte mahnun apn tyala np.vstack denr te 2 dimension madhay convert 
#karayla help krte 
#s#imilarities = cosine_similarity(np.vstack(df['embedding']) , [question_embedding]).flatten()
#print(similarities)
#top_result = 3
#max_index = similarities.argsort()[::-1][0 : top_result]
#print(max_index)
#new_df = df.loc[max_index]
#print(new_df[["number","title","text" ]])
