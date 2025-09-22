import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import requests

df = joblib.load('embedding.joblib')
def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    #print("Status code:", r.status_code)
    #print("Full response:", r.json())
    embedding = r.json()["embeddings"]
    return embedding
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    response = r.json()
    print(response)
    return response


incoming_query = input("Whats the question ?")
question_embedding = create_embeddings([incoming_query])[0]

#print(question_embedding)
#vector 2 dimension madhay kam krte mahnun apn tyala np.vstack denr te 2 dimension madhay convert 
#karayla help krte 

similarities = cosine_similarity(np.vstack(df['embedding']) , [question_embedding]).flatten()
top_result = 5
max_index = similarities.argsort()[::-1][0 : top_result]
#print(max_index)
new_df = df.loc[max_index]
#print(new_df[["number","title","text" ]])
promt = f'''This is a  web development course . Here are video subtitle chunks containing video title,
 video number , start time in seconds , end time in seconds , the text at that time :

{new_df[["title" , "number" , "start" , "end" , "text"]].reset_index(drop=True).to_json()}s
------------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks , you have to answer where and how much
content is taught in which video(in which video and at what timestamp) and guide the user to go 
to that particular video by using number eg : (01 , 02 ,03) take referance from this {new_df["number"]} don't guide by index chunk  . If user asks unrelated question , tell him that you can only asnwer question related to the course
'''
for index,item in new_df.iterrows():
    print(index , item["title"] , item["number"] ,item["text"], item["start"], item["end"])
with open("promt.txt" , "w")as f:
    f.write(promt)
#ha dictionary dete mhnun tya dict madhun response baher kadaycha ["response"] use krun
response = inference(promt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
print(new_df["number"])