from datasets import load_from_disk
from transformers import BertModel
from transformers import BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import csv


SummEmbeddings=[]

local_dataset_path ='DataSet'
multi_lexsum = load_from_disk(local_dataset_path)

all_scores = []
SumAfterModel=[]
reference_summary=[]
cosSim_scores = []


def compute_similarity(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)

def generate_bert_embeddings(text):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    tokenized_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True)


    with torch.no_grad():
        outputs = model(**tokenized_text)
        

    
    embeddings = outputs.last_hidden_state.mean(dim=1)  

    
    return embeddings  


    
trainData = multi_lexsum['train']
tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-led-base-16384")
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-led-base-16384")
    
    
padding = "max_length"
    
    
for i in range(min(100, len(trainData))):
        
    if i == 0:
        print("This is printed only in the first iteration")

    example = trainData[i]

    sources = example['sources']

    embeding=generate_bert_embeddings(sources)
   
    text = f"""{embeding}"""
    
    input_tokenized = tokenizer.encode(text, return_tensors='pt', padding=padding, pad_to_max_length=True, max_length=6144, truncation=True)

    summary_ids = model.generate(
    input_tokenized,
    num_beams=8,
    no_repeat_ngram_size=3,
    length_penalty=2,
    min_length=200,
    max_length=450
    )


    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
   
    summaryLong=multi_lexsum['validation'][i]['summary/long']
    
    SumAfterModel.append(generated_summary)
    reference_summary.append(summaryLong)

    print(f"Summary No {i}: {generated_summary}\n\n")



for i in range(len(SumAfterModel)):
    
    if i == 0:
        print("This is printed only in the first iteration : Begin Computing Cosin Similarities.")
        

    embedding1 = generate_bert_embeddings(SumAfterModel[i])
    embedding2 = generate_bert_embeddings(reference_summary[i])

    cosSim = compute_similarity(embedding1, embedding2)

    print(f"Document NO {i} Cosin Smiliarity Score : {cosSim}")
    cosSim_scores.append(cosSim)
        
    
csv_file_path = r"D:\Gam3a\Grad I\Imp\FineTuned_Cos_Similarity_Scores.csv"


with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Document No.", "Cosine Similarity Score"])  # Write header row
    for i in range(len(cosSim_scores)):
        if i == 0:
            print("This is printed only in the first iteration : Begin Writing in the csv.")
        score = cosSim_scores[i]
        writer.writerow([i, score])
            
    print("CSV file saved successfully.")
   


    