from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

tokenizer = BertTokenizer.from_pretrained('pile-of-law/legalbert-large-1.7M-2')
model = BertModel.from_pretrained('pile-of-law/legalbert-large-1.7M-2')

for i in range(min(50, len(trainData))):
    example = trainData[i]

    sources = example['summary/long']

    text=sources
    
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)

    output = model(**encoded_input)

    num_labels = 3
    classifier = LegalDocumentClassifier(hidden_size=model.config.hidden_size, num_labels=num_labels)

  
    class_labels = ["Contract", "Court Opinion", "Statute","Agreement"]
    
    pooler_output = output.pooler_output

    
    logits = classifier(pooler_output)
    predictions = torch.argmax(logits, dim=1)

  
    predicted_labels = [class_labels[pred.item()] for pred in predictions]

    print(f"Document Index {i}, Prediction {predicted_labels}")
