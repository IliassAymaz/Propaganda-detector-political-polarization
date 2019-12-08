"""
BERT for token classification
"""
import torch
from transformers import BertTokenizer, BertForTokenClassification
import pandas as pd
import numpy as np

train_table = pd.read_csv("train_table_.csv")
train_table['loss'] = ''
train_table['hidden_states'] = ''

#define the tokenizer & the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()

sentence_embeddings = []
for index, row in train_table.iterrows():
    #sentence_tokens
    train_sentence = row[0]
    train_tokens = tokenizer.tokenize(train_sentence)
    train_tokens_ids = tokenizer.convert_tokens_to_ids(train_tokens)
    train_tokens_ids_tensor = torch.tensor([train_tokens_ids])
    
    #label_tokens
    if pd.isnull(row[1]):
        train_labels = [0 for i in range(len(train_tokens_ids))]
        train_labels_tensor = torch.tensor([train_labels])
    else:
        spans = row[1]
        spans_tokens = tokenizer.tokenize(spans)
        spans_tokens_ids = tokenizer.convert_tokens_to_ids(spans_tokens)
        begin_offset = train_tokens_ids.index(spans_tokens_ids[0])
        end_offset = train_tokens_ids.index(spans_tokens_ids[-1]) + 1
        
        train_labels = [1 if i < end_offset and i >= begin_offset else 0 for i in range(len(train_tokens_ids))]
        train_labels_tensor = torch.tensor([train_labels])
        
    #pre-train the model
    with torch.no_grad():
        outputs = model(train_tokens_ids_tensor, labels=train_labels_tensor)
        loss, scores, hidden_states = outputs[:3]
        train_table.iloc[index, 2] = loss.item()
        train_table.iloc[index, 3] = hidden_states[0]
    

train_table.to_csv("train_table_embeddings.csv", index = False, encoding = "utf-8")


#    token_embeddings = []
#    print(hidden_states[0][0])
#    for token in range(len(train_tokens)):
#        hidden_layers = (hidden_states[0][0][token])
#        token_embeddings.append(hidden_layers)
#    
#    print(len(token_embeddings))
#    #all_layers = [torch.sum(torch.stack(layer)[:], 0) for layer in token_embeddings]
#    
#    sentence_embeddings.append(token_embeddings)


##define the tokenizer & the model
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForTokenClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)
#model.eval()
#
##train: tokens
#train_sentence = "Manchin says Democrats acted like babies at the SOTU (video) Personal Liberty Poll Exercise your right to vote."
#train_tokens = tokenizer.tokenize(train_sentence)
#train_tokens_ids = tokenizer.convert_tokens_to_ids(train_tokens)
#train_tokens_ids_tensor = torch.tensor([train_tokens_ids])
#
##train: labels
#begin_offset = 34
#end_offset = 40
#tagged_fragments = train_sentence[begin_offset:end_offset]
#tagged_fragments_tokens = tokenizer.tokenize(tagged_fragments)
#tagged_fragments_tokens_ids = tokenizer.convert_tokens_to_ids(tagged_fragments_tokens)
#begin_offset_token = train_tokens_ids.index(tagged_fragments_tokens_ids[0])
#end_offset_token = train_tokens_ids.index(tagged_fragments_tokens_ids[-1]) + 1
#
#train_labels = [1 if i < end_offset_token and i >= begin_offset_token else 0 for i in range(len(train_tokens_ids))]
#train_labels_tensor = torch.tensor([train_labels])
#
##pre-train the model
#outputs = model(train_tokens_ids_tensor, labels=train_labels_tensor)
#loss, scores, hidden_states = outputs[:3]
#print(loss.data)


#
#test_sentence = "In a glaring sign of just how stupid and petty things have become in Washington these days."
#
#test_tokens = tokenizer.tokenize(test_sentence)
#test_tokens_ids = tokenizer.convert_tokens_to_ids(test_tokens)
#test_tokens_ids_tensor = torch.tensor([test_tokens_ids])
#
#with torch.no_grad():
#    logits = model(test_tokens_ids_tensor)
#
#logits = logits.detach().cpu().numpy()
#
#predictions = []
#predictions.extend([list(p) for p in numpy.argmax(logits, axis=2)])
#
#print(predictions)

