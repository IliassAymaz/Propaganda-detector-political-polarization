"""
BERT for token classification
"""
import torch
from transformers import BertTokenizer, BertForTokenClassification
import pandas as pd
import numpy as np

print(torch.__version__)
train_table = pd.read_csv("train_table.csv")
train_table['hidden_states'] = ''
train_table['loss'] = ''

#define the tokenizer & the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()
print(model.config)

sentence_embeddings = {}
non_propaganda_table = {}
for index, row in train_table.iterrows():
    print(index)
    #sentence_tokens
    train_sentence = row[1]
    train_tokens = tokenizer.tokenize(train_sentence)
    train_tokens_ids = tokenizer.convert_tokens_to_ids(train_tokens)
    train_tokens_ids_tensor = torch.tensor([train_tokens_ids])
    
    #label_tokens
    spans = row[2]
    begin_offset = -1
    end_offset = -1
    if pd.isnull(spans):
        train_labels = [0 for i in range(len(train_tokens_ids))]
        train_labels_tensor = torch.tensor([train_labels])
    else:
        spans_tokens = tokenizer.tokenize(spans)
#        print(repr(train_sentence))
#        print(repr(spans))
        spans_tokens_ids = tokenizer.convert_tokens_to_ids(spans_tokens)
        begin_offset = train_tokens_ids.index(spans_tokens_ids[0])
        end_offset = train_tokens_ids.index(spans_tokens_ids[-1]) + 1
        
        train_labels = [1 if i < end_offset and i >= begin_offset else 0 for i in range(len(train_tokens_ids))]
        train_labels_tensor = torch.tensor([train_labels])
    
    #use the pre-trained model to get token embeddings
    with torch.no_grad():
        outputs = model(train_tokens_ids_tensor, labels=train_labels_tensor)
        loss, scores, hidden_states = outputs[:3]
        
        #hidden_states[layer][0][token][units]
        token_embeddings = []
        for token in range(len(train_tokens)):
            hidden_layers = []
            for layer in range(len(hidden_states)):
                hidden_layers.append(hidden_states[layer][0][token])
            token_embeddings.append(hidden_layers)
        
        #token_embeddings[token][layer][units]
        token_embeddings_all_layers = []
        for token in token_embeddings:
            token_embeddings_all_layers.append(torch.sum(torch.stack(token), dim=0))
        
        #propagandistic span embeddings
        if begin_offset != -1:
            sentence_embeddings[train_sentence] = (spans, row[3], row[4], token_embeddings_all_layers[begin_offset:end_offset])
        #non-propagandistic span embeddings
        else:
            sentence_embeddings[train_sentence] = token_embeddings_all_layers[:]
            non_propaganda_table[train_sentence] = -1


print("Writing to csv file...")
pd.DataFrame.from_dict(data=sentence_embeddings, orient='index').to_csv('train_table_embeddings.csv', header=False, encoding = "utf-8")
#torch.save(sentence_embeddings, "span_embeddings.pt")
pd.DataFrame.from_dict(data=non_propaganda_table, orient='index').to_csv('non_propaganda_table.csv', header=False, encoding = "utf-8")
print("Done.")
