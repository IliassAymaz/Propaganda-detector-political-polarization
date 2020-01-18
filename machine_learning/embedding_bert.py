"""
BERT for token classification
"""
import torch
from transformers import BertTokenizer, BertForTokenClassification
import pandas as pd
import numpy as np
    
def get_bert_embedding(model, tokenizer, sentence, span):
    #convert sentence into tokens
    sentence_tokens = tokenizer.tokenize(sentence)
    sentence_tokens_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    sentence_tokens_ids_tensor = torch.tensor([sentence_tokens_ids])
    
    #convert label into tokens
    begin_offset = -1
    end_offset = -1
    if pd.isnull(span):
        #non-propaganda
        labels = [0 for i in range(len(sentence_tokens_ids))]
        labels_tensor = torch.tensor([labels])
    else:
        #contains propaganda
        span_tokens = tokenizer.tokenize(span)
        span_tokens_ids = tokenizer.convert_tokens_to_ids(span_tokens)
        begin_offset = sentence_tokens_ids.index(span_tokens_ids[0])
        end_offset = sentence_tokens_ids.index(span_tokens_ids[-1]) + 1
        
        labels = [1 if i < end_offset and i >= begin_offset else 0 for i in range(len(sentence_tokens_ids))]
        labels_tensor = torch.tensor([labels])
    
    #use the pre-trained model to get sentence embeddings
    with torch.no_grad():
        outputs = model(sentence_tokens_ids_tensor, labels=labels_tensor)
        loss, scores, hidden_states = outputs[:3]
        
        #hidden_states[layer][0][token][units], 768 units for each token
        token_embeddings = []
        for token in range(len(sentence_tokens)):
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
            span_embeddings = token_embeddings_all_layers[begin_offset:end_offset]
        #non-propagandistic span embeddings
        else:
            span_embeddings = token_embeddings_all_layers[:]
    return span_embeddings

def main():
    data_table = pd.read_csv("train_table.csv")
    
    #define bert tokenizer && bert model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()
    print(model.config)
    
    data_embeddings = {}
    for index, row in data_table.iterrows():
        print(index)
        #get sentence_tokens && label_tokens
        sentence = row[1]
        span = row[2]
        
        #get bert embeddings
        span_embeddings = get_bert_embedding(model, tokenizer, sentence, span)
        
        #non-propagandistic span embeddings
        if pd.isnull(span):
            data_embeddings[sentence] = span_embeddings
        #propagandistic span embeddings
        else:
            data_embeddings[sentence] = (span, row[3], row[4], span_embeddings)

        
    print("Writing to output file...")
    torch.save(data_embeddings, "data_embeddings.pt")
    print("Done.")

if __name__ == "__main__":
    print(torch.__version__)
    main()