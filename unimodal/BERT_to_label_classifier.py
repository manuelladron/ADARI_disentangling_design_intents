import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import json
import datetime


IMG_LABEL_PATH = "../../ADARI/ADARI_furniture_tfidf_top3adjs.json"
IMG_TO_SENTENCE_PATH = "../../ADARI/ADARI_furniture_sents.json"
WORD_TO_INDEX_PATH = "../../ADARI/ADARI_furniture_onehots_w2i_3labels.json"

torch.manual_seed(42)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def open_json(path):
    f = open(path) 
    data = json.load(f) 
    f.close()
    return data 


class ADARIMultiHotSentsDataset(torch.utils.data.Dataset):
    def __init__(self, 
                img_to_sentences_path, 
                img_to_label_path, 
                word_to_index_path,
                tokenizer):
        super(ADARIMultiHotSentsDataset).__init__()
        
        self.img_to_sentences_path = img_to_sentences_path
        self.img_to_label_path = img_to_label_path
        self.word_to_index_path = word_to_index_path

        self.img_to_sent = open_json(self.img_to_sentences_path)
        self.img_to_labels = open_json(self.img_to_label_path)
        self.word_to_index = open_json(self.word_to_index_path)

        self.im_names = list(self.img_to_labels.keys())
        max_word = max(list(self.word_to_index.values()))
        self.num_classes = max_word        

        
    def __len__(self):
        return len(self.img_to_labels.keys())
        
    def __getitem__(self, idx):
        imname = self.im_names[idx]

        # one hot encode the labels
        l = torch.zeros((self.num_classes))
        for w in self.img_to_labels[imname]:
            l[self.word_to_index[w]] = 1.0

        tokens = tokenizer(
            "".join([s + ' ' for s in self.img_to_sent[imname][0]]),
            padding = 'max_length',
            max_length = 50,
            truncation = True,
            return_tensors = 'pt',
            return_attention_mask = True)

        return l, tokens.input_ids, tokens.attention_mask


print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Loading data...")
dataset = ADARIMultiHotSentsDataset(IMG_TO_SENTENCE_PATH, IMG_LABEL_PATH, WORD_TO_INDEX_PATH, tokenizer)
train_set, test_set = torch.utils.data.random_split(dataset, [int(.8 * len(dataset)), len(dataset) - int(.8 * len(dataset))])

print("Loading Pretrained Bert...")
pretrained_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)

# Create model with pretrained bert, not pretrained classifier
new_config = pretrained_bert.config
new_config.num_labels = dataset.num_classes
print("Creating new BertSeqClassification...")
bert_new_head = BertForSequenceClassification(new_config)
bert_new_head.bert = pretrained_bert.bert


batch_size = 16
lr = 0.001
num_epochs = 100

def train(bert_model, train_losses, test_losses):
    bert_model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    dataloader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    optimizer = torch.optim.Adam(bert_model.classifier.parameters(), lr=lr)

    for epoch in range(num_epochs):
        bert_model.train()
        losses = []
        for labels, input_ids, attn_mask in dataloader:
            labels.to(device)
            input_ids.to(device)
            attn_mask.to(device)

            input_ids = input_ids.reshape((input_ids.shape[0], input_ids.shape[2]))
            attn_mask = attn_mask.reshape((attn_mask.shape[0], attn_mask.shape[2]))

            optimizer.zero_grad()
            
            logits = bert_model(input_ids = input_ids, attention_mask = attn_mask).logits

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            
            losses.append(loss.item())
        train_losses.append(sum(losses) / len(losses))
        print(f"Avg Loss at Epoch {epoch}: {train_losses[-1]}")
        # Compute test loss
        with torch.no_grad():
            bert_model.eval()
            losses = []
            for labels, input_ids, attn_mask in dataloader:
                labels.to(device)
                input_ids.to(device)
                attn_mask.to(device)

                input_ids = input_ids.reshape((input_ids.shape[0], input_ids.shape[2]))
                attn_mask = attn_mask.reshape((attn_mask.shape[0], attn_mask.shape[2]))

                logits = bert_model(input_ids = input_ids, attention_mask = attn_mask).logits

                loss = criterion(logits, labels)
                
                losses.append(loss.item())
            test_losses.append(sum(losses) / len(losses))
        print(f"Avg Test Loss at Epoch {epoch}: {test_losses[-1]}")

model_name = datetime.datetime.now()
train_losses = []
test_losses = []
try:
    print("Training...")
    train(bert_new_head, train_losses, test_losses)
except KeyboardInterrupt:
    pass
bert_new_head.cpu()
bert_new_head.save_pretrained(f"BERT_classification_{model_name}")
with open(f"train_losses_{model_name}.json", "w") as f:
    json.dump(train_losses, f) 
with open(f"test_losses_{model_name}.json", "w") as f:
    json.dump(train_losses, f) 
    