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
BERT_TEST_MODEL_PATH = ""

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

print("Loading Bert Model to test...")
test_bert = BertForSequenceClassification.from_pretrained(BERT_TEST_MODEL_PATH, return_dict=True)

# Compute F1 Score
def test_score(model, test_set, threshold):
    model.eval()
    model.to(device)
    test_d = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    nonzero_words = {}
    avg_f1_score = []
    with torch.no_grad():
        bert_model.eval()
        for labels, input_ids, attn_mask in dataloader:
            input_ids = input_ids.reshape((input_ids.shape[0], input_ids.shape[2]))
            attn_mask = attn_mask.reshape((attn_mask.shape[0], attn_mask.shape[2]))

            labels = labels.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            out = torch.sigmoid(bert_model(input_ids = input_ids, attention_mask = attn_mask).logits)

            score = f1_score(labels.cpu(), (out > threshold).cpu(), average='samples')
            avg_f1_score.append(score)
    print(f"Threshold: {threshold}: {sum(avg_f1_score) / len(avg_f1_score)}")

for t in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
    print(f"Testing {t}...")
    test_score(test_bert, test_set, t)