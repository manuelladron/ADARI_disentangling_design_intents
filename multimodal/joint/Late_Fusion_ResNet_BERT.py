import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import json
import datetime
from PIL import Image
from torchvision import models, transforms

IMG_LABEL_PATH = "../../../ADARI/ADARI_furniture_tfidf_top3adjs.json"
IMG_TO_SENTENCE_PATH = "../../../ADARI/ADARI_furniture_sents.json"
WORD_TO_INDEX_PATH = "../../../ADARI/ADARI_furniture_onehots_w2i_3labels.json"
IMG_PATH = "../../../ADARI/full"

IMG_SIZE = 64


BERT_TEST_MODEL_PATH = "../../../ADARI/BERT_Classification_Data"
RESNET_TEST_MODEL_PATH = "../../../ADARI/resnet_28.pt"

torch.manual_seed(42)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def open_json(path):
    f = open(path) 
    data = json.load(f) 
    f.close()
    return data 


def initialize_resnet(num_classes):
    model_ft = models.resnet152()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    return model_ft

class LateFusionDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        img_to_sentences_path, 
        img_to_label_path, 
        word_to_index_path,
        tokenizer,
        img_path,
        img_size,
    ):
        super(LateFusionDataset).__init__()

        self.img_to_sentences_path = img_to_sentences_path
        self.img_to_label_path = img_to_label_path
        self.word_to_index_path = word_to_index_path

        self.img_to_sent = open_json(self.img_to_sentences_path)
        self.img_to_labels = open_json(self.img_to_label_path)
        self.word_to_index = open_json(self.word_to_index_path)

        self.im_names = list(self.img_to_labels.keys())
        max_word = max(list(self.word_to_index.values()))
        self.num_classes = max_word  

        self.img_path = img_path
        self.img_size = img_size

        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.img_to_labels.keys())

    def __getitem__(self, idx):
        imname = self.im_names[idx]

        # one hot encode the labels
        l = torch.zeros((self.num_classes))
        for w in self.img_to_labels[imname]:
            l[self.word_to_index[w] - 1] = 1.0

        tokens = self.tokenizer(
            "".join([s + ' ' for s in self.img_to_sent[imname][0]]),
            padding = 'max_length',
            max_length = 50,
            truncation = True,
            return_tensors = 'pt',
            return_attention_mask = True)

        img = Image.open(self.img_path + "/" + imname)
        img = transforms.Compose(
            [
                transforms.Resize(self.img_size), 
                transforms.CenterCrop(self.img_size), 
                transforms.ToTensor()
            ])(img)
        return l, tokens.input_ids, tokens.attention_mask, img


class LateFusionBERTResnet(torch.nn.Module):
    def __init__(
        self, 
        pretrained_bert : BertForSequenceClassification, 
        pretrained_resnet,
        vocab_size
    ):
        super(LateFusionBERTResnet, self).__init__()

        self.pretrained_bert = pretrained_bert
        self.pretrained_resnet = pretrained_resnet
        self.pretrained_bert.eval()
        self.pretrained_resnet.eval()

        modules = list(self.pretrained_resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.classifier = torch.nn.Linear(
            pretrained_bert.bert.config.hidden_size + pretrained_resnet.fc.in_features, 
            vocab_size
        )
        self.dropout = nn.Dropout(self.pretrained_bert.bert.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask,
        images
    ):
        # Get BERT features
        outputs = self.pretrained_bert.bert(
            input_ids,
            attention_mask=attention_mask
        )
        bert_output = outputs[1]

        # Get Resnet Features
        resnet_output = self.resnet(images)

        # Concatenate
        catted = torch.cat((bert_output, resnet_output.reshape(resnet_output.shape[0], -1)), dim=1)

        return self.classifier(self.dropout(catted))

    def save(self, filename):
        torch.save(self.classifier.state_dict(), filename)

    def load(self, filename):
        self.classifier.load_state_dict(torch.load(filename))

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Loading Data...")
dataset = LateFusionDataset(
    IMG_TO_SENTENCE_PATH, 
    IMG_LABEL_PATH, 
    WORD_TO_INDEX_PATH, 
    tokenizer, 
    IMG_PATH, 
    IMG_SIZE
)
train_set, test_set = torch.utils.data.random_split(
    dataset, 
    [int(.8 * len(dataset)), len(dataset) - int(.8 * len(dataset))]
)

print("Loading Bert Model...")
test_bert = BertForSequenceClassification.from_pretrained(BERT_TEST_MODEL_PATH, return_dict=True)

print("Loading Resnet model...")
# Resnet was trained with 1-indexed onehots, so adjust that
test_resnet = initialize_resnet(dataset.num_classes + 1)
resnet_checkpoint = torch.load(RESNET_TEST_MODEL_PATH, map_location=torch.device(device))
test_resnet.load_state_dict(resnet_checkpoint['model_state_dict'])

late_fusion_model = LateFusionBERTResnet(test_bert, test_resnet, dataset.num_classes)

batch_size = 16
lr = 0.001
num_epochs = 100

def train(model, train_losses, test_losses):
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    dataloader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for labels, input_ids, attn_mask, img in dataloader:
            input_ids = input_ids.reshape((input_ids.shape[0], input_ids.shape[2]))
            attn_mask = attn_mask.reshape((attn_mask.shape[0], attn_mask.shape[2]))

            labels = labels.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            img = img.to(device)

            optimizer.zero_grad()
            
            logits = model(
                input_ids = input_ids, 
                attention_mask = attn_mask,
                images = img
            )

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            
            losses.append(loss.item())
        train_losses.append(sum(losses) / len(losses))
        print(f"Avg Loss at Epoch {epoch}: {train_losses[-1]}")
        # Compute test loss
        with torch.no_grad():
            model.eval()
            losses = []
            for labels, input_ids, attn_mask, img in dataloader:
                input_ids = input_ids.reshape((input_ids.shape[0], input_ids.shape[2]))
                attn_mask = attn_mask.reshape((attn_mask.shape[0], attn_mask.shape[2]))

                labels = labels.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                img = img.to(device)

                logits = model(
                    input_ids = input_ids, 
                    attention_mask = attn_mask,
                    images = img
                )

                loss = criterion(logits, labels)
                
                losses.append(loss.item())
            test_losses.append(sum(losses) / len(losses))
        print(f"Avg Test Loss at Epoch {epoch}: {test_losses[-1]}")

model_name = datetime.datetime.now()
train_losses = []
test_losses = []
try:
    print("Training...")
    train(late_fusion_model, train_losses, test_losses)
except KeyboardInterrupt:
    pass
late_fusion_model.cpu()
late_fusion_model.save(f"Late_Fusion_classification_{model_name}.pth")
with open(f"Late_Fusion_train_losses_{model_name}.json", "w") as f:
    json.dump(train_losses, f) 
with open(f"Late_Fusion_test_losses_{model_name}.json", "w") as f:
    json.dump(train_losses, f) 
