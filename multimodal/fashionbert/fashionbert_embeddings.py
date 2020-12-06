import torch, torchvision
import sys
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AdamW
from transformers import BertTokenizer, BertModel
# from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from transformers.modeling_bert import BertPreTrainingHeads
from utils import construct_bert_input, PreprocessedADARI_evaluation, save_json

from fashionbert_model import FashionBert, FashionBertHead
import argparse
import datetime

from tqdm import tqdm
import numpy as np
from IPython.display import clear_output

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FashionBert_embedder(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config) 
    
        self.bert = BertModel(config)
        self.im_to_embedding = torch.nn.Linear(2048, 768)
        self.im_to_embedding_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.init_weights()

    def embed(self, embeds, plot_hidden=False):
        
        batch_size = embeds.shape[0]
        seq_length = embeds.shape[1]
        hidden_dim = embeds.shape[2]

        outputs = self.bert(inputs_embeds=embeds, return_dict=True)
        hidden_states = outputs[2]
        
        """
        ----hidden states analyzer--- https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#3-extracting-embeddings
        Number of layers: 13   (initial embeddings + 12 BERT layers)
        Number of batches: 1
        Number of tokens: 512
        Number of hidden units: 768
        """
        # Plot 
        if plot_hidden:
            token_i = 5
            layer_i = 5
            batch_i = 0
            vec = hidden_states[layer_i][batch_i][token_i]
            # Plot the values as a histogram to show their distribution.
            plt.figure(figsize=(10,10))
            plt.hist(vec, bins=200)
            plt.show()
        
        token_embeddings = torch.stack(hidden_states, dim=0) # [13, 1, 512, 768]
        token_embeddings = token_embeddings.squeeze(1)       # [13, 512, 768]
        # Permute to have tokens dimension first
        token_embeddings = token_embeddings.permute(1,0,2)   # [512, 13, 768]

        # hidden states corresponding to the text part
        text_embedding = token_embeddings[:448, :,  :]  # [448, 13, 768]
        # hidden states corresponding to the image part
        image_embedding = token_embeddings[448:, :, :]  # [64, 13, 768]

        # Average the second to last hidden layer 
        text_vec = text_embedding[:, -2, :] # [448, 768]
        text_vec = torch.mean(text_vec, dim=0) # [768]
        
        image_vec = image_embedding[:, -2, :] # [64, 768]
        image_vec = torch.mean(image_vec, dim=0) # [768]
        
        return text_vec, image_vec
            

def get_embeddings(dataset, save_file, num_samples= 1000, pretrained_model=None):
    torch.cuda.empty_cache()
    torch.manual_seed(0)    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        sampler = torch.utils.data.SubsetRandomSampler(
                    torch.randint(high=len(dataset), size=(len(dataset),))),
        )
        
       
    print('dataloader len: ', len(dataloader))
    
    if pretrained_model != None:
        embedder = FashionBert_embedder.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
    else:
        embedder = FashionBert_embedder.from_pretrained('bert-base-uncased', output_hidden_states=True, return_dict=True)
    
    embedder.to(device)
    embedder.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    fashionbert_embeds = dict()
    with torch.no_grad():
        for i, (patches, neg_patches, input_ids, is_paired, attention_mask, neg_input_ids, neg_attention_mask, img_name) in enumerate(tqdm(dataloader)):
            input_ids = input_ids.to(device)
            patches = patches.to(device)
            
            inputs = input_ids.squeeze(0).detach().tolist()
            seq = tokenizer.convert_ids_to_tokens(inputs)
            seq = tokenizer.convert_tokens_to_string(seq)
            embeds = construct_bert_input(patches, input_ids, embedder, device)
            text_emb, img_emb = embedder.embed(embeds)
            
            fashionbert_embeds[img_name[0]] = {'text': seq,
                                        'text_emb': text_emb.tolist(),
                                        'img_emb': img_emb.tolist()}
    
    save_json(save_file, fashionbert_embeds)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get embeddings FashionBert')
    parser.add_argument('--path_to_dataset', help='Absolute path to .pkl file')
    parser.add_argument('--path_to_pretrained_model', help='Path to pretrained directory', default=None)
    parser.add_argument('--num_samples', help='Number of samples for dataloader', default=1000)
    parser.add_argument('--save_file_name', help='Name for file with embeddings', default='fashionbert_vanilla_adaptive.json')
    args = parser.parse_args()
    
    print('Loading dataset...')
    dataset = PreprocessedADARI_evaluation(args.path_to_dataset)
    print('Getting embeddings...')
    get_embeddings(dataset, args.save_file_name, args.num_samples, args.path_to_pretrained_model)
    
    

