#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Evaluator(torch.nn.Module):     
    def __init__(self, pretrained_model=None):
        super(Evaluator, self).__init__()
        
        if pretrained_model != None:
            print('-- Loading fashionbert_pretrained model: {}'.format(pretrained_model))
            fashion_bert = FashionBert.from_pretrained(pretrained_model, return_dict=True)
        else:
            fashion_bert = FashionBert.from_pretrained('bert-base-uncased', return_dict=True)
        
        self.model = fashion_bert
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    
    def text2img_scores(self,
                        input_ids,
                        embeds,
                        att_mask,
                        embeds_n,    # list 
                        att_mask_n,    # list 
                        ):               
        """
        INPUTS:
            input_ids     [1, 448]
            embeds:       [1, 512, 768]
            att_mask:     [1, 448]
            embeds_n:     list with 100 of [1, 512, 768]
            att_mask_n:   list with 100 of [1, 448]
        """
        # Score for positive 
        query_dict_scores = []
        query_scores = []
        query_labels = []
        
        score_pos = self.get_scores_and_metrics(
            embeds=embeds.to(device), 
            attention_mask=att_mask.to(device), 
            labels=input_ids.to(device),  
            is_paired=torch.tensor(True).to(device),
            only_alignment=True,
            )
        
        #label = score_pos[1]
        score_p = score_pos[0].squeeze()
        score_p = score_p[1].detach().item() # confidence that is actually positive
        score_pos_dict = {'text': input_ids,
                         'score': score_p,
                         'label': True}
        query_dict_scores.append(score_pos_dict)
        query_scores.append(score_p)
        query_labels.append(True)
        
        # Scores for negative
        for n in range(len(embeds_n)):
            score_neg = self.get_scores_and_metrics(
                        embeds=embeds_n[n].to(device), 
                        attention_mask=att_mask_n[n].to(device), 
                        labels=input_ids.to(device), 
                        is_paired= torch.tensor(False).to(device),
                        only_alignment = True,
                        )
            
            score_n  = score_neg[0].squeeze()
            score_n  = score_n[1].detach().item() # confidence that is actually positive
            score_neg_dict = {'text': input_ids,
                             'score': score_n,
                             'label': False}
            
            query_dict_scores.append(score_neg_dict)
            query_scores.append(score_n)
            query_labels.append(False)
        
        #print(evaluator.tokenizer.convert_ids_to_tokens(ids))
        S = [(s,l) for s, l in sorted(zip(query_scores, query_labels), key=lambda x: x[0], reverse=True)]
        return S
    
    def img2text_scores(self, input_ids_p, embeds_p, att_mask_p, input_ids_n, embeds_n, att_mask_n):
        """
        INPUTS:
            input_ids_p : [1, 448]
            embeds_p:     [1, 512, 768]
            att_mask_p:   [1, 448]
            input_ids_n:  list with 100 of [1, 448]
            embeds_n:     list with 100 of [1, 512, 768]
            att_mask_n:   list with 100 of [1, 448]
        """
        # Score for positive 
        query_dict_scores = []
        query_scores = []
        query_labels = []
        
        score_pos = self.get_scores_and_metrics(
            embeds=embeds_p.to(device), 
            attention_mask=att_mask_p.to(device), 
            labels=input_ids_p.to(device), 
            is_paired=torch.tensor(True).to(device),
            only_alignment=True,
            )
        
        #label = score_pos[1]
        score_p = score_pos[0].squeeze()
        score_p = score_p[1].detach().item() # confidence that is actually positive
        score_pos_dict = {'text': input_ids_p,
                         'score': score_p,
                         'label': True}
        query_dict_scores.append(score_pos_dict)
        query_scores.append(score_p)
        query_labels.append(True)
        
        # Scores for negative
        for n in range(len(embeds_n)):
            score_neg = self.get_scores_and_metrics(
                embeds=embeds_n[n].to(device), 
                attention_mask=att_mask_n[n].to(device), 
                labels=input_ids_n[n].to(device), 
                is_paired= torch.tensor(False).to(device),
                only_alignment = True,
                )
            
            score_n  = score_neg[0].squeeze()
            score_n  = score_n[1].detach().item() # confidence that is actually positive
            score_neg_dict = {'text': input_ids_n[n],
                             'score': score_n,
                             'label': False}
            
            query_dict_scores.append(score_neg_dict)
            query_scores.append(score_n)
            query_labels.append(False)
        
        #print(evaluator.tokenizer.convert_ids_to_tokens(ids))
        S = [(s,l) for s, l in sorted(zip(query_scores, query_labels), key=lambda x: x[0], reverse=True)]

        return S
        
    def rank_at_K(self, dict_scores, img2text=True):
        if img2text:
            print('------ Image 2 Text ------')
        else:
            print('------ Text 2 Image ------')
        
        Ks = [1, 5, 10]
        for K in Ks:
            #print('------ Rank @ {} ------'.format(K))
            found = 0
            for key, val in dict_scores.items():
                tmp_range = K if K < len(val) else len(val)
                for i in range(tmp_range):
                    score, label = val[i]
                    if label:
                        found += 1
                        break
            print('------ Rank @ {} = {} ------'.format(K, (found/len(dict_scores.keys()) )))        
        
    def get_scores_and_metrics(
        self,
        embeds,                       # text + image embedded 
        attention_mask,
        labels=None,                  # [batch, 448]
        is_paired=None,               # [batch]
        only_alignment = False,
        ):
        
        batch_size = embeds.shape[0]
        seq_length = embeds.shape[1]
        hidden_dim = embeds.shape[2]

        outputs = self.model.bert(inputs_embeds=embeds, return_dict=True)
        sequence_output = outputs.last_hidden_state # [batch, seq_length, hidden_size]
        pooler_output = outputs.pooler_output      #  [batch_size, hidden_size] last layer of hidden-state of first token (CLS) + linear layer + tanh

        # hidden states corresponding to the text part
        text_output = sequence_output[:, :labels.shape[1], :]  # [batch, 448, 768]
        # hidden states corresponding to the image part
        image_output = sequence_output[:, labels.shape[1]:, :] # [batch, 64, 768]

        ### FOR TEXT 
        # Predict the masked text tokens and alignment scores (whether image and text match)
        prediction_scores, alignment_scores = self.model.cls(text_output, pooler_output)
        # prediction score is [batch, 448, vocab_size = 30522]
        # aligment score is [batch, 2] 2 with logits corresponding to 1 and  0
        
        if only_alignment:
            return alignment_scores, is_paired.double().detach().item()
        
        text_evaluator = {'text_pred_logits': prediction_scores, 
                         'text_labels': labels}
        
        alignment_evaluator = {'alignment_logits': alignment_scores,
                              'alignment_labels': is_paired}
        
        text_acc, alig_acc = self.accuracy_scores(text_evaluator, alignment_evaluator)
        return text_acc, alig_acc
    
    def accuracy_scores(self, text_evaluator, alignment_evaluator):
        """
        Text evaluator:  dictionary with preds and labels (aligned)
        Image evaluator: dictionary with image output and image patches (aligned)
        """
        # Text
        text_pred_logits = text_evaluator['text_pred_logits'] # [num_aligned, 448, vocab_size]
        text_labels = text_evaluator['text_labels']           # [num_aligned, 448]
        
        text_preds_logits = text_pred_logits.detach().cpu().numpy()
        text_labels = text_labels.cpu().numpy().flatten()
        text_preds = np.argmax(text_preds_logits, axis=2).flatten() # [num_algined, 448]
        
        # Alignment
        alig_pred_logits = alignment_evaluator['alignment_logits']      # [1, 2]
        alig_labels = alignment_evaluator['alignment_labels']           # [2]
        
        alig_pred_logits = alig_pred_logits.detach().cpu().numpy()
        alig_labels = alig_labels.double().cpu().numpy().flatten()
        alig_preds = np.argmax(alig_pred_logits, axis=1).flatten() # [1, 2]
        
        text_acc = accuracy_score(text_labels, text_preds)
        alig_acc = accuracy_score(alig_labels, alig_preds)
        
        return text_acc, alig_acc
        


# In[36]:


def image2text(i, patches, neg_patches, input_ids, is_paired, attention_mask, neg_input_ids, neg_attention_mask, evaluator):
    """
    image2text retrieval: 
        Query = Image
        Paired with: 1 positive text, 100 negative texts
    """
    im_seq_len = patches.shape[1]
    bs = input_ids.shape[0]
    len_neg_inputs = neg_input_ids.shape[1]

    # PAIRED PATCHES
    # mask image patches with prob 10%
    masked_patches = patches.detach().clone()
    masked_patches = masked_patches.view(-1, patches.shape[2])
    im_mask = torch.rand((masked_patches.shape[0], 1)) >= 0.1
    masked_patches *= im_mask

    try:
        masked_patches = masked_patches.view(bs, im_seq_len, patches.shape[2])
    except Exception as e:
        print(e)
        print(f"masked_patches: {masked_patches.shape}")
        print(f"im_mask: {im_mask.shape}")
        print(f"patches: {patches.shape}")
        return
        
    # Mask tokens with prob 15%, note id 103 is the [MASK] token
    token_mask = torch.rand(input_ids.shape)
    masked_input_ids = input_ids.detach().clone()
    masked_input_ids[token_mask < 0.15] = 103

    embeds = construct_bert_input(masked_patches, masked_input_ids, evaluator.model, device=device)
    # pad attention mask with 1s so model pays attention to the image parts
    attention_mask = F.pad(attention_mask, (0, embeds.shape[1] - input_ids.shape[1]), value = 1)

    # NEGATIVE SAMPLE # [batch, 100, 448]
    all_embeds_neg = []
    all_att_mask = []
    all_neg_inputs = []

    for j in range(len_neg_inputs):
        neg_input_id_sample = neg_input_ids[:, j, :] # [1, 448]
        neg_attention_mask_sample = neg_attention_mask[:, j, :]

        token_mask = torch.rand(neg_input_id_sample.shape)
        masked_input_ids = neg_input_id_sample.detach().clone()
        masked_input_ids[token_mask < 0.15] = 103

        embeds_neg = construct_bert_input(masked_patches, masked_input_ids, evaluator.model, device=device)
        attention_mask_neg = F.pad(neg_attention_mask_sample, (0, embeds_neg.shape[1] - neg_input_id_sample.shape[1]), value = 1)

        all_embeds_neg.append(embeds_neg)
        all_att_mask.append(attention_mask_neg)
        all_neg_inputs.append(neg_input_id_sample.detach())

    # Now I have all joint embeddings for 1 positive sample and 100 neg samples
    all_scores_query = evaluator.img2text_scores(
                input_ids_p = input_ids,
                embeds_p = embeds,
                att_mask_p = attention_mask,
                input_ids_n = all_neg_inputs,
                embeds_n = all_embeds_neg,
                att_mask_n = all_att_mask)

    # Accuracy: only in positive example
    txt_acc, alig_acc = evaluator.get_scores_and_metrics(
                        embeds,                       # text + image embedded 
                        attention_mask,
                        labels=input_ids,                  # [batch, 448]
                        is_paired=is_paired,               # [batch]
                        only_alignment = False,
                        )

    return all_scores_query, txt_acc, alig_acc
    
    
def text2image(i, patches, neg_patches, input_ids, is_paired, attention_mask, neg_input_ids, neg_attention_mask, evaluator):
    """
    text2image retrieval: 
        Query = Text
        Paired with: 1 positive image, 100 negative images
    """    
    im_seq_len = patches.shape[1]
    bs = input_ids.shape[0]
    len_neg_inputs = neg_input_ids.shape[1]
    
    # PAIRED PATCHES
    # mask image patches with prob 10%
    masked_patches = patches.detach().clone()
    masked_patches = masked_patches.view(-1, patches.shape[2])
    im_mask = torch.rand((masked_patches.shape[0], 1)) >= 0.1
    masked_patches *= im_mask

    try:
        masked_patches = masked_patches.view(bs, im_seq_len, patches.shape[2])
    except Exception as e:
        print(e)
        print(f"masked_patches: {masked_patches.shape}")
        print(f"im_mask: {im_mask.shape}")
        print(f"patches: {patches.shape}")
        return

    # UNPAIRED PATCHES
    neg_masked_patches = neg_patches.detach().clone()
    neg_masked_patches = neg_masked_patches.view(-1, neg_patches.shape[3])
    neg_im_mask = torch.rand((neg_masked_patches.shape[0], 1)) >= 0.1
    neg_masked_patches *= neg_im_mask

    try:
        neg_masked_patches = neg_masked_patches.view(bs, len_neg_inputs, im_seq_len, patches.shape[2])
    except Exception as e:
        print(e)
        print(f"masked_patches: {neg_masked_patches.shape}")
        print(f"im_mask: {neg_im_mask.shape}")
        print(f"patches: {neg_patches.shape}")
        return
        
    # Mask tokens with prob 15%, note id 103 is the [MASK] token
    token_mask = torch.rand(input_ids.shape)
    masked_input_ids = input_ids.detach().clone()
    masked_input_ids[token_mask < 0.15] = 103
    
    # POSITIVE IMAGE 
    embeds = construct_bert_input(masked_patches, masked_input_ids, evaluator.model, device=device)
    attention_mask = F.pad(attention_mask, (0, embeds.shape[1] - input_ids.shape[1]), value = 1)
        
    # NEGATIVE SAMPLES
    all_embeds_neg = []
    all_att_mask = []
    
    for p in range(len_neg_inputs):
        neg_masked_patches_sample = neg_masked_patches[:, p, :, :]
        embeds_neg = construct_bert_input(neg_masked_patches_sample, masked_input_ids, evaluator.model, device=device)
        attention_mask_neg = F.pad(attention_mask, (0, embeds_neg.shape[1] - input_ids.shape[1]), value = 1)

        all_embeds_neg.append(embeds_neg)
        all_att_mask.append(attention_mask_neg)

    
    # Now I have all joint embeddings for 1 positive sample and 100 neg samples
    all_scores_query = evaluator.text2img_scores(
                input_ids   = input_ids,
                embeds      = embeds,
                att_mask    = attention_mask,
                embeds_n    = all_embeds_neg, # list
                att_mask_n  = all_att_mask) # list
              
    
    # Accuracy: only in positive example
    txt_acc, alig_acc = evaluator.get_scores_and_metrics(
                        embeds,                       # text + image embedded 
                        attention_mask,
                        labels=input_ids,                  # [batch, 448]
                        is_paired=is_paired,               # [batch]
                        only_alignment = False,
                        )
    
    return all_scores_query, txt_acc, alig_acc
    
    
def test(dataset, device, num_samples, pretrained_model=None):
    torch.cuda.empty_cache()
    torch.manual_seed(0)    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        sampler = torch.utils.data.SubsetRandomSampler(
                    torch.randint(high=len(dataset), size=(num_samples,))),
        )
    if pretrained_model != None:
        evaluator = Evaluator(pretrained_model)
    else:
        evaluator = Evaluator()
    
    evaluator.model.to(device)
    evaluator.model.eval()

    query_dict_im2txt = {}
    query_dict_txt2im = {}
    running_acc_alignment_im2txt = 0.0
    running_acc_pred_im2txt = 0.0
    running_acc_alignment_txt2im = 0.0
    running_acc_pred_txt2im = 0.0
    
    with torch.no_grad():
        for i, (patches, neg_patches, input_ids, is_paired, attention_mask, neg_input_ids, neg_attention_mask, img_name) in enumerate(dataloader):

            # ****** Shapes ********
            # input_ids shape:     [1, 448]
            # neg_input_ids shape: [1, NUM_SAMPLES=100, 448]
            # neg_patches:         [1, NUM_SAMPLES=100, 64, 2048]
            
            # IMAGE 2 TEXT
            im2txt_query_scores, im2txt_pred_acc, im2txt_alig_acc = image2text(i, patches, neg_patches, input_ids, 
                                                                                is_paired, attention_mask, 
                                                                                neg_input_ids, neg_attention_mask,
                                                                                evaluator)
            # Accuracies 
            running_acc_pred_im2txt += im2txt_pred_acc
            running_acc_alignment_im2txt += im2txt_alig_acc
            
            # For Rank @ K
            query_dict_im2txt[img_name[0]] = im2txt_query_scores
            
            
            # TEXT 2 IMAGE
            txt2im_query_scores, txt2im_pred_acc, txt2im_alig_acc = text2image(i, patches, neg_patches, input_ids, 
                                                                                is_paired, attention_mask, 
                                                                                neg_input_ids, neg_attention_mask,
                                                                                evaluator)
            # Accuracies 
            running_acc_pred_txt2im += txt2im_pred_acc
            running_acc_alignment_txt2im += txt2im_alig_acc
            
            # For Rank @ K
            query_dict_txt2im[img_name[0]] = im2txt_query_scores
   

    im2txt_test_set_accuracy_pred = (running_acc_pred_im2txt / len(dataloader))
    im2txt_test_set_accuracy_alig = (running_acc_alignment_im2txt / len(dataloader))
    txt2im_test_set_accuracy_pred = (running_acc_pred_txt2im / len(dataloader))
    txt2im_test_set_accuracy_alig = (running_acc_alignment_txt2im / len(dataloader))
    
    print()
    print('---- IMAGE 2 TEXT EVALUATIONS ---------------------')
    evaluator.rank_at_K(query_dict_im2txt, True)
    print('---- Accuracy in token predictions: {} -----'.format(im2txt_test_set_accuracy_pred))
    print('---- Accuracy in text-image alignment: {} -----'.format(im2txt_test_set_accuracy_alig))
    print()
    print('---- TEXT 2 IMAGE EVALUATIONS ---------------------')
    evaluator.rank_at_K(query_dict_txt2im, False)
    print('---- Accuracy in token predictions: {} -----'.format(txt2im_test_set_accuracy_pred))
    print('---- Accuracy in text-image alignment: {} -----'.format(txt2im_test_set_accuracy_alig))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FashionBert.')
    parser.add_argument('--path_to_images', help='Path to images folder')
    parser.add_argument('--path_to_dict_pairs', help='Path to ADARI furniture dict .json file')
    parser.add_argument('--num_neg_samples', help='Number of negative samples int', default=10)
    parser.add_argument('--path_to_pretrained_model', help='Path to pretrained model', default=None)
    
    args = parser.parse_args()
    print('Processing the dataset...')
    dataset = PreprocessedADARI_evaluation(args.path_to_images, args.path_to_dict_pairs)
    print('Starting evaluation...')
    test(dataset, device, args.num_neg_samples, args.path_to_pretrained_model)
    print('Done!')


