#!/usr/bin/env python
# coding: utf-8

# In[57]:


from __future__ import print_function
from __future__ import division

import sys
import torch
import json
import spacy 
import string
import numpy as np
from torch.utils.data import TensorDataset, random_split
torch.manual_seed(42)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


# In[54]:


nlp = spacy.load('en_core_web_sm')


# In[12]:


# Path for file dset_dataloader.json
def open_json(path):
    f = open(path) 
    data = json.load(f) 
    f.close()
    return data 

def save_json(file_path, data):
    out_file = open(file_path, "w")
    json.dump(data, out_file)
    out_file.close()

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

def get_file(path):
    with open(path, encoding='utf-8') as f:
        data = json.loads(json.load(f))
    return data


# In[80]:


ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
NUMBERS = '0123456789'

class PreprocessedData(object):
    def __init__(self, train_file_path, test_file_path, tokenizer, max_sentence_length, eqs, save_path):
        """
        train_file_path = list with files
        test_file_path = list with files
        """
        
        self.train_path = train_file_path
        self.test_path = test_file_path
        self.save_path = save_path
        self.tokenizer = tokenizer
        
        self.VOCAB = None
        self.VOCAB_SIZE = None
        self.max_sentence_length = max_sentence_length
        self.eqs = eqs # whether or not +,- has same size
                
        # Automatically run it when making an instance
        self.RUN_for_dataset()
        
    def get_file(self, path):
        with open(path, encoding='utf-8') as f:
            data = json.loads(json.load(f))
        return data
    
    def save_json(self, file_path, data):
        out_file = open(file_path, "w")
        json.dump(data, out_file)
        out_file.close()
    
    def text_from_json(self, json_file):
        """
        Input: list of json files. 
        Returns:  a list with all sentences in all json files
        """
        all_text = []
        for file in json_file:
            for sample in file:
                text_l = sample['text']
                for sentence in text_l:
                    sent = sentence.lower()
                    all_text.append(sent)
        return all_text
    
    ############## PROCESSING DATA ##############
    def getNegativeTags(self, alphabet, positive_tags, all_text):
        """
        Takes in an alphabet, a list of tags and a list of sentences. Returns an alphabet (STRING) that correspond to the 
        negative samples by substracting the positive tags
        """
        # 1) Get alphabet cropped to the length of sentences
        idx_len = len(all_text)
        negative_tags = alphabet[:idx_len]
        
        # 2) Iterate over positive tags and remove them from cropped_alphabet. 
        for tag in positive_tags:
            negative_tags = negative_tags.replace(tag, "")

        return negative_tags #string of negative tags
    
    def cleanText(self, cur_text):
        """ Returns list of lowercase sentences """
        new_list = []
        for sentence in cur_text:
            # (1) Make everything lowercase
            new_s = sentence.lower()
            if len(new_s) != 0:
                new_list.append(new_s)
        return new_list
    
    def getTextValidity(self, positive_tags, negative_tags, text_list):
        """
        Inputs:
         - positive tags (type LIST)
         - negative tags (type STRING) —doesn't matter that both sources are different types.
         - list with all sentences 
        Outputs:
         - List of valid sentences
         - List of non-valid sentences 
        """
        v = [text_list[ALPHABET.index(letter)].lower() for letter in positive_tags if text_list[ALPHABET.index(letter)] != None]
        n = [text_list[ALPHABET.index(letter)].lower() for letter in negative_tags if text_list[ALPHABET.index(letter)] != None]
        return v, n
    
    def get_statistics(self):
        pos_sens = len(self.pos_sentences)
        neg_sens = len(self.neg_sentences)
        l = list(range(0, 256))

        # 1) average len
        max_sent = 0
        pos = []
        neg = []
        # for valid sentences
        for s in self.pos_sentences:
            len_sentence = len(s)
            pos.append(len_sentence)
            if len_sentence > max_sent:
                max_sent = len_sentence
        avg_pos_sen_len = sum(pos) / pos_sens
        
        # for nonvalid sentences
        for s in self.neg_sentences:
            len_sentence = len(s)
            neg.append(len_sentence)
            if len_sentence > max_sent:
                max_sent = len_sentence
        avg_neg_sen_len = sum(neg) / neg_sens
        
        print('Avg Positive Sentence Length: %.2f' % avg_pos_sen_len)
        print('Avg Negative Sentence Length: %.2f' % avg_neg_sen_len)
        print('Max Sentence Length: %.2f' % max_sent)
        
        # 2) avg num of adjs
        pos_adjs = [len([token for token in nlp(s) if token.pos_ == 'ADJ']) for s in self.pos_sentences]
        avg_pos_adjs = sum(pos_adjs) / pos_sens
        neg_adjs = [len([token for token in nlp(s) if token.pos_ == 'ADJ']) for s in self.neg_sentences]
        avg_neg_adjs = sum(neg_adjs) / neg_sens
        
        print('Avg Positive Adjectives: %.2f' % avg_pos_adjs)
        print('Avg Negative Adjectives: %.2f' % avg_neg_adjs)

    
    def get_all_text(self, files):
        """
        Parse json file and outputs train_data (text) and numpy array labels for binary classification
        """
        self.neg_sentences = []
        self.neg_sentences_labels = []
        self.pos_sentences = []
        self.pos_sentences_labels = []
        self.neg_sent_dict = dict()
        self.pos_sent_dict = dict()
        
        for file in files:
            # iterate over the examples in file and grab positive and negative samples
            for i in range(len(file)):
                positive_tags = file[i]['text-tags']
                text_list = file[i]['text']
                negative_tags = self.getNegativeTags(ALPHABET, positive_tags, text_list)
                text_list = self.cleanText(text_list)
                valid_text, nonvalid_text = self.getTextValidity(positive_tags, negative_tags, text_list) 

                # append sentences and labels
                for nv_text in nonvalid_text:
                    neg_label = np.array([0])
                    self.neg_sentences.append(nv_text)
                    self.neg_sentences_labels.append(neg_label)

                    self.neg_sent_dict[nv_text] = [0]

                for v_text in valid_text:
                    pos_label = np.array([1])
                    self.pos_sentences.append(v_text)
                    self.pos_sentences_labels.append(pos_label)

                    self.pos_sent_dict[v_text] = [0]
        
        if self.eqs:
            min_val = min(len(self.neg_sentences), len(self.pos_sentences))
            self.neg_sentences = self.neg_sentences[:min_val]
            self.pos_sentences = self.pos_sentences[:min_val]
            self.neg_sentences_labels = self.neg_sentences_labels[:min_val]
            self.pos_sentences_labels = self.pos_sentences_labels[:min_val]
        
        self.sentences = self.pos_sentences + self.neg_sentences
        self.sentences_labels = self.pos_sentences_labels + self.neg_sentences_labels
    
        # from list to array 
        self.sentences_labels = np.array(self.sentences_labels, dtype='int64')
        
    def save_dict(self, path):
        self.save_json(path + '/relevant_sents.json', self.pos_sent_dict)
        self.save_json(path + '/irrelevant_sents.json', self.neg_sent_dict)
        
        
    def tokenize_sentences(self):
        # Tokenize all sentences and map the tokens to their word ID 
        self.inputs_ids = []
        self.attention_masks = []
        
        for sent in self.sentences:                
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(sent,
                                                add_special_tokens = True, # add [CLS] and [SEP]
                                                max_length = self.max_sentence_length,
                                                pad_to_max_length = True,  # pad and truncate
                                                return_attention_mask = True,
                                                return_tensors = 'pt'
                                                )
            self.inputs_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])

        # Convert list to tensors
        self.inputs_ids = torch.cat(self.inputs_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.tensor(self.sentences_labels)
        
    
    def partition_data(self, train_percentage):
        assert len(self.inputs_ids) == len(self.sentences_labels)
        dataset = TensorDataset(self.inputs_ids, self.attention_masks, self.labels)
        
        train_size = int(train_percentage * len(dataset))
        dev_size = len(dataset) - train_size
        
        self.train_dataset, self.dev_dataset = random_split(dataset, [train_size, dev_size])
        
        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(dev_size))
    
    def RUN_for_dataset(self):
        # 1) get jsons
        train_raw = []
        for i in range(len(self.train_path)): # list with all training data from different sections
            train_raw.append(self.get_file(self.train_path[i]))
        
        # 2) get text
        self.get_all_text(train_raw)
        print(len(self.sentences), len(self.sentences_labels))
        
        # 3) tokenize at word-level
        self.tokenize_sentences()
        
        # 4) partition data
        self.partition_data(.8)
        
        # 5) Save json
        self.save_dict(self.save_path)


# In[76]:


#P = PreprocessedData([design_tagged_p], None, 0)


# In[ ]:


#P.save_dict('/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/SPRING 2020/DL_11785/project/Github_Ambiguity-in-Computational-Creativity-master/data')


# In[77]:


#P.get_statistics()


# In[ ]:




