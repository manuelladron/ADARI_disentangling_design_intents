#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, torchvision
import json
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import MultiModalBertDataset
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# directory containing all raw images
IMG_PATH = "/home/ubuntu/ADARI"
PAIRS_PATH = "/home/ubuntu/ADARI_furniture_pairs.json"

def open_json(path):
    f = open(path) 
    data = json.load(f) 
    f.close()
    return data 

def save_json(file_path, data):
    out_file = open(file_path, "w")
    json.dump(data, out_file)
    out_file.close()


# In[ ]:


dataset = MultiModalBertDataset(IMG_PATH, PAIRS_PATH, device=device)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

df = pd.DataFrame(columns=["patches", "input_ids", "is_paired", "attention_mask"])
for j, (patches, input_ids, is_paired, attention_mask) in enumerate(dataloader):
    for i in range(patches.shape[0]):
        df = df.append(
            {
                "patches": patches[i].cpu().numpy(),
                "input_ids": input_ids[i].cpu().numpy(),
                "is_paired": is_paired[i].cpu().numpy(),
                "attention_mask": attention_mask[i].cpu().numpy()
            },
            ignore_index=True
        )
        print(f"Finished sample")
    if j % 10 == 0:
        print(f"At batch # {j}")
    


# In[ ]:


df.to_pickle("./preprocessed_patches.pkl")

