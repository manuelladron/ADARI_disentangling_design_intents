#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, torchvision
import torchvision.transforms as transforms
from PIL import Image
import json, datetime
from sklearn.metrics import f1_score

CLASS_LABEL_PATH = "../../ADARI/ADARI_furniture_onehots.json"
IMAGE_FOLDER = "../../ADARI/full"

torch.manual_seed(42)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# In[2]:


def open_json(path):
    f = open(path) 
    data = json.load(f) 
    f.close()
    return data 

class ADARIMultiHotDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, class_label_file):
        super(ADARIMultiHotDataset).__init__()
        
        self.image_folder = image_folder
        self.class_label_file = class_label_file
        self.transform = transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                           ])
        self.im_to_one_hots = open_json(self.class_label_file)
        self.im_names = list(self.im_to_one_hots.keys())
        self.num_classes = len(self.im_to_one_hots[self.im_names[0]])
        
    def __len__(self):
        return len(self.im_names)
        
    def __getitem__(self, idx):
        imname = self.im_names[idx]
        
        img = Image.open(self.image_folder + '/' + imname)
        return self.transform(img), torch.tensor(self.im_to_one_hots[imname]).double()
        




data = ADARIMultiHotDataset(IMAGE_FOLDER, CLASS_LABEL_PATH)
vocab_size = data.num_classes

train_set, test_set = torch.utils.data.random_split(data, 
                                                    [int(.8 * len(data)), len(data) - int(.8 * len(data))])



# Create model
def build_model():
    vgg = torchvision.models.vgg16()
    vgg.classifier[6] = torch.nn.Linear(4096, vocab_size)
    return vgg


# Compute F1 Score
def test_score(model, test_set, threshold):
    model.eval()
    model.to(device)
    test_d = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    nonzero_words = {}
    avg_f1_score = []
    with torch.no_grad():
        for i, (im, l) in enumerate(test_d):
            im = im.to(device)
            imhat = torch.sigmoid(model(im))
            nonzeros = torch.nonzero((imhat > 0.1).cpu())
            score = f1_score(l.cpu(), (imhat > threshold).cpu(), average='samples')
            avg_f1_score.append(score)
            nonzero_words[i] = nonzeros[:,1].tolist()
    print(f"Threshold: {threshold}: {sum(avg_f1_score) / len(avg_f1_score)}")
    """
    with open("nonzero_words.json", "w") as f:
        json.dump(nonzero_words, f)
    """

print("Testing model...")
test_model = build_model()
test_model.load_state_dict(torch.load("../../final_vgg.pth"))
for t in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
    test_score(test_model, test_set, t)


# In[ ]:


#vgg


# In[ ]:




