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
        


# In[3]:


# Load Data

data = ADARIMultiHotDataset(IMAGE_FOLDER, CLASS_LABEL_PATH)
vocab_size = data.num_classes

train_set, test_set = torch.utils.data.random_split(data, 
                                                    [int(.8 * len(data)), len(data) - int(.8 * len(data))])


# In[4]:


# Create model
def build_model():
    vgg = torchvision.models.vgg16()
    vgg.classifier[6] = torch.nn.Linear(4096, vocab_size)
    return vgg
vgg = build_model()


# In[6]:


# Training Parameters
batch_size = 64
num_workers = 1
lr = 0.001
num_epochs = 100


# In[26]:


# Training loop

def train(model, train_loss, test_loss):
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    dataloader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(vgg.parameters(), lr=lr)
    
    
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for im, labels in dataloader:
            
            im = im.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            l_hat = model(im)
            loss = criterion(l_hat, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        train_loss.append(sum(losses) / len(losses))
        print(f"Avg Loss at Epoch {epoch}: {train_loss[-1]}")

        ep_test_losses = []
        model.eval()
        with torch.no_grad():
            for im, labels in test_loader:
                im = im.to(device)
                labels = labels.to(device)
                l_hat = model(im)
                loss = criterion(l_hat, labels)
                ep_test_losses.append(loss.item())
        test_loss.append(sum(ep_test_losses) / len(ep_test_losses))
        print(f"Avg Test Loss at Epoch {epoch}: {test_loss[-1]}")

    return train_loss, test_loss
        
        
        


# In[27]:

train_loss = []
test_loss = []
model_name = datetime.datetime.now()
try:
    train(vgg, train_loss, test_loss)
except KeyboardInterrupt:
    pass
vgg.cpu()
torch.save(vgg.state_dict(), f"VGG16_ADARI_{model_name}.pth")
with open(f"losses_{model_name}.json", "w") as f:
    json.dump({"train_loss": train_loss, "test_loss": test_loss}, f)


# In[ ]:

"""
# Compute F1 Score
def test_score(model, test_set):
    model.eval()
    model.to(device)
    test_d = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    for im, l in test_d:
        im = im.to(device)
        imhat = model(im)
        imhat.to('cpu')
        score = f1_score(l, imhat, average='weighted')
        print(f"F1 Score: {score}")

test_model = build_model()
test_model.load_state_dict(torch.load("../../final_vgg.pth"))
test_score(test_model, test_set)
"""

# In[ ]:


#vgg


# In[ ]:




