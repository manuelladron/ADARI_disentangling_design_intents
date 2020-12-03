import torch, torchvision
import json
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizer
import pandas as pd 
import pickle
import random


def open_json(path):
    f = open(path) 
    data = json.load(f) 
    f.close()
    return data 

def save_json(file_path, data):
    out_file = open(file_path, "w")
    json.dump(data, out_file)
    out_file.close()



def construct_bert_input(patches, input_ids, fashion_bert, device=None):
    # patches shape: batch size, im sequence length, embedding size
    # input_ids shape: batch size, sentence length

    # shape: batch size, sequence length, embedding size
    #word_embeddings = fashion_bert.bert.embeddings(
    #    input_ids.to(device), 
    #    token_type_ids=torch.zeros(input_ids.shape, dtype=torch.long).to(device), 
    #    position_ids=torch.arange(0, input_ids.shape[1], dtype=torch.long).to(device) * torch.ones(input_ids.shape, dtype=torch.long).to(device))
    word_embeddings = fashion_bert.bert.embeddings(input_ids.to(device))

    image_position_ids = torch.arange(1, patches.shape[1]+1, dtype=torch.long).view(-1, 1) * torch.ones(patches.shape[0], dtype=torch.long)
    image_position_ids = image_position_ids.T
    image_token_type_ids = torch.ones((patches.shape[0], patches.shape[1]), dtype=torch.long)

    image_position_embeds = fashion_bert.bert.embeddings.position_embeddings(image_position_ids.to(device))
    image_token_type_embeds = fashion_bert.bert.embeddings.token_type_embeddings(image_token_type_ids.to(device))

    # transforms patches into batch size, im sequence length, 768
    im_seq_len = patches.shape[1]
    patches = patches.view(-1, patches.shape[2])
    patches = fashion_bert.im_to_embedding(patches.to(device))
    # now shape batch size, im sequence length, 768
    patches = patches.view(word_embeddings.shape[0], im_seq_len, -1)

    # shape: batch size, im sequence length, embedding size
    image_embeddings = patches + image_position_embeds + image_token_type_embeds
    image_embeddings = fashion_bert.im_to_embedding_norm(image_embeddings)

    return torch.cat((word_embeddings, image_embeddings), dim=1)


class EncoderCNN(torch.nn.Module):
    def __init__(self):
        """Load the pretrained ResNet50 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.eval()
        modules = list(resnet.children())[:-1]      # delete the last fc (classification) layer.
        self.resnet = torch.nn.Sequential(*modules)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        return features


class MultiModalBertDataset(Dataset):
    def __init__(
        self, 
        path_to_images, 
        data_dict_path,
        patch_size = 8, 
        img_size = 64,
        device = None
    ):
        super(MultiModalBertDataset).__init__()
        self.img_path = path_to_images

        # list of dicts giving img name, sentence, 
        # and whether the sentence is paired or not
        self.data_dict = open_json(data_dict_path)

        self.patch_size = patch_size
        self.img_size = img_size

        self.im_encoder = EncoderCNN()
        self.im_encoder.eval()
        self.im_encoder.to(device)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = device

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        sample = self.data_dict[index]

        image_name = sample['id']
        text = sample['text']
        is_paired = sample['label']
        
        name = self.img_path + "/" + image_name
        img = Image.open(name)
        
        img = torchvision.transforms.Compose([
        torchvision.transforms.Resize(self.img_size),
        torchvision.transforms.CenterCrop(self.img_size),
        torchvision.transforms.ToTensor()])(img)
        
        # pad just in case
        img = F.pad(img, (img.shape[2] % self.patch_size // 2, img.shape[2] % self.patch_size // 2,
                         img.shape[1] % self.patch_size // 2, img.shape[1] % self.patch_size // 2))
        
        patches = []
        with torch.no_grad():
            for i in range(img.shape[1] // self.patch_size):
                for j in range(img.shape[2] // self.patch_size):
                    encoded_patch = self.im_encoder(img[:, 
                                            i*self.patch_size:(i+1)*self.patch_size, 
                                            j*self.patch_size:(j+1)*self.patch_size].reshape(1, img.shape[0], self.patch_size, self.patch_size).to(self.device))
                    patches.append(encoded_patch[0])
        
        tokens = self.tokenizer(
            "".join([s + ' ' for s in text[0]]),
            max_length = 448,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt')
    
        return torch.stack(patches), tokens['input_ids'][0], torch.tensor(is_paired), tokens['attention_mask'][0]


class FashionBertRandomPatchesDataset(Dataset):
    def __init__(
        self, 
        path_to_images, 
        data_dict_path,
        num_patches = 64, 
        img_size = 64,
        device = None
    ):
        super(MultiModalBertDataset).__init__()
        self.img_path = path_to_images

        # list of dicts giving img name, sentence, 
        # and whether the sentence is paired or not
        self.data_dict = open_json(data_dict_path)

        self.img_size = img_size
        self.num_patches = num_patches

        self.im_encoder = EncoderCNN()
        self.im_encoder.eval()
        self.im_encoder.to(device)

        self.min_patch_dim = 4
        self.max_patch_dim = 16
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = device

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        sample = self.data_dict[index]

        image_name = sample['id']
        text = sample['text']
        is_paired = sample['label']
        
        name = self.img_path + "/" + image_name
        img = Image.open(name)
        
        img = torchvision.transforms.Compose([
        torchvision.transforms.Resize(self.img_size),
        torchvision.transforms.CenterCrop(self.img_size),
        torchvision.transforms.ToTensor()])(img)
        
        patches = []
        with torch.no_grad():
            for _ in range(num_patches):
                height = random.randrange(self.min_patch_dim, self.max_patch_dim+1, 2)
                width = random.randrange(self.min_patch_dim, self.max_patch_dim+1, 2)

                start_x = random.randrange(0, self.img_size - height)
                start_y = random.randrange(0, self.img_size - width)

                patch = img[:, start_x:start_x + height, start_y:start_y + height]
                patches.append(self.im_encoder(patch.reshape(-1, patch.shape[0], patch.shape[1], patch.shape[2]))[0])
        
        tokens = self.tokenizer(
            "".join([s + ' ' for s in text[0]]),
            max_length = 448,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt')
    
        return torch.stack(patches), tokens['input_ids'][0], torch.tensor(is_paired), tokens['attention_mask'][0], image_name

class PreprocessedADARI(Dataset):
    def __init__(self, path_to_dataset):
        super(PreprocessedADARI).__init__()
        self.path_to_dataset = path_to_dataset
        
        f = open(self.path_to_dataset, "rb")
        self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataset.iloc[idx]

        return (
            torch.tensor(sample.patches).view(sample.patches.shape[0], sample.patches.shape[1]), 
            torch.tensor(sample.input_ids),
            torch.tensor(sample.is_paired),
            torch.tensor(sample.attention_mask)
            )
