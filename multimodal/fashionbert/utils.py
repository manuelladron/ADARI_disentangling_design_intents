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

def construct_bert_input(patches, input_ids, fashion_bert, device=None, random_patches=False):
    # patches shape: batch size, im sequence length, embedding size
    # input_ids shape: batch size, sentence length

    # shape: batch size, sequence length, embedding size
    #word_embeddings = fashion_bert.bert.embeddings(
    #    input_ids.to(device), 
    #    token_type_ids=torch.zeros(input_ids.shape, dtype=torch.long).to(device), 
    #    position_ids=torch.arange(0, input_ids.shape[1], dtype=torch.long).to(device) * torch.ones(input_ids.shape, dtype=torch.long).to(device))
    word_embeddings = fashion_bert.bert.embeddings(input_ids.to(device))

    if not random_patches:
        image_position_ids = torch.arange(1, patches.shape[1]+1, dtype=torch.long).view(-1, 1) * torch.ones(patches.shape[0], dtype=torch.long)
        image_position_ids = image_position_ids.T
    image_token_type_ids = torch.ones((patches.shape[0], patches.shape[1]), dtype=torch.long)

    if not random_patches:
        image_position_embeds = fashion_bert.bert.embeddings.position_embeddings(image_position_ids.to(device))
    image_token_type_embeds = fashion_bert.bert.embeddings.token_type_embeddings(image_token_type_ids.to(device))

    # transforms patches into batch size, im sequence length, 768
    im_seq_len = patches.shape[1]
    patches = patches.view(-1, patches.shape[2])
    patches = fashion_bert.im_to_embedding(patches.to(device))
    # now shape batch size, im sequence length, 768
    patches = patches.view(word_embeddings.shape[0], im_seq_len, -1)

    # shape: batch size, im sequence length, embedding size
    if not random_patches:
        image_embeddings = patches + image_position_embeds + image_token_type_embeds
    else:
        image_embeddings = patches + image_token_type_embeds
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

class Encoder_mCNN_resnet(torch.nn.Module):
    """
    Resnet152 to load weights from 11-777 midterm mCNN_V4.1 coordinated representation network

    """
    def __init__(self, path_to_pretrained):
        """Load the pretrained ResNet50 and replace top fc layer."""
        super(Encoder_mCNN_resnet, self).__init__()
        self.resnet = torchvision.models.resnet152(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, 50)
        self.load_pretrained(path_to_pretrained)
        self.resnet.eval()
        modules = list(self.resnet.children())[:-1]  # delete the last fc (classification) layer.
        self.resnet_ = torch.nn.Sequential(*modules)

    def load_pretrained(self, path_to_pretrained):
        checkpoint = torch.load(path_to_pretrained, map_location=torch.device('cpu'))
        self.resnet.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet_(images)
        return features
    
    
class MultiModalBertDataset(Dataset):
    def __init__(
        self, 
        path_to_images, 
        data_dict_path,
        path_to_encoder_model=None,
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

        if path_to_encoder_model == None:
            self.im_encoder = EncoderCNN()
        else:
            self.im_encoder = Encoder_mCNN_resnet(path_to_encoder_model)
        
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
        patch_positions = []
        with torch.no_grad():
            for i in range(img.shape[1] // self.patch_size):
                for j in range(img.shape[2] // self.patch_size):
                    patches.append(img[:, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size])
                    patch_positions.append((i*self.patch_size, j*self.patch_size, (i+1)*self.patch_size, (j+1)*self.patch_size))
            processed_patches = self.im_encoder(torch.stack(patches).to(self.device))
        
        tokens = self.tokenizer(
            "".join([s + ' ' for s in text[0]]),
            max_length = 448,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt')
    
        return processed_patches, tokens['input_ids'][0], torch.tensor(is_paired), tokens['attention_mask'][0], image_name, torch.tensor(patch_positions, dtype=torch.long)


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
 
class EvaluationDataset(Dataset):
    def __init__(self, path_to_dataset):
        super(EvaluationDataset).__init__()
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
            torch.tensor(sample.attention_mask),
            sample.img_name,
            torch.tensor(sample.patch_positions)
        )

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
        patch_positions = []
        with torch.no_grad():
            for _ in range(self.num_patches):
                height = random.randrange(self.min_patch_dim, self.max_patch_dim+1, 2)
                width = random.randrange(self.min_patch_dim, self.max_patch_dim+1, 2)

                start_x = random.randrange(0, self.img_size - height)
                start_y = random.randrange(0, self.img_size - width)

                patch = img[:, start_x:start_x + height, start_y:start_y + width]

                pad_up = (self.max_patch_dim - patch.shape[1]) // 2
                pad_bottom = self.max_patch_dim - patch.shape[1] - pad_up

                pad_left = (self.max_patch_dim - patch.shape[2]) // 2
                pad_right = self.max_patch_dim - patch.shape[2] - pad_left
                patches.append(F.pad(patch, (pad_left, pad_right, pad_up, pad_bottom)))

                patch_positions.append((start_x, start_y, start_x+height, start_y+width))
            processed_patches = self.im_encoder(torch.stack(patches).to(self.device))
        
        tokens = self.tokenizer(
            "".join([s + ' ' for s in text[0]]),
            max_length = 448,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt')
    
        return processed_patches, tokens['input_ids'][0], torch.tensor(is_paired), tokens['attention_mask'][0], image_name, torch.tensor(patch_positions, dtype=torch.long)


def mask_input_ids(
    input_ids, 
    pos, 
    adj_mask_prob, 
    other_mask_prob, 
    adj_value,
    attention_mask,
    mask_value=103
    ):
    """
    input_ids: Shape (batch_size, sequence length)
    pos: Shape (batch_size, sequence length)
    adj_mask_prob: probability of masking adjective words (must be in [0, 1])
    other_mask_prob: probability of masking non-adjective words (must be in [0, 1])
    adj_value: Value in pos tensor corresponding to an adjective
    attention_mask: Shape (batch_size, sequence length), The attention mask given as input
    mask_value: The value to set masked tokens to (probably 103)

    returns:
        masked_input_ids, labels
        masked_input_ids should be fed into construct_bert_input
        labels should be fed into fashion_bert label argument
    """

    masked_input_ids = input_ids.detach().clone()
    labels = input_ids.detach().clone()

    # mask the adjectives    
    # get all adjectives
    adj = masked_input_ids[pos == adj_value]
    # create mask
    adj_mask = torch.rand(adj.shape)
    # set them to mask_value with prob adj_mask_prob
    adj[adj_mask < adj_mask_prob] = mask_value
    # set the masked_input_ids adjectives to the masked adj vector
    masked_input_ids[pos == adj_value] = adj

    labels_adj = labels[pos == adj_value]
    labels_adj[adj_mask >= adj_mask_prob] = -100
    labels[pos == adj_value] = labels_adj

    # mask other words
    other = masked_input_ids[pos != adj_value]
    other_mask = torch.rand(other.shape)
    other[other_mask < other_mask_prob] = mask_value
    masked_input_ids[pos != adj_value] = other

    labels_other = labels[pos != adj_value]
    labels_other[other_mask >= other_mask_prob] = -100
    labels[pos != adj_value] = labels_other

    labels[attention_mask == 0] = -100

    return masked_input_ids, labels

def test():
    input_ids = torch.rand((64, 50))
    pos = (torch.rand((64, 50)) * 10).long()
    adj_value = 1
    adj_mask_prob = .5
    other_mask_prob = .1
    attention_mask = torch.ones(input_ids.shape)
    
    return mask_input_ids(input_ids, pos, adj_value, adj_mask_prob, adj_value, attention_mask)


class Pos:
    POS_MAP  = ['UNK', 'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 
                'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']
    
    TAG_MAP = [
        ".",        
        ",",        
        "-LRB-",    
        "-RRB-",    
        "``",       
        "\"\"",     
        "''",       
        ",",        
        "$",        
        "#",        
        "AFX",      
        "CC",       
        "CD",       
        "DT",       
        "EX",       
        "FW",       
        "HYPH",     
        "IN",       
        "JJ",       
        "JJR",      
        "JJS",      
        "LS",       
        "MD",       
        "NIL",      
        "NN",       
        "NNP",      
        "NNPS",     
        "NNS",   
        "PDT",   
        "POS",   
        "PRP",   
        "PRP$",  
        "RB",    
        "RBR",   
        "RBS",   
        "RP",    
        "SP",    
        "SYM",   
        "TO",    
        "UH",    
        "VB",    
        "VBD",  
        "VBG",  
        "VBN",  
        "VBP",  
        "VBZ",  
        "WDT",  
        "WP",   
        "WP$",  
        "WRB",  
        "ADD",  
        "NFP",   
        "GW",    
        "XX",    
        "BES",   
        "HVS",   
        "_SP",   
    ]
