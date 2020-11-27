import torch, torchvision
import json
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AdamW
from transformers import BertTokenizer, BertModel, BertOnlyMLMHead, BertOnlyNSPHead
from utils import construct_bert_input, MultiModalBertDataset


class FashionBert(torch.nn.Module):
    def __init__(self, pretrained_str = 'bert-base-uncased'):
        super(FashionBert).__init__()

        self.model = BertModel.from_pretrained(pretrained_str, return_dict=True)
        self.config = self.model.config

        self.embeddings = self.model.embeddings
        self.position_embeddings = self.model.embeddings.position_embeddings
        self.token_type_embeddings = self.model.token_type_embeddings

        self.im_patch_fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 768),
            torch.nn.LeakyReLU()
        )

        self.masked_prediction = BertOnlyMLMHead(self.config)
        self.alignment_prediction = BertOnlyNSPHead(self.config)

    def forward(
        self,
        embeds,
        labels=None,
        unmasked_patch_features=None,
        is_paired=None,
    ):
        """
            Args: 
                embeds
                    hidden embeddings to pass to the bert model
                        batch size, seq length, hidden dim
                labels
                    Unmasked tokenized token ids
                        batch size, word sequence length
                unmasked_patch_features
                    Unmasked image features
                        batch size, image sequence length, image embedding size
                is_paired
                    bool tensor, Whether the sample is aligned with the sentence
                        batch size, 1
        """

        outputs = self.model(input_embeds=embeds, return_dict=True)
        sequence_output = outputs.last_hidden_state
        pooler_output = outputs.pooler_output

        # hidden states corresponding to the text part
        text_output = sequence_output[:, :input_ids.shape[1]-1, :]
        # hidden states corresponding to the image part
        image_output = sequence_output[:, input_ids.shape[1]:, :]

        masked_lm_loss = None
        if labels is not None:
            prediction_scores = self.masked_prediction(text_output)
            loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        masked_patch_loss = None
        if unmasked_patch_features is not None:
            pred_probs = torch.nn.LogSoftmax(dim=2)(image_output)
            true_probs = torch.nn.LogSoftmax(dim=2)(unmasked_patch_features)
            loss_fct = torch.nn.KLDivLoss()
            masked_patch_loss = loss_fct(true_probs, pred_probs)
        
        alignment_loss = None
        if is_paired is not None:
            seq_relationship_scores = self.alignment_prediction(pooler_output)
            loss_fct = torch.nn.CrossEntropyLoss()
            alignment_loss = loss_fct(seq_relationship_scores.view(-1, 2), is_paired.double().view(-1))
            
        return {
            "raw_outputs": outputs, 
            "masked_lm_loss": masked_lm_loss,
            "masked_patch_loss": masked_patch_loss,
            "alignment_loss": alignment_loss
            }


def train(fashion_bert, dataset, params, device):
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=params.batch_size, 
        shuffle=True,
        )

    fashion_bert.to(device)
    fashion_bert.train()
    opt = AdamW(fashion_bert.params(), lr=params.lr)

    for _ in range(params.num_epochs):
        for patches, input_ids, is_paired in dataloader:
            patches = patches.to(device)
            input_ids = input_ids.to(device)
            is_paired = is_paired.to(device)

            opt.zero_grad()

            # mask image patches with prob 10%
            im_seq_len = patches.shape[1]
            masked_patches = patches.detach().clone().to(device)
            masked_patches = masked_patches.view(-1, patches.shape[2])
            im_mask = torch.rand((masked_patches.shape[0], 1)) >= 0.1
            masked_patches *= im_mask
            masked_patches = masked_patches.view(params.batch_size, im_seq_len, masked_patches.shape[2])

            # mask tokens with prob 15%, note id 103 is the [MASK] token
            token_mask = torch.rand(input_ids.shape)
            masked_input_ids = input_ids.detach().clone().to(device)
            masked_input_ids[token_mask < 0.15] = 103

            embeds = construct_bert_input(masked_patches, masked_input_ids, fashion_bert)
            outputs = fashion_bert(embeds, input_ids, patches, is_paired)

            loss = (1. / 3.) * outputs['masked_lm_loss'] \
                + (1. / 3.) * outputs['masked_patch_loss'] \
                + (1. / 3.) * outputs['alignment_loss']

            loss.backward()
            opt.step()

