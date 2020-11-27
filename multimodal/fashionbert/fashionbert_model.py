import torch, torchvision
import json
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizer, BertModel, BertOnlyMLMHead, BertOnlyNSPHead
from utils import construct_bert_input


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
        patches,
        input_ids,
        labels=None,
        unmasked_patch_features=None,
        is_paired=None,
    ):
        """
            Args: 
                patches
                    Masked image features
                        batch size, image sequence length, image embedding size
                input_ids
                    Masked tokenized token ids
                        batch size, word sequence length
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
        embeds = construct_bert_input(patches, input_ids, self)

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
