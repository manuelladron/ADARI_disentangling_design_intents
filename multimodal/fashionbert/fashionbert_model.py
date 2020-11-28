import torch, torchvision
import json
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AdamW
from transformers import BertTokenizer, BertModel
from transformers.modeling_bert import BertPreTrainingHeads
from utils import construct_bert_input, MultiModalBertDataset
from transformers import get_linear_schedule_with_warmup
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FashionBert(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.im_patch_fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 768),
            torch.nn.LeakyReLU()
        )

        self.cls = BertPreTrainingHeads(config)
        self.init_weights()

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

        outputs = self.bert(inputs_embeds=embeds, return_dict=True)
        sequence_output = outputs.last_hidden_state
        pooler_output = outputs.pooler_output

        # hidden states corresponding to the text part
        text_output = sequence_output[:, :labels.shape[1], :]
        # hidden states corresponding to the image part
        image_output = sequence_output[:, labels.shape[1]:, :]

        print(f"Text output: {text_output.shape}")
        print(f"Image output: {image_output.shape}")

        prediction_scores, alignment_scores = self.cls(text_output, pooler_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        masked_patch_loss = None
        if unmasked_patch_features is not None:
            pred_probs = torch.nn.LogSoftmax(dim=2)(image_output)
            true_probs = torch.nn.LogSoftmax(dim=2)(unmasked_patch_features)
            print(pred_probs.shape)
            print(true_probs.shape)
            loss_fct = torch.nn.KLDivLoss()
            masked_patch_loss = loss_fct(true_probs, pred_probs)
        
        alignment_loss = None
        if is_paired is not None:
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
    opt = AdamW(
        fashion_bert.parameters(), 
        lr=params.lr, 
        betas=(params.beta1, params.beta2), 
        weight_decay=params.weight_decay
        )
    scheduler = get_linear_schedule_with_warmup(opt, params.num_warmup_steps, params.num_epochs * len(dataloader))

    for _ in range(params.num_epochs):
        for patches, input_ids, is_paired in dataloader:
            patches = patches.to(device)
            input_ids = input_ids.to(device)
            is_paired = is_paired.to(device)

            print(f"Input id shape: {input_ids.shape}")

            opt.zero_grad()

            # mask image patches with prob 10%
            im_seq_len = patches.shape[1]
            masked_patches = patches.detach().clone().to(device)
            masked_patches = masked_patches.view(-1, patches.shape[2])
            im_mask = torch.rand((masked_patches.shape[0], 1)) >= 0.1
            masked_patches *= im_mask
            masked_patches = masked_patches.view(params.batch_size, im_seq_len, patches.shape[2])

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
            scheduler.step()

            print("one step")


class TrainParams:
    lr = 2e-5
    batch_size = 64
    beta1 = 0.95
    beta2 = .999
    weight_decay = 1e-4
    num_warmup_steps = 5000
    num_epochs = 10


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FashionBert.')
    parser.add_argument('--path_to_images', help='Absolute path to image directory', default='/Users/alexschneidman/Documents/CMU/CMU/F20/777/ADARI/v2/full')
    parser.add_argument('--path_to_data_dict', help='Absolute path to json containing img name, sentence pair dict', default='/Users/alexschneidman/Documents/CMU/CMU/F20/777/ADARI/ADARI_furniture_pairs.json')
    args = parser.parse_args()

    params = TrainParams()

    fashion_bert = FashionBert.from_pretrained('bert-base-uncased', return_dict=True)
    dataset = MultiModalBertDataset(
        args.path_to_images, 
        args.path_to_data_dict,
        device=device,
        )

    try:
        train(fashion_bert, dataset, params, device)
    except KeyboardInterrupt:
        fashion_bert.save_pretrained('pretrained_fashionbert')