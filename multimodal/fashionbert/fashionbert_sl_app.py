import torch, torchvision
import streamlit as st
from PIL import Image
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from utils import construct_bert_input
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import spacy

nlp = spacy.load("en_core_web_sm")
# disable everything except the tagger
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "tagger"]
nlp.disable_pipes(*other_pipes)


class FashionbertAtt(transformers.BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.im_to_embedding = torch.nn.Linear(2048, 768)
        self.im_to_embedding_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.init_weights()

    @st.cache(allow_output_mutation=True)
    def get_att(
            self,
            embeds,  # text + image embedded
            attention_mask,
            seq_number,
            attention_layer=10,
            attention_head=5,
    ):

        batch_size = embeds.shape[0]
        seq_length = embeds.shape[1]
        hidden_dim = embeds.shape[2]

        outputs = self.bert(inputs_embeds=embeds,
                            attention_mask=attention_mask,
                            return_dict=True)


        attentions = outputs.attentions  # list with 12 tensors of shape [batch=1, num_heads=12, seq_len=512, seq_len=512]
        attentions = torch.stack(attentions, dim=0).squeeze(1)  # [layers=12, heads=12, 512, 512]
        # print('inspecting att layer: {}, head: {}'.format(attention_layer, attention_head))

        att_l10 = attentions[attention_layer, attention_head, :, :]
        att_10_h5_w4 = att_l10[seq_number, :]  # [512]
        att_text = att_10_h5_w4[:448]
        att_im = att_10_h5_w4[448:]

        NUM_ATT = 10
        att_max_text = torch.topk(att_text, 10)
        att_min_text = torch.topk(att_text, 10, largest=False)

        att_max_img = torch.topk(att_im, 10)
        att_min_img = torch.topk(att_im, 10, largest=False)

        attention_mask_text = attention_mask[:, :448].squeeze(0)
        att_text_nopad = att_text[attention_mask_text == 1] # [1, 448]
        return att_max_text, att_min_text, att_max_img, att_min_img, att_text_nopad


    def highlight_random_patches(self, full_image_name, attention_idx_list, patches_positions):
        """
        Receives image name and list with N indices (from 0 to 64) corresponding to the N highest attention tokens
        in the image
        patches positions = [1, 64, 4]
        """
        img = Image.open(full_image_name)

        img = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor()])(img)  # [3, 64, 64]

        patches_positions_ = patches_positions.squeeze(0)  # [64, 4]
        intens = 0

        for idx in attention_idx_list:
            positions = patches_positions_[idx]  # [4]
            a = positions[0].detach().item() * 4
            b = positions[1].detach().item() * 4
            c = positions[2].detach().item() * 4
            d = positions[3].detach().item() * 4
            """
            a,b ------ a,d
            -           -
            -           -
            -           -
            c,b ------- c,d
            """
            R = img[0]  # [64, 64]
            G = img[1]  # [64, 64]
            B = img[2]  # [64, 64]

            # upper horizontal
            R[a, b:d] = 1.
            # Double pixels
            R[a+1, b:d] = 1.
            # left vertical
            R[a:c, b] = 1.
            # Double pixels
            R[a:c, b+1] = 1.
            # bottom horizontal
            R[c - 1, b:d] = 1.
            # Double pixels
            R[c-2, b:d] = 1.
            # right vertical
            R[a:c, d - 1] = 1.
            # Double pixels
            R[a:c, d-2] = 1.

            # upper horizontal
            G[a, b:d] = intens
            # Double pixels
            G[a + 1, b:d] = intens
            # left vertical
            G[a:c, b] = intens
            # Double pixels
            G[a:c, b + 1] = intens
            # bottom horizontal
            G[c - 1, b:d] = intens
            # Double pixels
            G[c-2, b:d] = intens
            # right vertical
            G[a:c, d - 1] = intens
            # Double pixels
            G[a:c, d-2] = intens

            # upper horizontal
            B[a, b:d] = intens
            # Double pixels
            B[a + 1, b:d] = intens
            # left vertical
            B[a:c, b] = intens
            # Double pixels
            B[a:c, b + 1] = intens
            # bottom horizontal
            B[c - 1, b:d] = intens
            # Double pixels
            B[c-2, b:d] = intens
            # right vertical
            B[a:c, d - 1] = intens
            # Double pixels
            B[a:c, d-2] = intens

            intens += 0.1

        to_pil = torchvision.transforms.ToPILImage()
        img = to_pil(img)
        # img.save('./img_with_att_patches.png')
        st.image(img, use_column_width=True)

@st.cache(allow_output_mutation=True)
def get_adjs(token_list):
    adjs = []
    occs = dict()
    i = 0
    for doc in nlp.pipe(token_list):
        if len(doc) > 0 and doc[0].pos_ == 'ADJ':
            # 1) Insert adj in dictionary and init counter or update
            if doc[0].text in occs.keys():
                occs[doc[0].text] += 1
            else:
                occs[doc[0].text] = 1
            # Append list with [ADJ, INDEX IN STRING, OCCURRENCE]
            adjs.append([doc[0].text, i, occs[doc[0].text]])
        i += 1
    return adjs

@st.cache(allow_output_mutation=True)
def get_all_words(token_list):
    words = []
    occs = dict()
    for i, token in enumerate(token_list):
        # 1) Insert adj in dictionary and init counter or update
        if not token.startswith('#'):
            if token in occs.keys():
                occs[token] += 1
            else:
                occs[token] = 1
            # Append list with [ADJ, INDEX IN STRING, OCCURRENCE]
            words.append([token, i, occs[token]])
    return words

@st.cache(allow_output_mutation=True)
def find_word_at_nth_occurrrence(s, keyword, occ):
    val = -1
    for i in range(occ):
        val = s.find(keyword, val + 1)
    return val

@st.cache(allow_output_mutation=True)
def highlight_and_bold(nopad_string, adj, occ):
    """
    nopadstring = string wihtout [PADS]
    adj = keyword
    adj_idx = index of adj in string
    """
     # Partition string
    keyword = ' ' + adj + ' '
    adj_len = len(keyword)
    idx_adj = find_word_at_nth_occurrrence(nopad_string, keyword, occ)
    # create new strings
    before = nopad_string[:idx_adj]
    after = nopad_string[idx_adj+adj_len:]

    # Find commas or periods
    before_reversed = before[::-1]
    before_comma = before_reversed.find(',')
    before_point = before_reversed.find('.')

    after_comma = after.find(',')
    after_point = after.find('.')

    # Start index highlight
    before_len = len(before)
    if before_comma != -1:
        # We need to do this operation since we found index in reversed string
        before_comma = before_len - before_comma
        marked_before = before_comma
    elif before_point != -1:
        before_point = before_len - before_point
        marked_before = before_point
    else:
        marked_before = 0

    # End index highlight
    if after_comma  != -1:
        marked_after = after_comma
    elif after_point != -1:
        marked_after = after_point
    else:
        marked_after = -1

    if marked_after != -1:
        marked_after += (before_len + adj_len)


    if marked_after != -1:
        highlighted_text = nopad_string[:marked_before] + '<mark>' + nopad_string[marked_before:marked_after] + '</mark>' + nopad_string[marked_after:]

    else:
        highlighted_text = nopad_string[:marked_before] + '<mark>' + nopad_string[marked_before:] + '</mark>'

    idx_adj = find_word_at_nth_occurrrence(highlighted_text, keyword, occ)
    highlighted_text_ = highlighted_text[:idx_adj] + '**' + highlighted_text[idx_adj:idx_adj+len(keyword)] + '**' + highlighted_text[idx_adj+len(keyword):]
    return highlighted_text_

def create_barchart_h(data, labels, word, title, topk=False):
    plt.rc('xtick', labelsize=12)
    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = ['salmon'] * len(labels)
    if topk != True:
        id = labels.index(word)
        colors[id] = 'gray'
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Attention Weights')
    ax.set_title(title)
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.legend()
    ax.barh(x, data, color=colors, align='center')
    st.pyplot(fig)

def create_barchart(data, labels):
    plt.rc('xtick', labelsize=5)
    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots(figsize=(5, 2))
    colors = ['salmon'] * len(labels)
    colors[len(labels)//2] = 'gray'
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Attention Weights')
    ax.set_title('Attention Language')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60)
    ax.legend()
    ax.bar(x, data, color=colors, align='center')
    st.pyplot(fig)

@st.cache(allow_output_mutation=True)
def set_model(pretrained_model, sample, device, path_to_dataset):
    if pretrained_model != None:
        viz = FashionbertAtt.from_pretrained(pretrained_model,
                                             output_hidden_states=True,
                                             output_attentions=True,
                                             return_dict=True)
    else:
        viz = FashionbertAtt.from_pretrained('bert-base-uncased',
                                             output_hidden_states=True,
                                             output_attentions=True,
                                             return_dict=True)

    viz.to(device)
    viz.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    f = open(path_to_dataset, "rb")
    dataset = pickle.load(f)  # dictionary of 5 elements

    all_patches = dataset['patches']
    all_input_ids = dataset['input_ids']
    all_att_mask = dataset['attention_masks']
    all_img_name = dataset['img_names']
    all_patch_pos = dataset['patch_positions']

    patches = all_patches[sample]
    input_ids = all_input_ids[sample]
    attention_mask = all_att_mask[sample]
    img_name = all_img_name[sample]
    patch_positions = all_patch_pos[sample]

    patches = patches.unsqueeze(0)                  # [1, 64, 2048]
    input_ids = input_ids.unsqueeze(0)              # [1, 448]
    attention_mask = attention_mask.unsqueeze(0)    # [1, 448]
    patch_positions = patch_positions.unsqueeze(0)  # [1, 64, 4]

    return viz, tokenizer, patches, input_ids, attention_mask, img_name, patch_positions

# @st.cache(allow_output_mutation=True)
def show_patches(full_image_name, indexes, positions, img_name=None):
    to_pil = torchvision.transforms.ToPILImage()
    im = Image.open(full_image_name)
    if img_name != None:
        im.save('_{}_.png'.format(img_name))
    im = torchvision.transforms.Compose([
        torchvision.transforms.Resize(512),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor()])(im)  # [3, 64, 64]

    patches_po = positions.clone().squeeze(0).detach() # [64, 4]
    patches_po *= 8
    patches_po = patches_po.tolist()
    all_patches = []
    av = indexes[0].detach().tolist()
    ai = indexes[1].detach().tolist()
    for i, id in enumerate(ai):
        start_x, start_y, end_x, end_y = patches_po[id]
        patch = im[:, start_x:end_x, start_y:end_y]
        patch = to_pil(patch)
        if img_name != None:
            patch.save('{}_patch_{}.png'.format(img_name, i))
        all_patches.append(patch)
    st.image(all_patches, use_column_width=False)

def test(path_to_dataset, sample, device, slider_att, slider_head, pretrained_model=None):
    im_path_fur = '/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/PHD-CD_Research/ADARI/images/ADARI_v2/furniture/full'
    viz, tokenizer, patches, input_ids, attention_mask, img_name, patch_positions = set_model(pretrained_model, sample, device, path_to_dataset)

    # Text
    inputs = input_ids.squeeze(0).detach().tolist()
    seq = tokenizer.convert_ids_to_tokens(inputs)
    seq_st = tokenizer.convert_tokens_to_string(seq)

    # Image
    image_name = im_path_fur + '/' + img_name

    embeds = construct_bert_input(patches, input_ids, viz, device=device)
    attention_mask = F.pad(attention_mask, (0, embeds.shape[1] - input_ids.shape[1]), value=1)  # [1, 512]

    # print('SEQUENCE')
    # print(seq_st)
    #
    # print('QUERY WORD')
    button = st.sidebar.radio('Limit words to adjectives?', ('Yes', 'No'))
    if button == 'Yes':
        adjs = get_adjs(seq) # list with [ADJ, INDEX IN STRING, OCCURRENCE]
    else:
        adjs = get_all_words(seq)
    adj_idx_pair = st.sidebar.selectbox('Select adjective to see attention', adjs)
    adj = adj_idx_pair[0]
    idx_adj = adj_idx_pair[1]
    adj_occ = adj_idx_pair[2]

    # Text without pad
    new_seq = []
    for token in seq:
        if token != '[PAD]':
            new_seq.append(token)
    nopad_string = tokenizer.convert_tokens_to_string(new_seq)

    #### Gets attention viz
    att_max_t, att_min_t, att_max_i, att_min_i, all_att_text = viz.get_att(embeds,  # text + image embedded
                                                                             attention_mask,
                                                                             seq_number=idx_adj,
                                                                             attention_layer=slider_att,
                                                                             attention_head=slider_head)

    att_max_t_id = att_max_t[1].tolist()
    att_max_t_v = att_max_t[0].tolist()
    att_min_t = att_min_t[1].tolist()
    att_max_i_id = att_max_i[1].tolist()
    att_min_i = att_min_i[1].tolist()

    N = 20
    # Get N attentions around word
    if idx_adj >= N:
        neighbors = all_att_text[idx_adj-N:idx_adj+N]
        if (idx_adj) + N >= 448:
            neighbors_ids = range(idx_adj - N, 448)
        else:
            neighbors_ids = range(idx_adj - N, idx_adj + N)
    else:
        neighbors = all_att_text[:idx_adj + N]
        neighbors_ids = range(0, idx_adj + N)


    neighbors = neighbors.tolist()
    labels = [seq[id] for id in neighbors_ids]
    # This line prevents from going over the limit of the text sequence
    labels = labels[:len(neighbors)]

    col1, col2, col3 = st.beta_columns(3)
    # Visualize patches that receive most attention
    with col1:
        viz.highlight_random_patches(image_name, att_max_i_id, patch_positions)
    with col2:
        create_barchart_h(neighbors, labels, adj, 'Attention weights in the 20 words around the query')
    with col3:
        labels2 = [seq[id] for id in att_max_t_id]
        create_barchart_h(att_max_t_v, labels2, adj, 'Attention weights of the top 10 words', topk=True)

    st.subheader('Top-10 Attention Patches')
    show_patches(image_name, att_max_i, patch_positions, None)
    t = highlight_and_bold(nopad_string, adj, adj_occ)
    st.subheader('Design Description')
    st.markdown(t, unsafe_allow_html=True)
    chart_data = pd.DataFrame(
                all_att_text.detach().numpy()
    )
    st.subheader('Attention Weights of the entire sequence')
    st.area_chart(chart_data, use_container_width=True)


def main():
    # Set up layout
    max_width = 1200
    padding_top = 0
    padding_right = 0
    padding_left = 0
    padding_bottom = 0
    COLOR = 'black'
    BACKGROUND_COLOR = 'white'
    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            color: {COLOR};
            background-color: {BACKGROUND_COLOR};
        }}
    </style>
    """,
    unsafe_allow_html = True
    )

    st.title('Object-Agnostic Bert')
    st.subheader('Attention-Weights Visualizer for disentangling Design-Intents')
    att_layer = st.sidebar.slider('Choose Attention Layer', 0, 11, 9, 1, '%d', 'att_layer_slider')
    att_head = st.sidebar.slider('Choose Attention Head', 0, 11, 11, 1, '%d', 'att_head_slider')
    # sample = st.sidebar.slider('Choose sample', 0, 1000, 14, 1, '%d', 'iteration_sample')
    samples = range(0,1000)
    sample = st.sidebar.selectbox('Choose sample', samples)

    # Path to models and data
    # dataset_path2 = '/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/PHD-CD_Research/github/__fashionbert_trained/pickle_files/rp_nm_res50/rp_nm_res50_eval_dataset.pkl'
    dataset_path = '/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/PHD-CD_Research/github/__fashionbert_trained/pickle_files/np_nm_res50/evaluation_set_fashionbert_vanilla.pkl'
    model_path = '/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/PHD-CD_Research/github/__fashionbert_trained/fashionbert_np_nm_10epochs/'
    # model_path2 = '/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/PHD-CD_Research/github/__fashionbert_trained/fashionbert_np_nm_10epochs'

    # model_path = '/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/PHD-CD_Research/github/__fashionbert_trained/fashionbert_07:33:58_rp_am_10e_manuel/'

    # Run app
    test(dataset_path, sample, device, att_layer, att_head, model_path)

main()