import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import random


class Evaluation_negpairs(Dataset):
    def __init__(self, path_to_dataset):
        super(Evaluation_negpairs).__init__()
        f = open(path_to_dataset, "rb")
        self.dataset = pickle.load(f)

        self.patches = self.dataset['patches']
        self.input_ids = self.dataset['input_ids']
        self.att_masks = self.dataset['attention_masks']
        self.images = self.dataset['img_names']
        self.patch_positions = self.dataset['patch_positions']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patches  = self.patches[idx]  # [64, 2048]
        input_ids = self.input_ids[idx]  # [448]
        att_masks = self.att_masks[idx]  # [448]
        im_name = self.images[idx]
        patch_pos = self.patch_positions[idx]

        # Generate 100 random indices
        negative_indices = random.sample(range(0, len(self.images)), 100)

        # Sample 100 negative pairs for sample
        neg_input_ids = [self.input_ids[i] for i in negative_indices if i != idx]
        neg_att_masks = [self.att_masks[i] for i in negative_indices if i != idx]

        # Sample 100 negative images
        neg_patches = [self.patches[i] for i in negative_indices if i != idx]

        neg_input_ids = torch.stack(neg_input_ids, dim=0)  # [100, 448]
        neg_att_masks = torch.stack(neg_att_masks, dim=0)  # [100, 448]
        neg_patches = torch.stack(neg_patches, dim=0)  # [NUM_SAMPLES, 64, 2048]

        return (
            patches.view(patches.shape[0], patches.shape[1]),  # [64, 2048]
            neg_patches.clone().detach(),  # [NUM_SAMPLES, 64, 2048]
            input_ids.clone().detach(),  # [448]
            att_masks.clone().detach(),  # [448]
            neg_input_ids.clone().detach(),  # [100, 448]
            neg_att_masks.clone().detach(),  # [100, 448]
            im_name
        )

def get_all_paired_test_set(dataset, savefile_path, num_samples=1000):
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    train_size = int(len(dataset) * .8)
    test_size = len(dataset) - train_size
    _, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    print('Original test set size: ', len(test_set))

    dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
    )

    all_patches = []
    all_ids = []
    all_masks = []
    all_im_names = []
    all_patch_positions = []

    paired = 0
    with torch.no_grad():
        for i, (patches, input_ids, is_paired, attention_mask, img_name, patch_positions) in enumerate(dataloader):
            if paired >= num_samples:
                print('Paired: ', paired)
                break

            # Convert tuple of images to numpy
            np_imgs = np.asarray(img_name)
            np_ispaired = is_paired.numpy()
            np_aligned_imgs = np_imgs[np_ispaired]
            aligned_imgs = np_aligned_imgs.tolist()

            # The rest
            aligned_ids= input_ids[is_paired]
            aligned_patches = patches[is_paired]
            aligned_att_mask = attention_mask[is_paired]
            aligned_patch_pos = patch_positions[is_paired]

            num_paired = aligned_ids.shape[0]

            if num_paired > 0:
                paired += num_paired
                all_patches.append(aligned_patches)
                all_ids.append(aligned_ids)
                all_masks.append(aligned_att_mask)
                all_im_names.extend(aligned_imgs)
                all_patch_positions.append(aligned_patch_pos)
                #aligned_test_set.append((aligned_patches, aligned_ids, aligned_att_mask, aligned_imgs, aligned_patch_pos))

            else: continue

    print(all_patches[0].shape)
    print(all_patches[10].shape)
    PATCHES = torch.cat(all_patches, dim=0)
    IDS = torch.cat(all_ids, dim=0)
    MASKS = torch.cat(all_masks, dim=0)
    PATCH_POS = torch.cat(all_patch_positions, dim=0)

    PATCHES = PATCHES[:1000, ]
    IDS = IDS[:1000, ]
    MASKS = MASKS[:1000, ]
    PATCH_POS = PATCH_POS[:1000, ]
    IMGS = all_im_names[:1000]
    print('Saving test set...')

    D = {'patches': PATCHES,
         'input_ids': IDS,
         'attention_masks': MASKS,
         'img_names':IMGS,
         'patch_positions':PATCH_POS}

    print('Length paired test set: ', PATCHES.shape[0])
    with open(savefile_path, 'wb') as handle:
        pickle.dump(D, handle)

    print('--dataset saved in: {}'.format(savefile_path))


# if __name__ == '__main__':

    # print('Processing the dataset...')
    # dataset = EvaluationDataset('../../../__fashionbert_trained/fashionbert_vanilla_adaptive/preprocessed_fashionbert_vanilla.pkl')
    # savefile_path = '../../../__fashionbert_trained/fashionbert_vanilla_adaptive/evaluation_set_fashionbert_vanilla.pkl'
    # print('Done!')
    # print('\nEvaluating...')
    # get_all_paired_test_set(dataset, savefile_path, num_samples=1000)
    # print('Done!')

