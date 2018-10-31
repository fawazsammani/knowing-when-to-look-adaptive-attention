import os
import h5py
import json
import torch
from torch.utils.data import Dataset
import torch.optim
import torch.utils.data

#A PyTorch Dataset class to be used in a PyTorch DataLoader to create mini-batches.
class CaptionDataset(Dataset):

    def __init__(self, split, transform=None):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join('caption data', self.split + '_IMAGES_' + '.hdf5'), 'r')
        self.imgs = self.h['images']
        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']
        # Load encoded captions (completely into memory)
        with open(os.path.join('caption data', self.split + '_CAPTIONS_' + '.json'), 'r') as j:
            self.captions = json.load(j)
        # Load caption lengths (completely into memory)
        with open(os.path.join('caption data', self.split + '_CAPLENS_' + '.json'), 'r') as j:
            self.caplens = json.load(j)
        self.transform = transform
        # Total number of datapoints
        self.dataset_size = len(self.captions)
        
    def __getitem__(self, i):
        # i is the caption. To get it's corresponding image, we do the following:
        # The Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)  #Manually create the torch tensor rather than transforms.ToTensor()
        if self.transform is not None:
            img = self.transform(img)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            #For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            #Example: Returning all the captions for caption 17 -->  
            #[(17//5)*5   :  (17//5)*5  +  5] --> [15:20]--> all captions of image 3
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size