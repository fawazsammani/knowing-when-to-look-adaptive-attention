import os
import h5py
import json
import torch
from torch.utils.data import Dataset
import torch.optim
import torch.utils.data

class COCOTrainDataset(Dataset):

    def __init__(self, transform=None):

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join('caption data', 'TRAIN_IMAGES_coco' + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join('caption data','TRAIN_CAPTIONS_coco.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join('caption data', 'TRAIN_CAPLENS_coco' + '.json'), 'r') as j:
            self.caplens = json.load(j)
            
        with open(os.path.join('caption data', 'TRAIN_names_coco' + '.json'), 'r') as j:
            self.image_names = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        """
        returns:
        img: the image convereted into a tensor of shape (batch_size,3, 256, 256)
        caption: the ground-truth caption of shape (batch_size, max_length)
        caplen: the valid length (without padding) of the ground-truth caption of shape (batch_size,1)
        """
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        img_name = self.image_names[i // self.cpi]
        
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        
        return img, caption, caplen

    def __len__(self):
        return self.dataset_size


class COCOValidationDataset(Dataset):

    def __init__(self, transform=None):

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join('caption data', 'VAL_IMAGES_coco' + '.hdf5'), 'r')
        self.imgs = self.h['images']

        with open(os.path.join('caption data', 'VAL_names_coco' + '.json'), 'r') as j:
            self.image_names = json.load(j)
            
        with open('caption data/VAL_NAMES_IDS.json', 'r') as j:
            self.name_ids = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.image_names)

    def __getitem__(self, i):
        """
        returns:
        img: the image convereted into a tensor of shape (batch_size,3, 256, 256)
        image_id: the respective id for the image of shape (batch_size, 1)
        """
        img = torch.FloatTensor(self.imgs[i] / 255.)
        img_name = self.image_names[i]
        
        if self.transform is not None:
            img = self.transform(img)
            
        image_id = torch.LongTensor([self.name_ids[img_name]])
        
        return img, image_id

    def __len__(self):
        return self.dataset_size
