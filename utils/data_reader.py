"""
Data reader: create the dataset object and read from folders
"""
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch
import os
import numpy as np
from skimage import io, transform

class EyeDataset(Dataset):
    def __init__(self, flags, transform=None,train=True):
        """
        The initialization function of the EyeDataset
        :param flags: The parameter flags
        :param transform: The transforms to apply
        """
        if train:
            self.root_dir = flags.train_root_dir
            self.labels = pd.read_csv(flags.train_label_file, index_col=0).astype('str')
        else:
            self.root_dir = flags.test_root_dir
            self.labels = pd.read_csv(flags.test_label_file, index_col=0).astype('str')
        self.img_l, self.img_w = flags.img_l, flags.img_w
        self.cut_square = flags.cut_square
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.labels.iloc[idx, 0]
        #print('Idx: ', idx)
        img_name = os.path.join(self.root_dir, name)

        label_name = os.path.join(self.root_dir, 'mask', name)
        labels = np.expand_dims(io.imread(label_name)>200, axis=0)              # 200 is the threshold for rounding
        #labels_inv = 1 - labels
        #labels_inv = np.logical_not(labels)
        #labels = np.concatenate([labels_inv, labels], axis=0)
        image = io.imread(img_name)
        #print("shape of read labels", np.shape(labels))
        #print("shape of read image", np.shape(image))
        # cut to square for simplicity now
        # print(np.shape(image))
        if self.cut_square:
            image = image[:self.img_w, :self.img_w]
            labels = labels[:, :self.img_w, :self.img_w]
        sample = {'image': image, 'labels': labels, 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels, names = sample['image'], sample['labels'], sample['name']
        # Make the image into 3 dimension
        image = np.expand_dims(image, axis=0)
        image = np.concatenate([image, image, image], axis=0).astype('float')
        labels = labels.astype('float')
        #labels = np.expand_dims(labels, axis=0)
        #labels = np.concatenate([labels, labels, labels], axis=0).astype('float')
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels),
                'name': names}

def read_data(flags):
    # Get the random seed manually set if random seed in flags is positive
    if flags.random_seed > 0:
        torch.manual_seed(flags.random_seed)
    trainSet = EyeDataset(flags, transform=ToTensor(),train=True)
    train_loader = DataLoader(trainSet, batch_size=flags.batch_size, shuffle=True,
                              num_workers=flags.num_workers)
    # get test set
    testSet = EyeDataset(flags, transform=ToTensor(),train=False)
    test_loader = DataLoader(testSet, batch_size=flags.batch_size, shuffle=True,
                              num_workers=flags.num_workers)
    return train_loader, test_loader
