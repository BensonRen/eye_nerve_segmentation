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
    def __init__(self, flags, transform=None):
        """
        The initialization function of the EyeDataset
        :param flags: The parameter flags
        :param transform: The transforms to apply
        """
        self.root_dir = flags.root_dir
        self.labels = pd.read_csv(flags.label_file, header=None, index_col=0).astype('str')
        self.img_l, self.img_w = flags.img_l, flags.img_w
        self.cut_square = flags.cut_square
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print('Idx: ', idx)
        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])

        label_name = os.path.join(self.root_dir, 'mask', self.labels.iloc[idx, 0])
        labels = np.expand_dims(io.imread(label_name)>200, axis=0)              # 200 is the threshold for rounding
        labels_inv = 1 - labels
        #labels_inv = np.logical_not(labels)
        labels = np.concatenate([labels, labels_inv], axis=0)
        image = io.imread(img_name)
        #print("shape of read labels", np.shape(labels))
        #print("shape of read image", np.shape(image))
        # cut to square for simplicity now
        # print(np.shape(image))
        if self.cut_square:
            image = image[ :self.img_w, :self.img_w]
            labels = labels[:, :self.img_w, :self.img_w]
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # Make the image into 3 dimension
        image = np.expand_dims(image, axis=0)
        image = np.concatenate([image, image, image], axis=0).astype('float')
        labels = labels.astype('float')
        #labels = np.expand_dims(labels, axis=0)
        #labels = np.concatenate([labels, labels, labels], axis=0).astype('float')
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

def read_data(flags):
    trainSet = EyeDataset(flags, transform=ToTensor())
    train_loader = DataLoader(trainSet, batch_size=flags.batch_size, shuffle=True)
    test_loader = train_loader
    return train_loader, test_loader
