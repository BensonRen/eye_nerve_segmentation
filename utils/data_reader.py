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
        self.labels = pd.read_csv(flags.label_file)
        self.img_l, self.img_w = flags.img_l, flags.img_w
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = io.imread(img_name)
        labels = self.labels.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, self.img_w, self.img_l)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]

# use the same transformations for train/val in this example
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

train_set = SimDataset(2000, transform = trans)
val_set = SimDataset(200, transform = trans)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 25

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}