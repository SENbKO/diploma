import torch
import numpy as np
from torch.utils.data import Dataset
import cv2


class PascalVOC_Dataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform = None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    #Load binary masks from dataset
    def load_binary_mask(self, mask_paths):
        mask =  cv2.imread(mask_paths, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 0).astype('float32')
        return binary_mask

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.load_binary_mask(self.mask_paths[index])

        #Perform augmentation if transmitted in the constructor
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.image_paths)
