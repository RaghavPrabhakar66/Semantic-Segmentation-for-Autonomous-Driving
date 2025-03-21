import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
class Comma10k(Dataset):
    """Comma 10k Semantic Segmentation dataset."""

    def __init__(self, df, root_dir, width=None, height=None, transform=None):
        """
        Args:
            df (dataframe): Dataframe containing list of images.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data        = df
        self.root_dir    = root_dir
        self.transform   = transform
        self.new_width   = width
        self.new_height  = height

        self.imgs_dir    = os.path.join(root_dir, 'images')
        self.masks_dir   = os.path.join(root_dir, 'masks')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name  = os.path.join(self.imgs_dir, self.data[idx])
        mask_name = os.path.join(self.masks_dir, self.data[idx])
        
        image = Image.open(img_name)
        mask  = Image.open(mask_name)

        #image = transform.resize(image, (self.new_height, self.new_width))
        #mask = transform.resize(mask, (self.new_height, self.new_width))

        #@mask = np.stack([(mask == v) for v in [41,  76,  90, 124, 161, 0]], axis=-1).astype('uint8')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image':image, 'gt_mask':mask}

