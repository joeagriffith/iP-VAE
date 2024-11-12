import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import os

from PIL import Image

def remove_to_tensor(transform):
    if type(transform) == transforms.ToTensor:
        transform = None

    if type(transform) == transforms.Compose:
        new_transforms = []
        for t in transform.transforms:
            if type(t) != transforms.ToTensor:
                new_transforms.append(t)
        transform = transforms.Compose(new_transforms)
    return transform


class PreloadedDataset(Dataset):
    def __init__(self, main_dir, shape, transform=None, device="cpu", shuffle=False, use_tqdm=True):
        self.main_dir = main_dir
        self.shape = shape
        self.transform = transform
        self.device = device
        self.classes = os.listdir(main_dir)
        self.shuffled = shuffle #  This flag is useful for cross_val_split_by_class()  
        if '.DS_Store' in self.classes:
            self.classes.remove('.DS_Store')
            
        self.images = torch.zeros(shape).to(self.device)
        self.targets = torch.zeros(0).type(torch.LongTensor).to(self.device)
        
        pre_transform = transforms.ToTensor()
        self.transform = remove_to_tensor(transform)                
        
        #  preload images
        if self.main_dir is not None:
            loop = tqdm(enumerate(self.classes), total=len(self.classes), leave=False) if use_tqdm else enumerate(self.classes)
            for class_idx, class_name in loop:
                class_dir = os.path.join(self.main_dir, class_name)
                image_names = os.listdir(class_dir)
                class_images = []
                for file_name in image_names:
                    img_loc = os.path.join(class_dir, file_name)
                    class_images.append(pre_transform(Image.open(img_loc).convert("RGB")))

                class_images = torch.stack(class_images).to(self.device)
                class_targets = (torch.ones(len(class_images)) * class_idx).type(torch.LongTensor).to(self.device)

                self.images = torch.cat([self.images, class_images])
                self.targets = torch.cat([self.targets, class_targets])
            
            #  Transformed_images stores a transformed copy of images.
            #  This enables us to keep a virgin copy of the original images
            #  which we can use at each epoch to generate different transformed images
            #  Note: we must remember to call dataset.transform() when requesting transformed images
        if self.transform is None:
            self.transformed_images = self.images
        else:
            self.transformed_images = self.transform(self.images)
            
        if shuffle:
            self._shuffle()
        
    #  Useful for loading data which is stored in a different format to TinyImageNet30
    def from_dataset(dataset, transform, device="cpu", use_tqdm=True):
        preloaded_dataset = PreloadedDataset(None, dataset.__getitem__(0)[0].shape, use_tqdm=use_tqdm)
        data = []
        targets = []
        loop = tqdm(range(len(dataset)), leave=False) if use_tqdm else range(len(dataset))
        for i in loop:
            d, t = dataset.__getitem__(i)
            if type(t) is not torch.Tensor:
                t = torch.tensor(t)
            data.append(d)
            targets.append(t)
            
        assert type(data[0]) == torch.Tensor, print(f"Data is {type(data[0])} not torch.Tensor")
        assert type(targets[0]) == torch.Tensor, print(f"Targets is {type(targets[0])} not torch.Tensor")
        transform = remove_to_tensor(transform)
        
        preloaded_dataset.shape = data[0].shape
        preloaded_dataset.device = device
        preloaded_dataset.transform = transform
        preloaded_dataset.images = torch.stack(data).to(device)
        preloaded_dataset.targets = torch.stack(targets).to(device)
        if transform is not None:
            preloaded_dataset.transformed_images = transform(torch.stack(data).to(device))
        else:
            preloaded_dataset.transformed_images = torch.stack(data).to(device)
        
        return preloaded_dataset
              
            
    #  Transforms the data in batches so as not to overload memory
    def apply_transform(self, device=None, batch_size=500):
        if self.transform is not None:
            if device is None:
                device = self.device
            
            low = 0
            high = batch_size
            while low < len(self.images):
                if high > len(self.images):
                    high = len(self.images)
                self.transformed_images[low:high] = self.transform(self.images[low:high].to(device)).to(self.device)
                low += batch_size
                high += batch_size
        
        
    #  Now a man who needs no introduction
    def __len__(self):
        return len(self.images)
    
    
    #  Returns images which have already been transformed - unless self.transform is none
    #  This saves us from transforming individual images, which is very slow.
    def __getitem__(self, idx):
        return self.transformed_images[idx], self.targets[idx]        
    
    
    def _shuffle(self):
        indices = torch.randperm(self.images.shape[0])
        self.images = self.images[indices]
        self.targets = self.targets[indices]
        self.transformed_images = self.transformed_images[indices]
        if not self.shuffled:
            self.shuffled = True  
    

    #  Returns a training and validation dataset, split with an equal number of each class.
    #  a val_idx from 0 -> 1/val_ratio determines which segment of each class is used for validation.
    def cross_val_split_by_class(self, val_ratio, val_idx, val_transform=None, device="cpu"):
        assert not self.shuffled, "Dataset must not be shuffled to split by class"
        max_idx = int(1/val_ratio) - 1
        assert val_idx >= 0 and val_idx <= max_idx, f"Invalid val_idx: {val_idx} for ratio: {val_ratio}"
        
        train_dataset = PreloadedDataset(None, self.shape, None, device)
        val_dataset = PreloadedDataset(None, self.shape, None, device)

        trans_shape = self.transformed_images.shape
        train_dataset.transformed_images = torch.zeros(0, trans_shape[1], trans_shape[2], trans_shape[3]).to(device)
        val_dataset.transformed_images = torch.zeros(0, trans_shape[1], trans_shape[2], trans_shape[3]).to(device)
        
        train_dataset.transform = self.transform
        val_dataset.transform = val_transform
        val_transform = remove_to_tensor(val_transform)
        
        num_per_class = len(self.images)//len(self.classes)
        split_size = int(num_per_class*val_ratio)
        indices = torch.arange(num_per_class).split(split_size)
        assert len(indices[-1]) == split_size, "Unable to split by class with current parameters"
        
        #  Iterate over data in batches of class_size
        #  Split into segments and allocate the segments to either train/val datasets
        for class_idx in range(len(self.classes)):
            class_indices = []
            for i in range(len(indices)):
                class_indices.append(indices[i] + num_per_class*class_idx)
            assert len(class_indices) == len(indices)
            val_indices = class_indices[val_idx]
            if val_idx == 0:
                train_indices = torch.cat(class_indices[val_idx+1:])
            elif val_idx == max_idx:
                train_indices = torch.cat(class_indices[:val_idx])
            else:
                train_indices = torch.cat([torch.cat(class_indices[:val_idx]), torch.cat(class_indices[val_idx+1:])])

            train_dataset.images = torch.cat([
                train_dataset.images, 
                self.images[train_indices].to(device)
            ])
            train_dataset.targets = torch.cat([
                train_dataset.targets, 
                self.targets[train_indices].to(device)
            ])
            train_dataset.transformed_images = torch.cat([
                train_dataset.transformed_images, 
                self.transformed_images[train_indices].to(device)
            ])

            val_dataset.images = torch.cat([
                val_dataset.images,
                self.images[val_indices].to(device)
            ])
            val_dataset.targets = torch.cat([
                val_dataset.targets,
                self.targets[val_indices].to(device)
            ])
            val_dataset.transformed_images = torch.cat([
                val_dataset.transformed_images,
                val_transform(self.images[val_indices]).to(device) if val_transform is not None else self.images[val_indices].to(device)# Untransformed, as we want to test on un-augmented data
            ])
        
        return train_dataset, val_dataset