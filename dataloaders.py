from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import random
from collections import defaultdict
import numpy as np
from torchvision import transforms
import torch

from constants import *

# Dataset Classes
class ISIC2018Dataset(Dataset):
    """Dataset class for ISIC 2018 images and labels."""

    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        if 'label' not in df.columns:
            df['label'] = df[df.columns[1:]].idxmax(axis=1)
        self.data = df
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['label'].map(LABEL_MAPPING_ISIC2018)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.data.iloc[idx, 0]}.jpg")
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.data.loc[idx, "label_encoded"], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


class ISIC2019Dataset(Dataset):
    """Dataset for ISIC2019."""
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        cols_list = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
        LABEL_MAPPING_ISIC2019
        if 'label' not in df.columns:
            df['label'] = df[cols_list].idxmax(axis=1)
        self.data = df
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['label'].str.lower().map(LABEL_MAPPING_ISIC2019)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.loc[idx, "label_encoded"], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


class TripletISIC2018Dataset(Dataset):
    """Triplet dataset class for ISIC 2018."""

    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        if 'label' not in df.columns:
            df['label'] = df[df.columns[1:]].idxmax(axis=1)
        self.data = df
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['label'].map(LABEL_MAPPING_ISIC2018)
        self.pos_neg_map = {
            label: {
                'positive': self.data[self.data['label_encoded'] == label],
                'negative': self.data[self.data['label_encoded'] != label]
            }
            for label in self.data['label_encoded'].unique()
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data.iloc[idx]
        label = anchor['label_encoded']
        pos_candidates = self.pos_neg_map[label]['positive']
        neg_candidates = self.pos_neg_map[label]['negative']

        positive = pos_candidates.sample(1).iloc[0]
        negative = neg_candidates.sample(1).iloc[0]

        anchor_img = self._load_image(anchor['image'])
        positive_img = self._load_image(positive['image'])
        negative_img = self._load_image(negative['image'])

        return anchor_img, positive_img, negative_img

    def _load_image(self, image_id):
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class SupConISIC2018Dataset(Dataset):
    """Triplet dataset class for ISIC 2018."""

    def __init__(self, csv_file, img_dir, transforms=None, aug_size=1):
        df = pd.read_csv(csv_file)
        if 'label' not in df.columns:
            df['label'] = df[df.columns[1:]].idxmax(axis=1)
        self.data = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.data['label_encoded'] = self.data['label'].map(LABEL_MAPPING_ISIC2018)
        self.aug_size = aug_size
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.data.iloc[idx, 0]}.jpg")
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.data.loc[idx, "label_encoded"], dtype=torch.long)
        
        img_augs = []
        for _ in range(self.aug_size):
            _transform = random.choice(self.transforms)
            img_augs.append(_transform(image))
        if self.aug_size == 1:
            return img_augs[-1], label 
        
        return tuple(img_augs), label


    def _load_image(self, image_id):
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image



class ISIC2018DatasetBalancedPairing(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, pair_factor=1):
        df = pd.read_csv(csv_file)
        df['label'] = df['label'] if 'label' in df.columns else df.iloc[:, 1:].idxmax(axis=1)
        
        # Filter out NV label
        self.data = df[df['label'] != 'NV']
        
        self.img_dir = img_dir
        self.transform = transform
        self.pair_factor = pair_factor
        
        self.data['label_encoded'] = self.data['label'].map(LABEL_MAPPING_ISIC2018)
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.data['label_encoded']):
            self.class_to_indices[label].append(idx)
        
        # total samples and class counts
        self.class_counts = {label: len(indices) for label, indices in self.class_to_indices.items()}
        self.total_samples = sum(self.class_counts.values())
        self.label_keys = [k for k in self.class_to_indices.keys() if k != 1]  # Exclude NV label

    def _image_fusion(self, img1, img2, alpha=None):
        alpha = random.uniform(0.3, 0.7) if alpha is None else alpha
        img1_np = np.array(img1, dtype=np.float32)
        img2_np = np.array(img2, dtype=np.float32)
        
        img_mixed = (1 - alpha) * img1_np + alpha * img2_np
        return Image.fromarray(np.clip(img_mixed, 0, 255).astype(np.uint8).transpose(1,2,0))
    
    def _get_image(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.loc[idx, "label_encoded"], dtype=torch.long)
        return (self.transform(image) if self.transform else image), label

    def __len__(self):
        return self.total_samples + int(self.total_samples * self.pair_factor)

    def __getitem__(self, idx):
        if idx < self.total_samples:
            return self._get_image(idx)

        # excluding NV label
        label_to_fuse = random.choice(self.label_keys)
        label_indices = self.class_to_indices[label_to_fuse]
        
        if len(label_indices) < 2:
            return self._get_image(random.randint(0, self.total_samples - 1))
        
        base_idx, pair_idx = random.sample(label_indices, 2)
        base_img, base_label = self._get_image(base_idx)
        pair_img, _ = self._get_image(pair_idx)
        
        base_img_pil = transforms.ToPILImage()(base_img.cpu()) if torch.is_tensor(base_img) else base_img
        pair_img_pil = transforms.ToPILImage()(pair_img.cpu()) if torch.is_tensor(pair_img) else pair_img
        
        fused_img_pil = self._image_fusion(base_img_pil, pair_img_pil)
        return (self.transform(fused_img_pil) if self.transform else fused_img_pil), base_label
     
