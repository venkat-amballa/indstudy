import os
import sys
import time
import datetime
import warnings
from PIL import Image

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import random
import numpy as np
from collections import defaultdict
import cv2
from sklearn.metrics import roc_auc_score

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCALER = GradScaler()

# Label mappings
LABEL_MAPPING_HAM = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
}
LABEL_MAPPING_ISIC2018 = {
    'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6
}
LABEL_MAPPING_ISIC2019 = {
    'mel': 0, 'nv': 1, 'bcc': 2, 'ak': 3, 'bkl': 4, 'df': 5, 'vasc': 6, 'scc': 7, 'unk': 8
}

def add_gaussian_noise(img, mean=0, std=0.1):
    img_np = np.array(img)
    noise = np.random.normal(mean, std, img_np.shape).astype(np.uint8)
    img_np = np.clip(img_np + noise, 0, 255)  # clip
    return Image.fromarray(img_np)

def convert_to_hsv(img):
    img = np.array(img)  # Convert PIL to numpy
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return Image.fromarray(img_hsv)


def gamma_correction(img, gamma=None):
    if gamma is None:
        gamma=random.uniform(0.5, 2.5)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_corrected = np.power(img_np, gamma)
    img_corrected = np.clip(img_corrected * 255, 0, 255).astype(np.uint8) # clip
    return Image.fromarray(img_corrected)

# Dataset Classes
class HAM10000Dataset(Dataset):
    """Dataset for HAM10000."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['dx'].map(LABEL_MAPPING_HAM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 1] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


class ISIC2018Dataset(Dataset):
    """Dataset for ISIC2018."""
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
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.loc[idx, "label_encoded"], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image

import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms

LABEL_MAPPING_ISIC2018 = {
    'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6
}

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
     

class ISIC2019Dataset(Dataset):
    """Dataset for ISIC2019."""
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        cols_list = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
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


# Loss Functions
class FocalLoss(nn.Module):
    """Implementation of Focal Loss."""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Implementation of Label Smoothing Loss."""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(inputs, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), confidence)
        targets += self.smoothing / (inputs.size(-1) - 1)
        loss = (-targets * log_probs).sum(dim=-1).mean()
        return loss


# Training and Evaluation
def train_and_evaluate(loss_functions, loss_fn_name, model, train_loader, val_loader, optimizer, 
                       num_epochs=10, checkpoint_path="best_model.pth",
                       writer=None, device=DEVICE, skip_batches=None):
    """Train and evaluate the model."""
    start_epoch, best_val_accuracy = 0, 0.0

    # For mixed precision training to speed up the model 
    grad_scaler = torch.amp.GradScaler()

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        print(f"Resuming training from epoch {start_epoch} with best validation accuracy: {best_val_accuracy:.4f}")

    criterion = loss_functions[loss_fn_name]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode="min", 
                                                           factor=0.6, 
                                                           patience=4)
    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs), desc="Epochs"):
        model.train()
        running_loss, all_preds, all_labels = 0.0, [], []
        all_probs = []

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Batches")):
            if skip_batches and batch_idx >= skip_batches:
                print(f"Skipping remaining batches after {skip_batches}")
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # with torch.amp.autocast(device_type=device, dtype=torch.float16):
            #     outputs = model(images)
            #     loss = criterion(outputs, labels)
            # grad_scaler.scale(loss).backward()
            # grad_scaler.step(optimizer)
            # grad_scaler.update()


            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            prob = torch.softmax(outputs, dim=1)
            all_probs.extend(prob.detach().cpu().numpy())
        # Log training metrics
        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
        train_accuracy = accuracy_score(all_labels, all_preds)

        # Evaluate on validation set
        val_loss, val_f1, val_accuracy, val_roc_auc = evaluate_model(model, val_loader, criterion, device=device)
        # scheduler.step(val_accuracy)
        scheduler.step(val_loss)
        # OVR; macro - treats all classes equally;
        train_roc_auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr', average='macro')
    
        if writer:
            writer.add_scalar(f"Train_Loss", train_loss,  global_step=epoch)
            writer.add_scalar(f"Train_Accuracy", train_accuracy,  global_step=epoch)
            writer.add_scalar(f"Train_F1", train_f1,  global_step=epoch)
            writer.add_scalar(f"Train_roc_auc", train_roc_auc,  global_step=epoch)
            writer.add_scalar(f"Validation_Loss", val_loss,  global_step=epoch)
            writer.add_scalar(f"Validation_Accuracy", val_accuracy,  global_step=epoch)
            writer.add_scalar(f"Validation_F1", val_f1,  global_step=epoch)
            writer.add_scalar(f"Validation_ROC_AUC", val_roc_auc, global_step=epoch)
            current_lr = scheduler.get_last_lr()[0]  # Get the learning rate for the first parameter group
            writer.add_scalar(f"Learning_Rate", current_lr, global_step=epoch)


        print(f"Epoch {epoch + 1}/{num_epochs}\n"
              f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2f}, Train F1: {train_f1:.2f}, Train_roc_auc: {train_roc_auc:.2f}\n"
              f"Valid Loss: {val_loss:.2f},   Valid Accuracy: {val_accuracy:.2f},   Valid F1: {val_f1:.2f},   Valid_roc_auc: {val_roc_auc:.2f}\n")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy
            }, checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")


def evaluate_model(model, data_loader, criterion, device=DEVICE):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())   
            all_labels.extend(labels.cpu().numpy())
            # probabilites for roc_auc
            prob = torch.softmax(outputs, dim=1)
            all_probs.extend(prob.detach().cpu().numpy())

    val_loss /= len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr', average='macro')
    return val_loss, f1, accuracy, roc_auc