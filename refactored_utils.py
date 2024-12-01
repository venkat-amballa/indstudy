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
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

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

# Augmentation helpers
class ImageFusion:
    def __call__(self, img1, img2):
        alpha = 0.5  # blending ratio
        return Image.blend(img1, img2, alpha)

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

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
        model.train()
        running_loss, all_preds, all_labels = 0.0, [], []

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
            # # loss.backward()
            # # optimizer.step()
            # grad_scaler.scale(loss).backward()
            # grad_scaler.step(optimizer)
            # grad_scaler.update()


            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if writer:
                writer.add_scalar(f"{loss_fn_name}/Batch_Loss", loss.item(), global_step=epoch * len(train_loader) + batch_idx)

        # Log training metrics
        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
        train_accuracy = accuracy_score(all_labels, all_preds)

        # Evaluate on validation set
        val_f1, val_accuracy = evaluate_model(model, val_loader, device=device)

        if writer:
            writer.add_scalar(f"{loss_fn_name}/Train_Loss", train_loss,  global_step=epoch)
            writer.add_scalar(f"{loss_fn_name}/Train_F1", train_f1,  global_step=epoch)
            writer.add_scalar(f"{loss_fn_name}/Train_Accuracy", train_accuracy,  global_step=epoch)
            writer.add_scalar(f"{loss_fn_name}/Validation_F1", val_f1,  global_step=epoch)
            writer.add_scalar(f"{loss_fn_name}/Validation_Accuracy", val_accuracy,  global_step=epoch)

        print(f"[{loss_fn_name}] Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}")

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


def evaluate_model(model, data_loader, device=DEVICE):
    """Evaluate the model."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Evaluated accuracy: {accuracy:.4f}")

    # with open(report_path, 'w') as f:
    #     f.write(f"Accuracy: {accuracy:.4f}\n")
    #     f.write(classification_report(all_labels, all_preds, zero_division=1))

    return f1, accuracy
