import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
import datetime
import time
import sys
import argparse

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.amp import GradScaler, autocast

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

scaler = GradScaler()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device:",device)

# Define label mapping
label_mapping_ham = {
    
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

label_mapping_isic2018 = {
    'MEL': 0,
    'NV': 1,
    'BCC': 2,
    'AKIEC': 3,
    'BKL': 4,
    'DF': 5,
    'VASC': 6
}

label_mapping = {
    'mel': 0,
    'nv': 1,
    'bcc': 2,
    'ak': 3,
    'bkl': 4,
    'df': 5,
    'vasc': 6,
    'scc':7,
    'unk':8
}

# Define the Dataset class
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['dx'].map(label_mapping)

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
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        if 'label' not in df.columns:
            df['label'] = df[df.columns[1:]].idxmax(axis=1)

        self.data = df
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['label'].map(label_mapping_isic2018)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.loc[idx, "label_encoded"], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the Dataset class
class ISIC2019Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        cols_list = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
        if 'label' not in df.columns:
            df['label'] = df[cols_list].idxmax(axis=1)
        self.data = df
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['label'].str.lower().map(label_mapping)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.loc[idx, "label_encoded"], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


################################################### LOSS FUNCTIONS ##################################################
# Define loss functions
class FocalLoss(nn.Module):
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

# Use this instead of CrossEntropyLoss or FocalLoss

##################################################### LOSS FUNCTIONS END ############################################


# Initialize the SummaryWriter
# Define the train and evaluate function with TensorBoard logging
def train_and_evaluate(loss_functions, loss_fn_name, model, train_loader, val_loader, optimizer, 
                       RESULTS_PTH, REPORT_PTH, 
                       num_epochs=10,
                       checkpoint_path="best_model.pth",
                       writer = None):
    start_epoch = 0
    best_val_accuracy = 0.0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        print(f"Resuming training from epoch {start_epoch + 1} with best validation accuracy: {best_val_accuracy:.4f}")

    criterion = loss_functions[loss_fn_name]
    results = {'epoch': [], 'train_loss': [], 'train_f1': [], 'train_accuracy': [], 'val_f1': [], 'val_accuracy': []}

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        batch = 0
        # Training loop
        for images, labels in train_loader:
            batch += 1
            if batch > 2: break
            if (batch + 1) % 50 == 0:
                print("*", end="", flush=True)

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
        train_accuracy = accuracy_score(all_labels, all_preds)

        if writer:
            # Log training metrics to TensorBoard
            writer.add_scalar(f'{loss_fn_name}/Train_Loss', train_loss, epoch)
            writer.add_scalar(f'{loss_fn_name}/Train_F1', train_f1, epoch)
            writer.add_scalar(f'{loss_fn_name}/Train_Accuracy', train_accuracy, epoch)

        # Evaluate on validation data
        val_f1, val_accuracy = evaluate_model(model, val_loader, REPORT_PTH)
        if writer:
            # Log validation metrics to TensorBoard
            writer.add_scalar(f'{loss_fn_name}/Val_F1', val_f1, epoch)
            writer.add_scalar(f'{loss_fn_name}/Val_Accuracy', val_accuracy, epoch)

        # Log metrics to results
        results['epoch'].append(epoch + 1)
        results['train_loss'].append(train_loss)
        results['train_f1'].append(train_f1)
        results['train_accuracy'].append(train_accuracy)
        results['val_f1'].append(val_f1)
        results['val_accuracy'].append(val_accuracy)
        print("\r", end="")
        print(f"[{loss_fn_name}] Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.7f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
              f"Val F1: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if not os.path.exists(RESULTS_PTH):
            results_df = pd.DataFrame({'epoch': [], 'train_loss': [], 'train_f1': [], 'train_accuracy': [], 'val_f1': [], 'val_accuracy': []})
        else:
            results_df = pd.read_csv(RESULTS_PTH)
    
        results_df.loc[len(results_df)] = [epoch+1, train_loss, train_f1, train_accuracy, val_f1, val_accuracy]
            
        results_df.to_csv(RESULTS_PTH, index=False)
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy
            }, checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
    if writer:
        writer.flush()
        writer.close()
    return results_df



# Evaluation function for validation/test F1 score and Accuracy
def evaluate_model(model, data_loader, REPORT_PTH):
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
    print("evaluted accuracy:", accuracy)
    with open(REPORT_PTH, 'w') as fp:
        fp.write(f"Accuracy: {accuracy:.4f}\n")    
        fp.write(classification_report(all_labels, all_preds, zero_division=1))

    return f1, accuracy
