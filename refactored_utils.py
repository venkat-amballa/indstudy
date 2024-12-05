import os
import sys
import time
import datetime
import warnings
from PIL import Image

import pandas as pd
import torch
import torch.nn as nn

import timm
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import random
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

from losses import *
from constants import *
# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

SCALER = GradScaler()

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