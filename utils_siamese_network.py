import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from torchvision import transforms
from PIL import Image
import os
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.model_selection import train_test_split
import datetime
import time
from tqdm import tqdm
import argparse
from torch.amp import GradScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore", category=UserWarning)

# Define label mapping
label_mapping_isic2018 = {
    'MEL': 0,
    'NV': 1,
    'BCC': 2,
    'AKIEC': 3,
    'BKL': 4,
    'DF': 5,
    'VASC': 6
}
DEVICE = 'cuda'

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
    
# Triplet Dataset Class
class TripletISIC2018Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        if 'label' not in df.columns:
            df['label'] = df[df.columns[1:]].idxmax(axis=1)

        self.data = df
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['label'].map(label_mapping_isic2018)
        pos_neg_map = {}
        for label in self.data['label_encoded'].unique():
            pos_neg_map[label] = {
                'positive': self.data[self.data['label_encoded'] == label],
                'negative': self.data[self.data['label_encoded'] != label]
            }
        self.pos_neg_map  = pos_neg_map
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
        img_name = os.path.join(self.img_dir, image_id + '.jpg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)
        
        loss = torch.mean((pos_dist)**2 + torch.relu(self.margin - neg_dist)**2)
        return loss

class DualMSLoss(nn.Module):
    def __init__(self, margin_positive=0.5, margin_negative=1.5):
        super(DualMSLoss, self).__init__()
        self.margin_positive = margin_positive
        self.margin_negative = margin_negative

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)  # Positive pair distance
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)  # Negative pair distance

        positive_loss = torch.clamp(pos_dist - self.margin_positive, min=0) ** 2

        negative_loss = torch.clamp(self.margin_negative - neg_dist, min=0) ** 2

        loss = torch.mean(positive_loss + negative_loss)

        return loss


# Modified Training and Evaluation Functions for Triplet Network
def train_triplet_network(model, train_loader, val_loader, optimizer, criteria, 
                          RESULTS_PTH, 
                          num_epochs=10, 
                          checkpoint_path="triplet_best_model.pth",
                          device=DEVICE,
                          writer = None,
                          train_loader_normal = None):
    
    best_acc = 0
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resuming training from epoch {start_epoch + 1} with best loss: {best_acc:.4f}")

    # results = {'epoch':[], 'loss':[], 'val_acc':[]}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode="max", 
                                                           factor=0.1, 
                                                           patience=3, 
                                                           verbose=True)
    
    

    grad_scaler = GradScaler()
    
    AMP_TRAIN = True
    accumulation_steps = 4
    
    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        batch = 0
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
            # if batch_idx>10:
            #     break
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_embed, positive_embed, negative_embed = model(anchor), model(positive), model(negative)

            if AMP_TRAIN:
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    anchor_embed, positive_embed, negative_embed = model(anchor), model(positive), model(negative)
                    loss = criteria(anchor_embed, positive_embed, negative_embed)
                    loss = loss / accumulation_steps
                grad_scaler.scale(loss).backward()                
                # grad_scaler.step(optimizer)
                # grad_scaler.update()
            else:
                anchor_embed, positive_embed, negative_embed = model(anchor), model(positive), model(negative)
                loss = criteria(anchor_embed, positive_embed, negative_embed)
                loss = loss / accumulation_steps  # Scale loss
                loss.backward()
                # optimizer.step()

            # Accumulate gradients and step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx+1 == len(train_loader):
                if AMP_TRAIN:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()  # Reset gradients

            running_loss += loss.item()* accumulation_steps

        avg_loss = running_loss / len(train_loader)
        # val_acc = evaluate_triplet_network(model, criteria, val_loader, device=device)
        val_f1, val_acc = evaluate_triplet_model(model, train_loader_normal, val_loader, device=device)
        
        # Log metrics to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('F1_Score/val', val_f1, epoch)
        # results['epoch'] = epoch+1
        # results['loss'].append(avg_loss)
        # results['val_acc'].append(val_acc)
        # results['val_f1'].append(val_f1)

        # if not os.path.exists(RESULTS_PTH):
        #     results_df = pd.DataFrame({'epoch':[], 'loss':[], 'val_acc':[]})
        # else:
        #     results_df = pd.read_csv(RESULTS_PTH)
    
        # results_df.loc[len(results_df)] = [epoch+1, avg_loss, val_acc]
            
        # results_df.to_csv(RESULTS_PTH, index=False)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if val_acc < best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, checkpoint_path)
            print(f"Model checkpoint saved with loss {best_acc:.4f}")

    # return results

def generate_reference_embeddings(model, data_loader, device=DEVICE):
    """
    Generate and save the average embedding vector for each label (class).
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader):    
            images, labels = images.to(device), labels.to(device)

            batch_embeddings = model(images)  # (batch_size, embedding_size)

            all_embeddings.append(batch_embeddings)
            all_labels.append(labels.squeeze())  # remove last dim (batch_size, 1)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (total_samples, embedding_size)
        all_labels = torch.cat(all_labels, dim=0)  # (total_samples,)

    # Calculate the average embedding for each unique label
    label_embeddings = {}
    for label in all_labels.unique():
        label_indices = (all_labels == label).nonzero(as_tuple=True)[0]        
        average_embedding = all_embeddings[label_indices].mean(dim=0).cpu()
        label_embeddings[label.item()] = average_embedding

    return label_embeddings

def predict_class(model, data_loader, reference_embeddings, device=DEVICE):
    """
    function to predict the class of each image in data_loader
    by finding the nearest reference embedding.
    """
    model.eval()
    predictions = {}
    predictions['true_label'] = []
    predictions['predicted_label'] = []
    predictions['distance'] = []
    
    labels = list(reference_embeddings.keys())
    ref_embeddings = torch.stack([reference_embeddings[label].to(device) for label in labels])  # (num_classes, embedding_size)

    with torch.no_grad():
        for images, true_labels in data_loader:
            images, true_labels = images.to(device), true_labels.to(device)
            
            batch_embeddings = model(images)  # (batch_size, embedding_size)
            
            # compute pairwise distances in batch
            distances = torch.cdist(batch_embeddings, ref_embeddings)
            
            # min distance and corresponding label for each image
            min_distances, min_indices = distances.min(dim=1)
            for i in range(len(images)):
                predicted_label = labels[min_indices[i].item()]
                
                predictions['true_label'].append(true_labels[i].item())
                predictions['predicted_label'].append(predicted_label)
                predictions['distance'].append(min_distances[i].item())
    return predictions



def evaluate_triplet_network(model, criteria, data_loader, device=DEVICE):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for anchor, positive, negative in data_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_embed, positive_embed, negative_embed = model(anchor), model(positive), model(negative)
            loss = criteria(anchor_embed, positive_embed, negative_embed)
            running_loss += loss.item()
    
    avg_loss = running_loss / len(data_loader)
    return avg_loss



def evaluate_triplet_model(model, train_loader, data_loader, device=DEVICE, report=False):
    """Evaluate the model."""
    model.eval()
    all_preds, all_labels = [], []
    reference_embeddings = generate_reference_embeddings(model, train_loader, device=device)
    predictions_dict = predict_class(model, data_loader, reference_embeddings, device=device)

    # Compute and print metrics
    y_true, y_pred = predictions_dict['true_label'], predictions_dict['predicted_label']
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\nEvaluated accuracy: {accuracy:.4f}")
    print(f"Evaluated f1_score: {f1:.4f}")
    if report:
        print(classification_report(y_true, y_pred))
    return f1, accuracy
