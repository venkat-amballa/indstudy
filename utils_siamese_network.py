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

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device:", device)

# Define label mapping
label_mapping = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

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


# Triplet Dataset Class
class TripletHAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.data['label_encoded'] = self.data['dx'].map(label_mapping)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data.iloc[idx]
        pos_candidates = self.data[self.data['label_encoded'] == anchor['label_encoded']]
        neg_candidates = self.data[self.data['label_encoded'] != anchor['label_encoded']]
        
        positive = pos_candidates.sample(1).iloc[0]
        negative = neg_candidates.sample(1).iloc[0]

        anchor_img = self._load_image(anchor['image_id'])
        positive_img = self._load_image(positive['image_id'])
        negative_img = self._load_image(negative['image_id'])

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


# Modified Training and Evaluation Functions for Triplet Network
def train_triplet_network(model, train_loader, val_loader, optimizer, loss_fn, 
                          RESULTS_PTH, 
                          num_epochs=10, 
                          checkpoint_path="triplet_best_model.pth"):
    
    best_loss = float('inf')
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming training from epoch {start_epoch + 1} with best loss: {best_loss:.4f}")

    results = {'epoch':[], 'loss':[], 'val_loss':[]}
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        batch = 0
        for anchor, positive, negative in train_loader:
            batch += 1
            # if batch > 3: break
            if (batch + 1) % 10 == 0:
                print("*", end="", flush=True)
            
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            
            anchor_embed, positive_embed, negative_embed = model(anchor), model(positive), model(negative)
            loss = loss_fn(anchor_embed, positive_embed, negative_embed)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_loss = evaluate_triplet_network(model, loss_fn, val_loader)

        results['epoch'] = epoch+1
        results['loss'].append(avg_loss)
        results['val_loss'].append(val_loss)
        print("\r", end="")
        if not os.path.exists(RESULTS_PTH):
            results_df = pd.DataFrame({'epoch':[], 'loss':[], 'val_loss':[]})
        else:
            results_df = pd.read_csv(RESULTS_PTH)
    
        results_df.loc[len(results_df)] = [epoch+1, avg_loss, val_loss]
            
        results_df.to_csv(RESULTS_PTH, index=False)
    
        # print(f"Results saved to {RESULTS_PTH}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_path)
            print(f"Model checkpoint saved with loss {best_loss:.4f}")

    return results

def generate_reference_embeddings(model, data_loader):
    """
    Generate and save the average embedding vector for each label (class).
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    i = 0
    with torch.no_grad():
        for images, labels in data_loader:    
            i+= 1
            if i>4:
                break
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

def predict_class(model, data_loader, reference_embeddings):
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
            
            # Get the predicted label and corresponding distance for each image
            min_distances, min_indices = distances.min(dim=1)
            for i in range(len(images)):
                predicted_label = labels[min_indices[i].item()]
                
                predictions['true_label'].append(true_labels[i].item())
                predictions['predicted_label'].append(predicted_label)
                predictions['distance'].append(min_distances[i].item())
    return predictions



def evaluate_triplet_network(model, loss_fn, data_loader):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for anchor, positive, negative in data_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_embed, positive_embed, negative_embed = model(anchor), model(positive), model(negative)
            loss = loss_fn(anchor_embed, positive_embed, negative_embed)
            running_loss += loss.item()
    
    avg_loss = running_loss / len(data_loader)
    return avg_loss

