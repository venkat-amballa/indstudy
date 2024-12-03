import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from torch.amp import GradScaler

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label Mapping
LABEL_MAPPING_ISIC2018 = {
    'MEL': 0,
    'NV': 1,
    'BCC': 2,
    'AKIEC': 3,
    'BKL': 4,
    'DF': 5,
    'VASC': 6
}

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


# Loss Functions
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # p=2, euclidean dist
        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, p=2) 
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        
        # calc triplet loss
        loss = torch.nn.functional.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2, p=2)
        loss = torch.mean((1 - label) * 0.5 * torch.pow(euclidean_distance, 2) +
                          (label) * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


class DualMSLoss(nn.Module):
    """Dual Margin Separation Loss."""

    def __init__(self, margin_positive=0.5, margin_negative=1.5):
        super(DualMSLoss, self).__init__()
        self.margin_positive = margin_positive
        self.margin_negative = margin_negative

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)

        positive_loss = torch.clamp(pos_dist - self.margin_positive, min=0) ** 2
        negative_loss = torch.clamp(self.margin_negative - neg_dist, min=0) ** 2

        return torch.mean(positive_loss + negative_loss)


# Training and Evaluation
def train_triplet_network(
    model, train_loader, val_loader, optimizer, 
    loss_fn, 
    results_path,
    num_epochs, 
    checkpoint_path, device, writer=None, train_loader_normal=None
):
    """Train the triplet network."""
    best_acc = 0
    grad_scaler = GradScaler()
    AMP_TRAIN = True
    accumulation_steps = 4

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                anchor_embed = model(anchor)
                positive_embed = model(positive)
                negative_embed = model(negative)
                loss = loss_fn(anchor_embed, positive_embed, negative_embed) / accumulation_steps

            grad_scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        avg_loss = running_loss / len(train_loader)
        val_f1, val_acc = evaluate_triplet_model(model, train_loader_normal, val_loader, device=device)

        # Log metrics to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('F1_Score/    val', val_f1, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, checkpoint_path)
            print(f"Model checkpoint saved with accuracy: {best_acc:.4f}")


def evaluate_triplet_model(model, train_loader, val_loader, device=DEVICE, report=False):
    """Evaluate the triplet model using reference embeddings."""
    model.eval()
    reference_embeddings = generate_reference_embeddings(model, train_loader, device=device)
    predictions = predict_class(model, val_loader, reference_embeddings, device=device)

    y_true = predictions['true_label']
    y_pred = predictions['predicted_label']

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    if report:
        print(classification_report(y_true, y_pred))

    return f1, accuracy


def generate_reference_embeddings(model, data_loader, device=DEVICE):
    """Generate reference embeddings for each class."""
    model.eval()
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    label_embeddings = {
        label.item(): all_embeddings[all_labels == label].mean(dim=0).cpu()
        for label in all_labels.unique()
    }

    return label_embeddings


def predict_class(model, data_loader, reference_embeddings, device=DEVICE):
    """Predict classes by comparing to reference embeddings."""
    model.eval()
    predictions = {'true_label': [], 'predicted_label': [], 'distance': []}
    ref_embeddings = torch.stack([emb.to(device) for emb in reference_embeddings.values()])
    labels = list(reference_embeddings.keys())

    with torch.no_grad():
        for images, true_labels in data_loader:
            images, true_labels = images.to(device), true_labels.to(device)
            embeddings = model(images)

            distances = torch.cdist(embeddings, ref_embeddings)
            min_distances, min_indices = distances.min(dim=1)

            for i in range(len(images)):
                predictions['true_label'].append(true_labels[i].item())
                predictions['predicted_label'].append(labels[min_indices[i].item()])
                predictions['distance'].append(min_distances[i].item())

    return predictions
