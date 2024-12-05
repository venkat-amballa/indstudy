import warnings
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torchvision import transforms
import os
from tqdm import tqdm
from torch.amp import GradScaler

# from losses import *
from constants import *
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Training and Evaluation
def train_triplet_network(
    model, train_loader, val_loader, optimizer, 
    criteria, 
    results_path,
    num_epochs, 
    checkpoint_path, device, writer=None, train_loader_normal=None
):
    """Train the triplet network."""
    best_acc = 0
    best_loss = 1e8
    grad_scaler = GradScaler()
    AMP_TRAIN = True
    accumulation_steps = 4
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        print(f"Resuming training from epoch {start_epoch} with best validation accuracy: {best_val_accuracy:.4f}")

    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            # if batch_idx > 4:
            #     break
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            embeddings = model(images)
            loss = criteria(embeddings, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        # val_f1, val_acc = evaluate_triplet_model(model, train_loader_normal, val_loader, device=device)
        val_f1, val_acc = None, None
        # Log metrics to TensorBoard
        if writer:
            writer.add_scalar('Train_Loss', avg_loss, epoch)
            # writer.add_scalar('Val_Accuracy', val_acc, epoch)
            # writer.add_scalar('Val_F1_Score', val_f1, epoch)
        # print(f"Epoch {epoch + 1}/{num_epochs}\n"
        # f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2f}, Train F1: {train_f1:.2f}, Train_roc_auc: {train_roc_auc:.2f}\n"
        # f"Valid Loss: {val_loss:.2f},   Valid Accuracy: {val_accuracy:.2f},   Valid F1: {val_f1:.2f},   Valid_roc_auc: {val_roc_auc:.2f}\n")
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss}")

        if checkpoint_path and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_path)
            print(f"Model checkpoint saved with accuracy: {best_loss:.4f}")
    
    val_f1, val_acc = evaluate_triplet_model(model, train_loader, val_loader, device=device, report=True)
    return val_f1, val_acc

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
