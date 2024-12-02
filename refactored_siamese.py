import os
import time
import datetime
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report
from refactored_utils_siamese import *
import timm
from torch.utils.tensorboard import SummaryWriter
import argparse

# Constants and Hyperparameters
DATASET_DIR = 'datasets'
BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-5
DEFAULT_MODEL_TYPE = 'efficientnet_b0'
CHANGES = '-dualmsloss'
LOSS_FN_NAME = f'-mean-std-interpol-{CHANGES}'

LOSS_FN = TripletLoss(margin=2)
# LOSS_FN = DualMSLoss()

# Experiment Directory Setup
def setup_experiment_dir(model_type, changes):
    experiment_name = f'triplet_{model_type}{changes}'
    experiment_dir = os.path.join(experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir, experiment_name

# Dataset Paths
def get_dataset_paths(dataset_dir):
    return {
        "train_images": os.path.join(dataset_dir, 'HAM10000_images'),
        "train_labels": os.path.join(dataset_dir, 'ISIC2018_Task3_Training_GroundTruth.csv'),
        "val_images": os.path.join(dataset_dir, 'ISIC2018_Task3_Validation_Input'),
        "val_labels": os.path.join(dataset_dir, 'ISIC2018_Task3_Validation_GroundTruth.csv'),
        "test_images": os.path.join(dataset_dir, 'ISIC2018_Task3_Test_Input'),
        "test_labels": os.path.join(dataset_dir, 'ISIC2018_Task3_Test_GroundTruth.csv')
    }

# Transformations
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    return train_transforms, test_transform

# Initialize TensorBoard Writer
def initialize_writer(experiment_name, uid):
    log_dir = f"runs/{experiment_name}_{uid}"
    return SummaryWriter(log_dir=log_dir)

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")
    parser.add_argument("-d", "--device", type=str, help="Device to use (cpu/cuda)")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("-uid", "--uid", type=str, help="UID for resuming the training")
    return parser.parse_args()

# Main Function
def main():
    args = parse_arguments()

    # Device Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs or DEFAULT_EPOCHS
    lr = args.learning_rate or DEFAULT_LR
    uid = args.uid or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print("Using Device:", device)

    # Experiment Setup
    experiment_dir, experiment_name = setup_experiment_dir(DEFAULT_MODEL_TYPE, CHANGES)
    writer = initialize_writer(experiment_name, uid)

    paths = get_dataset_paths(DATASET_DIR)
    train_transforms, test_transform = get_transforms()

    # Datasets and DataLoaders
    train_dataset = TripletISIC2018Dataset(csv_file=paths['train_labels'], 
                                           img_dir=paths['train_images'], 
                                           transform=train_transforms)
    train_dataset_normal = ISIC2018Dataset(csv_file=paths['train_labels'], 
                                           img_dir=paths['train_images'], 
                                           transform=train_transforms)
    val_dataset = ISIC2018Dataset(csv_file=paths['val_labels'], 
                                   img_dir=paths['val_images'], 
                                   transform=train_transforms)
    test_dataset = ISIC2018Dataset(csv_file=paths['test_labels'], 
                                    img_dir=paths['test_images'], 
                                    transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    train_loader_normal = DataLoader(train_dataset_normal, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Paths for saving results
    checkpoint_path = os.path.join(experiment_dir, f"triplet_best_model_{uid}.pth")
    results_path = os.path.join(experiment_dir, f"results_{DEFAULT_MODEL_TYPE}_{LOSS_FN_NAME}_{uid}.csv")
    predictions_path = os.path.join(experiment_dir, f"predictions_{DEFAULT_MODEL_TYPE}_{uid}.csv")

    # Model Initialization
    model = timm.create_model(DEFAULT_MODEL_TYPE, pretrained=True, num_classes=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # Training and Evaluation
    start_time = time.time()
    train_results = train_triplet_network(model, train_loader, val_loader, optimizer, 
                                          LOSS_FN, 
                                          results_path,
                                          epochs, 
                                          checkpoint_path,
                                          device,
                                          writer=writer,
                                          train_loader_normal=train_loader_normal)

    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    # Test Evaluation
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    test_f1, test_acc = evaluate_triplet_model(model, train_loader, test_loader, device=device, report=True)
    print(f"Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")

    writer.add_text("Experiment Results", f"Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
