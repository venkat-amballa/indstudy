import json
import os
import time
import datetime
import pandas as pd
import torch
import random
import numpy as np
import argparse
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report
import timm
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import optuna

from constants import *
from dataloaders import SupConISIC2018Dataset, ISIC2018Dataset
from refactored_utils_siamese import *  # Assuming this has necessary utility functions

# Constants and Hyperparameters
DEFAULT_MODEL_TYPE = 'efficientnet_b0'
DATASET_DIR = 'datasets'

# Experiment Directory Setup
def setup_experiment_dir(model_type, changes=""):
    experiment_name = f'triplet_{model_type}{changes}'
    experiment_dir = os.path.join(experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir, experiment_name

# Set Seed for Reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

# Define Transforms for Data Augmentation
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
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-lr", "--learning_rate", default=5e-4, type=float, help="Learning rate")
    parser.add_argument("-batch", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-workers", "--workers", type=int, default=3, help="Number of data loading workers")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use (cpu/cuda)")
    parser.add_argument("-uid", "--uid", type=str, help="UID for resuming the training")
    parser.add_argument("-writer", "--enable_writer", action="store_true", help="Enable tensorboard writer")
    parser.add_argument("-skip_after", "--skip_batches_after", type=int, help="Run the training loop only for given number of batches; Testing only!")
    parser.add_argument("-imb_fusion", "--handle_imbalance_by_fusion", action="store_true", help="Handle imbalance using image fusion")
    parser.add_argument("-info", "--info", type=str, help="Information for the experiment directory")
    return parser.parse_args()

# Dataset and DataLoader Initialization
def create_dataloaders(dataset_paths, train_transforms, test_transform, batch_size, workers):
    train_dataset = SupConISIC2018Dataset(csv_file=dataset_paths['train_labels'], 
                                           img_dir=dataset_paths['train_images'], 
                                           transforms=[train_transforms])
    val_dataset = ISIC2018Dataset(csv_file=dataset_paths['val_labels'], 
                                   img_dir=dataset_paths['val_images'], 
                                   transform=train_transforms)
    test_dataset = ISIC2018Dataset(csv_file=dataset_paths['test_labels'], 
                                    img_dir=dataset_paths['test_images'], 
                                    transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# Main Function
def train_and_evaluate(args):
    print("Using Device:", device)

    # Setup Experiment
    experiment_dir, experiment_name = setup_experiment_dir(DEFAULT_MODEL_TYPE, info)
    writer = initialize_writer(experiment_name, uid) if enable_writer else None

    # Get Dataset Paths and Transforms
    dataset_paths = get_dataset_paths(DATASET_DIR)
    train_transforms, test_transform = get_transforms()

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset_paths, train_transforms, test_transform, batch_size, workers)

    # Model Initialization
    model = timm.create_model(DEFAULT_MODEL_TYPE, pretrained=True, num_classes=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    criteria = losses.SupConLoss(temperature=0.1)  # Loss function for contrastive learning

    # Paths for Saving Results
    checkpoint_path = os.path.join(experiment_dir, f"triplet_best_model_{uid}.pth")
    results_path = os.path.join(experiment_dir, f"results_{DEFAULT_MODEL_TYPE}_{info}_{uid}.csv")
    predictions_path = os.path.join(experiment_dir, f"predictions_{DEFAULT_MODEL_TYPE}_{uid}.csv")

    # Training and Evaluation
    start_time = time.time()
    val_f1, val_acc = train_triplet_network(
        model, train_loader, val_loader, optimizer, criteria, results_path, epochs, checkpoint_path, device,
        writer=writer
    )
    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    print(f"Validation F1 Score: {val_f1:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Log results to TensorBoard
    if writer:
        writer.add_text("Experiment Results", f"Validation F1 Score: {val_f1:.4f}, Validation Accuracy: {val_acc:.4f}")
        writer.flush()
        writer.close()
    return val_f1

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    temperature = trial.suggest_uniform('temperature', 0.01, 0.1)
    model_type = trial.suggest_categorical('model_type', ['efficientnet_b0', 'resnet18', 'efficientnet_b1', 'resnet50'])
    # epochs = trial.suggest_int('epochs', 2, 5)
    epochs = 15
    info = f"trial_{trial.number}"
    uid = "optuna_optimization"
    enable_writer = True
    device = 'cuda'
    # Setup Experiment
    experiment_dir, experiment_name = setup_experiment_dir(DEFAULT_MODEL_TYPE, info)
    writer = initialize_writer(experiment_name, uid) if enable_writer else None


    # Get Dataset Paths and Transforms
    dataset_paths = get_dataset_paths(DATASET_DIR)
    train_transforms, test_transform = get_transforms()

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset_paths, train_transforms, test_transform, batch_size, workers)

    # Model Initialization
    model = timm.create_model(model_type, pretrained=True, num_classes=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criteria = losses.SupConLoss(temperature=temperature)  # Loss function for contrastive learning

    # Paths for Saving Results
    checkpoint_path = "" # os.path.join(experiment_dir, f"triplet_best_model_{uid}.pth")
    results_path = None # os.path.join(experiment_dir, f"results_{DEFAULT_MODEL_TYPE}_{info}_{uid}.csv")
    # predictions_path = os.path.join(experiment_dir, f"predictions_{DEFAULT_MODEL_TYPE}_{uid}.csv")

    # Training and Evaluation
    start_time = time.time()
    val_f1, val_acc = train_triplet_network(
        model, train_loader, val_loader, optimizer, criteria, results_path, epochs, checkpoint_path, device,
        writer=writer
    )
    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    print(f"Validation F1 Score: {val_f1:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Log results to TensorBoard
    if writer:
        writer.add_text("Experiment Results", f"Validation F1 Score: {val_f1:.4f}, Validation Accuracy: {val_acc:.4f}")
        writer.flush()
        writer.close()

    return val_f1

if __name__ == '__main__':
    
    # Parse Arguments
    args = parse_arguments()
    print(args)

    # Constants and Hyperparameters
    device = args.device
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    workers = args.workers
    handle_imbalance = args.handle_imbalance_by_fusion
    enable_writer = args.enable_writer
    skip_batches = args.skip_batches_after if args.skip_batches_after else None
    info = args.info if args.info else ""
    uid = args.uid if args.uid else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set Seed 
    set_seed()

    # train_and_evaluate(args)
    study = optuna.create_study(direction='maximize')  # You can also use 'minimize' for error metrics
    study.optimize(objective, n_trials=20)  # Run 20 trials

    # Print the best trial's results
    print("Best Trial:")
    best_trial = study.best_trial
    print(f"Value: {best_trial.value}")
        
    # Save the model associated with the best trial
    # best_model_path = 'best_trial_model.pth'
    # best_model_state_dict = best_trial.user_attrs['model_state_dict']
    
    # # Save the model's state dict (weights)
    # torch.save(best_model_state_dict, best_model_path)
    # print(f"Best model saved to {best_model_path}")

    # Optionally, save hyperparameters used in the best trial
    best_hyperparameters = best_trial.params
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_hyperparameters, f)
    print("Best hyperparameters saved to best_hyperparameters.json")
