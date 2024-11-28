import os
import time
import datetime
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report
from utils_siamese_network import *
import timm

# Constants and Hyperparameters
UID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
DATASET_DIR = 'datasets'
BATCH_SIZE = 32
EPOCHS = 50
MODEL_TYPE = 'resnet50'
CHANGES = '-margin-2'
LOSS_FN_NAME = f'-mean-std-interpol-constrastiveloss{CHANGES}'
# Loss function
loss_fn = ContrastiveLoss(margin=2)

EXPERIMENT_DIR = os.path.join(f'triplet_{MODEL_TYPE}{CHANGES}')
CHECKPOINT_PATH = os.path.join(EXPERIMENT_DIR, f"triplet_best_model_{UID}.pth")
RESULTS_PTH = os.path.join(EXPERIMENT_DIR, f"results_{MODEL_TYPE}_{LOSS_FN_NAME}_{UID}.csv")

PREDICTIONS_PTH = os.path.join(EXPERIMENT_DIR, f"predictions_{MODEL_TYPE}_{UID}.csv")
REPORT_PTH = os.path.join(EXPERIMENT_DIR, f"predictions_{MODEL_TYPE}_{UID}.txt")

# Paths for dataset files
METADATA_FILE_PTH = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')

TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, 'HAM10000_images')
TRAIN_GROUND_TRUTH_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Training_GroundTruth.csv')

VAL_IMAGES_DIR = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Validation_Input')
VAL_GROUND_TRUTH_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Validation_GroundTruth.csv')

TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Test_Input')
TEST_GROUND_TRUTH_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Test_GroundTruth.csv')

parser = argparse.ArgumentParser(description="args")
parser.add_argument("-e", "--epochs", type=int, help="number of epochs")
args = parser.parse_args()

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

# Set up experiment directory
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Load datasets and dataloaders
train_dataset = TripletISIC2018Dataset(csv_file=TRAIN_GROUND_TRUTH_PTH, 
                                       img_dir=TRAIN_IMAGES_DIR, 
                                       transform=train_transforms)

val_dataset = TripletISIC2018Dataset(csv_file=VAL_GROUND_TRUTH_PTH, 
                                     img_dir=VAL_IMAGES_DIR, 
                                     transform=train_transforms)

test_dataset = ISIC2018Dataset(csv_file=TEST_GROUND_TRUTH_PTH, 
                               img_dir=TEST_IMAGES_DIR, 
                               transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Model initialization
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device:", device)

    model = timm.create_model(MODEL_TYPE, pretrained=True, num_classes=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-7)

    # Training and evaluation
    start_time = time.time()
    train_results = train_triplet_network(model, train_loader, val_loader, optimizer, 
                                        loss_fn, 
                                        RESULTS_PTH,
                                        num_epochs=EPOCHS, 
                                        checkpoint_path=CHECKPOINT_PATH,
                                        device=device
                                        )

    # test_loss = evaluate_triplet_network(model, loss_fn, test_loader)

    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    #####################
    # Prediction on test data
    #####################
    # Load checkpoint for testing
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Generate reference embeddings and predictions
    reference_embeddings = generate_reference_embeddings(model, train_loader, device=device)
    predictions = predict_class(model, test_loader, reference_embeddings, device=device)

    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(PREDICTIONS_PTH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PTH}")

    # Compute and print metrics
    y_true, y_pred = predictions_df['true_label'], predictions_df['predicted_label']
    accuracy = accuracy_score(y_true, y_pred)
    print("=" * 30)

    with open(REPORT_PTH, 'w') as fp:
        fp.write(f"Accuracy: {accuracy:.4f}\n")    
        fp.write(classification_report(y_true, y_pred))

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    main()