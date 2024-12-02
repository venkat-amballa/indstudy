from utils_siamese_network import *
import torch
import timm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])


UID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # 20241128_123508

DATASET_DIR = 'datasets'
BATCH_SIZE = 32
EPOCHS = 50
# MODEL_TYPE = 'resnet18'
MODEL_TYPE = 'resnet50'
CHANGES = '-margin-2'
device = 'cuda'

EXPERIMENT_DIR = f'triplet_{MODEL_TYPE}{CHANGES}'
checkpoint_path = os.path.join(EXPERIMENT_DIR, "triplet_best_model_20241128_123508.pth")


results_pth = os.path.join(EXPERIMENT_DIR, f"results_{MODEL_TYPE}_{UID}.csv")   ## <-------------


METADATA_FILE_PTH = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')
# IMAGES_DIR = os.path.join(DATASET_DIR, 'HAM10000_images')

# TEST_DIR_PTH = os.path.join(DATASET_DIR, 'test_data.csv')

TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Test_Input')
TEST_DIR_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Test_GroundTruth.csv')

# test_dataset = HAM10000Dataset(csv_file=TEST_DIR_PTH, 
#                                img_dir=IMAGES_DIR, 
#                                transform=test_transforms)
def main():

    test_dataset = ISIC2018Dataset(csv_file=TEST_DIR_PTH, 
                                img_dir=TEST_IMAGES_DIR, 
                                transform=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


    # model
    model = timm.create_model(MODEL_TYPE, pretrained=True, num_classes=0)  # num_classes=0 for feature extraction
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    checkpoint = torch.load(checkpoint_path)

    # Load the model state  
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load the optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    reference_embeddings = generate_reference_embeddings(model, test_loader, device=device)

    predictions = predict_class(model, test_loader, reference_embeddings, device=device)
    results = pd.DataFrame(predictions)
    results.to_csv(results_pth, index=False)

    # metrics
    y_true, y_pred = results['true_label'], results['predicted_label']
    accuracy = accuracy_score(y_true, y_pred)
    print("="*30)
    print(f"{accuracy=}")
    print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        print("exception:",e)
        sys.exit(0)