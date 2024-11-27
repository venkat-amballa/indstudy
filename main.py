from main_utils import *


# Define data preprocessing transformations
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

test_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])
parser = argparse.ArgumentParser(description="args")
parser.add_argument("-e", "--epochs", type=int, help="number of epochs")
args = parser.parse_args()

# Load and split data
DATASET_DIR = 'datasets'
BATCH_SIZE = 32

METADATA_FILE_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Training_GroundTruth.csv')
IMAGES_DIR = os.path.join(DATASET_DIR, 'HAM10000_images')

TRAIN_DIR_PTH = os.path.join(DATASET_DIR, 'train_data.csv')
VAL_DIR_PTH = os.path.join(DATASET_DIR, 'val_data.csv')
TEST_DIR_PTH = os.path.join(DATASET_DIR, 'test_data.csv')

all_results = []
UID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# ##########################################################

EPOCHS = args.epochs if args.epochs else 50 

# MODEL_TYPE = 'vit_small'
# MODEL_TYPE = 'vit_base_patch16_224'
# MODEL_TYPE = 'resnet50'
# MODEL_TYPE = 'convnext_tiny'
# MODEL_TYPE = 'convnext_xlarge'
# MODEL_TYPE = 'convnextv2_large'
# MODEL_TYPE = 'efficientnet_b0'
MODEL_TYPE = 'swin_base_patch4_window7_224'
# CHANGES = 'using_timm_mean_std_interpol'
CHANGES = 'using_timm_mean_std_interpol'

EXPERIMENT_DIR = os.path.join(f'{MODEL_TYPE}_{CHANGES}')

loss_functions = {
    # 'CrossEntropy': nn.CrossEntropyLoss(),
    'FocalLoss': FocalLoss(alpha=1, gamma=2),
    # 'LabelSmoothing': LabelSmoothingLoss(smoothing=0.1),
    # 'TverskyLoss': TverskyLoss(alpha=0.7, beta=0.3),
    # 'F1Loss': F1Loss(),
}

# #######################################################

if not os.path.exists(EXPERIMENT_DIR):
    os.mkdir(EXPERIMENT_DIR)

data = pd.read_csv(METADATA_FILE_PTH)

train_data, remaining_data = train_test_split(data, test_size=0.15, random_state=42, stratify=data['dx'])
val_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=42, stratify=remaining_data['dx'])

# Save the datasets

train_data.to_csv(TRAIN_DIR_PTH, index=False)
val_data.to_csv(VAL_DIR_PTH, index=False)
test_data.to_csv(TEST_DIR_PTH, index=False)

# Create datasets and dataloaders
train_dataset = HAM10000Dataset(csv_file=TRAIN_DIR_PTH, 
                                img_dir=IMAGES_DIR, 
                                transform=train_transforms)
val_dataset = HAM10000Dataset(csv_file=VAL_DIR_PTH, 
                              img_dir=IMAGES_DIR, 
                              transform=test_transforms)
test_dataset = HAM10000Dataset(csv_file=TEST_DIR_PTH, 
                               img_dir=IMAGES_DIR, 
                               transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


start_time = time.time()
for loss_fn_name in loss_functions:
    model_checkpoint_name = f"model_{MODEL_TYPE}_{loss_fn_name}_{UID}.pth"      ## <-------------
    model_checkpoint_pth = os.path.join(EXPERIMENT_DIR, model_checkpoint_name)

    # Initialize the model and optimizer for each experiment
    print("*"*10+loss_fn_name+"*"*10)
    model = timm.create_model(MODEL_TYPE, pretrained=True, num_classes=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)# use lower lr

    # Train and log results
    results = train_and_evaluate(loss_functions,
                                 loss_fn_name, 
                                 model, 
                                 train_loader, val_loader, optimizer, 
                                 num_epochs=EPOCHS, 
                                 checkpoint_path=model_checkpoint_pth)
    
    all_results.append(pd.DataFrame(results).assign(loss_fn=loss_fn_name))

print(f"Time taken in {(time.time()-start_time)//60} mins")

# Combine results for comparison
all_results_df = pd.concat(all_results, ignore_index=True)
test_f1, test_acc = evaluate_model(model, test_loader)

all_results_df['Test_f1'] = test_f1
all_results_df['Test_acc'] = test_acc

results_pth = os.path.join(EXPERIMENT_DIR, f"results_{MODEL_TYPE}_{UID}.csv")   ## <-------------

all_results_df.to_csv(results_pth, index=False)
print("Experiment completed and results saved.")