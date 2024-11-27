from main_utils import *
from torch.utils.tensorboard import SummaryWriter


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

TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, 'HAM10000_images')
TRAIN_DIR_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Training_GroundTruth.csv')

VAL_IMAGES_DIR = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Validation_Input')
VAL_DIR_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Validation_GroundTruth.csv')

TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Test_Input')
TEST_DIR_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Test_GroundTruth.csv')

all_results = []
UID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# ##########################################################

EPOCHS = args.epochs if args.epochs else 50 

# MODEL_TYPE = 'vit_small'
# MODEL_TYPE = 'vit_base_patch16_224'
# MODEL_TYPE = 'resnet50'
MODEL_TYPE = 'inception_resnet_v2'
# MODEL_TYPE = 'convnext_tiny'
# MODEL_TYPE = 'convnext_xlarge'
# MODEL_TYPE = 'convnextv2_large'
# MODEL_TYPE = 'efficientnet_b0'
# MODEL_TYPE = 'swin_base_patch4_window7_224'


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

# Create datasets and dataloaders
train_dataset = ISIC2018Dataset(csv_file=TRAIN_DIR_PTH, 
                                img_dir=TRAIN_IMAGES_DIR, 
                                transform=train_transforms)
val_dataset = ISIC2018Dataset(csv_file=VAL_DIR_PTH, 
                              img_dir=VAL_IMAGES_DIR, 
                              transform=test_transforms)
test_dataset = ISIC2018Dataset(csv_file=TEST_DIR_PTH, 
                               img_dir=TEST_IMAGES_DIR, 
                               transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


writer = SummaryWriter(log_dir=f'logs/{MODEL_TYPE}_{UID}')

start_time = time.time()
for LOSS_FN_NAME in loss_functions:
    model_checkpoint_name = f"model_{MODEL_TYPE}_{LOSS_FN_NAME}_{UID}.pth"      ## <-------------
    RESULTS_PTH = os.path.join(EXPERIMENT_DIR, f"results_{MODEL_TYPE}_{LOSS_FN_NAME}_{UID}.csv")
    PREDICTIONS_PTH = os.path.join(EXPERIMENT_DIR, f"predictions_{MODEL_TYPE}_{UID}.csv")
    REPORT_PTH = os.path.join(EXPERIMENT_DIR, f"predictions_{MODEL_TYPE}_{UID}.txt")

    model_checkpoint_pth = os.path.join(EXPERIMENT_DIR, model_checkpoint_name)

    # Initialize the model and optimizer for each experiment
    print("*"*10+LOSS_FN_NAME+"*"*10)
    model = timm.create_model(MODEL_TYPE, pretrained=True, num_classes=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)# use lower lr

    # Train and log results
    results_df = train_and_evaluate(loss_functions,
                                 LOSS_FN_NAME, 
                                 model, 
                                 train_loader, val_loader, optimizer, 
                                 RESULTS_PTH,
                                 REPORT_PTH,
                                 writer,
                                 num_epochs=EPOCHS, 
                                 checkpoint_path=model_checkpoint_pth,
                                 )
    
    checkpoint = torch.load(model_checkpoint_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    test_f1, test_acc = evaluate_model(model, test_loader, REPORT_PTH)
    
    results_df['Test_f1'] = test_f1
    results_df['Test_acc'] = test_acc
    results_df.to_csv(RESULTS_PTH, index=False)


    # all_results.append(pd.DataFrame(results).assign(loss_fn=loss_fn_name))

print(f"Time taken in {(time.time()-start_time)//60} mins")
print("Experiment completed and results saved.")