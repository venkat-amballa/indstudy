from main_utils import *
import torchvision

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

# Load and split data
DATASET_DIR = 'datasets'
BATCH_SIZE = 64
NUM_CLASSES = 7

TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, 'HAM10000_images')
TRAIN_DIR_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Training_GroundTruth.csv')

VAL_IMAGES_DIR = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Validation_Input')
VAL_DIR_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Validation_GroundTruth.csv')

TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Test_Input')
TEST_DIR_PTH = os.path.join(DATASET_DIR, 'ISIC2018_Task3_Test_GroundTruth.csv')

all_results = []

# ##########################################################

# MODEL_TYPE = 'vit_small'
# MODEL_TYPE = 'vit_base_patch16_224'
# MODEL_TYPE = 'resnet50'
# MODEL_TYPE = 'inception_resnet_v2'
# MODEL_TYPE = 'convnext_tiny'
# MODEL_TYPE = 'convnext_xlarge'
# MODEL_TYPE = 'convnextv2_large'
MODEL_TYPE = 'efficientnet_b0'
# MODEL_TYPE = 'swin_base_patch4_window7_224'


# CHANGES = 'using_timm_mean_std_interpol'
CHANGES = ''

EXPERIMENT_DIR = os.path.join(f"{MODEL_TYPE}_{CHANGES}" if CHANGES else MODEL_TYPE)

loss_functions = {
    'CrossEntropy': nn.CrossEntropyLoss(),
    # 'FocalLoss': FocalLoss(alpha=1, gamma=2),
    # 'LabelSmoothing': LabelSmoothingLoss(smoothing=0.1),
    # 'TverskyLoss': TverskyLoss(alpha=0.7, beta=0.3),
    # 'F1Loss': F1Loss(),
}

# #######################################################

if not os.path.exists(EXPERIMENT_DIR):
    os.makedirs(EXPERIMENT_DIR)

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

args = {
    "num_workers": 3,
    "pin_memory": True,
    "batch_size": BATCH_SIZE
}
train_loader = DataLoader(train_dataset, shuffle=True,**args)
val_loader = DataLoader(val_dataset, shuffle=False, **args)
test_loader = DataLoader(test_dataset, shuffle=False, **args)


def main():
    ################ PARSER #############################
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs")
    parser.add_argument("-d", "--device", type=str, help="device to use(cpu/cuda)")
    parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate to use")
    parser.add_argument("-wd", "--weight_decay", type=float, help="weight decay value")
    parser.add_argument("-uid", "--uid", type=str, help="uid of the previous run, pass this to resume the training")
    parser.add_argument("-skip", "--skip_batches", action="store_true", help="Use 3 batches per epoch if set, for testing.")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device:",device)

    EPOCHS = args.epochs if args.epochs else 50 
    DEVICE = args.device if args.device else device
    LR = args.learning_rate if args.learning_rate else 1e-5
    WEIGHT_DECAY = args.weight_decay if args.weight_decay else 1e-6
    UID = args.uid if args.uid else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    skip_batches = args.skip_batches


    print(f"{EXPERIMENT_DIR=}")
    print(f"{UID=}")
    writer = SummaryWriter(log_dir=f"runs/{EXPERIMENT_DIR}_{UID}")

    examples = iter(train_loader)

    batch_data, batch_labels = examples.__next__()
    
    img_grid = torchvision.utils.make_grid(batch_data)
    writer.add_image("sample_batch_data", img_grid)
    writer.add_text("Hyperparameters", f"EPOCHS={EPOCHS}, LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}")

    start_time = time.time()
    for LOSS_FN_NAME in loss_functions:
        model_checkpoint_name = f"model_{MODEL_TYPE}_{LOSS_FN_NAME}_{UID}.pth"      ## <-------------
        RESULTS_PTH = os.path.join(EXPERIMENT_DIR, f"results_{MODEL_TYPE}_{LOSS_FN_NAME}_{UID}.csv")
        PREDICTIONS_PTH = os.path.join(EXPERIMENT_DIR, f"predictions_{MODEL_TYPE}_{UID}.csv")
        REPORT_PTH = os.path.join(EXPERIMENT_DIR, f"predictions_{MODEL_TYPE}_{UID}.txt")

        model_checkpoint_pth = os.path.join(EXPERIMENT_DIR, model_checkpoint_name)

        # Initialize the model and optimizer for each experiment
        print("*"*10+LOSS_FN_NAME+"*"*10)
        model = timm.create_model(MODEL_TYPE, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)# use lower lr

        # Train and log results
        train_and_evaluate(loss_functions,
                                    LOSS_FN_NAME, 
                                    model, 
                                    train_loader, val_loader, optimizer, 
                                    RESULTS_PTH,
                                    REPORT_PTH,
                                    num_epochs=EPOCHS, 
                                    checkpoint_path=model_checkpoint_pth,
                                    writer = writer,
                                    device=DEVICE,
                                    skip_batches = skip_batches
                                    )
        
        checkpoint = torch.load(model_checkpoint_pth)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        test_f1, test_acc = evaluate_model(model, test_loader, REPORT_PTH, device=DEVICE)
        if writer:
            writer.add_text("Test_F1_Score", f"Test: f1_score={test_f1}")
            writer.add_text("Test_Accuracy", f"Test: Accuracy={test_acc}")

        # results_df['Test_f1'] = test_f1
        # results_df['Test_acc'] = test_acc
        # results_df.to_csv(RESULTS_PTH, index=False)


        # all_results.append(pd.DataFrame(results).assign(loss_fn=loss_fn_name))

    print(f"Time taken in {(time.time()-start_time)//60} mins")
    print("Experiment completed and results saved.")
    writer.flush()
    writer.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up...")
        sys.exit(0)
