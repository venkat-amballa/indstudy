from refactored_utils import *
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

NUM_CLASSES = 7
BATCH_SIZE = 64
DATASET_DIR = 'datasets'
NUM_WORKERS = 3 # for dataloader

loss_functions = {
    'CrossEntropy': nn.CrossEntropyLoss(),
    'FocalLoss': FocalLoss(alpha=1, gamma=2),
    'LabelSmoothing': LabelSmoothingLoss(smoothing=0.1),
    # 'TverskyLoss': TverskyLoss(alpha=0.7, beta=0.3),
    # 'F1Loss': F1Loss(),
}

def convert_to_hsv(img):
    """Converts a PIL image to HSV using OpenCV."""
    img = np.array(img)  # Convert PIL to numpy
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return Image.fromarray(img_hsv)

def get_transforms():
    """Define data preprocessing transformations."""
    # # 1. Median filtering/ guassian filtering
    # # MedianFiltering()
    

    # # 3 . Color space hsv or LAB or RGB (select them randomly while training)
    # transforms.Lambda(lambda img: img.convert('HSV'))
    # transforms.Lambda(lambda img: Image.fromarray(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)))
    
    # # 6. Randomcrop
    # transforms.RandomCrop(200),
    # transforms.Resize((224, 224)),

    # # 2. Contrast enhancement


    # # RandomEqualize - Equalizes image histograms to enhance contrast.
    # transforms.RandomEqualize(),
    # # RandomErasing  - Erases random regions to improve robustness.
    # transforms.RandomErasing(p=0.2), 

    # # 5. mixup - Combines two images using a weighted average

    # # 4. Image Fusion - Overlaying two images to produce more data points.
    # # SoftAugementation (CVPR, 2023)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img: convert_to_hsv(img)),  # Convert to HSV
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    return train_transforms, test_transforms


def get_dataloaders(dataset_dir, batch_size, train_transforms, test_transforms, num_workers=NUM_WORKERS):
    """Create datasets and dataloaders."""
    train_dataset = ISIC2018Dataset(
        csv_file=os.path.join(dataset_dir, 'ISIC2018_Task3_Training_GroundTruth.csv'),
        img_dir=os.path.join(dataset_dir, 'HAM10000_images'),
        transform=train_transforms
    )

    val_dataset = ISIC2018Dataset(
        csv_file=os.path.join(dataset_dir, 'ISIC2018_Task3_Validation_GroundTruth.csv'),
        img_dir=os.path.join(dataset_dir, 'ISIC2018_Task3_Validation_Input'),
        transform=test_transforms
    )

    test_dataset = ISIC2018Dataset(
        csv_file=os.path.join(dataset_dir, 'ISIC2018_Task3_Test_GroundTruth.csv'),
        img_dir=os.path.join(dataset_dir, 'ISIC2018_Task3_Test_Input'),
        transform=test_transforms
    )

    args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "batch_size": batch_size,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **args)
    val_loader = DataLoader(val_dataset, shuffle=False, **args)
    test_loader = DataLoader(test_dataset, shuffle=False, **args)

    return train_loader, val_loader, test_loader


def setup_experiment_dir(experiment_name):
    """Set up experiment directory."""
    experiment_dir = f"experiments/{experiment_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    return experiment_dir


def initialize_writer(experiment_name, uid):
    """Initialize TensorBoard SummaryWriter."""
    log_dir = f"runs/{experiment_name}_{uid}"
    return SummaryWriter(log_dir=log_dir)

def get_class_weights():
    
    pass
def run_experiment(experiment, train_loader, val_loader, test_loader):
    """Run a single experiment."""
    model_type = experiment["model_type"]
    loss_fn_name = experiment["loss_fn"]
    optimizer_name = experiment["optimizer"]
    hyperparams = experiment["hyperparams"]
    info = experiment.get("info", "")
    # Variable to accumlate experiments text results
    experiment_text = ""

    # Prepare logging
    experiment_name = f"{model_type}_{loss_fn_name}_{optimizer_name}"
    experiment_name = experiment_name + f"_{info}" if info else experiment_name

    experiment_dir = setup_experiment_dir(experiment_name)
    uid = experiment.get('UID', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print(f"======={experiment_name=}, {uid=} ==========")
    writer = initialize_writer(experiment_name, uid)

    # Save a sample batch for visualisation
    examples = iter(train_loader)
    batch_data, batch_labels = examples.__next__()
    img_grid = torchvision.utils.make_grid(batch_data)
    writer.add_image("sample_batch_data", img_grid)


    # Initialize model, loss, and optimizer
    model = timm.create_model(model_type, pretrained=True, num_classes=NUM_CLASSES).to(hyperparams["device"])
    criterion = loss_functions[loss_fn_name]
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), **hyperparams["optimizer_params"])

    # Performing Lr range test to find the optimal Learning rate
    if experiment.get('lr_find', False):
        amp_config = {
            'device_type': hyperparams['device'],
            'dtype': torch.float16,
        }
        grad_scaler = torch.amp.GradScaler()
        
        lr_finder = LRFinder(
            model, optimizer, criterion, device=hyperparams['device'],
            amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
        )
        try:
            lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode='exp')
            fig, axs = plt.subplots()
            axs, lr_suggest = lr_finder.plot(show_lr=1.2, ax=axs, suggest_lr=True)

            # Log the lr range plot to tensor board
            writer.add_figure("Lr_range_test", fig)
            writer.flush()
            
            experiment_text += f"lr_range_test: lr_suggest={lr_suggest}\n"
            print(experiment_text)

            # Use the lr_min as the new learning rate
            hyperparams["optimizer_params"]["lr"] = lr_suggest/100 # a safe lr value that is not aggressive
            optimizer = optimizer_class(model.parameters(), **hyperparams["optimizer_params"])
        finally:
            lr_finder.reset()
    # Paths
    checkpoint_path = os.path.join(experiment_dir, f"model_{uid}.pth")
    # results_path = os.path.join(experiment_dir, f"results_{uid}.csv")
    # report_path = os.path.join(experiment_dir, f"report_{uid}.txt")

    # Train and evaluate
    train_and_evaluate(
        loss_functions={loss_fn_name: criterion},
        loss_fn_name=loss_fn_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=hyperparams["epochs"],
        checkpoint_path=checkpoint_path,
        writer=writer,
        device=hyperparams["device"],
        skip_batches=hyperparams.get("skip_batches", None)
    )

    # Evaluate on test set
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_f1, test_acc = evaluate_model(model, test_loader, device=hyperparams["device"])

    # Append test results and hyperparameters to experiment text
    experiment_text += f"Test: f1_score={test_f1}, Test: Accuracy={test_acc}\n"
    experiment_text += f"Experiment: {str(experiment)}\n"

    # Loging the info at once
    writer.add_text("Experiment Results", experiment_text)

    writer.flush()
    writer.close()


def main():
    """Main function to run multiple experiments."""
    # Define experiments
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiments = [
        {
            "model_type": "efficientnet_b0",
            "loss_fn": "CrossEntropy",
            "optimizer": "AdamW",
            "hyperparams": {
                "epochs": 20,
                "device": DEVICE,
                "optimizer_params": {"lr": 1e-4, "weight_decay": 1e-6},
                "batch_size": 64,
            },
            "skip": True,
            "lr_find": True,
            # "info": "gradient_accumulation"
            # "UID": "20241129_143457"
        },
        {   # best 87% testing acc
            "model_type": "efficientnet_b0",
            "loss_fn": "CrossEntropy",
            "optimizer": "Adam",
            "hyperparams": {
                "epochs": 20,
                "device": DEVICE,
                "optimizer_params": {"lr": 5e-4},
                "batch_size": 64,
            },
            # "skip": True,
            "lr_find": True,
            # "info": "gradient_accumulation"
            # "UID": "20241129_145429"
        },
        # {    
        #     "model_type": "resnet50",
        #     "loss_fn": "CrossEntropy",
        #     "optimizer": "Adam",
        #     "hyperparams": {
        #         "epochs": 20,
        #         "device": DEVICE,
        #         "optimizer_params": {"lr": 5e-4},
        #         "batch_size": 64,
        #     },
        #     # "info": "gradient_accumulation"
        #     # "skip": False,
        #     # "lr_find": True,
        #     "UID": "20241130_023847"
        # },
    ]

    # Get data loaders
    train_transforms, test_transforms = get_transforms()
    train_loader, val_loader, test_loader = get_dataloaders(DATASET_DIR, BATCH_SIZE, train_transforms, test_transforms)

    # Run each experiment
    for experiment in experiments:
        if experiment.get('skip', False):
            print(f"---Skipping experiment-->: {experiment['model_type']} with {experiment['loss_fn']}")
        else:
            print(f"Running experiment: {experiment['model_type']} with {experiment['loss_fn']}")
            run_experiment(experiment, train_loader, val_loader, test_loader)
            

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up...")
        sys.exit(0)
