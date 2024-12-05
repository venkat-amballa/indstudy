from refactored_utils import *
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import argparse
import albumentations as A
from torchvision import transforms

from dataloaders import ISIC2018Dataset, ISIC2018DatasetBalancedPairing

NUM_CLASSES = 0 #7
DATASET_DIR = 'datasets'

loss_functions = {
    'CrossEntropy': nn.CrossEntropyLoss(),
    'FocalLoss': FocalLoss(alpha=1, gamma=2),
    'LabelSmoothing': LabelSmoothingLoss(smoothing=0.1),
    # 'TverskyLoss': TverskyLoss(alpha=0.7, beta=0.3),
    # 'F1Loss': F1Loss(),
}

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
    # Image.blend()
    # # SoftAugementation (CVPR, 2023)

    # train_transforms = transforms.Compose([
    #     transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    #     # transforms.Lambda(convert_to_hsv),  # Convert to HSV, no improvement
    #     transforms.RandomEqualize(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    #     transforms.ToTensor(),
    #     transforms.RandomErasing(p=0.2),
        # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    # ])

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  
        # transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),  
        # transforms.RandomCrop(224),                                                          
        transforms.RandomHorizontalFlip(),                                                   
        transforms.RandomVerticalFlip(),                                                     
        transforms.RandomRotation(30),                                                       
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3),        
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),          
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.3, interpolation=3),
        # transforms.Lambda(lambda img: add_gaussian_noise(img, mean=0, std=0.1)),  # Gaussian noise
        # transforms.Lambda(lambda img: convert_to_hsv(img)),  # Convert image to HSV
        # transforms.Lambda(lambda img: gamma_correction(img, gamma=random.uniform(0.5, 2.5))),  # Gamma correction
        transforms.ToTensor(),                                                                
        # transforms.RandomErasing(p=0.2),  # Randomly erase portions of the image
        # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         
        # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    return train_transforms, test_transforms


def get_dataloaders(dataset_dir, batch_size, train_transforms, test_transforms, num_workers, required_labels=False, handle_imbalance=False):
    """Create datasets and dataloaders."""
    if handle_imbalance:
        print("Handling imbalance with fusion.....!")
        train_dataset = ISIC2018DatasetBalancedPairing(
            csv_file=os.path.join(dataset_dir, 'ISIC2018_Task3_Training_GroundTruth.csv'),
            img_dir=os.path.join(dataset_dir, 'HAM10000_images'),
            transform=train_transforms
        )
        print(f"Total size after: {len(train_dataset)=}")
    else:
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

    if required_labels:
        labels = train_dataset.data['label_encoded'].tolist()
        return train_loader, val_loader, test_loader, labels
    
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


def run_experiment(experiment, loss_functions, train_loader, val_loader, test_loader, enable_writer=False, skip_batches=None):
    """Run a single experiment."""
    model_type = experiment["model_type"]
    loss_fn_name = experiment["loss_fn"]
    optimizer_name = experiment["optimizer"]
    hyperparams = experiment["hyperparams"]
    epochs = experiment['epochs']
    device = experiment['device']
    batch_size = experiment['batch_size']

    info = experiment.get("info", "")
    # Variable to accumlate experiments text results
    experiment_text = ""

    # Prepare logging
    experiment_name = f"{model_type}_{loss_fn_name}_{optimizer_name}"
    experiment_name = experiment_name + f"_{info}" if info else experiment_name

    experiment_dir = setup_experiment_dir(experiment_name)
    uid = experiment['uid']
    print(f"======={experiment_name=}, {uid=} ==========")
    writer = None
    if enable_writer:
        writer = initialize_writer(experiment_name, uid)

    # # Save a sample batch for visualisation
    # examples = iter(train_loader)
    # batch_data, batch_labels = examples.__next__()
    # img_grid = torchvision.utils.make_grid(batch_data)
    # writer.add_image("sample_batch_data", img_grid)


    # Initialize model, loss, and optimizer
    model = timm.create_model(model_type, pretrained=True, num_classes=NUM_CLASSES).to(device)
    criterion = loss_functions[loss_fn_name]
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), **hyperparams)

    # Performing Lr range test to find the optimal Learning rate
    if experiment.get('lr_find', False):
        amp_config = {
            'device_type': device,
            'dtype': torch.float16,
        }
        grad_scaler = torch.amp.GradScaler()
        
        lr_finder = LRFinder(
            model, optimizer, criterion, device=device,
            amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
        )
        try:
            lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode='exp')
            fig, axs = plt.subplots()
            axs, lr_suggest = lr_finder.plot(show_lr=1.2, ax=axs, suggest_lr=True)

            # Log the lr range plot to tensor board
            if writer:
                writer.add_figure("Lr_range_test", fig)
                writer.flush()
            
            experiment_text += f"lr_range_test: lr_suggest={lr_suggest}\n"
            print(experiment_text)

            # Use the lr_min as the new learning rate
            hyperparams["lr"] = lr_suggest/100 # a safe lr value that is not aggressive
            print(f"{lr_suggest=}, Using safe lr by : {lr_suggest/100=}")
            optimizer = optimizer_class(model.parameters(), **hyperparams)
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
        num_epochs=epochs,
        checkpoint_path=checkpoint_path,
        writer=writer,
        device=device,
        skip_batches=skip_batches
    )

    # Evaluate on test set
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_f1, test_acc, test_roc_auc = evaluate_model(model, test_loader, criterion, device=device)
    print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}, Test F1: {test_f1:.2f}, Test_roc_auc: {test_roc_auc:.2f}\n")

    # Append test results and hyperparameters to experiment text
    experiment_text += f"Test | loss={test_loss}, f1_score={test_f1}, accuracy={test_acc}, roc_auc={test_roc_auc}\n"
    experiment_text += f"Experiment: {str(experiment)}\n"

    # Loging the info at once
    if writer:
        writer.add_text("Experiment Results", experiment_text)
        writer.flush()
        writer.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-lr", "--learning_rate", default=5e-4, type=float, help="Learning rate")
    parser.add_argument("-batch", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-workers", "--workers", type=int, default=3, help="Batch size")

    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use (cpu/cuda)")
    parser.add_argument("-uid", "--uid", type=str, help="UID for resuming the training")
    parser.add_argument("-writer", "--enable_writer", action="store_true", help="Enable tensorboard writer")
    parser.add_argument("-skip_after", "--skip_batches_after", type=int, help="Run the training loop only for given no.of batches; Testing only!")
    parser.add_argument("-imb_fusion", "--handle_imbalance_by_fusion", action="store_true", help="To handle imbalance using image fusion")
    parser.add_argument("-info", "--info", type=str, help="This will be used while creating the experiemnt dir and in tensorboard.")

    return parser.parse_args()


def main():
    """Main function to run multiple experiments."""
    args = parse_arguments()
    print(args)
    # Device Configuration
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    workers = args.workers
    DEVICE = args.device
    handle_imbalance = args.handle_imbalance_by_fusion
    enable_writer = args.enable_writer
    skip_batches = args.skip_batches_after if args.skip_batches_after else None
    INFO = args.info if args.info else ""
    
    uid = args.uid if args.uid else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print("Using Device:", DEVICE)

    # Get data loaders 
    train_transforms, test_transforms = get_transforms()
    train_loader, val_loader, test_loader, labels = get_dataloaders(DATASET_DIR, batch_size, 
                                                            train_transforms, 
                                                            test_transforms, 
                                                            num_workers=workers,
                                                            required_labels=True,
                                                            handle_imbalance=handle_imbalance)

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    loss_functions = {
        # 'CrossEntropy': nn.CrossEntropyLoss(weight=class_weights),
        # 'FocalLoss': FocalLoss(alpha=1, gamma=2),
        # 'LabelSmoothing': LabelSmoothingLoss(smoothing=0.1),
        'SupConLoss': SupConLoss(temperature=0.1)
        # 'TverskyLoss': TverskyLoss(alpha=0.7, beta=0.3),
        # 'F1Loss': F1Loss(),
    }

    experiments = [
        {   # best 87% testing acc
            "model_type": "efficientnet_b0",
            "loss_fn": "SupConLoss",
            "optimizer": "Adam",
            "epochs": epochs,
            "device": DEVICE,
            "batch_size": batch_size,
            "hyperparams": {
                "lr": 5e-4
            },
            # "skip": True,
            # "lr_find": True,
            "info": INFO,
            "uid": uid
        }
    ]

    # Run each experiment
    for experiment in experiments:
        if experiment.get('skip', False):
            print(f"---Skipping experiment-->: {experiment['model_type']} with {experiment['loss_fn']}")
        else:
            print(f"Running experiment: {experiment['model_type']} with {experiment['loss_fn']}")
            run_experiment(experiment, loss_functions, 
                           train_loader, val_loader, test_loader, 
                           enable_writer=enable_writer,
                           skip_batches=skip_batches)
            

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up...")
        sys.exit(0)
