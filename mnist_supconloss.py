from __future__ import print_function
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
import albumentations as A
import numpy as np

# Hyperparameters
embedding_dim = 128  
num_classes = 10     
temperature = 0.1    
batch_size = 256
num_workers = 1
learning_rate = 0.001
epochs = 10

# Define a simple CNN model for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)  # 7x7 is the down-sampled image size after conv layers
        self.fc2 = nn.Linear(512, embedding_dim)  # Feature embedding

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # print(x.shape)
        x = torch.relu(self.conv2(x))
        # print(x.shape)
        x = torch.max_pool2d(x, 2)
        # print(x.shape)
        x = torch.relu(self.conv3(x))
        # print(x.shape)
        x = torch.max_pool2d(x, 2)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)  # Embedding 
        # print(x.shape)
        return x

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform1, transform2, aug_size=4):
        self.dataset = dataset
        self.transforms = [transform1, transform2]
        self.aug_size = aug_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        img = np.array(img)  
        if img.ndim == 2:  # If grayscale image, add channel
            img = np.expand_dims(img, axis=-1)
        
        img_augs = []
        for i in range(self.aug_size):
            _transform = self.transforms[1] #random.choice(self.transforms)
            img_augs.append(_transform(image=img)["image"])
            print(img_augs[-1].shape)
        return tuple(img_augs), label

if __name__ == '__main__':
    # train_transforms = transforms.Compose([
    #     transforms.RandomRotation(10),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    # train_transforms2 = transforms.Compose([
    #     transforms.RandomAffine(15),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    # test_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    image_size = 224
    train_transforms = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=0.7),

        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        # A.Resize(image_size, image_size),
        # A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),    
        A.Normalize()
    ])

    train_transforms2 = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize()
    ])

    test_transforms = A.Compose([
        # A.Resize(image_size, image_size),
        A.Normalize()
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transforms)

    augmented_train_dataset = AugmentedDataset(train_dataset, train_transforms, train_transforms2)

    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    feature_extractor = FeatureExtractor(embedding_dim=embedding_dim).cuda()

    optimizer = optim.Adam(feature_extractor.parameters())
    criterion = SupConLoss(temperature=temperature)

    # train
    for epoch in range(epochs):
        feature_extractor.train()
        running_loss = 0.0
        for img_augs, labels in tqdm(train_loader):
            labels = labels.cuda()
            optimizer.zero_grad()

            embeddings = []
            for _img in img_augs:
                _img = _img.cuda()
                embeddings.append(feature_extractor(_img))

            features_n_views = torch.stack(embeddings, dim=1)
            loss = criterion(features_n_views, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, SupConLoss: {avg_loss:.4f}")

    # save it
    torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')
    print("saved the model!")
    # evaluate
    feature_extractor.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, target_labels in test_loader:
            embedding = feature_extractor(images.cuda())
            embeddings.append(embedding)
            labels.append(target_labels)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    # apply t-SNE 
    tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
    reduced_embeddings = tsne.fit_transform(embeddings.cpu().numpy())
    print("reduced the embeddings")
    # embeddings shape (num_samples, 2)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar()
    plt.title('t-SNE visualization of MNIST Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('tsne_supconloss.png')
    plt.show()