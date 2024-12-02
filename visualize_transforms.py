import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import random

def fetch_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except requests.RequestException as e:
        print(f"Error fetching image: {e}")
        return None

def median_filtering(img):
    return Image.fromarray(cv2.medianBlur(np.array(img), 5))

def gaussian_filtering(img):
    return Image.fromarray(cv2.GaussianBlur(np.array(img), (5, 5), 0))

def random_color_space(img):
    img_np = np.array(img)
    if random.choice([True, False]):
        return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV))
    else:
        return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB))

def mixup(img1, img2, alpha=0.5):
    img2_resized = img2.resize(img1.size)
    img1_np = np.array(img1, dtype=np.float32)
    img2_np = np.array(img2_resized, dtype=np.float32)
    img_mixed = (1 - alpha) * img1_np + alpha * img2_np
    return Image.fromarray(np.clip(img_mixed, 0, 255).astype(np.uint8))

def fuse_images(img1, img2, alpha=0.5):
    img2_resized = img2.resize(img1.size)
    return Image.blend(img1, img2_resized, alpha)

class NamedTransform:
    def __init__(self, func, name):
        self.func = func
        self.name = name

    def __call__(self, img):
        return self.func(img)

    def __repr__(self):
        return self.name

def get_train_transforms(img2):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        NamedTransform(median_filtering, "MedianFiltering"),
        NamedTransform(gaussian_filtering, "GaussianFiltering"),
        NamedTransform(random_color_space, "RandomColorSpace"),
        NamedTransform(lambda img: mixup(img, img2), "Mixup"),
        NamedTransform(lambda img: fuse_images(img, img2), "FuseImages"),
    ])

def visualize_transforms(data_transforms, image_url, second_image_url, writer=None):
    original_image = fetch_image_from_url(image_url)
    second_image = fetch_image_from_url(second_image_url)
    if not original_image or not second_image:
        print("Failed to fetch images.")
        return
    fig, axes = plt.subplots(len(data_transforms.transforms),1, figsize=(5, 15))
    for i, transform in enumerate(data_transforms.transforms):
        try:
            transformed_image = transform(original_image.copy())
            transformed_image_np = np.array(transformed_image)
            axes[i].imshow(transformed_image_np)
            axes[i].set_title(str(transform))
            axes[i].axis('off')
        except Exception as e:
            print(f"Error applying transformation {i+1}: {e}")
            axes[i].set_title(f"Error {i+1}")
            axes[i].axis('off')
    plt.tight_layout()
    # if writer:
    #     writer.add_figure("Image Transformations", fig)
    plt.show()

def main():
    primary_image_url = 'https://post.medicalnewstoday.com/wp-content/uploads/sites/3/2020/05/GettyImages-114329113_slide-800x728.jpg'
    secondary_image_url = 'https://th.bing.com/th/id/R.ba1a0074c838cd4484e348f0f67cf74a?rik=Wu%2f1rcldfXD4jg&pid=ImgRaw&r=0'
    img2 = fetch_image_from_url(secondary_image_url)
    if not img2:
        print("Failed to fetch the secondary image.")
        return
    train_transforms = get_train_transforms(img2)
    writer = SummaryWriter(log_dir='./logs')
    visualize_transforms(train_transforms, primary_image_url, secondary_image_url, writer)
    writer.close()

if __name__ == "__main__":
    main()