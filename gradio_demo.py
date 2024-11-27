import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import torch.nn.functional as F
import timm
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

MODEL_TYPE = "resnet50"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:", device)


model = timm.create_model(MODEL_TYPE, pretrained=True, num_classes=7)

# Load the saved model checkpoint
checkpoint = torch.load(r'.\old_models\resnet50_using_timm_mean_std_interpol\model_resnet50_FocalLoss_20241109_124133.pth')

# Load the model state
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  


# Define label mapping
label_mapping_ham = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}
inv_label_mapping = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
}

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

def predict_image(img):
    # Apply the necessary transforms
    # print(f"{img=}")
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    print(f"{img_tensor.shape=}")
    print(f"{type(img_tensor)=}")
    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)
        print(f"{outputs.shape}")
        probabilities = F.softmax(outputs[0], dim=0)  # Convert outputs to probabilities
    # Get the top 3 predictions
    print(f"{probabilities.shape=}")
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    confidences = {inv_label_mapping[catid]: float(prob) for prob, catid in zip(top3_prob.cpu().numpy(), top3_catid.cpu().numpy())}
    print(f"{confidences=}")
    return confidences

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title=" Dermatological Lesions Detection and classification",
    description="Upload an image!"
)

if __name__ == "__main__":
    demo.launch(share=True)  
