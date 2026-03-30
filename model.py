import torch
from PIL import Image
import torchvision.transforms as transforms
import json
from model_architecture import PatchRangeTransformer

# Load classes
with open("classes.json", "r") as f:
    classes = json.load(f)

# Load model
model = PatchRangeTransformer(
    img_size=224, patch_size=16,
    num_classes=4, embed_dim=384,
    num_heads=12, num_layers=8, range_r=3
)

model.load_state_dict(torch.load("prt_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(1).item()
        confidence = probs[0][pred].item()

    return classes[pred], confidence