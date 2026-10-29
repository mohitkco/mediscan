# predict.py - Arogya Diagnostics Pneumonia Screening
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import numpy as np

class ArogyaNetInference(nn.Module):
    def __init__(self):
        super(ArogyaNetInference, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=32)
        self.block2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_model(weights_path="arogya_net.pth"):
    model = ArogyaNetInference()
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    except:
        print("[!] Weights file not found or uninitialized. Using randomly initialized weights for demonstration.")
    model.eval()
    return model

def predict(image_path, threshold=0.35):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pneumonia_prob = probs[0][1].item()

    # Apply cost-matrix optimized threshold (e.g., 0.35 to prioritize sensitivity)
    prediction = "PNEUMONIA" if pneumonia_prob >= threshold else "NORMAL"
    return prediction, pneumonia_prob

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        pred, prob = predict(img_path)
        print(f"Prediction: {pred} (Confidence: {prob:.4f})")
    else:
        print("Usage: python predict.py <path_to_xray.jpg>")
