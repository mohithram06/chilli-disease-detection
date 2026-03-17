import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

st.title("🌶️ Chilli Disease Detection")

classes = ["chilli early", "chilli healthy", "chilli mild", "chilli severe"]

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    st.write("Prediction:", classes[predicted.item()])