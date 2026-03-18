import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import gdown

# Page config
st.set_page_config(page_title="Chilli Disease Detection", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: green;'>🌶️ Chilli Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.write("Upload a leaf image to detect disease severity and get suggestions.")

# Classes
classes = ["Early", "Healthy", "Mild", "Severe"]

# Treatments
treatments = {
    "Early": "Use mild fungicide spray and monitor regularly.",
    "Healthy": "No disease detected. Maintain good farming practices.",
    "Mild": "Apply recommended fungicide and remove infected leaves.",
    "Severe": "Use strong fungicide and isolate infected plants immediately."
}

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

    # Download model
    url = "https://drive.google.com/uc?id=1JF9vLsBaBM3oOwrFNcwfqWAJTww623yQ"
    gdown.download(url, "model.pth", quiet=False)

    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Upload
uploaded_file = st.file_uploader(" Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted = torch.argmax(probs).item()

    # Prediction
    st.markdown("## Prediction Result")
    st.success(f"**Disease Stage:** {classes[predicted]}")

    # Confidence
    confidence = probs[predicted].item() * 100
    st.info(f"Confidence: {confidence:.2f}%")

    # Treatment
    st.markdown("## 🌿 Suggested Treatment")
    st.warning(treatments[classes[predicted]])

    # All probabilities
    st.markdown("## All Class Probabilities")
    import pandas as pd

prob_dict = {classes[i]: probs[i].item()*100 for i in range(len(classes))}
df = pd.DataFrame(list(prob_dict.items()), columns=["Class", "Probability"])

import plotly.express as px

fig = px.bar(
    df,
    x="Class",
    y="Probability",
    color="Class",
    text="Probability",
    title="Prediction Confidence (%)",
)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(
    yaxis_title="Confidence (%)",
    xaxis_title="Disease Class",
    showlegend=False
)

st.plotly_chart(fig)