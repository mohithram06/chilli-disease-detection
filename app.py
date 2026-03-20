import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import gdown
import os
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="Chilli Disease Detection", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: green;'>🌶️ Chilli Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.write("Upload a leaf image to detect disease and get exact treatment guidance.")

# Classes
classes = ["Early", "Healthy", "Mild", "Severe"]

# Disease info with DOSE
disease_info = {
    "Early": {
        "type": "Fungal Infection",
        "reason": "Initial fungal growth due to moisture",
        "treatment": "Mancozeb",
        "dose": 2  # g per liter
    },
    "Healthy": {
        "type": "No Disease",
        "reason": "Healthy plant",
        "treatment": "No spray needed",
        "dose": 0
    },
    "Mild": {
        "type": "Bacterial/Fungal",
        "reason": "Spreading infection",
        "treatment": "Copper Oxychloride",
        "dose": 3
    },
    "Severe": {
        "type": "Severe Fungal/Viral",
        "reason": "Heavy infection spread",
        "treatment": "Carbendazim",
        "dose": 2
    },
}

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

    if not os.path.exists("model.pth"):
        url = "https://drive.google.com/uc?id=1JF9vLsBaBM3oOwrFNcwfqWAJTww623yQ"
        gdown.download(url, "model.pth", quiet=False)

    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

# MAIN APP
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted = torch.argmax(probs).item()

    # Input
    num_plants = st.number_input("🌱 Enter number of infected plants", min_value=1, value=10)

    # Prediction
    st.markdown("## 🧠 Prediction Result")
    st.success(f"**Disease Stage:** {classes[predicted]}")

    confidence = probs[predicted].item() * 100
    st.info(f"Confidence: {confidence:.2f}%")

    # Disease details
    info = disease_info[classes[predicted]]

    st.markdown("## 🧬 Disease Details")
    st.write(f"**Type:** {info['type']}")
    st.write(f"**Cause:** {info['reason']}")

    # Treatment
    st.markdown("## 💊 Treatment Plan")

    if info["dose"] == 0:
        st.success("No spray needed. Maintain healthy practices.")
    else:
        st.warning(f"Use {info['treatment']} ({info['dose']}g per liter)")

    # Spray calculation
    st.markdown("## 🧪 Spray Calculation")

    water_per_plant = 0.5  # liters
    total_water = num_plants * water_per_plant

    if info["dose"] > 0:
        total_medicine = total_water * info["dose"]

        st.write(f"💧 Total water needed: **{total_water:.2f} liters**")
        st.write(f"💊 Medicine required: **{total_medicine:.2f} grams**")

        # Convert to kg if large
        if total_medicine > 1000:
            st.success(
                f"👉 Mix **{total_medicine/1000:.2f} kg {info['treatment']}** in **{total_water:.0f}L water**"
            )
        else:
            st.success(
                f"👉 Mix **{total_medicine:.0f} g {info['treatment']}** in **{total_water:.0f}L water**"
            )

    # Tips
    st.markdown("## ☔ Protection Tips")
    st.write("• Avoid watering on leaves")
    st.write("• Protect from heavy rain")
    st.write("• Remove infected leaves early")
    st.write("• Ensure good sunlight")

    # Graph
    st.markdown("## 📊 Prediction Confidence")

    prob_dict = {classes[i]: probs[i].item()*100 for i in range(len(classes))}
    df = pd.DataFrame(list(prob_dict.items()), columns=["Class", "Probability"])

    fig = px.bar(
        df,
        x="Class",
        y="Probability",
        color="Class",
        text="Probability",
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig)