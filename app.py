import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import gdown
import os
import pandas as pd
import plotly.express as px
import urllib.parse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------- LANGUAGE ----------
language = st.selectbox("🌐 Select Language", ["English", "Hindi", "Kannada"])

translations = {
    "English": {"title": "Chilli Disease Detection"},
    "Hindi": {"title": "मिर्ची रोग पहचान"},
    "Kannada": {"title": "ಮೆಣಸಿನಕಾಯಿ ರೋಗ ಗುರುತು"}
}

text = translations[language]

# ---------- PAGE ----------
st.set_page_config(page_title="Chilli Disease", layout="centered")
st.markdown(f"<h1 style='text-align: center; color: green;'>🌶️ {text['title']}</h1>", unsafe_allow_html=True)

# ---------- CLASSES ----------
classes = ["Early", "Healthy", "Mild", "Severe"]

# ---------- DISEASE INFO ----------
disease_info = {
    "Early": {"type": "Fungal", "reason": "Moisture", "treatment": "Mancozeb", "dose": 2},
    "Healthy": {"type": "None", "reason": "Healthy", "treatment": "None", "dose": 0},
    "Mild": {"type": "Bacterial", "reason": "Spread", "treatment": "Copper Oxychloride", "dose": 3},
    "Severe": {"type": "Severe Infection", "reason": "High spread", "treatment": "Carbendazim", "dose": 2},
}

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

    if not os.path.exists("model.pth"):
        gdown.download("https://drive.google.com/uc?id=1JF9vLsBaBM3oOwrFNcwfqWAJTww623yQ", "model.pth", quiet=False)

    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------- IMAGE INPUT ----------
image_file = st.camera_input("📷 Take Photo")

if image_file is None:
    image_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

# ---------- MAIN ----------
if image_file:

    img = Image.open(image_file).convert("RGB")
    st.image(img, use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted = torch.argmax(probs).item()

    num_plants = st.number_input("🌱 Number of infected plants", min_value=1, value=10)

    # Prediction
    st.success(f"Disease: {classes[predicted]}")
    confidence = probs[predicted].item()*100
    st.info(f"Confidence: {confidence:.2f}%")

    info = disease_info[classes[predicted]]

    # Spray calc
    water = num_plants * 0.5
    medicine = water * info["dose"]

    st.markdown("## 💊 Treatment")

    if info["dose"] == 0:
        st.success("No spray needed")
    else:
        st.warning(f"{info['treatment']} ({info['dose']}g per liter)")

        if medicine > 1000:
            st.success(f"👉 Mix {medicine/1000:.2f} kg {info['treatment']} in {water:.0f}L water")
        else:
            st.success(f"👉 Mix {medicine:.0f} g {info['treatment']} in {water:.0f}L water")

    # Graph
    df = pd.DataFrame({
        "Class": classes,
        "Probability": [p.item()*100 for p in probs]
    })

    fig = px.bar(df, x="Class", y="Probability", color="Class", text="Probability")
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig)

    # WhatsApp
    report = f"Disease: {classes[predicted]}, Use {info['treatment']}"
    url = "https://wa.me/?text=" + urllib.parse.quote(report)
    st.markdown(f"[📲 Share WhatsApp]({url})")

    # PDF
    def create_pdf(text):
        path = "/mnt/data/report.pdf"
        doc = SimpleDocTemplate(path)
        styles = getSampleStyleSheet()
        content = [Paragraph(line, styles["Normal"]) for line in text.split("\n")]
        doc.build(content)
        return path

    pdf_text = f"""
    Disease: {classes[predicted]}
    Type: {info['type']}
    Treatment: {info['treatment']}
    Water: {water}L
    Medicine: {medicine}g
    """

    pdf_path = create_pdf(pdf_text)

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download PDF", f, file_name="report.pdf")