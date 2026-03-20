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
from io import BytesIO
import re

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ---------------- SETTINGS ----------------
st.set_page_config(page_title="Smart Farmer AI", layout="centered")

# ---------------- TITLE ----------------
st.title("🌶️ Smart Chilli Disease Detection")

st.info("💡 Tip: You can use your mobile keyboard mic 🎤 to speak instead of typing")

# ---------------- INPUTS ----------------
farmer_name = st.text_input("👨‍🌾 Farmer Name")
field_size = st.number_input("🌾 Field Size (acre)", min_value=0.1, value=1.0)
num_plants = st.number_input("🌱 Number of infected plants", min_value=1, value=10)

# ---------------- AI TEXT EXTRACT (optional typing support) ----------------
def extract_details(text):
    if not isinstance(text, str):
        return "", 1.0, 10

    text = text.lower()
    name, field, plants = "", 1.0, 10

    name_match = re.search(r"name\s*(is)?\s*(\w+)", text)
    if name_match:
        name = name_match.group(2)

    field_match = re.search(r"(\d+)\s*(acre|acres)", text)
    if field_match:
        field = float(field_match.group(1))

    plant_match = re.search(r"(\d+)\s*(plant|plants)", text)
    if plant_match:
        plants = int(plant_match.group(1))

    return name, field, plants

# ---------------- MODEL ----------------
classes = ["Early", "Healthy", "Mild", "Severe"]

disease_info = {
    "Early": {"type": "Fungal", "reason": "Moisture", "treatment": "Mancozeb", "dose": 2},
    "Healthy": {"type": "None", "reason": "Healthy", "treatment": "None", "dose": 0},
    "Mild": {"type": "Bacterial", "reason": "Spread", "treatment": "Copper Oxychloride", "dose": 3},
    "Severe": {"type": "Severe Infection", "reason": "High spread", "treatment": "Carbendazim", "dose": 2},
}

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

    if not os.path.exists("model.pth"):
        gdown.download("https://drive.google.com/uc?id=1JF9vLsBaBM3oOwrFNcwfqWAJTww623yQ", "model.pth")

    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- IMAGE INPUT ----------------
image_file = st.camera_input("📷 Take Photo")

if image_file is None:
    image_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

# ---------------- AI CHAT ----------------
def farmer_ai_response(q, disease, info):
    q = q.lower()
    if "medicine" in q:
        return f"Use {info['treatment']} {info['dose']} grams per liter."
    elif "days" in q:
        return "Spray every 4 to 5 days."
    elif "rain" in q:
        return "Avoid spraying before rain."
    elif "water" in q:
        return "Water near roots only."
    elif "disease" in q:
        return f"Disease is {disease}"
    return "Follow treatment instructions properly."

# ---------------- MAIN ----------------
if image_file:

    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs[0], dim=0)
        predicted = torch.argmax(probs).item()

    disease = classes[predicted]
    info = disease_info[disease]

    st.success(f"🌿 Disease Detected: {disease}")

    confidence = probs[predicted].item()*100
    st.info(f"Confidence: {confidence:.2f}%")

    # ---------------- SPRAY CALCULATION ----------------
    st.markdown("## 💊 Spray Calculation")

    water = field_size * 200
    medicine = water * info["dose"]

    st.write(f"🌾 Field Size: {field_size} acre")
    st.write(f"💧 Water Needed: {water:.2f} L")

    if medicine > 1000:
        st.write(f"💊 Medicine Required: {medicine/1000:.2f} kg")
    else:
        st.write(f"💊 Medicine Required: {medicine:.2f} g")

    st.success(f"👉 Use {info['treatment']} mixed in water")

    # ---------------- GRAPH ----------------
    df = pd.DataFrame({
        "Class": classes,
        "Probability": [p.item()*100 for p in probs]
    })

    fig = px.bar(df, x="Class", y="Probability", color="Class", text="Probability")
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig)

    # ---------------- AI CHAT ----------------
    st.markdown("## 🤖 Ask AI Assistant")

    user_q = st.text_input("Ask your question")

    if user_q:
        answer = farmer_ai_response(user_q, disease, info)
        st.info(f"🤖 {answer}")

    # ---------------- WHATSAPP ----------------
    msg = f"Disease: {disease}, Use {info['treatment']}"
    st.markdown(f"[📲 Share on WhatsApp](https://wa.me/?text={urllib.parse.quote(msg)})")

    # ---------------- PDF ----------------
    def create_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        img_path = "temp.jpg"
        img.save(img_path)

        content = [
            Paragraph("Chilli Disease Report", styles["Title"]),
            Spacer(1,10),
            RLImage(img_path, 3*inch,3*inch),
            Spacer(1,10),
            Paragraph(f"Farmer: {farmer_name}", styles["Normal"]),
            Paragraph(f"Field Size: {field_size} acre", styles["Normal"]),
            Paragraph(f"Plants: {num_plants}", styles["Normal"]),
            Paragraph(f"Disease: {disease}", styles["Normal"]),
            Paragraph(f"Type: {info['type']}", styles["Normal"]),
            Paragraph(f"Cause: {info['reason']}", styles["Normal"]),
            Paragraph(f"Treatment: {info['treatment']}", styles["Normal"]),
            Paragraph(f"Water: {water:.2f} L", styles["Normal"]),
            Paragraph(f"Medicine: {medicine:.2f} g", styles["Normal"]),
            Spacer(1,10),
            Paragraph("Instructions:", styles["Heading2"]),
            Paragraph("• Spray every 4-5 days", styles["Normal"]),
            Paragraph("• Avoid rain after spraying", styles["Normal"]),
            Paragraph("• Remove infected leaves", styles["Normal"]),
            Paragraph("• Ensure proper sunlight", styles["Normal"]),
        ]

        doc.build(content)
        buffer.seek(0)
        return buffer

    pdf = create_pdf()

    st.download_button("📥 Download Full Report", pdf, file_name="report.pdf")