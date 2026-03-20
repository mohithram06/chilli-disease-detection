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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- LANGUAGE ----------------
language = st.selectbox("🌐 Select Language", ["English", "Hindi", "Kannada"])

translations = {
    "English": {
        "title": "Chilli Disease Detection",
        "camera": "Take Photo",
        "upload": "Upload Image",
        "plants": "Number of infected plants",
        "disease": "Disease",
        "confidence": "Confidence",
        "treatment": "Treatment Plan",
        "water": "Total water needed",
        "medicine": "Medicine required",
        "download": "Download PDF",
        "share": "Share WhatsApp",
        "speak": "Speak Result"
    },
    "Hindi": {
        "title": "मिर्ची रोग पहचान",
        "camera": "फोटो लें",
        "upload": "छवि अपलोड करें",
        "plants": "संक्रमित पौधों की संख्या",
        "disease": "रोग",
        "confidence": "विश्वास स्तर",
        "treatment": "उपचार योजना",
        "water": "कुल पानी",
        "medicine": "दवा मात्रा",
        "download": "पीडीएफ डाउनलोड करें",
        "share": "व्हाट्सएप शेयर करें",
        "speak": "बोलें"
    },
    "Kannada": {
        "title": "ಮೆಣಸಿನಕಾಯಿ ರೋಗ ಗುರುತು",
        "camera": "ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ",
        "upload": "ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
        "plants": "ಸಂಕ್ರಮಿತ ಸಸಿಗಳ ಸಂಖ್ಯೆ",
        "disease": "ರೋಗ",
        "confidence": "ವಿಶ್ವಾಸ ಮಟ್ಟ",
        "treatment": "ಚಿಕಿತ್ಸೆ ಯೋಜನೆ",
        "water": "ಒಟ್ಟು ನೀರು",
        "medicine": "ಔಷಧ ಪ್ರಮಾಣ",
        "download": "PDF ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "share": "ವಾಟ್ಸಾಪ್ ಹಂಚಿಕೊಳ್ಳಿ",
        "speak": "ಮಾತನಾಡಿ"
    }
}

text = translations[language]

# ---------------- PAGE ----------------
st.set_page_config(page_title="Chilli Disease", layout="centered")
st.markdown(f"<h1 style='text-align:center; color:green;'>🌶️ {text['title']}</h1>", unsafe_allow_html=True)

# ---------------- SPEAK ----------------
def speak(text_to_speak, lang="en-IN"):
    st.markdown(
        f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{text_to_speak}");
        msg.lang = "{lang}";
        window.speechSynthesis.speak(msg);
        </script>
        """,
        unsafe_allow_html=True
    )

lang_map = {"English": "en-IN", "Hindi": "hi-IN", "Kannada": "kn-IN"}

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
image_file = st.camera_input(f"📷 {text['camera']}")

if image_file is None:
    image_file = st.file_uploader(f"📤 {text['upload']}", type=["jpg","png","jpeg"])

# ---------------- MAIN ----------------
if image_file:

    img = Image.open(image_file).convert("RGB")
    st.image(img, use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted = torch.argmax(probs).item()

    num_plants = st.number_input(f"🌱 {text['plants']}", min_value=1, value=10)

    # Prediction
    st.success(f"{text['disease']}: {classes[predicted]}")
    confidence = probs[predicted].item()*100
    st.info(f"{text['confidence']}: {confidence:.2f}%")

    info = disease_info[classes[predicted]]

    # Voice
    if st.button(f"🔊 {text['speak']}"):
        speak(f"Disease is {classes[predicted]}. Use {info['treatment']}", lang_map[language])

    # Spray
    water = num_plants * 0.5
    medicine = water * info["dose"]

    st.markdown(f"## 💊 {text['treatment']}")

    if info["dose"] == 0:
        st.success("No spray needed")
    else:
        if medicine > 1000:
            st.success(f"👉 Mix {medicine/1000:.2f} kg {info['treatment']} in {water:.0f}L water")
        else:
            st.success(f"👉 Mix {medicine:.0f} g {info['treatment']} in {water:.0f}L water")

    st.write(f"💧 {text['water']}: {water:.2f} L")
    st.write(f"💊 {text['medicine']}: {medicine:.2f} g")

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
    st.markdown(f"[📲 {text['share']}]({url})")

    # PDF (FIXED)
    def create_pdf_buffer(report_text):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = []
        for line in report_text.split("\n"):
            content.append(Paragraph(line, styles["Normal"]))
            content.append(Spacer(1, 10))

        doc.build(content)
        buffer.seek(0)
        return buffer

    pdf_text = f"""
    Disease: {classes[predicted]}
    Type: {info['type']}
    Treatment: {info['treatment']}
    Water: {water}L
    Medicine: {medicine}g
    """

    pdf_buffer = create_pdf_buffer(pdf_text)

    st.download_button(
        label=f"📥 {text['download']}",
        data=pdf_buffer,
        file_name="report.pdf",
        mime="application/pdf"
    )