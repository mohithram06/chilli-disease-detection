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
import streamlit.components.v1 as components

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ---------------- LANGUAGE ----------------
language = st.selectbox("🌐 Select Language", ["English", "Hindi", "Kannada"])

lang_map = {"English": "en-IN", "Hindi": "hi-IN", "Kannada": "kn-IN"}

# ---------------- VOICE SPEAK ----------------
def speak(text, lang):
    st.markdown(f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.lang = "{lang}";
    window.speechSynthesis.speak(msg);
    </script>
    """, unsafe_allow_html=True)

# ---------------- VOICE INPUT ----------------
def voice_ai_input(language):
    lang_code = lang_map[language]
    return components.html(f"""
    <button onclick="startRecognition()">🎤 Speak</button>
    <p id="out"></p>
    <script>
    function startRecognition() {{
        var rec = new webkitSpeechRecognition();
        rec.lang = "{lang_code}";
        rec.start();
        rec.onresult = function(e) {{
            var text = e.results[0][0].transcript;
            window.parent.postMessage({{type:'streamlit:setComponentValue', value:text}}, '*');
        }};
    }}
    </script>
    """, height=100)

# ---------------- NLP EXTRACT ----------------
def extract_details(text):
    name, field, plants = "", 1, 10
    text = text.lower()

    if "name" in text:
        words = text.split()
        if "name" in words:
            idx = words.index("name")
            if idx+1 < len(words):
                name = words[idx+1]

    field_match = re.search(r"(\d+)\s*(acre|acres)", text)
    if field_match:
        field = float(field_match.group(1))

    plant_match = re.search(r"(\d+)\s*(plant|plants)", text)
    if plant_match:
        plants = int(plant_match.group(1))

    return name, field, plants

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

# ---------------- UI ----------------
st.title("🌶️ Smart Farmer Assistant")

# Voice AI Input
st.markdown("## 🎤 Speak Details")
voice_text = voice_ai_input(language)

farmer_name, field_size, num_plants = "", 1.0, 10

if voice_text:
    st.success(f"You said: {voice_text}")
    farmer_name, field_size, num_plants = extract_details(voice_text)

# Manual override
farmer_name = st.text_input("👨‍🌾 Farmer Name", value=farmer_name)
field_size = st.number_input("🌾 Field Size (acre)", value=field_size)
num_plants = st.number_input("🌱 Plants", value=num_plants)

# Image input
image_file = st.camera_input("📷 Take Photo")

if image_file is None:
    image_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

# ---------------- MAIN ----------------
if image_file:

    img = Image.open(image_file).convert("RGB")
    st.image(img)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs[0], dim=0)
        predicted = torch.argmax(probs).item()

    disease = classes[predicted]
    info = disease_info[disease]

    st.success(f"Disease: {disease}")

    # Spray calc
    water = field_size * 200
    medicine = water * info["dose"]

    st.write(f"Water: {water} L")

    if medicine > 1000:
        st.write(f"Medicine: {medicine/1000:.2f} kg")
    else:
        st.write(f"Medicine: {medicine:.2f} g")

    # Voice result
    if st.button("🔊 Speak Result"):
        speak(f"Disease is {disease}. Use {info['treatment']}", lang_map[language])

    # AI Chat
    st.markdown("## 🤖 Ask AI")

    q = st.text_input("Ask question")

    if q:
        ans = farmer_ai_response(q, disease, info)
        st.info(ans)

    # Graph
    df = pd.DataFrame({"Class": classes, "Prob":[p.item()*100 for p in probs]})
    st.plotly_chart(px.bar(df, x="Class", y="Prob"))

    # WhatsApp
    msg = f"Disease: {disease}, Use {info['treatment']}"
    st.markdown(f"[📲 Share WhatsApp](https://wa.me/?text={urllib.parse.quote(msg)})")

    # PDF
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
            Paragraph(f"Farmer: {farmer_name}", styles["Normal"]),
            Paragraph(f"Field: {field_size} acre", styles["Normal"]),
            Paragraph(f"Disease: {disease}", styles["Normal"]),
            Paragraph(f"Medicine: {info['treatment']}", styles["Normal"]),
        ]

        doc.build(content)
        buffer.seek(0)
        return buffer

    pdf = create_pdf()

    st.download_button("📥 Download PDF", pdf, file_name="report.pdf")