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

# ---------------- SETTINGS ----------------
st.set_page_config(page_title="Smart Farmer AI", layout="centered")

# ---------------- LANGUAGE ----------------
language = st.selectbox("🌐 Select Language", ["English", "Hindi", "Kannada"])
lang_map = {"English": "en-IN", "Hindi": "hi-IN", "Kannada": "kn-IN"}

# ---------------- VOICE OUTPUT ----------------
def speak(text, lang):
    st.markdown(f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.lang = "{lang}";
    window.speechSynthesis.speak(msg);
    </script>
    """, unsafe_allow_html=True)

# ---------------- REAL-TIME VOICE ----------------
def real_time_voice(language):
    lang_code = lang_map[language]

    return components.html(f"""
    <button onclick="startListening()">🎤 Start Speaking</button>
    <button onclick="stopListening()">🛑 Stop</button>
    <p><b>Live Speech:</b></p>
    <div id="liveText" style="color:green;"></div>

    <script>
    var recognition;

    function startListening() {{
        recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "{lang_code}";

        recognition.onresult = function(event) {{
            let finalText = "";
            for (let i = event.resultIndex; i < event.results.length; ++i) {{
                finalText += event.results[i][0].transcript;
            }}
            document.getElementById("liveText").innerHTML = finalText;

            window.parent.postMessage({{
                type: "streamlit:setComponentValue",
                value: finalText
            }}, "*");
        }};

        recognition.start();
    }}

    function stopListening() {{
        if (recognition) recognition.stop();
    }}
    </script>
    """, height=200)

# ---------------- NLP ----------------
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

st.warning("🎤 Voice works best in Chrome browser")

# Voice
voice_text = real_time_voice(language)

farmer_name, field_size, num_plants = "", 1.0, 10

if isinstance(voice_text, str) and voice_text.strip():
    st.success(f"You said: {voice_text}")
    farmer_name, field_size, num_plants = extract_details(voice_text)

# Manual fallback
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

    # Spray calculation
    water = field_size * 200
    medicine = water * info["dose"]

    st.write(f"💧 Water: {water:.2f} L")

    if medicine > 1000:
        st.write(f"💊 Medicine: {medicine/1000:.2f} kg")
    else:
        st.write(f"💊 Medicine: {medicine:.2f} g")

    # Voice answer
    if st.button("🔊 Speak Result"):
        speak(f"Disease is {disease}. Use {info['treatment']}", lang_map[language])

    # AI Chat
    st.markdown("## 🤖 Ask AI")

    user_q = st.text_input("Ask your question")

    if user_q:
        ans = farmer_ai_response(user_q, disease, info)
        st.info(ans)

    # Voice auto response
    if isinstance(voice_text, str) and voice_text.strip():
        response = farmer_ai_response(voice_text, disease, info)
        st.info(f"🤖 {response}")
        speak(response, lang_map[language])

    # Graph
    df = pd.DataFrame({"Class": classes, "Probability":[p.item()*100 for p in probs]})
    st.plotly_chart(px.bar(df, x="Class", y="Probability", color="Class"))

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
            Spacer(1,10),
            Paragraph(f"Farmer: {farmer_name}", styles["Normal"]),
            Paragraph(f"Field: {field_size} acre", styles["Normal"]),
            Paragraph(f"Plants: {num_plants}", styles["Normal"]),
            Paragraph(f"Disease: {disease}", styles["Normal"]),
            Paragraph(f"Treatment: {info['treatment']}", styles["Normal"]),
            Paragraph(f"Water: {water} L", styles["Normal"]),
            Paragraph(f"Medicine: {medicine} g", styles["Normal"]),
            Spacer(1,10),
            Paragraph("Instructions:", styles["Heading2"]),
            Paragraph("• Spray every 4-5 days", styles["Normal"]),
            Paragraph("• Avoid rain after spraying", styles["Normal"]),
            Paragraph("• Remove infected leaves", styles["Normal"]),
        ]

        doc.build(content)
        buffer.seek(0)
        return buffer

    pdf = create_pdf()

    st.download_button("📥 Download PDF", pdf, file_name="report.pdf")