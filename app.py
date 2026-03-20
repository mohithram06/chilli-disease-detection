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

disease_info = {
    "Early": {
        "type": "Fungal Infection",
        "reason": "Caused by excess moisture and poor air circulation.",
        "spray": "Mancozeb or Carbendazim",
        "dose": "2 grams per liter of water",
        "days": "Spray every 5 days for 2 weeks",
        "watering": "Avoid overwatering. Keep soil slightly dry.",
        "rain": "Avoid spraying before rain. Reapply after rain."
    },
    "Healthy": {
        "type": "No Disease",
        "reason": "Plant is healthy.",
        "spray": "No spray needed",
        "dose": "-",
        "days": "-",
        "watering": "Normal watering",
        "rain": "No precautions needed"
    },
    "Mild": {
        "type": "Bacterial Infection",
        "reason": "Spreads through water splash and infected tools.",
        "spray": "Copper oxychloride",
        "dose": "3 grams per liter",
        "days": "Spray every 4–5 days",
        "watering": "Avoid leaf wetting",
        "rain": "Cover plants if possible"
    },
    "Severe": {
        "type": "Viral Infection",
        "reason": "Spread by insects like aphids/whiteflies.",
        "spray": "Imidacloprid (insecticide)",
        "dose": "0.5 ml per liter",
        "days": "Spray every 3 days for 2 weeks",
        "watering": "Normal watering but avoid stress",
        "rain": "Avoid rain exposure after spray"
    }
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
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted = torch.argmax(probs).item()

num_plants = st.number_input("🌱 Enter number of infected plants", min_value=1, value=10)

    # Prediction
    st.markdown("## 🧠 Prediction Result")
    st.success(f"**Disease Stage:** {classes[predicted]}")
   
   info = disease_info[classes[predicted]]

st.markdown("## 🧬 Disease Details")

st.write(f"**Type:** {info['type']}")
st.write(f"**Cause:** {info['reason']}")

st.markdown("## 💊 Treatment Plan")

st.write(f"**Recommended Spray:** {info['spray']}")
st.write(f"**Dosage:** {info['dose']}")
st.write(f"**Duration:** {info['days']}")

# Calculation (simple logic)
water_needed = num_plants * 0.5  # liters per plant (approx)

st.markdown("## 📊 Spray Calculation")
st.info(f"For {num_plants} plants, you need approx **{water_needed:.1f} liters** of solution.")

st.markdown("## 💧 Watering Advice")
st.write(info['watering'])

st.markdown("## 🌧️ Rain Protection")
st.write(info['rain'])

    # Confidence
    confidence = probs[predicted].item() * 100
    st.info(f"Confidence: {confidence:.2f}%")

    # Treatment
    st.markdown("## 🌿 Suggested Treatment")
    st.warning(treatments[classes[predicted]])

    # All probabilities
    st.markdown("## 📊 All Class Probabilities")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probs[i]*100:.2f}%")