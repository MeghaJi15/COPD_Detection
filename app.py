import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="PulmoScan AI | COPD Detection",
    page_icon="🫁",
    layout="centered"
)

# Custom CSS for a professional medical look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #007bff;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    # Adding a small delay to simulate loading or handle complex models
    try:
        model = tf.keras.models.load_model("lung_xray_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -----------------------------
# Image Preprocessing
# -----------------------------
IMG_SIZE = 224

def preprocess_image(image):
    # Convert to RGB if it's grayscale or RGBA
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Sidebar Information
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864344.png", width=100)
    st.title("About PulmoScan")
    st.info("""
    **COPD (Chronic Obstructive Pulmonary Disease)** is a chronic inflammatory lung disease that causes obstructed airflow from the lungs. 
    
    This AI tool uses Deep Learning to assist in identifying potential indicators in X-ray imagery.
    """)
    st.divider()
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.")

# -----------------------------
# Main UI
# -----------------------------
st.title("🫁 PulmoScan: COPD Analysis")
st.write("Scan chest X-rays for Chronic Obstructive Pulmonary Disease using neural networks.")

# Layout Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Section")
    uploaded_file = st.file_uploader(
        "Drop X-ray here...",
        type=["jpg", "png", "jpeg"]
    )

with col2:
    st.subheader("Image Preview")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Upload", use_container_width=True)
    else:
        st.info("Please upload an image to see the preview.")

# -----------------------------
# Prediction Logic
# -----------------------------
if uploaded_file is not None:
    processed = preprocess_image(image)
    
    if st.button("Analyze X-ray"):
        if model is None:
            st.error("Model not loaded. Check your 'lung_xray_model.keras' file.")
        else:
            with st.spinner('Analyzing patterns in pulmonary tissue...'):
                # Simulate a slight delay for better UX (feels like "thinking")
                time.sleep(1.5) 
                prediction = model.predict(processed)[0][0]
                
            st.divider()
            
            # Result Display
            conf_percent = float(prediction * 100)
            
            if prediction > 0.5:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: #d9534f;">⚠ COPD Indicators Detected</h2>
                    <p>The model identified features consistent with COPD pathology.</p>
                </div>
                """, unsafe_allow_html=True)
                st.error(f"Confidence Score: {conf_percent:.2f}%")
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: #5cb85c;">✅ Normal Results</h2>
                    <p>No significant signs of COPD were detected in the provided scan.</p>
                </div>
                """, unsafe_allow_html=True)
                st.success(f"Confidence Score: {100 - conf_percent:.2f}%")
            
            st.write("Confidence Meter:")
            st.progress(float(prediction))

# Footer
st.markdown("---")
st.caption("Developed by AI Health Labs | Built with Streamlit & TensorFlow")