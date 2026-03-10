import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
# Load TFLite Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="lung_model.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

interpreter, input_details, output_details = load_model()

# -----------------------------
# Image Preprocessing
# -----------------------------
IMG_SIZE = 224

def preprocess_image(image):
    
    # Convert to RGB if grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize
    image = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy
    image = np.array(image)
    
    # MobileNetV2 preprocessing
    image = preprocess_input(image)
    
    # Add batch dimension
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
        
        if interpreter is None:
            st.error("Model not loaded. Check your 'lung_model.tflite' file.")
        
        else:
            with st.spinner('Analyzing patterns in pulmonary tissue...'):
                
                time.sleep(1.5)
                
                try:
                    interpreter.set_tensor(input_details[0]['index'], processed.astype(np.float32))
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()
                
            st.divider()
            
            # -----------------------------
            # Multi-class Prediction Logic
            # -----------------------------
            classes = ["Normal", "COPD", "Pneumonia"]

            class_index = np.argmax(prediction)
            predicted_class = classes[class_index]

            confidence = float(np.max(prediction)) * 100

            st.markdown(f"""
            <div class="prediction-card">
                <h2>Prediction: {predicted_class}</h2>
                <p>The AI model analyzed the uploaded chest X-ray.</p>
            </div>
            """, unsafe_allow_html=True)

            if predicted_class == "Normal":
                st.success(f"Confidence Score: {confidence:.2f}%")
            elif predicted_class == "COPD":
                st.error(f"Confidence Score: {confidence:.2f}%")
            else:
                st.warning(f"Confidence Score: {confidence:.2f}%")

            st.write("Confidence Meter:")
            st.progress(float(confidence/100))

# Footer
st.markdown("---")
st.caption("Developed by AI Health Labs | Built with Streamlit & TensorFlow")
