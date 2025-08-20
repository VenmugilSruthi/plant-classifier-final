import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2 # Import for OpenCV image processing
import base64 # Import to handle the background image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plant Classifier",
    page_icon="🌿",
    layout="wide"
)

# --- CUSTOM CSS FOR HOVER EFFECTS AND STYLING ---
st.markdown("""
<style>
/* Class for the hover effect on images */
.hover-effect {
    transition: transform .2s; /* Animation */
    border-radius: 10px; /* Rounded corners for the images */
}
.hover-effect:hover {
    transform: scale(1.05); /* (105% zoom - Feel free to change this value) */
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
}
/* Style for the main content block to improve readability with background */
.main .block-container {
    background-color: rgba(0, 0, 0, 0.6); /* Dark semi-transparent overlay */
    backdrop-filter: blur(5px); /* Blur effect */
    padding: 2rem;
    border-radius: 10px;
}
/* NEW: This rule hides the fullscreen button that appears on hover */
[data-testid="stImageToolbar"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# --- FUNCTION TO ADD BACKGROUND IMAGE ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    """Reads a binary file and returns its base64 encoded string."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """Sets a background image."""
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp > header {{
        background-color: transparent;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function to set the background
try:
    set_png_as_page_bg('bg.jpg')
except FileNotFoundError:
    st.warning("bg.jpg not found. Please add it to the folder for the background to work.")


# --- SIDEBAR ---
st.sidebar.title("About")
st.sidebar.info(
    """
    This application demonstrates a plant classification model using Deep Learning.
    - **Model:** MobileNetV2 (fine-tuned)
    - **Framework:** TensorFlow/Keras
    - **App:** Streamlit
    """
)
st.sidebar.success("Project for Image and Video Analytics Course")

# --- CLASS NAMES ---
CLASS_NAMES = {
    "0": "aloevera", "1": "banana", "2": "bilimbi", "3": "cantaloupe", "4": "cassava", "5": "coconut",
    "6": "corn", "7": "cucumber", "8": "curcuma", "9": "eggplant", "10": "galangal", "11": "ginger",
    "12": "guava", "13": "kale", "14": "longbeans", "15": "mango", "16": "melon", "17": "orange",
    "18": "paddy", "19": "papaya", "20": "peper chili", "21": "pineapple", "22": "pomelo",
    "23": "shallot", "24": "soybeans", "25": "spinach", "26": "sweet potatoes", "27": "tobacco",
    "28": "waterapple", "29": "watermelon"
}

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model from disk."""
    try:
        model = tf.keras.models.load_model('plant_classification_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- IMAGE PROCESSING FUNCTIONS ---
def apply_clahe(pil_img):
    cv_img = np.array(pil_img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(cv_img)
    return Image.fromarray(enhanced_img)

def detect_edges(pil_img):
    cv_img = np.array(pil_img.convert('L'))
    edges = cv2.Canny(cv_img, 100, 200)
    return Image.fromarray(edges)

def remove_background(pil_img):
    cv_img = np.array(pil_img.convert('RGB'))
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(cv_img, cv_img, mask=mask)
    return Image.fromarray(result)

# --- HELPER FUNCTION FOR PREDICTION ---
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- MAIN INTERFACE ---
st.title("🌿 Plant Type Classifier & Image Analyzer")
st.write("Upload an image of a plant to classify it and apply various image processing techniques.")

uploaded_file = st.file_uploader("Upload your image here...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    st.write("---")
    st.markdown("### 1. Model Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_file)
        # Applying the hover effect class to the image
        st.markdown('<div class="hover-effect">', unsafe_allow_html=True)
        st.image(image, caption='Original Uploaded Image', width=300)
        st.markdown('</div>', unsafe_allow_html=True)


    with col2:
        with st.spinner('Classifying...'):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            top_k = 3
            top_indices = np.argsort(prediction[0])[-top_k:][::-1]

            st.subheader("Top Predictions:")
            top_class_name = CLASS_NAMES[str(top_indices[0])]
            top_confidence = prediction[0][top_indices[0]] * 100
            st.success(f"**{top_class_name.capitalize()}** ({top_confidence:.2f}%)")
            st.write("---")

            chart_data = []
            for i in top_indices:
                class_name = CLASS_NAMES[str(i)]
                confidence = prediction[0][i] * 100
                chart_data.append([class_name.capitalize(), confidence])
            df = pd.DataFrame(chart_data, columns=['Plant Type', 'Confidence'])
            st.bar_chart(df.set_index('Plant Type'))

    st.write("---")
    st.markdown("### 2. Image Analytics Showcase")
    with st.expander("Click here to see various image processing techniques applied"):
        
        st.markdown("#### Contrast Enhancement (CLAHE)")
        enhanced_image = apply_clahe(image)
        st.markdown('<div class="hover-effect">', unsafe_allow_html=True)
        st.image(enhanced_image, caption='Contrast Enhanced Image', width=300)
        st.markdown('</div>', unsafe_allow_html=True)


        st.markdown("#### Edge Detection (Canny)")
        edge_image = detect_edges(image)
        st.markdown('<div class="hover-effect">', unsafe_allow_html=True)
        st.image(edge_image, caption='Edge Detected Image', width=300)
        st.markdown('</div>', unsafe_allow_html=True)


        st.markdown("#### Background Removal (Green-based Segmentation)")
        bg_removed_image = remove_background(image)
        st.markdown('<div class="hover-effect">', unsafe_allow_html=True)
        st.image(bg_removed_image, caption='Background Removed Image', width=300)
        st.markdown('</div>', unsafe_allow_html=True)
        st.info("Note: This is a simple segmentation that isolates green pixels.")

st.write("---")
with st.expander("Showcase: Click here to see the Project Details"):
    st.markdown("### 📄 Project Overview")
    st.write("""
    - **Problem Definition:** The goal is to accurately classify plant images into 30 categories using Deep Learning, automating a key task in agriculture and botany.
    - **Dataset:** The "Plants Type Datasets" from Kaggle was used, featuring a diverse set of images.
    - **Preprocessing:** Steps included resizing images to 224x224, normalizing pixel values, and applying data augmentation (rotation, zoom, etc.) to enhance model robustness.
    """)
    st.markdown("### ⚙️ Model and Performance")
    st.write("""
    - **Model Architecture:** The project uses a **MobileNetV2** model with transfer learning. The pre-trained model was fine-tuned on the plant dataset for this specific task.
    - **Evaluation:** The model achieved a **final validation accuracy of ~89%**. This demonstrates its ability to generalize to new, unseen images.
    """)
    try:
        st.image('training_history.png', caption='Model Training and Validation History')
    except FileNotFoundError:
        st.warning("'training_history.png' not found. Please add it for a complete report.")
    st.markdown("### 💡 Findings and Interpretation")
    st.write("""
    - The model performs well on clear, centered images but can be challenged by poor lighting or unusual angles.
    - The use of transfer learning significantly reduced training time and improved accuracy compared to training from scratch.
    - **Real-world Application:** This tool can be integrated into a mobile app for farmers, researchers, or gardeners for quick plant identification in the field.
    """)
