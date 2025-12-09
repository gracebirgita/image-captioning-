import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import os
import joblib

from captioning_model_classes import(
    CaptionTrainer, 
    TransformerDecoder, 
    DecoderLayer, 
    TransformerEncoderBlock,
    WarmUpLinear,
    build_visual_encoder_all
)


# Set page configuration
st.set_page_config(
    page_title="EfficientNet-Transformer Social Media Captioning",
    layout="wide",
    page_icon="📸"
)

# --- MODEL LOADING & CACHING ---
@st.cache_resource
def load_keras_model():
    model_path= "models/model_epoch22.keras"

    custom_objects={
        "CaptionTrainer": CaptionTrainer,
        "TransformerDecoder": TransformerDecoder,
        "DecoderLayer": DecoderLayer,
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "WarmUpLinear": WarmUpLinear,
        "build_visual_encoder_all": build_visual_encoder_all,
    }

    try:
        # load model
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        st.success("Model success loaded")
        return model
    except FileNotFoundError:
        st.error(f"File model not found : {model_path}")
    except Exception as e:
        st.error(f"Fail load keras model : {e}")
        return None


# --- PREPROCESSING & FEATURE EXTRACTION (No changes) ---
def preprocess(img):
    pass #sesuaikan ipynb
    # img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # img_proc = clahe.apply(img_gray)
    # return img_proc

def extract_features(img):
    pass # sesuaikan code build all encoder complete
    # orb = cv2.ORB_create(nfeatures=100)
    # kp, des = orb.detectAndCompute(img, None)
    # if des is None: des = np.zeros((1,32), dtype=np.uint8)
    # flat = des.flatten()
    # if flat.shape[0] < 3200: flat = np.pad(flat, (0, 3200 - flat.shape[0]))
    # if flat.shape[0] > 3200: flat = flat[:3200]
    # return flat

# --- LLM SIMULATION FUNCTION ---
def generate_caption(model, image_tensor, tokenizer, max_capt_len):
    """Fungsi Placeholder untuk meniru output caption asli"""
    pass

def generate_llm_caption():
    """Fungsi Placeholder untuk meniru output dari Model LLM Caption Generator"""
    pass

# ----------------------------------------------------
# APLIKASI UTAMA STREAMLIT
# ----------------------------------------------------

st.title("📸 Social Media Post Captioning")
st.markdown("---")


model_captioning = load_keras_model()
# 1. Upload Image (Di kolom Kiri)
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded_file and model_captioning is not None:
    img = Image.open(uploaded_file)
    
    # Menggunakan st.columns untuk tata letak yang lebih baik
    col_img, col_results = st.columns([1, 1.5]) 

    # --- Kolom Kiri: Display Gambar ---
    with col_img:
        st.subheader("Uploaded Image")
        
        st.image(img, caption="Uploaded Image", use_container_width=False)
        st.markdown("---")
        
        # 2. Tombol Generate Caption (Pemicu)
        if st.button("▶️ Generate Caption", type="primary"):
            st.session_state['run_analysis'] = True
            
        if 'run_analysis' not in st.session_state:
             st.session_state['run_analysis'] = False

    # --- Kolom Kanan: Hasil Analisis ---
    with col_results:
        st.subheader("Analysis Result")

        # Inisialisasi placeholder
        placeholder_caption = st.empty()
        placeholder_llm = st.empty()
        
        placeholder_caption.info("Press 'Generate Caption' to generate capton.")
        placeholder_llm.markdown("*(loading caption result...)*")

        # 3. Logika Analisis (Terpicu oleh Tombol)
        if st.session_state.run_analysis:
            
            # --- Tahap CV (Object Detection/Klasifikasi) ---
            with st.spinner('1/2. Load basic caption from image...'):
                img_preprocessed = preprocess(img)
                caption=generate_caption(model_captioning, img_preprocessed,tokenizer=None, max_capt_len=20)

                # features = extract_features(img_proc)
                # features_full = np.concatenate([features, np.zeros(8), np.zeros(512)]).reshape(1,-1)
                
                # Prediction
                # pred = svm_model.predict(features_full)
                # labels = le.inverse_transform(pred)
                # labels_str = ', '.join([str(label) for label in labels])

            # 3. Kasi placeholder caption generated dr model murni
            placeholder_caption.success(f"Basic caption result: ")
            st.markdown(f"**Caption: ** *{caption}*")

            # placeholder_cv.success(f"✅ CV Model Detected Objects:")
            # st.markdown(f"**Objek Terdeteksi:** `{labels_str}`")

            # --- Tahap NLP (Caption Generation) ---
            with st.spinner('2/2. Load Engaging Caption...'):
                pass
                # 4. Kasi text buat munculin caption dr LLM nya
                # llm_caption = generate_llm_caption(labels)

            # placeholder_llm.success(f"🤖 LLM Generated Caption:")
            # st.markdown(f"**Caption:** *{llm_caption}*")

            st.session_state.run_analysis = False # Reset state