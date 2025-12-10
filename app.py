import streamlit as st
from PIL import Image
from openai import OpenAI
import requests
import json
import time

import re
import cv2
import numpy as np
import tensorflow as tf
import os
import pickle
import tempfile
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess


from huggingface_hub import hf_hub_download

# path dr hugging face
model_path = "grcb05/image-captioning-transformer/model_epoch21.keras"
from captioning_model_classes import(
    CaptionTrainer, 
    TransformerDecoder, 
    DecoderLayer, 
    TransformerEncoderBlock,
    WarmUpLinear,
    build_visual_encoder_all
)

MAX_SEQ_LEN = 20
VOCAB_SIZE=10000
HF_TOKEN=st.secrets["HF_TOKEN"]
api_key = st.secrets["OPENROUTER_API_KEY"]
# print("API Key loaded:", api_key[:4] + "..." )

# Set page configuration
st.set_page_config(
    page_title="EfficientNet-Transformer Social Media Captioning",
    layout="wide",
    page_icon="📸"
)

# --- MODEL LOADING & CACHING ---
@st.cache_data
def download_model(hf_token):
    model_path = hf_hub_download(
        repo_id="grcb05/image-captioning-transformer",
        filename="model_epoch21.keras",
        use_auth_token=hf_token
    )
    return model_path
model_path = download_model(HF_TOKEN)

@st.cache_resource
def load_keras_model():
    # model_path= "models/model_epoch22.keras"
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
def preprocess(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224,224))
    img_converted = efficient_preprocess(img)
    return img_converted

# vectorizer
def custom_standardization(input_string):
    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")

    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def build_text_vectorizer(pkl_path="vectorizer_data.pkl", max_length=20):
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        vocab=data["vocab"]
        print(f"Loaded vectorizer from {pkl_path}, vocab size: {len(vocab)}")
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=len(vocab),
            output_sequence_length=max_length,
            standardize=custom_standardization,
            output_mode="int"
        )
        # adapt dummy data untuk build internal weights (vocab)
        vectorizer.adapt(tf.constant(vocab))
        return vectorizer, vocab    
    
    except FileNotFoundError:
        raise ValueError("Vectorizer pickle not found...")
    
vectorizer, vocab = build_text_vectorizer(pkl_path="vectorizer_data.pkl", max_length=20)
# vocab = tokenizer.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))

# --- LLM SIMULATION FUNCTION ---
def generate_caption(model, image_path,index_lookup, max_capt_len):
    """Fungsi Placeholder untuk meniru output caption asli"""
    img = preprocess(image_path)
    
    # pass image -> cnn
    start_id = 3
    end_id=4
    decoded_ids=[start_id]

    img_batch = tf.expand_dims(img, axis=0)           # (1, H, W, 3)
    # output encoder
    enc_output = model.encoder(img_batch, training=False) #1,seqlen, dmodel

    for _ in range(max_capt_len):
        tokenized_input = tf.expand_dims(decoded_ids, axis=0)  # (1, seq)
        mask = (tokenized_input != 0)

        # dec_mask =  tf.cast(tokenized != 0, tf.bool) # (1, seq_len)
        predictions = model.decoder(
            tokenized_input,
            enc_output, 
            mask=mask,
            training=False,
        ) #1,seqlen, vocab
        next_id = tf.argmax(predictions[0, -1]).numpy()
        # next_id = int(tf.argmax(predictions[0, len(decoded_ids-1)]).numpy())
        # 7. Greedy sampling (keras)
        # stop jika <end>
        if next_id == end_id:
            break
        # decoded_sentence += " " + next_word
        decoded_ids.append(next_id)

    decoded_sentence = " ".join(index_lookup[i] for i in decoded_ids[1:])
    # print("Predicted Caption: ", decoded_sentence)
    return decoded_sentence

def generate_llm_caption(raw_caption):
    """Fungsi Placeholder untuk meniru output dari Model LLM Caption Generator"""

    messages=[
        {
        "role": "user",
        "content": (
            f"You are a social media content expert. "
            f"Given the following context, generate a catchy, engaging caption for a social media post. "
            f"The caption should be concise, attention-grabbing, and encourage interaction (like, comment, share). "
            f"Include 5-10 relevant hashtags at the end, based on the content.\n\n"
            f"Context:\n{raw_caption}\n\n"
            f"Output format:\n"
            f"Caption: <your caption here>\n"
            f"Hashtags: <relevant hashtags separated by spaces>"
            )
        }
    ]
    
    for _ in range(3):  # coba maksimal 3 kali
        response=requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "mistralai/mistral-small-3.1-24b-instruct:free",
                "messages": messages
            })
        )
        if response.status_code == 429:
            print("Rate limited, retrying in 10 seconds...")
            time.sleep(10)
            continue
        break

    # get response
    result = response.json() # choice[0] messages, content

    print(result)
    # contentnya
    if "choices" in result: 
        content=result["choices"][0]["message"]["content"]
    else:
        error_msg = result.get("error", {}).get("message", "Unknown error")
        print("API returned an error:", error_msg)
        content = ""


    caption_match = re.search(r'Caption:\s*"(.*?)"', content, re.DOTALL)
    hashtags_match = re.search(r'Hashtags:\s*(.*)', content, re.DOTALL)

    caption = caption_match.group(1).strip() if caption_match else ""
    hashtags = hashtags_match.group(1).strip() if hashtags_match else ""

    return {"caption": caption, "hashtags":hashtags}
    
# ----------------------------------------------------
# APLIKASI UTAMA STREAMLIT
# ----------------------------------------------------
st.title("📸 Social Media Image Captioning")
st.markdown("---")


model_captioning = load_keras_model()
# 1. Upload Image (Di kolom Kiri)
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded_file and model_captioning is not None:
    img = Image.open(uploaded_file).convert("RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    # Menggunakan st.columns untuk tata letak yang lebih baik
    st.write()
    st.write()
    st.write()
    col_img, col_results = st.columns([1, 1.5]) 

    # --- Kolom Kiri: Display Gambar ---
    with col_img:
        st.subheader("Uploaded Image")
        
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.markdown("---")

        # 2. Tombol Generate Caption (Pemicu)
        if st.button("▶️ Generate Caption", type="primary"):
            st.session_state['run_analysis'] = True
            
        if 'run_analysis' not in st.session_state:
             st.session_state['run_analysis'] = False

    # --- Kolom Kanan: Hasil Analisis ---
    with col_results:
        st.subheader("Generated Caption")


        # Inisialisasi placeholder
        placeholder_caption = st.empty()
        placeholder_llm = st.empty()
        
        if not st.session_state.run_analysis:
            placeholder_caption.info("Press 'Generate Caption' to generate capton.")
        # placeholder_llm.markdown("*(loading caption result...)*")

        # 3. Logika Analisis (Terpicu oleh Tombol)
        if st.session_state.run_analysis:
            
            # --- Tahap captioning ---
            with st.spinner('1/2. Generating basic caption from image...'):
                # img_preprocessed = preprocess(img)
                caption=generate_caption(model_captioning, 
                                         image_path=tmp_path,
                                         index_lookup=index_lookup,
                                         max_capt_len=20)
                
            placeholder_caption.success(f"{caption}")            
            st.markdown("---")

            # --- Tahap NLP (Caption Generation) ---
            with st.spinner('2/2. Refining into an engaging social media post...'):
                # 4. Kasi text buat munculin caption dr LLM nya
                if caption:
                    llm_output=generate_llm_caption(caption)
                    caption_text = llm_output["caption"]
                    hashtags_text = llm_output["hashtags"]
                else:
                    st.write("raw caption not found...")
                    caption_text = ""
                    hashtags_text = ""
            
            st.subheader("Social Media Caption (LLM Refined)")
            with st.expander("Click to see the caption...", expanded=False):
                st.markdown(f"{caption_text}\n\n {hashtags_text}")

                if caption_text =="" and hashtags_text=="":
                    st.markdown("no caption generated, try again...")

            st.session_state.run_analysis = False # Reset state