import io
import base64
import os
import pickle

import gdown
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import pad_sequences


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Afaan Oromo Image Captioner",
    page_icon="🇪🇹",
    layout="centered",
)


# ─────────────────────────────────────────────────────────────────────────────
# Download models from Google Drive
# ─────────────────────────────────────────────────────────────────────────────
def download_models():
    model_files = {
        "model (1).keras":             "1Muy2szHvErl60Zi3gs0qdo6PISfK_nsq",
        "tokenizer (2).pkl":           "1LoTcsh6A8CwW9N7DH1Br_Ib8NDChXZtA",
        "feature_extractor (1).keras": "1q5tDGe4Bu_lf7ynajjGeW_br3JbSwB1Q",
    }

    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            st.info(f"Downloading {filename}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}&confirm=t",
                filename,
                quiet=False,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Generate Caption  (original logic — unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def generate_caption(image_path, model_path, tokenizer_path,
                     feature_extractor_path, max_length=42, img_size=224):
    """Return the Afaan Oromo caption string for the given image."""
    caption_model     = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img       = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    image_features = feature_extractor.predict(img_array, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat     = caption_model.predict([image_features, sequence], verbose=0)
        word     = tokenizer.index_word.get(int(np.argmax(yhat)), None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Text-to-Speech  — Afaan Oromo  (lang="om")
# ─────────────────────────────────────────────────────────────────────────────
def text_to_speech(caption):
    """
    Convert caption to MP3 with gTTS using Oromo locale (lang='om').
    Falls back to English ('en') if Oromo locale is unavailable.
    Returns a base64 data-URI string, or None on failure.
    """
    try:
        from gtts import gTTS
    except ImportError:
        return None

    if not caption.strip():
        return None

    for lang in ("om", "en"):
        try:
            buf = io.BytesIO()
            gTTS(text=caption, lang=lang, slow=False).write_to_fp(buf)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            return "data:audio/mpeg;base64," + b64
        except Exception:
            continue

    return None


def render_audio_player(data_uri):
    """Inject an HTML audio player + download link into the Streamlit page."""
    st.markdown(
        """
        <div style="
            margin-top: 1rem;
            padding: 0.9rem 1.1rem;
            background: rgba(25, 118, 210, 0.07);
            border: 1px solid rgba(25, 118, 210, 0.30);
            border-radius: 12px;
        ">
          <p style="margin:0 0 0.45rem; font-size:0.78rem; font-weight:700;
                    letter-spacing:0.12em; text-transform:uppercase; color:#1976D2;">
            🔊 Listen — Afaan Oromo
          </p>
          <audio controls autoplay style="width:100%; border-radius:8px;">
            <source src="{uri}" type="audio/mpeg">
            Your browser does not support the audio element.
          </audio>
          <br>
          <a href="{uri}" download="afaan_oromo_caption.mp3"
             style="display:inline-block; margin-top:0.5rem;
                    font-size:0.75rem; color:#1976D2; text-decoration:none;
                    border:1px solid #1976D2; padding:0.2rem 0.75rem;
                    border-radius:50px;">
            ⬇ Download MP3
          </a>
        </div>
        """.replace("{uri}", data_uri),
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("Afaan Oromo Image Captioner Model")
    st.write("Upload an image and generate a caption using the trained model.")

    download_models()

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        tmp_path = "uploaded_image.jpg"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        model_path             = "model (1).keras"
        tokenizer_path         = "tokenizer (2).pkl"
        feature_extractor_path = "feature_extractor (1).keras"

        with st.spinner("Generating caption..."):
            try:
                caption = generate_caption(
                    tmp_path, model_path, tokenizer_path, feature_extractor_path
                )
            except Exception as exc:
                st.error("Caption generation failed: " + str(exc))
                return

        # Original: show image with caption as matplotlib title
        img_display = load_img(tmp_path, target_size=(224, 224))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_display)
        ax.axis("off")
        ax.set_title(caption, fontsize=16, color="blue", wrap=True)
        st.pyplot(fig)
        plt.close(fig)

        # Show caption as readable text
        st.markdown("### Generated Caption")
        st.info(caption)

        # NEW: Text-to-Speech audio player
        with st.spinner("Synthesising speech..."):
            data_uri = text_to_speech(caption)

        if data_uri:
            render_audio_player(data_uri)
        else:
            st.warning(
                "Audio unavailable. "
                "Make sure gTTS is installed: pip install gTTS"
            )


if __name__ == "__main__":
    main()
