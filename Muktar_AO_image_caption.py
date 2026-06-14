import io
import base64
import os
import pickle

import gdown
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import pad_sequences


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Afaan Oromo Image Captioner",
    page_icon="🇪🇹",
    layout="centered",
)

st.title("🇪🇹 Afaan Oromo Image Captioner")
st.write("Upload an image and get an AI-generated caption in **Afaan Oromo**, read aloud automatically.")


# ─────────────────────────────────────────────
# Download models from Google Drive
# ─────────────────────────────────────────────
MODEL_FILES = {
    "model (1).keras":               "1Muy2szHvErl60Zi3gs0qdo6PISfK_nsq",
    "tokenizer (2).pkl":             "1LoTcsh6A8CwW9N7DH1Br_Ib8NDChXZtA",
    "feature_extractor (1).keras":   "1q5tDGe4Bu_lf7ynajjGeW_br3JbSwB1Q",
}


def download_models() -> None:
    """Download model files from Google Drive if not already present."""
    for filename, file_id in MODEL_FILES.items():
        if not os.path.exists(filename):
            st.info(f"⬇️ Downloading {filename} …")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}&confirm=t",
                filename,
                quiet=False,
            )


# ─────────────────────────────────────────────
# Caption generation
# ─────────────────────────────────────────────
def generate_caption(
    image_path: str,
    model_path: str,
    tokenizer_path: str,
    feature_extractor_path: str,
    max_length: int = 42,
    img_size: int = 224,
) -> str:
    """Return the generated Afaan Oromo caption for *image_path*."""
    caption_model     = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    image_features = feature_extractor.predict(img_array, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat     = caption_model.predict([image_features, sequence], verbose=0)
        word     = tokenizer.index_word.get(int(np.argmax(yhat)))
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


# ─────────────────────────────────────────────
# Text-to-Speech  (Afaan Oromo — lang="om")
# ─────────────────────────────────────────────
def caption_to_audio_html(caption: str) -> str:
    """
    Convert *caption* to an MP3 using gTTS (Oromo locale 'om') and return
    an HTML snippet with an <audio> player + download link.

    Falls back to English TTS if the Oromo locale is unavailable.
    Returns an empty string when gTTS is not installed.
    """
    try:
        from gtts import gTTS
    except ImportError:
        return (
            "<p style='color:#e57373;font-size:0.85rem;'>"
            "⚠️ <code>gTTS</code> not installed — run "
            "<code>pip install gTTS</code> to enable audio.</p>"
        )

    if not caption.strip():
        return ""

    buf = io.BytesIO()
    # Try Oromo first; fall back to English if the locale is missing
    for lang in ("om", "en"):
        try:
            gTTS(text=caption, lang=lang, slow=False).write_to_fp(buf)
            buf.seek(0)
            break
        except Exception:
            buf = io.BytesIO()
            continue
    else:
        return "<p style='color:#e57373;font-size:0.85rem;'>⚠️ Could not synthesise audio.</p>"

    b64 = base64.b64encode(buf.read()).decode()
    data_uri = f"data:audio/mpeg;base64,{b64}"

    return f"""
<div style="
    margin-top:1rem;
    padding:0.9rem 1rem;
    background:rgba(25,118,210,0.08);
    border:1px solid rgba(25,118,210,0.30);
    border-radius:12px;
">
  <p style="margin:0 0 0.5rem;font-size:0.78rem;font-weight:700;
            letter-spacing:0.12em;text-transform:uppercase;color:#1976D2;">
    🔊 Listen to caption (Afaan Oromo)
  </p>
  <audio controls autoplay style="width:100%;border-radius:8px;">
    <source src="{data_uri}" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <br>
  <a href="{data_uri}" download="afaan_oromo_caption.mp3"
     style="display:inline-block;margin-top:0.5rem;
            font-size:0.75rem;color:#1976D2;text-decoration:none;
            border:1px solid #1976D2;padding:0.2rem 0.7rem;
            border-radius:50px;">
    ⬇ Download MP3
  </a>
</div>
"""


# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────
def main() -> None:
    # Download model weights on first run
    download_models()

    uploaded_image = st.file_uploader(
        "📂 Choose an image …", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is None:
        return

    # Save to disk so Keras can read it
    tmp_path = "uploaded_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    model_path             = "model (1).keras"
    tokenizer_path         = "tokenizer (2).pkl"
    feature_extractor_path = "feature_extractor (1).keras"

    with st.spinner("✨ Generating caption …"):
        try:
            caption = generate_caption(
                tmp_path, model_path, tokenizer_path, feature_extractor_path
            )
        except Exception as exc:
            st.error(f"❌ Caption generation failed: {exc}")
            return

    # ── Display image with caption as title ──────────────────────────────────
    img_display = load_img(tmp_path, target_size=(224, 224))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_display)
    ax.axis("off")
    ax.set_title(caption, fontsize=14, color="blue", wrap=True)
    st.pyplot(fig)
    plt.close(fig)

    # ── Show caption as text ─────────────────────────────────────────────────
    st.markdown("### 📝 Generated Caption")
    st.info(caption)

    # ── Text-to-Speech audio player ──────────────────────────────────────────
    audio_html = caption_to_audio_html(caption)
    if audio_html:
        st.markdown(audio_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
