import streamlit as st
import numpy as np
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import gdown
import os
from gtts import gTTS
import tempfile
import base64


# ---- Page Configuration ----
st.set_page_config(
    page_title="Afaan Oromo Image Captioner",
    page_icon="🖼️",
    layout="centered"
)


# ---- Download models from Google Drive ----
def download_models():
    model_files = {
        "model (1).keras": "1Muy2szHvErl60Zi3gs0qdo6PISfK_nsq",
        "tokenizer (2).pkl": "1LoTcsh6A8CwW9N7DH1Br_Ib8NDChXZtA",
        "feature_extractor (1).keras": "1q5tDGe4Bu_lf7ynajjGeW_br3JbSwB1Q",
    }

    for filename, file_id in model_files.items():
        if not os.path.exists(filename):
            st.info(f"Downloading {filename}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                filename,
                quiet=False
            )


# ---- Text-to-Speech: Caption to Audio (Afaan Oromo) ----
def text_to_speech_afaan_oromo(caption_text: str) -> str:
    """
    Convert the generated caption to speech using gTTS.
    Afaan Oromo (Oromo language) is not natively supported by gTTS,
    so we use the closest supported language ('af' for Afrikaans is not suitable).
    The best available approach is to use a language that can approximate
    the phonetics, or use a neutral TTS.

    NOTE: gTTS does NOT support Afaan Oromo (om) natively.
    We use 'en' as fallback for universal deployment.
    If you have an Afaan Oromo-specific TTS API, replace this function.

    For Google Translate TTS, Oromo (om) is supported in some regions.
    We attempt 'om' first and fall back to 'en' if unavailable.

    Returns the path to the generated audio file.
    """
    # Create a temp file to store audio
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp_file.name
    tmp_file.close()

    # Try Oromo language code first
    try:
        tts = gTTS(text=caption_text, lang="om", slow=False)
        tts.save(tmp_path)
        return tmp_path, "om"
    except Exception:
        pass

    # Fallback: English pronunciation
    try:
        tts = gTTS(text=caption_text, lang="en", slow=False)
        tts.save(tmp_path)
        return tmp_path, "en (fallback)"
    except Exception as e:
        return None, f"TTS Error: {e}"


def get_audio_player_html(audio_path: str) -> str:
    """
    Read audio file and return an HTML audio player with base64-encoded audio.
    This avoids Streamlit file-serving issues.
    """
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
    <audio controls autoplay style="width:100%; margin-top:10px;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """


# ---- Generate Caption ----
def generate_caption(image_path, model_path, tokenizer_path,
                     feature_extractor_path, max_length=42, img_size=224):
    """
    Load models and generate a caption for the given image.
    Returns the caption string.
    """
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    image_features = feature_extractor.predict(img_array, verbose=0)

    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


def display_image_with_caption(image_path, caption, img_size=224):
    """
    Display the image with the generated caption using matplotlib.
    """
    img = load_img(image_path, target_size=(img_size, img_size))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=14, color="blue", wrap=True, pad=10)
    st.pyplot(fig)
    plt.close(fig)


# ---- Streamlit App ----
def main():
    # ---------- Header ----------
    st.title("🖼️ Afaan Oromo Image Captioner")
    st.markdown(
        """
        Upload an image and the model will automatically generate a caption in **Afaan Oromo**,
        then read it aloud for you using text-to-speech.
        """
    )

    st.divider()

    # ---------- Download Models ----------
    with st.status("🔄 Checking model files...", expanded=False) as status:
        try:
            download_models()
            status.update(label="✅ Models ready!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"❌ Model download failed: {e}", state="error")
            st.stop()

    # ---------- Upload Image ----------
    st.subheader("📤 Upload an Image")
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_image is not None:
        # Save the uploaded image to disk
        upload_path = "uploaded_image.jpg"
        with open(upload_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Show a preview of the uploaded image
        st.subheader("🔍 Uploaded Image Preview")
        st.image(upload_path, use_column_width=True)

        st.divider()

        # ---------- Generate Caption ----------
        st.subheader("📝 Generated Caption (Afaan Oromo)")

        model_path = "model (1).keras"
        tokenizer_path = "tokenizer (2).pkl"
        feature_extractor_path = "feature_extractor (1).keras"

        with st.spinner("🤖 Generating caption... Please wait."):
            try:
                caption = generate_caption(
                    upload_path, model_path, tokenizer_path, feature_extractor_path
                )
            except Exception as e:
                st.error(f"❌ Caption generation failed: {e}")
                st.stop()

        if caption:
            # Display image with caption overlay (matplotlib)
            display_image_with_caption(upload_path, caption)

            # Show caption text in a styled box
            st.success(f"**Caption:** {caption}")

            st.divider()

            # ---------- Text-to-Speech ----------
            st.subheader("🔊 Listen to the Caption (Afaan Oromo Speech)")

            with st.spinner("🎙️ Converting caption to speech..."):
                audio_path, lang_used = text_to_speech_afaan_oromo(caption)

            if audio_path and os.path.exists(audio_path):
                if lang_used == "om":
                    st.info("🗣️ Speaking in **Afaan Oromo** (Oromo language TTS)")
                else:
                    st.warning(
                        f"⚠️ Afaan Oromo TTS not available in this environment. "
                        f"Using **{lang_used}** pronunciation as fallback. "
                        "For native Afaan Oromo audio, integrate a dedicated Oromo TTS API."
                    )

                # Embed audio player using base64 (works reliably on Streamlit Cloud)
                audio_html = get_audio_player_html(audio_path)
                st.markdown(audio_html, unsafe_allow_html=True)

                # Also provide a download button for the audio
                with open(audio_path, "rb") as audio_file:
                    st.download_button(
                        label="⬇️ Download Audio",
                        data=audio_file.read(),
                        file_name="caption_audio.mp3",
                        mime="audio/mp3"
                    )

                # Cleanup temp file
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            else:
                st.error(f"❌ Audio generation failed: {lang_used}")

        else:
            st.warning("⚠️ No caption was generated. Please try a different image.")

    st.divider()

    # ---------- Footer ----------
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.85em;'>
            Afaan Oromo Image Captioner · Developed by Muktar AO · Powered by Deep Learning & gTTS
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
