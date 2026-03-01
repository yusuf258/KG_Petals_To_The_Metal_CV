import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image


# --- 1. Page Config ---
st.set_page_config(
    page_title="Petals to the Metal: Flower Classification",
    page_icon="🌸",
    layout="wide"
)

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 104  # adjust if needed


# --- 2. Smart Path Resolver ---
def resolve_path(filename: str) -> str | None:
    """
    Supports BOTH layouts:

    Local:
      - project_root/models/<filename>
      - project_root/src/streamlit_app.py

    HF Space:
      - project_root/src/<filename>
      - project_root/src/streamlit_app.py

    Also supports optional src/models/<filename>.
    """
    here = Path(__file__).resolve().parent  # .../src

    candidates = [
        # HF layout (model is in src/)
        here / filename,
        # HF alternative (src/models/)
        here / "models" / filename,

        # Local layout (models/ folder next to src/)
        here / ".." / "models" / filename,

        # CWD-based fallbacks (docker workdir is /app)
        Path("src") / filename,
        Path("src") / "models" / filename,
        Path("models") / filename,
        Path(filename),
    ]

    for p in candidates:
        try:
            p2 = p.resolve()
        except Exception:
            p2 = p
        if p2.exists():
            return str(p2)
    return None


# --- 3. TFRecord Processing Functions ---
def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    return image, example["id"]


def extract_images_from_tfrec(tfrec_path, num_images=20):
    dataset = tf.data.TFRecordDataset(tfrec_path)
    dataset = dataset.map(parse_tfrecord_fn)

    extracted_data = []
    for image_tensor, id_tensor in dataset.take(num_images):
        img_array = image_tensor.numpy()
        id_str = id_tensor.numpy().decode("utf-8")
        pil_image = Image.fromarray(img_array)
        extracted_data.append({"image": pil_image, "id": id_str})

    return extracted_data


# --- 4. Model and Preprocessing ---
def _build_model_for_weights(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Rebuilds the training architecture:
      MobileNetV2(include_top=False, imagenet) -> GAP -> Dense(num_classes, softmax)
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.build((None, 224, 224, 3))
    return model


class PatchedDense(tf.keras.layers.Dense):
    """
    Some Keras/TF combos (common on HF) don't recognize Dense(config['quantization_config']).
    If the saved .keras contains this key, we strip it during from_config().
    """
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


@st.cache_resource(show_spinner=False)
def load_model():
    """
    HF-safe strategy:

    1) Prefer weights if present (most robust across Keras/TensorFlow minor version drift)
       - looks for best_model.weights.h5
    2) If weights not present, try best_model.keras with:
       - compile=False
       - custom PatchedDense to ignore quantization_config
    """
    # 1) WEIGHTS FIRST
    weights_path = resolve_path("best_model.weights.h5")
    if weights_path:
        try:
            model = _build_model_for_weights(num_classes=NUM_CLASSES)
            model.load_weights(weights_path)
            return model
        except Exception as e:
            st.warning(f"Weights load failed ({weights_path}). Will try .keras. Error: {e}")

    # 2) .KERAS FALLBACK
    keras_path = resolve_path("best_model.keras")
    if keras_path:
        try:
            return tf.keras.models.load_model(
                keras_path,
                compile=False,
                custom_objects={"Dense": PatchedDense, "PatchedDense": PatchedDense},
                safe_mode=False,
            )
        except Exception as e:
            st.error(f"Model (.keras) yüklenemedi: {e}")
            return None

    return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Keep preprocessing consistent with training.

    Your original app used /255 scaling. Keep it unless training used MobileNetV2 preprocess_input.
    If you trained with preprocess_input, change to:
      arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    """
    image = image.convert("RGB").resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- 5. Main App ---
def main():
    st.title("🌸 Petals to the Metal: Flower Classification")
    st.markdown("Upload a flower image (`jpg`, `png`) or a dataset file (`tfrec`) to classify.")

    with st.spinner("Loading model..."):
        model = load_model()

    if model is None:
        st.error(
            "🚨 Model file not found or could not be loaded.\n\n"
            "Expected one of these (both supported layouts):\n"
            "- Local:  models/best_model.weights.h5 (recommended) OR models/best_model.keras\n"
            "- HF:     src/best_model.weights.h5   (recommended) OR src/best_model.keras\n\n"
            "Not: HF'de .keras deserialize hatası sık olduğundan weights dosyasını mutlaka koymanı öneririm."
        )
        st.stop()

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "tfrec"])

    selected_image = None
    caption_text = ""

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".tfrec"):
            st.info("📦 TFRecord file detected. Reading content...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".tfrec") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:
                images_list = extract_images_from_tfrec(temp_file_path, num_images=20)

                if images_list:
                    st.success(f"Successfully extracted {len(images_list)} sample images from the file.")
                    selected_index = st.slider(
                        "Select an image to analyze:",
                        min_value=0,
                        max_value=len(images_list) - 1,
                        value=0,
                    )
                    selected_data = images_list[selected_index]
                    selected_image = selected_data["image"]
                    caption_text = f"TFRecord ID: {selected_data['id']}"
                else:
                    st.warning("Could not read images from this TFRecord file.")
            except Exception as e:
                st.error(f"TFRecord reading error: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        else:
            selected_image = Image.open(uploaded_file)
            caption_text = "Uploaded Image"

        if selected_image:
            st.divider()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🖼️ Preview")
                st.image(selected_image, caption=caption_text, use_column_width=True)
                predict_btn = st.button("🔍 Predict Flower", type="primary")

            if predict_btn:
                with col2:
                    st.subheader("📊 Analysis Result")
                    with st.spinner("AI is thinking..."):
                        processed_img = preprocess_image(selected_image)
                        predictions = model.predict(processed_img, verbose=0)

                        class_id = int(np.argmax(predictions[0]))
                        confidence = float(np.max(predictions[0]))

                        st.success(f"**Predicted Class: {class_id}**")
                        st.metric("Confidence Score", f"{confidence:.2%}")

                        top_5_idx = predictions[0].argsort()[-5:][::-1]
                        top_5_val = predictions[0][top_5_idx]

                        chart_df = pd.DataFrame(
                            {"Class ID": [str(x) for x in top_5_idx], "Probability": top_5_val}
                        )

                        fig = px.bar(
                            chart_df,
                            x="Probability",
                            y="Class ID",
                            orientation="h",
                            text_auto=".2%",
                            title="Top 5 Probabilities",
                        )
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_column_width=True)


if __name__ == "__main__":
    main()
