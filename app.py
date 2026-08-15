import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PlantGuard AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hero ---------- */

    .hero {
        padding: 2.2rem 2rem;
        border-radius: 22px;
        margin-bottom: 2rem;
        background: linear-gradient(
            135deg,
            #0f5132 0%,
            #198754 55%,
            #20c997 100%
        );
        color: white;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.12);
    }

    .hero h1 {
        font-size: 42px;
        margin: 0;
        font-weight: 800;
        color: white !important;
    }

    .hero p {
        font-size: 18px;
        margin-top: 10px;
        color: #ecfdf5 !important;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        color: white;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* ---------- Cards ---------- */

    .card {
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 16px;
        background: rgba(128,128,128,0.04);
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.04);
    }

    .metric-value {
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 13px;
        opacity: 0.7;
    }

    /* ---------- Result cards ---------- */

    .result-card {
        padding: 24px;
        border-radius: 18px;
        margin-top: 18px;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .result-high {
        background: rgba(25, 135, 84, 0.12);
        border-left: 6px solid #198754;
    }

    .result-medium {
        background: rgba(255, 193, 7, 0.12);
        border-left: 6px solid #ffc107;
    }

    .result-low {
        background: rgba(220, 53, 69, 0.12);
        border-left: 6px solid #dc3545;
    }

    .result-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.65;
        margin-bottom: 8px;
    }

    .disease-name {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .confidence-text {
        font-size: 18px;
        font-weight: 700;
    }

    /* ---------- Status ---------- */

    .status-high {
        color: #198754;
        font-weight: 800;
    }

    .status-medium {
        color: #b58100;
        font-weight: 800;
    }

    .status-low {
        color: #dc3545;
        font-weight: 800;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        opacity: 0.6;
        font-size: 13px;
        padding-top: 30px;
    }

    /* ---------- Sidebar ---------- */

    .sidebar-title {
        font-size: 20px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "models/plant_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = "class_names.json"

IMG_SIZE = (224, 224)

HIGH_CONFIDENCE = 0.70
MEDIUM_CONFIDENCE = 0.40


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


@st.cache_data
def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


model = load_model()
class_names = load_class_names()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_class_name(class_name):
    """
    Converts PlantVillage class names into
    user-friendly names.
    """

    if "___" in class_name:

        crop, disease = class_name.split(
            "___",
            1
        )

    else:

        crop = "Unknown"
        disease = class_name

    crop = crop.replace("_", " ")

    disease = disease.replace("_", " ")

    return crop, disease


def get_confidence_status(confidence):

    if confidence >= HIGH_CONFIDENCE:

        return (
            "High Confidence",
            "result-high",
            "status-high"
        )

    elif confidence >= MEDIUM_CONFIDENCE:

        return (
            "Moderate Confidence",
            "result-medium",
            "status-medium"
        )

    else:

        return (
            "Low Confidence",
            "result-low",
            "status-low"
        )


def predict_disease(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(
        image_array
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    results = []

    for index in top_indices:

        results.append(
            {
                "class": class_names[index],
                "confidence": float(
                    predictions[index]
                )
            }
        )

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🌿 PlantGuard AI</div>',
        unsafe_allow_html=True
    )

    st.write(
        "AI-powered plant disease classification "
        "using deep learning."
    )

    st.divider()

    st.markdown("### Model")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Training:** Transfer Learning")
    st.write("**Classes:** 38")
    st.write("**Input:** 224 × 224 RGB")

    st.divider()

    st.markdown("### Dataset")

    st.write("**PlantVillage**")
    st.write("54,305 color leaf images")
    st.write("38 plant/disease classes")

    st.divider()

    st.markdown("### Confidence")

    st.write("🟢 ≥ 70% — High")
    st.write("🟡 40–70% — Moderate")
    st.write("🔴 < 40% — Low")


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">AI • COMPUTER VISION • DEEP LEARNING</div>'
    '<h1>Plant Disease Detection</h1>'
    '<p>Upload a plant leaf image and let a deep learning model identify the most likely plant disease.</p>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROJECT METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">38</div>
            <div class="metric-label">Classes</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">54K+</div>
            <div class="metric-label">Training Images</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">CNN</div>
            <div class="metric-label">Deep Learning</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">224²</div>
            <div class="metric-label">Image Input</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown("### 📷 Analyze a Leaf")

st.write(
    "Upload a clear image of a plant leaf. "
    "For best results, use good lighting and "
    "keep the leaf clearly visible."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# IMAGE + PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    image_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_col:

        st.markdown(
            "#### Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"Image size: {image.size[0]} × "
            f"{image.size[1]} pixels"
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with result_col:

        st.markdown(
            "#### AI Analysis"
        )

        predict_button = st.button(
            "🔍 Analyze Leaf",
            use_container_width=True,
            type="primary"
        )

        if predict_button:

            with st.spinner(
                "Analyzing leaf characteristics..."
            ):

                results = predict_disease(
                    image
                )

            best = results[0]

            crop, disease = format_class_name(
                best["class"]
            )

            confidence = best[
                "confidence"
            ]

            status, result_class, status_class = (
                get_confidence_status(
                    confidence
                )
            )

            is_healthy = (
                "healthy"
                in best["class"].lower()
            )

            if is_healthy:

                condition_label = (
                    "Healthy Leaf"
                )

            else:

                condition_label = (
                    "Potential Disease Detected"
                )


            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            result_html = (
                f'<div class="result-card {result_class}">'
                f'<div class="result-title">{condition_label}</div>'
                f'<div class="disease-name">{crop} — {disease}</div>'
                f'<div class="{status_class}">{status}</div>'
                f'<div class="confidence-text">'
                f'Confidence: {confidence:.2%}'
                f'</div>'
                f'</div>'
            )

            st.markdown(
                result_html,
                unsafe_allow_html=True
            )

            st.progress(
                confidence
            )

            # ------------------------------------------------
            # LOW CONFIDENCE MESSAGE
            # ------------------------------------------------

            if confidence < MEDIUM_CONFIDENCE:

                st.warning(
                    "The model is uncertain about this "
                    "prediction. Try a clearer image with "
                    "better lighting and minimal background."
                )

            elif confidence < HIGH_CONFIDENCE:

                st.info(
                    "The model has moderate confidence. "
                    "Consider verifying the result with "
                    "an agricultural expert."
                )

            else:

                st.success(
                    "The model has high confidence in "
                    "this prediction."
                )


# ============================================================
# TOP PREDICTIONS
# ============================================================

            st.markdown("### Top 3 Predictions")

            for rank, result in enumerate(
                results,
                start=1
            ):

                crop_name, disease_name = (
                    format_class_name(
                        result["class"]
                    )
                )

                percentage = (
                    result["confidence"] * 100
                )

                st.write(
                    f"**{rank}. {crop_name} — "
                    f"{disease_name}**"
                )

                st.progress(
                    result["confidence"]
                )

                st.caption(
                    f"{percentage:.2f}% confidence"
                )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.markdown("### How It Works")

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown("### 01")
    st.write("**Upload**")
    st.caption(
        "Upload a clear plant leaf image."
    )

with step2:
    st.markdown("### 02")
    st.write("**Analyze**")
    st.caption(
        "MobileNetV2 extracts visual features."
    )

with step3:
    st.markdown("### 03")
    st.write("**Predict**")
    st.caption(
        "The model returns disease and confidence."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ Educational and demonstration purposes only. "
    "Predictions should not be treated as a professional "
    "agricultural diagnosis."
)

st.markdown(
    """
    <div class="footer">
        PlantGuard AI • Plant Disease Detection •
        MobileNetV2 Transfer Learning
    </div>
    """,
    unsafe_allow_html=True
)