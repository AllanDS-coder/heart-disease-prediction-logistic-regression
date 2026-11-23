
import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path

# ============================================================
# CardioLens
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"
IMAGE_PATH = ROOT / "image" / "heart.png"

st.set_page_config(
    page_title="CardioLens",
    page_icon="♥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Background image
# ============================================================

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


background_image = get_image_base64(IMAGE_PATH)

# ============================================================
# Styling
# ============================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --navy: #16324F;
        --teal: #078A83;
        --muted: #65788C;
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-image:
            linear-gradient(
                rgba(244,248,250,0.86),
                rgba(244,248,250,0.86)
            ),
            url("data:image/png;base64,{background_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        max-width: 760px;
        padding: 1.2rem 1rem 2.5rem;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* BRAND */

    h1, h2, h3, h4 {{
        color: var(--navy) !important;
    }}

    /* INPUTS */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {{
        border-radius: 12px;
        background-color: white;
    }}

    label {{
        color: #344C64 !important;
        font-weight: 600 !important;
    }}

    /* FORM */

    [data-testid="stForm"] {{
        background: rgba(255,255,255,0.94);
        border: 1px solid #DCE6EE;
        border-radius: 22px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(22,50,79,0.06);
    }}

    div.stFormSubmitButton > button {{
        border-radius: 14px;
        min-height: 54px;
        background: #078A83;
        color: white;
        border: 0;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(7,138,131,0.20);
        transition: all 0.2s ease;
    }}

    div.stFormSubmitButton > button:hover {{
        background: #066E69;
        color: white;
        transform: translateY(-1px);
    }}

    /* METRIC CARD */

    div[data-testid="stMetric"] {{
        background: #F0EAFE;
        border: 1px solid #DED2FA;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 5px 18px rgba(65,48,105,0.05);
    }}

    div[data-testid="stMetricLabel"] {{
        color: #594A7D;
    }}

    div[data-testid="stMetricValue"] {{
        color: #34245E;
    }}

    /* PROGRESS BAR */

    div[data-testid="stProgress"] > div > div {{
        background-color: #078A83;
    }}

    /* INFO MESSAGE */

    div[data-testid="stAlert"] {{
        border-radius: 16px;
    }}

    /* TABLE */

    [data-testid="stTable"] {{
        border-radius: 14px;
        overflow: hidden;
    }}

    /* EXPANDER */

    [data-testid="stExpander"] {{
        background: rgba(255,255,255,0.95);
        border: 1px solid #DCE6EE;
        border-radius: 16px;
    }}

    /* COLOUR SECTION LABELS */

    .section-heading {{
        padding: 12px 16px;
        border-radius: 12px;
        margin: 12px 0 18px 0;
        font-size: 15px;
        font-weight: 700;
    }}

    .blue-heading {{
        background: #E8F3FF;
        color: #24557A;
        border-left: 5px solid #4A90C2;
    }}

    .green-heading {{
        background: #E4F5EE;
        color: #226C55;
        border-left: 5px solid #46A987;
    }}

    .orange-heading {{
        background: #FFF1DF;
        color: #875A25;
        border-left: 5px solid #E5A253;
    }}

    .purple-heading {{
        background: #F0EAFE;
        color: #594A7D;
        border-left: 5px solid #9071D4;
    }}

    .teal-heading {{
        background: #DFF5F1;
        color: #176C67;
        border-left: 5px solid #078A83;
    }}

    /* INTRO CARD */

    .intro-card {{
        background: linear-gradient(135deg, #16324F, #176D85);
        padding: 25px;
        border-radius: 22px;
        margin: 12px 0 22px 0;
        color: white;
        box-shadow: 0 10px 28px rgba(22,50,79,0.14);
    }}

    .intro-card h2 {{
        color: white !important;
        font-size: 26px;
        margin-bottom: 8px;
    }}

    .intro-card p {{
        color: #E4F1F8;
        line-height: 1.7;
        font-size: 14px;
    }}

    .brand-caption {{
        color: #078A83;
        font-size: 11px;
        letter-spacing: 2px;
        font-weight: 700;
    }}

    /* SAFETY NOTICE */

    .safety-box {{
        background: #FFF4D9;
        border: 1px solid #EED9A2;
        border-radius: 17px;
        padding: 18px;
        color: #725821;
        line-height: 1.7;
        font-size: 13px;
    }}

    /* MOBILE */

    @media (max-width: 640px) {{
        .block-container {{
            padding-left: .8rem;
            padding-right: .8rem;
        }}

        .stApp {{
            background-attachment: scroll;
        }}

        .intro-card {{
            padding: 20px;
            border-radius: 18px;
        }}

        .intro-card h2 {{
            font-size: 23px;
        }}

        [data-testid="stForm"] {{
            padding: 14px;
            border-radius: 18px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ============================================================
# Brand and introduction
# ============================================================

st.markdown("## ♥ CardioLens")

st.markdown(
    '<div class="brand-caption">A HEART HEALTH DATA DEMONSTRATION</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="intro-card">
        <h2>Explore a heart health prediction</h2>
        <p>
            Enter the information below to see how a computer model
            classifies a set of health-related measurements.
            This is an educational demonstration, not a medical assessment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This is an educational demonstration, not a medical assessment."
)

# ============================================================
# Input form
# ============================================================

st.markdown(
    '<div class="section-heading blue-heading">01 · About the person</div>',
    unsafe_allow_html=True,
)

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50,
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["Female", "Male"],
        )

    st.markdown(
        '<div class="section-heading green-heading">02 · Measurements</div>',
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns(2)

    with col3:
        trestbps = st.number_input(
            "Resting blood pressure (mmHg)",
            min_value=50.0,
            max_value=300.0,
            value=120.0,
        )

    with col4:
        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=50.0,
            max_value=1000.0,
            value=200.0,
        )

    col5, col6 = st.columns(2)

    with col5:
        thalch = st.number_input(
            "Maximum heart rate",
            min_value=30.0,
            max_value=250.0,
            value=150.0,
        )

    with col6:
        oldpeak = st.number_input(
            "Oldpeak measurement",
            min_value=-5.0,
            max_value=10.0,
            value=0.0,
        )

    st.markdown(
        '<div class="section-heading orange-heading">03 · Additional information</div>',
        unsafe_allow_html=True,
    )

    cp = st.selectbox(
        "Chest pain description",
        [
            "typical angina",
            "atypical angina",
            "non-anginal",
            "asymptomatic",
        ],
    )

    col7, col8 = st.columns(2)

    with col7:
        fbs = st.selectbox(
            "Fasting blood sugar above 120 mg/dl",
            [False, True],
            format_func=lambda value: "No" if not value else "Yes",
        )

    with col8:
        exang = st.selectbox(
            "Chest pain during exercise",
            [False, True],
            format_func=lambda value: "No" if not value else "Yes",
        )

    restecg = st.selectbox(
        "Resting ECG description",
        [
            "normal",
            "st-t abnormality",
            "lv hypertrophy",
        ],
    )

    slope = st.selectbox(
        "ST segment slope",
        [
            "upsloping",
            "flat",
            "downsloping",
        ],
    )

    submitted = st.form_submit_button(
        "View result",
        use_container_width=True,
    )

# ============================================================
# Prediction
# ============================================================

if submitted:

    input_data = pd.DataFrame([{
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
        "sex": sex,
        "cp": cp,
        "fbs": fbs,
        "restecg": restecg,
        "exang": exang,
        "slope": slope,
    }])

    prediction = int(model.predict(input_data)[0])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    st.session_state["cardiolens_result"] = {
        "prediction": prediction,
        "probability": probability,
        "patient": input_data.to_dict("records")[0],
    }

# ============================================================
# Results
# ============================================================

if "cardiolens_result" in st.session_state:

    result = st.session_state["cardiolens_result"]

    probability = result["probability"]
    prediction = result["prediction"]
    patient = result["patient"]

    st.markdown("---")

    st.markdown(
        '<div class="section-heading purple-heading">04 · Your result</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    st.metric(
        label="Model-estimated probability of the positive class",
        value=f"{probability:.1%}",
    )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    st.caption(
        "This represents the model's estimated probability "
        "of the positive class. "
        "It is not a validated estimate of personal medical risk."
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if prediction == 1:

        st.info(
            "**The model returned a positive classification.**\n\n"
            "The model placed this set of inputs in the positive "
            "category."
        )

    else:

        st.info(
            "**The model returned a negative classification.**\n\n"
            "The model placed this set of inputs in the negative "
            "category."
        )

    # --------------------------------------------------------
    # Information entered
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading teal-heading">05 · Information you entered</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Review the values submitted for this demonstration."
    )

    summary = pd.DataFrame(
        [
            ["Age", f"{patient['age']} years"],
            ["Sex", patient["sex"]],
            [
                "Resting blood pressure",
                f"{patient['trestbps']} mmHg",
            ],
            [
                "Cholesterol",
                f"{patient['chol']} mg/dl",
            ],
            [
                "Maximum heart rate",
                str(patient["thalch"]),
            ],
            [
                "Oldpeak",
                str(patient["oldpeak"]),
            ],
            [
                "Chest pain description",
                patient["cp"],
            ],
            [
                "Fasting blood sugar above 120 mg/dl",
                "Yes" if patient["fbs"] else "No",
            ],
            [
                "Chest pain during exercise",
                "Yes" if patient["exang"] else "No",
            ],
            [
                "Resting ECG description",
                patient["restecg"],
            ],
            [
                "ST segment slope",
                patient["slope"],
            ],
        ],
        columns=["Information", "Value"],
    )

    st.table(summary)

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    with st.expander("What does this result mean?"):

        st.write(
            "This application uses a computer model trained "
            "on a dataset to produce a classification and a "
            "probability for its positive class. That output "
            "is not a diagnosis, a reliable estimate of personal "
            "medical risk, or a substitute for advice from a "
            "healthcare professional."
        )

# ============================================================
# Safety notice
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-heading orange-heading">Safety and responsible use</div>',
    unsafe_allow_html=True,
)

st.warning(
    "**Important:** CardioLens is an educational demonstration. "
    "It is not a medical device and has not been clinically validated. "
    "Do not use its output for diagnosis, treatment, screening, or "
    "medical decisions. Model results can be wrong and may not apply "
    "to different people or settings. Do not enter identifiable or "
    "confidential patient information."
)

st.caption(
    "CardioLens · Educational demonstration · Built with Python."
)

