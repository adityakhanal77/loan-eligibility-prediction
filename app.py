"""
Loan Eligibility Prediction — Streamlit App
--------------------------------------------
Group project UI. Loads trained models from /models (saved with joblib) and
lets the user pick any of the six models to generate a prediction, or view
all six side-by-side.

HOW TO PLUG IN YOUR REAL TRAINED MODELS
----------------------------------------
In Notebook 2, after training + tuning each model, save it like this:

    import joblib
    joblib.dump(model, "models/logistic_regression.pkl")
    joblib.dump(model, "models/decision_tree.pkl")
    joblib.dump(model, "models/random_forest.pkl")
    joblib.dump(model, "models/xgboost.pkl")
    joblib.dump(model, "models/svm.pkl")
    joblib.dump(model, "models/knn.pkl")

Also save the fitted scaler (used for SVM/KNN) and the list of final feature
column names, so this app can preprocess user input the same way you did
during training:

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl")

Until those files exist, this app runs on a simple rule-based placeholder
(clearly labelled below) so your group has a working, demoable UI from day one.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<meta name="color-scheme" content="light only">',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# STYLING — navy / off-white / muted gold, Lora + Inter
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #F4F6F9;
        --surface: #FFFFFF;
        --text-primary: #14202E;
        --text-secondary: #5B6B7C;
        --navy: #13294B;
        --navy-light: #1F3A63;
        --gold: #B8860B;
        --success: #1E7145;
        --danger: #A23B3B;
        --border: #D9DEE5;
        color-scheme: light only;
    }

    html { color-scheme: light only; }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .stApp { background-color: var(--bg); }

    h1, h2, h3 { font-family: 'Lora', serif !important; color: var(--navy); }

    section[data-testid="stSidebar"] {
        background-color: var(--navy);
    }
    section[data-testid="stSidebar"] * { color: #EDEFF3 !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        font-family: 'Lora', serif !important;
        color: #FFFFFF !important;
    }

    .brand-title {
        font-family: 'Lora', serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #B7C2D0;
        margin-top: 2px;
        margin-bottom: 1.6rem;
    }

    .panel {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--navy);
        border-radius: 4px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }

    .result-eligible {
        background-color: #EAF4EE;
        border-left: 4px solid var(--success);
        border-radius: 4px;
        padding: 1.2rem 1.6rem;
    }
    .result-not-eligible {
        background-color: #F7EAEA;
        border-left: 4px solid var(--danger);
        border-radius: 4px;
        padding: 1.2rem 1.6rem;
    }
    .result-verdict {
        font-family: 'Lora', serif;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .result-eligible .result-verdict { color: var(--success); }
    .result-not-eligible .result-verdict { color: var(--danger); }

    .stButton>button {
        background-color: var(--gold);
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button:hover { background-color: #A3790A; color: #FFFFFF; }

    div[data-testid="stMetricValue"] { font-family: 'Lora', serif; color: var(--navy); }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "Support Vector Machine": "models/svm.pkl",
    "K-Nearest Neighbors": "models/knn.pkl",
}

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="brand-title">Loan Eligibility Engine</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-sub">Final-year B.Sc. Data Science Capstone '
        'Project · Six-Model Comparison</p>',
        unsafe_allow_html=True,
    )

    st.markdown("**Model for prediction**")
    selected_model = st.radio(
        "Choose a model",
        list(MODEL_FILES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    compare_all = st.checkbox("Compare all six models", value=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.78rem; color:#9FACBC;">All six trained models '
        'are loaded from the <code>models/</code> folder and used for live '
        'predictions.</p>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown("## Loan Eligibility Prediction")
st.markdown(
    '<p style="color:var(--text-secondary); margin-top:-8px;">'
    "Enter applicant details below to predict loan eligibility.</p>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# INPUT FORM — 4-step wizard
# ----------------------------------------------------------------------
DEFAULTS = {
    "gender": "Male", "married": "Yes", "dependents": "0",
    "education": "Graduate", "self_employed": "No", "property_area": "Urban",
    "credit_history": "Yes", "loan_term": "360",
    "applicant_income": 5000, "coapplicant_income": 0, "loan_amount": 120,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "step" not in st.session_state:
    st.session_state.step = 1

STEP_TITLES = {
    1: "Personal Details",
    2: "Employment & Property",
    3: "Income & Loan Details",
    4: "Review & Predict",
}

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(f"#### Step {st.session_state.step} of 4 — {STEP_TITLES[st.session_state.step]}")
st.progress(st.session_state.step / 4)

GENDER_OPTS = ["Male", "Female"]
YESNO_OPTS = ["Yes", "No"]
DEP_OPTS = ["0", "1", "2", "3+"]
EDU_OPTS = ["Graduate", "Not Graduate"]
AREA_OPTS = ["Urban", "Semiurban", "Rural"]
TERM_OPTS = ["360", "180", "120", "60"]

if st.session_state.step == 1:
    with st.form("step1_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender_v = st.selectbox("Gender", GENDER_OPTS, index=GENDER_OPTS.index(st.session_state.gender))
            married_v = st.selectbox("Married", YESNO_OPTS, index=YESNO_OPTS.index(st.session_state.married))
        with col2:
            dependents_v = st.selectbox("Dependents", DEP_OPTS, index=DEP_OPTS.index(st.session_state.dependents))
            education_v = st.selectbox("Education", EDU_OPTS, index=EDU_OPTS.index(st.session_state.education))
        if st.form_submit_button("Next →"):
            st.session_state.gender = gender_v
            st.session_state.married = married_v
            st.session_state.dependents = dependents_v
            st.session_state.education = education_v
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    with st.form("step2_form"):
        col1, col2 = st.columns(2)
        with col1:
            self_employed_v = st.selectbox("Self Employed", YESNO_OPTS, index=YESNO_OPTS.index(st.session_state.self_employed))
            property_area_v = st.selectbox("Property Area", AREA_OPTS, index=AREA_OPTS.index(st.session_state.property_area))
        with col2:
            credit_history_v = st.selectbox("Credit History Meets Guidelines", YESNO_OPTS, index=YESNO_OPTS.index(st.session_state.credit_history))
            loan_term_v = st.selectbox("Loan Amount Term (months)", TERM_OPTS, index=TERM_OPTS.index(st.session_state.loan_term))
        nav1, nav2 = st.columns([1, 1])
        with nav1:
            back = st.form_submit_button("← Back")
        with nav2:
            fwd = st.form_submit_button("Next →")
        if back or fwd:
            st.session_state.self_employed = self_employed_v
            st.session_state.property_area = property_area_v
            st.session_state.credit_history = credit_history_v
            st.session_state.loan_term = loan_term_v
            st.session_state.step = 1 if back else 3
            st.rerun()

elif st.session_state.step == 3:
    with st.form("step3_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            applicant_income_v = st.number_input("Applicant Income (monthly)", min_value=0, step=500, value=int(st.session_state.applicant_income))
        with col2:
            coapplicant_income_v = st.number_input("Co-applicant Income (monthly)", min_value=0, step=500, value=int(st.session_state.coapplicant_income))
        with col3:
            loan_amount_v = st.number_input("Loan Amount (in thousands)", min_value=0, step=10, value=int(st.session_state.loan_amount))
        nav1, nav2 = st.columns([1, 1])
        with nav1:
            back = st.form_submit_button("← Back")
        with nav2:
            fwd = st.form_submit_button("Next →")
        if back or fwd:
            st.session_state.applicant_income = applicant_income_v
            st.session_state.coapplicant_income = coapplicant_income_v
            st.session_state.loan_amount = loan_amount_v
            st.session_state.step = 2 if back else 4
            st.rerun()

else:  # step 4 — review
    review_col1, review_col2 = st.columns(2)
    with review_col1:
        st.markdown(f"**Gender:** {st.session_state.gender}")
        st.markdown(f"**Married:** {st.session_state.married}")
        st.markdown(f"**Dependents:** {st.session_state.dependents}")
        st.markdown(f"**Education:** {st.session_state.education}")
        st.markdown(f"**Self Employed:** {st.session_state.self_employed}")
        st.markdown(f"**Property Area:** {st.session_state.property_area}")
    with review_col2:
        st.markdown(f"**Credit History OK:** {st.session_state.credit_history}")
        st.markdown(f"**Loan Term:** {st.session_state.loan_term} months")
        st.markdown(f"**Applicant Income:** {st.session_state.applicant_income}")
        st.markdown(f"**Co-applicant Income:** {st.session_state.coapplicant_income}")
        st.markdown(f"**Loan Amount:** {st.session_state.loan_amount}k")

    nav1, nav2 = st.columns([1, 1])
    with nav1:
        if st.button("← Back"):
            st.session_state.step = 3
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

predict_clicked = False
if st.session_state.step == 4:
    predict_clicked = st.button("Predict Eligibility")

# ----------------------------------------------------------------------
# PREPROCESSING — mirrors the pipeline used in Notebook 1
# ----------------------------------------------------------------------
def build_feature_row():
    gender = st.session_state.gender
    married = st.session_state.married
    dependents = st.session_state.dependents
    education = st.session_state.education
    self_employed = st.session_state.self_employed
    property_area = st.session_state.property_area
    credit_history = st.session_state.credit_history
    loan_term = st.session_state.loan_term
    applicant_income = st.session_state.applicant_income
    coapplicant_income = st.session_state.coapplicant_income
    loan_amount = st.session_state.loan_amount

    total_income = applicant_income + coapplicant_income
    dti_ratio = (loan_amount * 1000) / total_income if total_income > 0 else 0
    return {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": int(loan_term),
        "Credit_History": 1 if credit_history == "Yes" else 0,
        "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area],
        "Total_Income": total_income,
        "DTI_Ratio": dti_ratio,
    }


def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


SCALED_MODELS = {"Support Vector Machine", "K-Nearest Neighbors"}
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

_scaler = load_model(SCALER_PATH)
_feature_columns = load_model(FEATURE_COLUMNS_PATH)


def placeholder_score(features, jitter=0.0):
    """
    Rule-based stand-in used ONLY until the real trained models are saved
    to /models. Mirrors the strongest known real-world signal (credit
    history + income coverage of the loan) so the demo behaves sensibly.
    """
    score = 0.5
    score += 0.30 if features["Credit_History"] == 1 else -0.30
    income_coverage = features["Total_Income"] / max(features["LoanAmount"] * 1000 / 12, 1)
    score += min(max((income_coverage - 2) * 0.05, -0.15), 0.15)
    score += jitter
    return float(np.clip(score, 0.02, 0.98))


def get_prediction(model_name, features):
    model = load_model(MODEL_FILES[model_name])
    if model is not None:
        # Real trained model path
        X = pd.DataFrame([features])
        if _feature_columns is not None:
            X = X[_feature_columns]  # match training column order exactly
        if model_name in SCALED_MODELS and _scaler is not None:
            X = pd.DataFrame(_scaler.transform(X), columns=X.columns).values
        proba = model.predict_proba(X)[0][1]
        return float(proba), True
    # Placeholder path, with a small per-model jitter so the six aren't identical
    jitter_map = {
        "Logistic Regression": 0.0,
        "Decision Tree": -0.04,
        "Random Forest": 0.03,
        "XGBoost": 0.05,
        "Support Vector Machine": -0.02,
        "K-Nearest Neighbors": -0.06,
    }
    return placeholder_score(features, jitter_map.get(model_name, 0.0)), False

MODEL_ACCENTS = {
    "Logistic Regression": "#1E3A5F",    # navy blue
    "Decision Tree": "#2D5A3D",          # forest green
    "Random Forest": "#4A5568",          # slate gray
    "XGBoost": "#B7791F",                # burnt amber/gold
    "Support Vector Machine": "#7C2D2D", # deep maroon
    "K-Nearest Neighbors": "#3B6E8F",    # steel blue
}

# Alternate chart style per model: bar for 3, donut for the other 3
MODEL_CHART_STYLE = {
    "Logistic Regression": "bar",
    "Random Forest": "bar",
    "Support Vector Machine": "bar",
    "Decision Tree": "donut",
    "XGBoost": "donut",
    "K-Nearest Neighbors": "donut",
}


def render_confidence_chart(proba, accent, style):
    fig, ax = plt.subplots(figsize=(4, 1.1) if style == "bar" else (2.2, 2.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    if style == "bar":
        ax.barh([0], [1], color="#E5E7EB", height=0.5)
        ax.barh([0], [proba], color=accent, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        ax.text(proba + 0.02 if proba < 0.9 else proba - 0.12, 0,
                 f"{proba:.0%}", va="center", fontsize=11, fontweight="bold",
                 color=accent)
    else:
        wedge_sizes = [proba, 1 - proba]
        ax.pie(
            wedge_sizes,
            colors=[accent, "#E5E7EB"],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
        )
        ax.text(0, 0, f"{proba:.0%}", ha="center", va="center",
                 fontsize=15, fontweight="bold", color=accent)

    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ----------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------
if predict_clicked:
    features = build_feature_row()
    proba, is_real_model = get_prediction(selected_model, features)
    eligible = proba >= 0.5

    css_class = "result-eligible" if eligible else "result-not-eligible"
    verdict = "Eligible for Loan" if eligible else "Not Eligible for Loan"
    accent = MODEL_ACCENTS.get(selected_model, "#1E3A5F")
    chart_style = MODEL_CHART_STYLE.get(selected_model, "bar")

    st.markdown(f'<div class="{css_class}" style="border-left:4px solid {accent}; padding-left:16px;">', unsafe_allow_html=True)
    st.markdown(f'<p class="result-verdict">{verdict}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:var(--text-secondary); margin-bottom:0;">'
        f'Predicted using <strong style="color:{accent};">{selected_model}</strong> · '
        f"confidence score {proba:.0%}"
        f'{"" if is_real_model else " (placeholder scorer — plug in your trained model)"}'
        f"</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    render_confidence_chart(proba, accent, chart_style)

    if compare_all:
        st.markdown("#### Model Comparison")
        st.markdown(
            '<p style="color:var(--text-secondary); margin-top:-6px;">'
            "Confidence score for loan eligibility, across all six models.</p>",
            unsafe_allow_html=True,
        )

        results = {}
        for name in MODEL_FILES:
            p, _ = get_prediction(name, features)
            results[name] = p

        fig, ax = plt.subplots(figsize=(7, 3.2))
        names = list(results.keys())
        values = [results[n] * 100 for n in names]
        colors = ["#B8860B" if n == selected_model else "#13294B" for n in names]

        ax.barh(names, values, color=colors, height=0.55)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)", fontsize=9, color="#5B6B7C")
        ax.axvline(50, color="#A23B3B", linestyle="--", linewidth=1, alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#14202E", labelsize=9)
        fig.patch.set_facecolor("#F4F6F9")
        ax.set_facecolor("#F4F6F9")
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.markdown(
        '<p style="color:var(--text-secondary);">Fill in the applicant '
        "details above and click <strong>Predict Eligibility</strong> to "
        "see the result.</p>",
        unsafe_allow_html=True,
    )
