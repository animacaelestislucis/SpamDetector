import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------

BG_BASE = "#0A0E17"
BG_DEEP = "#060911"
PANEL = "rgba(255, 255, 255, 0.035)"
PANEL_BORDER = "rgba(255, 255, 255, 0.09)"
INPUT_BG = "#0D1220"
TEXT_PRIMARY = "#E8EAF2"
TEXT_MUTED = "#8890A6"
INDIGO = "#6C7CF6"
ELECTRIC_BLUE = "#4C8DFF"
CRIMSON = "#E5484D"
CRIMSON_DIM = "rgba(229, 72, 77, 0.10)"
EMERALD = "#3DD68C"
EMERALD_DIM = "rgba(61, 214, 140, 0.10)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    div[data-testid="stDecoration"] {{display: none;}}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(ellipse 900px 500px at 50% -10%, rgba(108, 124, 246, 0.16), transparent 60%),
            radial-gradient(ellipse 700px 400px at 85% 15%, rgba(76, 141, 255, 0.08), transparent 55%),
            {BG_DEEP};
    }}

    .block-container {{
        max-width: 620px;
        padding-top: 5.5rem;
        padding-bottom: 4rem;
    }}

    /* ---------- Icon ---------- */
    .icon-wrap {{
        display: flex;
        justify-content: center;
        margin-bottom: 1.6rem;
    }}
    .icon-glow {{
        filter: drop-shadow(0 0 18px rgba(108, 124, 246, 0.45));
    }}

    /* ---------- Header ---------- */
    .app-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 0.55rem;
    }}
    .app-subtitle {{
        font-size: 0.98rem;
        color: {TEXT_MUTED};
        text-align: center;
        margin-bottom: 2.6rem;
        font-weight: 400;
    }}

    /* ---------- Workspace panel ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {PANEL} !important;
        border: 1px solid {PANEL_BORDER} !important;
        border-radius: 20px !important;
        padding: 1.6rem 1.6rem 1.5rem 1.6rem;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        box-shadow:
            0 1px 0 0 rgba(255, 255, 255, 0.04) inset,
            0 20px 60px -20px rgba(0, 0, 0, 0.6) !important;
    }}

    div[data-testid="stTextArea"] textarea {{
        background: {INPUT_BG} !important;
        border-radius: 13px !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        padding: 1rem 1.1rem !important;
        font-size: 0.96rem !important;
        line-height: 1.55 !important;
        color: {TEXT_PRIMARY} !important;
        min-height: 176px !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.35) !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    }}
    div[data-testid="stTextArea"] textarea::placeholder {{
        color: #5C6478 !important;
    }}
    div[data-testid="stTextArea"] textarea:focus {{
        border-color: rgba(108, 124, 246, 0.55) !important;
        box-shadow:
            inset 0 1px 3px rgba(0, 0, 0, 0.35),
            0 0 0 3px rgba(108, 124, 246, 0.14) !important;
    }}

    /* ---------- Button ---------- */
    div.stButton {{
        display: flex;
        justify-content: center;
        margin-top: 1.3rem;
    }}
    div.stButton > button {{
        background: linear-gradient(135deg, {INDIGO} 0%, {ELECTRIC_BLUE} 100%);
        color: white;
        border: none;
        border-radius: 11px;
        padding: 0.72rem 2.7rem;
        font-weight: 600;
        font-size: 0.94rem;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 24px -8px rgba(76, 141, 255, 0.55);
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px);
        filter: brightness(1.08);
        box-shadow: 0 10px 30px -6px rgba(76, 141, 255, 0.7);
        color: white;
    }}
    div.stButton > button:active {{
        transform: translateY(0px);
        filter: brightness(0.98);
    }}
    div.stButton > button:focus:not(:active) {{
        color: white;
    }}

    /* ---------- Result card ---------- */
    .result-card {{
        margin-top: 1.5rem;
        padding: 1.3rem 1.5rem;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid;
        backdrop-filter: blur(14px);
        animation: rise 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    }}
    @keyframes rise {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .result-left {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }}
    .result-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    .result-label {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.08rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}
    .result-confidence {{
        font-size: 0.86rem;
        font-weight: 500;
        color: {TEXT_MUTED};
        font-variant-numeric: tabular-nums;
    }}

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] {{
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 12px !important;
        color: {TEXT_MUTED} !important;
    }}

    .footer-note {{
        text-align: center;
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        margin-top: 3rem;
        opacity: 0.55;
        letter-spacing: 0.01em;
    }}

    /* ---------- Responsive ---------- */
    @media (max-width: 480px) {{
        .block-container {{
            padding-top: 3.2rem;
            padding-left: 1.1rem;
            padding-right: 1.1rem;
        }}
        .app-title {{
            font-size: 1.6rem;
        }}
        .app-subtitle {{
            font-size: 0.88rem;
            margin-bottom: 1.8rem;
        }}
        .workspace {{
            padding: 1.2rem 1.1rem 1.2rem 1.1rem;
            border-radius: 16px;
        }}
        div.stButton > button {{
            width: 100%;
            padding: 0.75rem 1rem;
        }}
        .result-card {{
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ICON (shield + envelope, inline SVG)
# ---------------------------------------------------------

st.markdown(f"""
<div class="icon-wrap">
  <svg class="icon-glow" width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="shieldGrad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="{INDIGO}"/>
        <stop offset="100%" stop-color="{ELECTRIC_BLUE}"/>
      </linearGradient>
    </defs>
    <path d="M26 3L44 9.5V22C44 33.5 36.8 42.4 26 47C15.2 42.4 8 33.5 8 22V9.5L26 3Z"
          stroke="url(#shieldGrad)" stroke-width="2" stroke-linejoin="round" fill="rgba(108,124,246,0.05)"/>
    <path d="M15.5 18.5H36.5C37.3 18.5 38 19.2 38 20V30C38 30.8 37.3 31.5 36.5 31.5H15.5C14.7 31.5 14 30.8 14 30V20C14 19.2 14.7 18.5 15.5 18.5Z"
          stroke="url(#shieldGrad)" stroke-width="1.6" stroke-linejoin="round"/>
    <path d="M14.6 19.2L26 27L37.4 19.2" stroke="url(#shieldGrad)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL (TRAINED SILENTLY IN THE BACKGROUND)
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    data = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    data = data.dropna()

    data["label"] = data["label"].map({
        "ham": 0,
        "spam": 1
    })

    X_train, _, y_train, _ = train_test_split(
        data["message"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"]
    )

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


vectorizer, model = load_model()

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.markdown('<div class="app-title">Spam Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Paste in a message to check whether it\'s spam</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    message = st.text_area(
        "Message",
        placeholder="Enter your message here...",
        label_visibility="collapsed"
    )

    classify_clicked = st.button("Analyze Message")

    if classify_clicked:

        if message.strip() == "":
            st.warning("Please enter a message to analyze.")

        else:
            message_tfidf = vectorizer.transform([message])
            prediction = model.predict(message_tfidf)[0]
            probability = model.predict_proba(message_tfidf)[0]

            if prediction == 1:
                confidence = probability[1] * 100
                st.markdown(f"""
                <div class="result-card" style="background:{CRIMSON_DIM}; border-color:rgba(229, 72, 77, 0.35);">
                    <div class="result-left">
                        <span class="result-dot" style="background:{CRIMSON}; box-shadow:0 0 8px {CRIMSON};"></span>
                        <span class="result-label" style="color:{CRIMSON};">Spam</span>
                    </div>
                    <span class="result-confidence">{confidence:.1f}% confidence</span>
                </div>
                """, unsafe_allow_html=True)

            else:
                confidence = probability[0] * 100
                st.markdown(f"""
                <div class="result-card" style="background:{EMERALD_DIM}; border-color:rgba(61, 214, 140, 0.30);">
                    <div class="result-left">
                        <span class="result-dot" style="background:{EMERALD}; box-shadow:0 0 8px {EMERALD};"></span>
                        <span class="result-label" style="color:{EMERALD};">Not Spam</span>
                    </div>
                    <span class="result-confidence">{confidence:.1f}% confidence</span>
                </div>
                """, unsafe_allow_html=True)

st.markdown('<div class="footer-note">Powered by TF-IDF + Naive Bayes</div>', unsafe_allow_html=True)