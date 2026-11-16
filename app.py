from __future__ import annotations

import streamlit as st

from config import DISEASES
from dashboards import breast_cancer, diabetes, heart_disease
from dashboards.shared import render_model_performance

NAV_ITEMS = (
    "Overview",
    "Breast Cancer",
    "Heart Disease",
    "Diabetes",
    "Analytics",
    "About",
)

DISEASE_CARDS = (
    (
        "Breast Cancer Prediction",
        "An educational model using digitized fine-needle aspirate measurements.",
        "Breast Cancer",
        '<svg viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="24" r="17" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M24 15v18M15 24h18" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/></svg>',
        "violet",
    ),
    (
        "Heart Disease Prediction",
        "Explore the Cleveland dataset's clinical and exercise-related features.",
        "Heart Disease",
        '<svg viewBox="0 0 48 48" aria-hidden="true"><path d="M24 39S7 29 7 17.5C7 8.5 18.5 6 24 15c5.5-9 17-6.5 17 2.5C41 29 24 39 24 39Z" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linejoin="round"/><path d="M12 24h8l4-7 5 14 4-7h4" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "cyan",
    ),
    (
        "Diabetes Prediction",
        "Review the Pima dataset inputs with their original meanings and ranges.",
        "Diabetes",
        '<svg viewBox="0 0 48 48" aria-hidden="true"><path d="M24 6c-5.8 8.5-13 16.4-13 24a13 13 0 0 0 26 0C37 22.4 29.8 14.5 24 6Z" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M18 29h12M24 23v12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/></svg>',
        "mint",
    ),
    (
        "Model Performance Analytics",
        "Review saved held-out metrics, confusion matrices, and model charts.",
        "Analytics",
        '<svg viewBox="0 0 48 48" aria-hidden="true"><path d="M9 39V25h8v14M20 39V11h8v28M31 39V19h8v20" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M6 40h36" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/></svg>',
        "charcoal",
    ),
)


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ivory: #F7F5F2;
            --white: #FFFFFF;
            --violet: #5B4BCE;
            --violet-soft: #F0EDFF;
            --charcoal: #25213A;
            --mint: #239D84;
            --mint-soft: #E8F5F1;
            --coral: #E47B79;
            --coral-soft: #FFF0EE;
            --border: #EAE7EF;
            --muted: #777487;
            --ink: #302D40;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: var(--ivory);
            color: var(--ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1.2rem; }
        .block-container {
            width: 100%;
            max-width: 1540px;
            padding: 1.2rem clamp(1rem, 3vw, 3.2rem) 4rem;
        }
        .st-key-top_navigation {
            position: sticky;
            top: 0;
            z-index: 999;
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 5px 22px rgba(37, 33, 58, .055);
            padding: .65rem 1rem;
            margin-bottom: 2rem;
        }
        .brand-lockup { display: flex; align-items: center; gap: .68rem; min-width: 175px; }
        .brand-mark {
            display: inline-flex;
            width: 40px;
            height: 40px;
            align-items: center;
            justify-content: center;
            border-radius: 13px;
            color: #fff;
            background: var(--violet);
            font-size: 1.15rem;
            font-weight: 800;
        }
        .brand-name { color: var(--charcoal); font-size: 1.04rem; font-weight: 760; line-height: 1.1; }
        .brand-subtitle {
            color: var(--muted);
            font-size: .59rem;
            font-weight: 730;
            letter-spacing: .11em;
            margin-top: .24rem;
        }
        .profile-chip {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: auto;
            width: 38px;
            height: 38px;
            color: var(--violet);
            border: 1px solid #E1DDF8;
            border-radius: 50%;
            background: #F8F7FF;
            font-size: .85rem;
            font-weight: 750;
        }
        .st-key-top_navigation [data-testid="stRadio"] { margin: 0; }
        .st-key-top_navigation [data-testid="stRadio"] > label { display: none; }
        .st-key-top_navigation [data-testid="stSelectbox"] { display: none; }
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            justify-content: center;
            gap: .2rem;
        }
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label {
            border: 1px solid transparent;
            border-radius: 999px;
            padding: .42rem .7rem;
            margin: 0;
            transition: background .16s ease, color .16s ease, border-color .16s ease;
        }
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            color: var(--violet);
            background: var(--violet-soft);
            border-color: #DED9FA;
        }
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label p {
            color: inherit;
            font-size: .86rem;
            font-weight: 630;
            white-space: nowrap;
        }
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }
        h1, h2, h3, h4 { color: var(--charcoal); letter-spacing: -.035em; }
        h1 { font-weight: 740; }
        h2, h3 { font-weight: 700; }
        .st-key-top_navigation p { margin-bottom: 0; }
        [data-testid="stMetric"] {
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 20px rgba(37, 33, 58, .045);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--charcoal); }
        [data-testid="stButton"] > button,
        [data-testid="stFormSubmitButton"] > button {
            min-height: 2.72rem;
            border-radius: 11px;
            border: 1px solid var(--violet);
            background: var(--violet);
            color: var(--white);
            font-weight: 670;
            transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
        }
        [data-testid="stButton"] > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            background: #4F40BD;
            border-color: #4F40BD;
            color: var(--white);
            box-shadow: 0 7px 18px rgba(91, 75, 206, .2);
            transform: translateY(-1px);
        }
        [data-testid="stFileUploader"] section {
            background: var(--white);
            border: 1px dashed #BBB4E9;
            border-radius: 16px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border);
            border-radius: 17px;
            background: var(--white);
            box-shadow: 0 7px 22px rgba(37, 33, 58, .045);
        }
        [data-testid="stForm"] {
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: clamp(1rem, 2vw, 1.8rem);
        }
        .eyebrow {
            color: var(--violet);
            font-size: .74rem;
            font-weight: 760;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .48rem;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(240px, .7fr);
            align-items: center;
            gap: 2rem;
            min-height: 340px;
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: clamp(1.7rem, 5vw, 4rem);
            margin: .25rem 0 2.3rem;
            box-shadow: 0 12px 35px rgba(37, 33, 58, .055);
        }
        .hero-copy { position: relative; z-index: 1; }
        .hero-copy h1 { max-width: 620px; font-size: clamp(2.4rem, 4.5vw, 4rem); line-height: 1.04; margin: .35rem 0 .8rem; }
        .hero-copy p { max-width: 590px; color: var(--muted); font-size: 1.05rem; line-height: 1.7; }
        .st-key-home_hero {
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: clamp(1.2rem, 3.8vw, 3rem);
            margin: .25rem 0 2.3rem;
            box-shadow: 0 12px 35px rgba(37, 33, 58, .055);
        }
        .hero-art {
            display: flex;
            min-height: 220px;
            align-items: center;
            justify-content: center;
            color: var(--violet);
        }
        .hero-art svg { width: min(100%, 340px); height: auto; }
        .feature-card {
            height: 100%;
            min-height: 218px;
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 19px;
            padding: 1.35rem 1.35rem .65rem;
            box-shadow: 0 8px 26px rgba(37, 33, 58, .05);
            transition: transform .18s ease, box-shadow .18s ease;
        }
        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 13px 30px rgba(37, 33, 58, .09);
        }
        .feature-icon {
            display: inline-flex;
            width: 48px;
            height: 48px;
            align-items: center;
            justify-content: center;
            border-radius: 15px;
            background: var(--violet-soft);
            color: var(--violet);
        }
        .feature-icon svg { width: 27px; height: 27px; }
        .feature-icon.coral { color: var(--coral); background: var(--coral-soft); }
        .feature-icon.mint { color: var(--mint); background: var(--mint-soft); }
        .feature-icon.charcoal { color: var(--charcoal); background: #F0EFF3; }
        .feature-card h3 { font-size: 1.08rem; margin: .9rem 0 .4rem; }
        .feature-card p { color: var(--muted); font-size: .9rem; line-height: 1.55; min-height: 3.2rem; }
        .step-card { min-height: 154px; }
        .step-number {
            display: inline-flex;
            width: 30px;
            height: 30px;
            align-items: center;
            justify-content: center;
            color: var(--violet);
            border-radius: 10px;
            background: var(--violet-soft);
            font-size: .8rem;
            font-weight: 800;
        }
        .page-hero {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-left: 5px solid var(--violet);
            border-radius: 19px;
            padding: 1.5rem 1.8rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 7px 22px rgba(37, 33, 58, .045);
        }
        .page-hero p { color: var(--muted); margin: .3rem 0 0; }
        .section-note { color: var(--muted); font-size: .92rem; }
        .result-card {
            background: var(--white);
            border: 1px solid var(--border);
            border-left: 5px solid var(--mint);
            border-radius: 17px;
            padding: 1.3rem 1.5rem;
            margin: .5rem 0 1rem;
            box-shadow: 0 7px 22px rgba(37, 33, 58, .05);
        }
        .result-card.risk { border-left-color: var(--coral); }
        .result-label { color: var(--muted); font-size: .82rem; font-weight: 650; }
        .result-value { color: var(--charcoal); font-size: 1.4rem; font-weight: 760; margin-top: .2rem; }
        .privacy-note {
            background: var(--mint-soft);
            border: 1px solid #CDE9DF;
            border-radius: 14px;
            color: #276B5A;
            padding: .95rem 1.1rem;
            font-size: .9rem;
        }
        @media (max-width: 1000px) {
            .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] {
                justify-content: flex-start;
            }
            .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label {
                padding: .38rem .55rem;
            }
            .hero-shell { grid-template-columns: 1fr minmax(190px, .55fr); gap: 1rem; }
        }
        @media (max-width: 680px) {
            .block-container { padding: .7rem .75rem 2rem; }
            .st-key-top_navigation { padding: .55rem; margin-bottom: 1.2rem; }
            .st-key-top_navigation [data-testid="stHorizontalBlock"] {
                display: flex;
                flex-wrap: wrap;
                gap: .35rem;
            }
            .st-key-top_navigation [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child,
            .st-key-top_navigation [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
                flex: 1 0 100% !important;
                width: 100% !important;
                min-width: 0 !important;
            }
            .st-key-top_navigation [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) {
                display: none;
            }
            .st-key-top_navigation [data-testid="stRadio"] { display: none; }
            .st-key-top_navigation [data-testid="stSelectbox"] { display: block; }
            .st-key-top_navigation [data-testid="stSelectbox"] > label { display: none; }
            .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] {
                gap: .1rem;
            }
            .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label p {
                font-size: .76rem;
            }
            .profile-chip { display: none; }
            .hero-shell { display: block; min-height: auto; padding: 1.5rem; border-radius: 21px; }
            .st-key-home_hero { padding: 1.25rem; border-radius: 21px; margin-bottom: 1.4rem; }
            .hero-copy h1 { font-size: 2.45rem; }
            .hero-art { min-height: 160px; margin-top: .6rem; }
            .page-hero { padding: 1.2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        :root {
            --ivory: #090B10;
            --white: #11151D;
            --violet: #8B5CF6;
            --violet-soft: #211933;
            --charcoal: #FFFFFF;
            --mint: #34D399;
            --mint-soft: #102A24;
            --coral: #FB7185;
            --coral-soft: #321B23;
            --border: #293244;
            --muted: #B8C1D1;
            --ink: #FFFFFF;
            --secondary-bg: #0D1118;
            --input-bg: #1B2230;
            --cyan: #22D3EE;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp,
        [data-testid="stMain"], section.main, [data-testid="stHeader"] {
            background: #090B10 !important;
            color: #FFFFFF !important;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
            width: 0 !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        .block-container {
            width: 100%;
            max-width: 1540px;
            padding: 1.15rem clamp(1rem, 3vw, 3.2rem) 4rem;
        }
        .st-key-top_navigation {
            background: #0D1118 !important;
            border: 1px solid #293244 !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, .24);
        }
        .brand-mark { color: #FFFFFF !important; background: #8B5CF6 !important; }
        .brand-name, h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
        .brand-subtitle, .hero-copy p, .page-hero p, .section-note,
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
        .feature-card p { color: #B8C1D1 !important; }
        .profile-chip {
            color: #22D3EE !important;
            border-color: #27404C !important;
            background: #10202A !important;
        }
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label,
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #E5E7EB !important;
        }
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            color: #FFFFFF !important;
            background: #2A2041 !important;
            border-color: #59429A !important;
        }
        .st-key-top_navigation [data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: #1B2230 !important;
        }
        .st-key-top_navigation [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            color: #FFFFFF !important;
            background: #1B2230 !important;
            border-color: #293244 !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"] {
            background: #11151D !important;
            border-color: #293244 !important;
            box-shadow: 0 12px 30px rgba(0, 0, 0, .16) !important;
        }
        [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
        .stApp label, .stApp legend { color: #F3F4F6 !important; }
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
        .stApp p { color: #D1D5DB; }
        .eyebrow { color: #22D3EE !important; }
        .hero-shell, .st-key-home_hero {
            background: radial-gradient(ellipse at 82% 20%, rgba(139, 92, 246, .16), transparent 40%), #0D1118 !important;
            border-color: #293244 !important;
            box-shadow: 0 18px 44px rgba(0, 0, 0, .25) !important;
        }
        .hero-copy h1 { color: #FFFFFF !important; }
        .hero-copy p { color: #B8C1D1 !important; }
        .feature-card {
            background: #11151D !important;
            border-color: #293244 !important;
            box-shadow: 0 12px 28px rgba(0, 0, 0, .19) !important;
        }
        .feature-card:hover { border-color: #59429A !important; }
        .feature-card h3 { color: #FFFFFF !important; }
        .feature-icon { background: #211933 !important; color: #B69AFF !important; }
        .feature-icon.coral { color: #FB7185 !important; background: #321B23 !important; }
        .feature-icon.cyan { color: #22D3EE !important; background: #10202A !important; }
        .feature-icon.mint { color: #34D399 !important; background: #102A24 !important; }
        .feature-icon.charcoal { color: #22D3EE !important; background: #10202A !important; }
        .step-number { color: #D7C8FF !important; background: #211933 !important; }
        .page-hero {
            background: linear-gradient(110deg, #11151D 0%, #171D28 70%, #11151D 100%) !important;
            border-color: #293244 !important;
        }
        .result-card { background: #102A24 !important; border-color: #225343 !important; }
        .result-card.risk { background: #321B23 !important; border-color: #713343 !important; }
        .result-label { color: #B8C1D1 !important; }
        .result-value { color: #FFFFFF !important; }
        [data-testid="stMetric"] {
            background: #11151D !important;
            border-color: #293244 !important;
            box-shadow: 0 10px 24px rgba(0, 0, 0, .17) !important;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #B8C1D1 !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: #FFFFFF !important; }
        [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea, [data-testid="stDateInput"] input {
            color: #FFFFFF !important;
            background: #1B2230 !important;
            border: 1px solid #3A465C !important;
            caret-color: #22D3EE !important;
        }
        [data-testid="stNumberInput"] input::placeholder,
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color: #929CAF !important;
            opacity: 1 !important;
        }
        [data-testid="stNumberInput"] input:focus, [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus,
        [data-baseweb="select"] > div:focus-within {
            border-color: #8B5CF6 !important;
            box-shadow: 0 0 0 1px #8B5CF6 !important;
        }
        [data-testid="stNumberInput"] button {
            color: #E5E7EB !important;
            background: #252D3B !important;
            border-color: #3A465C !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-baseweb="select"] [role="combobox"] {
            color: #FFFFFF !important;
            background: #1B2230 !important;
            border-color: #3A465C !important;
        }
        [data-baseweb="select"] [role="combobox"] *,
        [data-baseweb="select"] svg { color: #FFFFFF !important; fill: #FFFFFF !important; }
        [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"],
        [role="listbox"], [role="option"] {
            color: #FFFFFF !important;
            background: #171D28 !important;
            border-color: #3A465C !important;
        }
        [role="option"]:hover, [role="option"][aria-selected="true"] {
            color: #FFFFFF !important;
            background: #2A2041 !important;
        }
        [data-testid="stButton"] > button, [data-testid="stFormSubmitButton"] > button {
            color: #FFFFFF !important;
            background: #8B5CF6 !important;
            border: 1px solid #9A7BFF !important;
            font-weight: 700;
        }
        [data-testid="stButton"] > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            color: #FFFFFF !important;
            background: #7546E8 !important;
            border-color: #B69AFF !important;
            box-shadow: 0 8px 22px rgba(139, 92, 246, .25) !important;
        }
        [data-testid="stAlert"] {
            color: #F3F4F6 !important;
            background: #171D28 !important;
            border: 1px solid #3A465C !important;
        }
        [data-testid="stAlert"] p, [data-testid="stAlert"] div,
        [data-testid="stAlert"] span { color: inherit !important; }
        [data-testid="stAlert"][kind="warning"] {
            color: #FDE68A !important;
            background: #30240A !important;
            border-color: #806113 !important;
        }
        [data-testid="stAlert"][kind="error"] {
            color: #FECDD3 !important;
            background: #321B23 !important;
            border-color: #713343 !important;
        }
        [data-testid="stAlert"][kind="success"] {
            color: #A7F3D0 !important;
            background: #102A24 !important;
            border-color: #225343 !important;
        }
        [data-testid="stAlert"][kind="info"] {
            color: #A5F3FC !important;
            background: #10202A !important;
            border-color: #27505C !important;
        }
        [data-testid="stExpander"] {
            color: #FFFFFF !important;
            background: #11151D !important;
            border: 1px solid #293244 !important;
        }
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
            color: #F3F4F6 !important;
        }
        [data-testid="stTabs"] button, [data-testid="stTabs"] button p { color: #B8C1D1 !important; }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #FFFFFF !important;
            border-bottom-color: #8B5CF6 !important;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid #293244 !important;
            border-radius: 12px;
            overflow: hidden;
        }
        [data-testid="stImage"] img {
            border-radius: 12px;
            border: 1px solid #293244;
        }
        [role="tooltip"], [data-baseweb="tooltip"] {
            color: #FFFFFF !important;
            background: #171D28 !important;
            border: 1px solid #3A465C !important;
        }
        @media (max-width: 680px) {
            .block-container { padding: .7rem .75rem 2rem; }
            .st-key-top_navigation { padding: .55rem; margin-bottom: 1.2rem; }
            .hero-copy h1 { font-size: 2.45rem; }
            .hero-art { min-height: 160px; margin-top: .6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _select_page(page: str) -> None:
    st.session_state["main_navigation"] = page
    st.session_state["main_navigation_desktop"] = page
    st.session_state["main_navigation_mobile"] = page


def _sync_desktop_navigation() -> None:
    page = st.session_state["main_navigation_desktop"]
    st.session_state["main_navigation"] = page
    st.session_state["main_navigation_mobile"] = page


def _sync_mobile_navigation() -> None:
    page = st.session_state["main_navigation_mobile"]
    st.session_state["main_navigation"] = page
    st.session_state["main_navigation_desktop"] = page


def _render_hero_art() -> str:
    return """
    <svg viewBox="0 0 360 260" role="img" aria-label="Abstract connected health data illustration">
        <circle cx="180" cy="130" r="96" fill="#171D28"/>
        <circle cx="180" cy="130" r="68" fill="#0D1118" stroke="#59429A" stroke-width="2"/>
        <circle cx="180" cy="130" r="42" fill="#8B5CF6"/>
        <path d="M158 132h13l9-16 13 31 10-19h18" fill="none" stroke="#FFFFFF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="82" cy="74" r="13" fill="#102A24" stroke="#34D399" stroke-width="3"/>
        <circle cx="282" cy="72" r="11" fill="#321B23" stroke="#FB7185" stroke-width="3"/>
        <circle cx="285" cy="196" r="14" fill="#10202A" stroke="#22D3EE" stroke-width="3"/>
        <circle cx="82" cy="191" r="10" fill="#211933" stroke="#8B5CF6" stroke-width="3"/>
        <path d="M96 82l38 23M263 81l-35 22M266 187l-34-28M94 184l38-25" stroke="#596579" stroke-width="2" stroke-dasharray="4 6"/>
        <rect x="41" y="115" width="56" height="31" rx="11" fill="#1B2230" stroke="#3A465C"/>
        <path d="M51 131h10l5-8 7 14 5-7h10" fill="none" stroke="#34D399" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        <rect x="260" y="115" width="61" height="31" rx="11" fill="#1B2230" stroke="#3A465C"/>
        <path d="M272 136v-9h7v9m4 0v-15h7v15m4 0v-6h7v6" fill="none" stroke="#22D3EE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """


def _render_home() -> None:
    with st.container(key="home_hero"):
        hero_copy, hero_art = st.columns([1.35, 0.65], vertical_alignment="center")
        with hero_copy:
            st.markdown('<div class="eyebrow">Health intelligence platform</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="hero-copy"><h1>Your Health. Smarter Insights.</h1></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="section-note">Explore AI-powered health prediction and analytics.</p>',
                unsafe_allow_html=True,
            )
            st.write(
                "HealthAI brings three disease-specific machine-learning models and "
                "their saved evaluations into a focused workspace for manual exploration."
            )
            st.button(
                "Explore Predictions",
                key="explore_predictions",
                on_click=_select_page,
                args=("Breast Cancer",),
            )
        with hero_art:
            st.markdown(
                f'<div class="hero-art">{_render_hero_art()}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("## Explore the platform")
    st.caption("Choose a workspace to continue.")
    card_columns = st.columns(4, gap="medium")
    for column, (title, description, page, icon, accent) in zip(
        card_columns, DISEASE_CARDS
    ):
        with column:
            st.markdown(
                f"""
                <article class="feature-card">
                    <span class="feature-icon {accent}">{icon}</span>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </article>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "Open dashboard",
                key=f"open_{page}",
                use_container_width=True,
                on_click=_select_page,
                args=(page,),
            )

    st.markdown("## A simple, transparent workflow")
    step_columns = st.columns(4, gap="medium")
    steps = (
        ("Enter health features", "Use the source dataset fields and their documented ranges."),
        ("Choose a model", "Select one of the saved disease-specific classifiers."),
        ("Generate a prediction", "Submit complete, validated values for the selected model."),
        ("Review model analytics", "Compare saved evaluation metrics and confusion matrices."),
    )
    for index, (column, (title, description)) in enumerate(zip(step_columns, steps), 1):
        with column, st.container(border=True):
            st.markdown(f'<span class="step-number">{index:02d}</span>', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.caption(description)

def _render_about() -> None:
    st.markdown('<div class="eyebrow">Platform information</div>', unsafe_allow_html=True)
    st.title("About HealthAI")
    st.write(
        "HealthAI is a multi-disease machine-learning demonstration that combines manual "
        "feature entry, trained classifiers, and saved evaluation reports."
    )
    overview, objectives = st.columns(2, gap="large")
    with overview, st.container(border=True):
        st.subheader("Project overview")
        st.write(
            "Each dashboard uses its own preserved preprocessing pipeline, model "
            "artifacts, feature schema, and held-out evaluation reports."
        )
        st.caption("Educational software; not intended for clinical use.")
    with objectives, st.container(border=True):
        st.subheader("Project objectives")
        st.markdown(
            "- Demonstrate disease-specific prediction workflows.\n"
            "- Preserve each model's original feature definitions and input order.\n"
            "- Make evaluation metrics and limitations easy to inspect."
        )
    st.subheader("Technologies and algorithms")
    technology_columns = st.columns(2, gap="large")
    with technology_columns[0], st.container(border=True):
        st.markdown("**Application stack**")
        st.write("Python · Streamlit · pandas · NumPy · scikit-learn · Joblib")
        st.write("Evaluation charts and summaries are generated with Matplotlib.")
    with technology_columns[1], st.container(border=True):
        st.markdown("**Machine-learning algorithms**")
        st.write("Logistic Regression · Random Forest · Support Vector Machine (SVM)")
        st.caption("Saved model artifacts are loaded; the application does not retrain them.")
    st.subheader("Dataset information")
    for disease in DISEASES.values():
        with st.container(border=True):
            st.markdown(f"**[{disease['source']}]({disease['source_url']})**")
            st.caption(disease["limitations"])
    st.subheader("Application workflow")
    st.write(
        "Choose a disease, enter all required features manually, select a saved model, "
        "review its prediction output, and compare the saved evaluation results in Analytics."
    )


def main() -> None:
    st.set_page_config(
        page_title="HealthAI · AI Healthcare Intelligence Platform",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()
    st.session_state.setdefault("main_navigation", "Overview")
    st.session_state.setdefault(
        "main_navigation_desktop",
        st.session_state["main_navigation"],
    )
    st.session_state.setdefault(
        "main_navigation_mobile",
        st.session_state["main_navigation"],
    )

    with st.container(key="top_navigation"):
        brand_column, navigation_column, profile_column = st.columns(
            [2.1, 8.7, 0.45],
            gap="small",
            vertical_alignment="center",
        )
        with brand_column:
            st.markdown(
                """
                <div class="brand-lockup">
                    <span class="brand-mark">H</span>
                    <div>
                        <div class="brand-name">HealthAI</div>
                        <div class="brand-subtitle">AI HEALTHCARE INTELLIGENCE</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with navigation_column:
            page = st.radio(
                "Top navigation",
                NAV_ITEMS,
                horizontal=True,
                key="main_navigation_desktop",
                on_change=_sync_desktop_navigation,
                label_visibility="collapsed",
            )
            st.selectbox(
                "Navigate",
                NAV_ITEMS,
                key="main_navigation_mobile",
                on_change=_sync_mobile_navigation,
                label_visibility="collapsed",
            )
        with profile_column:
            st.markdown('<div class="profile-chip">AI</div>', unsafe_allow_html=True)

    page = st.session_state["main_navigation"]
    if page == "Overview":
        _render_home()
    elif page == "Breast Cancer":
        breast_cancer.render()
    elif page == "Heart Disease":
        heart_disease.render()
    elif page == "Diabetes":
        diabetes.render()
    elif page == "Analytics":
        render_model_performance()
    else:
        _render_about()


if __name__ == "__main__":
    main()
