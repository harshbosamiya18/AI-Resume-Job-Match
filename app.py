import os
import logging
import warnings
import html
import tempfile
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# Suppress Hugging Face / Transformers docstring and general warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# FIX: Streamlit/Markdown treats any line indented 4+ spaces as an
# indented code block (CommonMark rule), which overrides
# unsafe_allow_html=True and renders the raw HTML tags as text with a
# copy-button code box instead of actual styled HTML. All of the HTML
# strings below are written with nested indentation for readability,
# so we patch st.markdown to strip leading whitespace from each line
# whenever unsafe_allow_html=True is used, without touching every
# individual call site.
_original_markdown = st.markdown


def _markdown_no_indent(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = "\n".join(line.lstrip() for line in body.splitlines())
    return _original_markdown(body, *args, **kwargs)


st.markdown = _markdown_no_indent

from src.resume_parser import extract_resume_text
from src.llm import generate_ai_analysis

from utils.helpers import (
    clean_ai_response,
    clean_whitespace,
    format_percentage,
    format_skill_list,
    get_score_description,
    get_score_label,
    normalize_skill_list,
    safe_list,
    validate_text_input,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Resume & Job Match",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# LOCAL HELPER FUNCTIONS
# ==================================================

def safe_score(value):
    """
    Convert a value into a safe float between 0 and 100.
    """

    try:
        score = float(value)

    except (
        TypeError,
        ValueError,
    ):
        score = 0.0

    return max(
        0.0,
        min(
            score,
            100.0,
        ),
    )


def get_score_progress(score):
    """
    Convert a percentage score into a Streamlit
    progress value between 0.0 and 1.0.
    """

    return safe_score(score) / 100.0


def get_score_class(score):
    """
    Return a CSS class based on the score.
    """

    score = safe_score(score)

    if score >= 85:
        return "excellent"

    if score >= 70:
        return "good"

    if score >= 50:
        return "moderate"

    return "low"


def get_score_status(score):
    """
    Return a short status label based on the score.
    """

    score = safe_score(score)

    if score >= 85:
        return "Excellent Match"

    if score >= 70:
        return "Strong Match"

    if score >= 50:
        return "Moderate Match"

    if score >= 35:
        return "Low Match"

    return "Needs Improvement"


def create_skill_chips(
    skills,
    chip_type="default",
):
    """
    Convert a list of skills into safe styled HTML chips.
    """

    skills = normalize_skill_list(
        safe_list(skills)
    )

    if not skills:
        return (
            '<span class="empty-skills">'
            'No skills detected.'
            '</span>'
        )

    chips = []

    for skill in skills:

        safe_skill = html.escape(
            str(skill)
        )

        chips.append(
            f"""
            <span class="skill-chip {chip_type}">
                {safe_skill}
            </span>
            """
        )

    return "".join(chips)


def display_metric_card(
    title,
    score,
    subtitle,
    icon,
):
    """
    Display a modern score metric card.
    """

    score = safe_score(score)

    score_class = get_score_class(
        score
    )

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-top">

                <div class="metric-icon">
                    {icon}
                </div>

                <div class="metric-title">
                    {html.escape(str(title))}
                </div>

            </div>

            <div class="metric-score {score_class}">
                {format_percentage(score)}
            </div>

            <div class="metric-subtitle">
                {html.escape(str(subtitle))}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        get_score_progress(score)
    )


def display_skill_panel(
    title,
    skills,
    panel_class,
    empty_message,
):
    """
    Display a skill panel with styled chips.
    """

    skills = normalize_skill_list(
        safe_list(skills)
    )

    if skills:

        content = create_skill_chips(
            skills=skills,
            chip_type=panel_class,
        )

    else:

        content = (
            '<div class="empty-panel">'
            f'{html.escape(str(empty_message))}'
            '</div>'
        )

    st.markdown(
        f"""
        <div class="skill-panel">

            <div class="skill-panel-title">
                {html.escape(str(title))}
            </div>

            <div class="skill-chip-container">
                {content}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def display_analysis_content(ai_analysis):
    """
    Display AI-generated analysis safely.
    """

    cleaned_analysis = clean_ai_response(
        ai_analysis
    )

    if cleaned_analysis:

        st.markdown(
            cleaned_analysis
        )

    else:

        st.warning(
            "AI analysis was not generated."
        )


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    /* ==============================================
       GLOBAL
    ============================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(124, 58, 237, 0.10),
                transparent 32%
            ),
            radial-gradient(
                circle at top right,
                rgba(59, 130, 246, 0.08),
                transparent 28%
            );
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ==============================================
       SIDEBAR
    ============================================== */

    [data-testid="stSidebar"] {
        border-right:
            1px solid rgba(128, 128, 128, 0.15);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        padding: 1rem;
        border-radius: 18px;
        margin-bottom: 1rem;

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.16),
                rgba(59, 130, 246, 0.10)
            );

        border:
            1px solid rgba(124, 58, 237, 0.20);
    }

    .sidebar-brand h2 {
        margin: 0;
        font-size: 1.3rem;
    }

    .sidebar-brand p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 0.88rem;
        opacity: 0.75;
        line-height: 1.5;
    }

    .pipeline-step {
        padding: 0.65rem 0.75rem;
        margin-bottom: 0.45rem;

        border-radius: 10px;

        background:
            rgba(128, 128, 128, 0.06);

        border:
            1px solid rgba(128, 128, 128, 0.08);

        font-size: 0.9rem;
    }

    .tech-badge {
        display: inline-block;

        padding:
            0.35rem 0.65rem;

        margin:
            0.18rem;

        border-radius:
            999px;

        font-size:
            0.78rem;

        background:
            rgba(124, 58, 237, 0.10);

        border:
            1px solid rgba(124, 58, 237, 0.18);
    }


    /* ==============================================
       HERO
    ============================================== */

    .hero-card {
        position: relative;

        overflow: hidden;

        padding:
            2.8rem
            2.5rem;

        margin-bottom:
            2rem;

        border-radius:
            28px;

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.13),
                rgba(59, 130, 246, 0.08)
            );

        border:
            1px solid rgba(124, 58, 237, 0.20);

        box-shadow:
            0 20px 50px
            rgba(0, 0, 0, 0.06);
    }

    .hero-card::before {
        content: "";

        position: absolute;

        width: 320px;
        height: 320px;

        border-radius: 50%;

        top: -180px;
        right: -100px;

        background:
            rgba(124, 58, 237, 0.12);

        filter:
            blur(30px);
    }

    .hero-content {
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;

        padding:
            0.45rem
            0.85rem;

        border-radius:
            999px;

        margin-bottom:
            1rem;

        font-size:
            0.85rem;

        font-weight:
            600;

        background:
            rgba(124, 58, 237, 0.14);

        border:
            1px solid rgba(124, 58, 237, 0.25);
    }

    .hero-card h1 {
        margin:
            0
            0
            0.7rem
            0;

        font-size:
            clamp(
                2rem,
                4vw,
                3.5rem
            );

        line-height:
            1.1;

        letter-spacing:
            -0.04em;
    }

    .hero-card p {
        max-width:
            800px;

        margin:
            0;

        font-size:
            1.05rem;

        line-height:
            1.7;

        opacity:
            0.78;
    }

    .hero-stats {
        display:
            flex;

        flex-wrap:
            wrap;

        gap:
            0.8rem;

        margin-top:
            1.5rem;
    }

    .hero-stat {
        padding:
            0.55rem
            0.9rem;

        border-radius:
            999px;

        font-size:
            0.82rem;

        background:
            rgba(255, 255, 255, 0.04);

        border:
            1px solid rgba(128, 128, 128, 0.15);
    }


    /* ==============================================
       SECTION HEADINGS
    ============================================== */

    .section-header {
        margin-top:
            2rem;

        margin-bottom:
            1rem;
    }

    .section-header h2 {
        margin:
            0;

        font-size:
            1.5rem;

        letter-spacing:
            -0.02em;
    }

    .section-header p {
        margin:
            0.35rem
            0
            0
            0;

        opacity:
            0.7;

        font-size:
            0.95rem;
    }


    /* ==============================================
       INPUT CARDS
    ============================================== */

    .input-card {
        min-height:
            160px;

        padding:
            1.4rem;

        border-radius:
            20px;

        background:
            rgba(255, 255, 255, 0.025);

        border:
            1px solid rgba(128, 128, 128, 0.15);

        box-shadow:
            0 10px 30px
            rgba(0, 0, 0, 0.03);
    }

    .input-card-title {
        display:
            flex;

        align-items:
            center;

        gap:
            0.6rem;

        margin-bottom:
            0.5rem;

        font-size:
            1.15rem;

        font-weight:
            700;
    }

    .input-card-description {
        margin-bottom:
            1rem;

        font-size:
            0.9rem;

        opacity:
            0.7;

        line-height:
            1.5;
    }


    /* ==============================================
       BUTTON
    ============================================== */

    div.stButton > button {
        min-height:
            56px;

        border-radius:
            16px;

        font-size:
            1rem;

        font-weight:
            700;

        letter-spacing:
            0.01em;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    div.stButton > button:hover {
        transform:
            translateY(-2px);

        box-shadow:
            0 10px 30px
            rgba(124, 58, 237, 0.22);
    }


    /* ==============================================
       RESULT HERO
    ============================================== */

    .result-hero {
        padding:
            2rem;

        margin-bottom:
            1.5rem;

        border-radius:
            24px;

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.12),
                rgba(59, 130, 246, 0.07)
            );

        border:
            1px solid rgba(124, 58, 237, 0.18);
    }

    .result-grid {
        display:
            grid;

        grid-template-columns:
            minmax(180px, 0.8fr)
            1fr;

        gap:
            2rem;

        align-items:
            center;
    }

    .main-score {
        width:
            170px;

        height:
            170px;

        border-radius:
            50%;

        display:
            flex;

        flex-direction:
            column;

        align-items:
            center;

        justify-content:
            center;

        margin:
            auto;

        background:
            rgba(255, 255, 255, 0.04);

        border:
            8px solid rgba(124, 58, 237, 0.25);

        box-shadow:
            inset 0 0 30px
            rgba(124, 58, 237, 0.08);
    }

    .main-score-value {
        font-size:
            2.4rem;

        font-weight:
            800;
    }

    .main-score-label {
        margin-top:
            0.15rem;

        font-size:
            0.75rem;

        opacity:
            0.65;

        text-transform:
            uppercase;

        letter-spacing:
            0.08em;
    }

    .result-status {
        display:
            inline-block;

        padding:
            0.4rem
            0.8rem;

        border-radius:
            999px;

        margin-bottom:
            0.7rem;

        background:
            rgba(124, 58, 237, 0.12);

        font-size:
            0.82rem;

        font-weight:
            700;
    }

    .result-hero h2 {
        margin:
            0
            0
            0.5rem
            0;

        font-size:
            1.8rem;
    }

    .result-hero p {
        margin:
            0;

        opacity:
            0.75;

        line-height:
            1.7;
    }


    /* ==============================================
       METRIC CARDS
    ============================================== */

    .metric-card {
        min-height:
            155px;

        padding:
            1.2rem;

        border-radius:
            18px;

        margin-bottom:
            0.65rem;

        background:
            rgba(255, 255, 255, 0.025);

        border:
            1px solid rgba(128, 128, 128, 0.14);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }

    .metric-card:hover {
        transform:
            translateY(-3px);

        border-color:
            rgba(124, 58, 237, 0.35);
    }

    .metric-top {
        display:
            flex;

        align-items:
            center;

        gap:
            0.6rem;
    }

    .metric-icon {
        width:
            38px;

        height:
            38px;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        border-radius:
            12px;

        background:
            rgba(124, 58, 237, 0.12);

        font-size:
            1.1rem;
    }

    .metric-title {
        font-size:
            0.85rem;

        font-weight:
            600;

        opacity:
            0.8;
    }

    .metric-score {
        margin-top:
            1rem;

        font-size:
            2rem;

        font-weight:
            800;

        letter-spacing:
            -0.03em;
    }

    .metric-score.excellent {
        color:
            #22c55e;
    }

    .metric-score.good {
        color:
            #3b82f6;
    }

    .metric-score.moderate {
        color:
            #f59e0b;
    }

    .metric-score.low {
        color:
            #ef4444;
    }

    .metric-subtitle {
        margin-top:
            0.25rem;

        font-size:
            0.78rem;

        opacity:
            0.65;
    }


    /* ==============================================
       SKILL PANELS
    ============================================== */

    .skill-panel {
        min-height:
            220px;

        padding:
            1.4rem;

        border-radius:
            20px;

        background:
            rgba(255, 255, 255, 0.025);

        border:
            1px solid rgba(128, 128, 128, 0.14);
    }

    .skill-panel-title {
        margin-bottom:
            1rem;

        font-size:
            1.05rem;

        font-weight:
            700;
    }

    .skill-chip-container {
        display:
            flex;

        flex-wrap:
            wrap;

        gap:
            0.55rem;
    }

    .skill-chip {
        display:
            inline-block;

        padding:
            0.45rem
            0.75rem;

        border-radius:
            999px;

        font-size:
            0.82rem;

        font-weight:
            600;

        background:
            rgba(128, 128, 128, 0.08);

        border:
            1px solid rgba(128, 128, 128, 0.14);
    }

    .skill-chip.match {
        background:
            rgba(34, 197, 94, 0.12);

        border-color:
            rgba(34, 197, 94, 0.28);
    }

    .skill-chip.missing {
        background:
            rgba(245, 158, 11, 0.10);

        border-color:
            rgba(245, 158, 11, 0.25);
    }

    .skill-chip.resume {
        background:
            rgba(59, 130, 246, 0.10);

        border-color:
            rgba(59, 130, 246, 0.24);
    }

    .skill-chip.job {
        background:
            rgba(124, 58, 237, 0.11);

        border-color:
            rgba(124, 58, 237, 0.24);
    }

    .empty-panel {
        padding:
            1rem;

        border-radius:
            12px;

        opacity:
            0.7;

        background:
            rgba(128, 128, 128, 0.05);
    }


    /* ==============================================
       AI ANALYSIS
    ============================================== */

    .ai-analysis-header {
        padding:
            1.5rem;

        border-radius:
            20px;

        margin-bottom:
            1rem;

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.12),
                rgba(59, 130, 246, 0.06)
            );

        border:
            1px solid rgba(124, 58, 237, 0.18);
    }

    .ai-analysis-header h2 {
        margin:
            0
            0
            0.5rem
            0;
    }

    .ai-analysis-header p {
        margin:
            0;

        opacity:
            0.75;

        line-height:
            1.6;
    }

    .analysis-container {
        padding:
            1.8rem;

        border-radius:
            20px;

        background:
            rgba(255, 255, 255, 0.025);

        border:
            1px solid rgba(128, 128, 128, 0.14);
    }

    .analysis-container h2,
    .analysis-container h3 {
        margin-top:
            1.4rem;

        padding-bottom:
            0.5rem;

        border-bottom:
            1px solid rgba(128, 128, 128, 0.12);
    }

    .analysis-container h2:first-child,
    .analysis-container h3:first-child {
        margin-top:
            0;
    }


    /* ==============================================
       FOOTER
    ============================================== */

    .footer {
        margin-top:
            3rem;

        padding:
            2rem
            1rem;

        text-align:
            center;

        border-top:
            1px solid rgba(128, 128, 128, 0.12);

        opacity:
            0.7;

        font-size:
            0.85rem;
    }


    /* ==============================================
       MOBILE
    ============================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left:
                1rem;

            padding-right:
                1rem;
        }

        .hero-card {
            padding:
                1.8rem
                1.4rem;
        }

        .result-grid {
            grid-template-columns:
                1fr;

            text-align:
                center;
        }

        .main-score {
            margin-bottom:
                0.5rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# HERO SECTION
# ==================================================

st.markdown(
    """
    <div class="hero-card">

        <div class="hero-content">

            <div class="hero-badge">
                ✨ Intelligent Resume Matching System
            </div>

            <h1>
                AI Resume & Job Match
            </h1>

            <p>
                Analyze how well a candidate matches a job using
                NLP, Machine Learning, semantic embeddings,
                Retrieval-Augmented Generation and Generative AI.
            </p>

            <div class="hero-stats">

                <div class="hero-stat">
                    🧠 NLP
                </div>

                <div class="hero-stat">
                    📊 Machine Learning
                </div>

                <div class="hero-stat">
                    🔍 Semantic Search
                </div>

                <div class="hero-stat">
                    📚 RAG
                </div>

                <div class="hero-stat">
                    ✨ Generative AI
                </div>

                <div class="hero-stat">
                    💻 CS & IT Skills
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <h2>
                🤖 AI Match Engine
            </h2>

            <p>
                An intelligent resume analysis system designed
                to compare candidate skills with job requirements.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "🧠 AI Pipeline"
    )

    pipeline_steps = [
        "1. Resume Parsing",
        "2. NLP Preprocessing",
        "3. Skill Extraction",
        "4. TF-IDF Similarity",
        "5. Semantic Similarity",
        "6. Exact Skill Matching",
        "7. Advanced AI Scoring",
        "8. RAG Knowledge Retrieval",
        "9. Gemini AI Analysis",
    ]

    for step in pipeline_steps:

        st.markdown(
            f"""
            <div class="pipeline-step">
                {html.escape(step)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader(
        "🛠️ Technologies"
    )

    technologies = [
        "Python",
        "spaCy",
        "Pandas",
        "Scikit-learn",
        "Sentence Transformers",
        "TF-IDF",
        "Cosine Similarity",
        "Embeddings",
        "RAG",
        "Gemini",
        "Streamlit",
    ]

    technology_html = ""

    for technology in technologies:

        technology_html += (
            '<span class="tech-badge">'
            f'{html.escape(technology)}'
            '</span>'
        )

    st.markdown(
        technology_html,
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader(
        "📌 How It Works"
    )

    st.caption(
        "1. Upload a PDF or TXT resume"
    )

    st.caption(
        "2. Paste the target job description"
    )

    st.caption(
        "3. Run AI-powered analysis"
    )

    st.caption(
        "4. Review scores, skills and AI insights"
    )

    st.divider()

    st.caption(
        "Built with NLP, Machine Learning, "
        "RAG and Generative AI."
    )


# ==================================================
# INPUT SECTION
# ==================================================

st.markdown(
    """
    <div class="section-header">

        <h2>
            📥 Start Your Analysis
        </h2>

        <p>
            Provide the candidate resume and target job description.
            The AI pipeline will perform the complete analysis.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


left_column, right_column = st.columns(
    2,
    gap="large",
)


# ==================================================
# RESUME INPUT
# ==================================================

with left_column:

    st.markdown(
        """
        <div class="input-card">

            <div class="input-card-title">
                📄 Upload Resume
            </div>

            <div class="input-card-description">
                Upload the candidate's resume in PDF or TXT format.
                The application will extract the text and analyze
                skills and technical alignment.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_resume = st.file_uploader(
        "Choose Resume File",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_resume is not None:

        st.success(
            f"Resume ready: {uploaded_resume.name}"
        )


# ==================================================
# JOB DESCRIPTION INPUT
# ==================================================

with right_column:

    st.markdown(
        """
        <div class="input-card">

            <div class="input-card-title">
                💼 Target Job Description
            </div>

            <div class="input-card-description">
                Paste the complete job description. Include
                responsibilities, required skills, preferred
                qualifications and technical requirements.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    job_description = st.text_area(
        "Job Description",
        height=250,
        placeholder=(
            "Paste the complete job description here...\n\n"
            "Example:\n\n"
            "We are looking for a Machine Learning Engineer "
            "with experience in Python, SQL, Machine Learning, "
            "TensorFlow, Docker and cloud technologies."
        ),
        label_visibility="collapsed",
    )


# ==================================================
# ANALYZE BUTTON
# ==================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True,
)

analyze_button = st.button(
    "✨ Analyze Resume with AI",
    type="primary",
    use_container_width=True,
)


# ==================================================
# ANALYSIS
# ==================================================

if analyze_button:

    if uploaded_resume is None:

        st.error(
            "📄 Please upload a resume PDF before "
            "starting the analysis."
        )

    else:

        try:

            job_description = validate_text_input(
                text=job_description,
                field_name="Job Description",
                min_length=20,
            )

        except ValueError as error:

            st.error(
                f"💼 {error}"
            )

        else:

            temporary_pdf_path = None

            try:

                # ==========================================
                # SAVE TEMPORARY PDF
                # ==========================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_resume.name).suffix.lower() or ".pdf",
                ) as temporary_file:

                    temporary_file.write(
                        uploaded_resume.getbuffer()
                    )

                    temporary_pdf_path = (
                        temporary_file.name
                    )


                # ==========================================
                # PROCESSING STATUS
                # ==========================================

                st.markdown(
                    """
                    <div class="section-header">

                        <h2>
                            ⚙️ AI Analysis in Progress
                        </h2>

                        <p>
                            Processing your resume through the
                            NLP, Machine Learning, RAG and
                            Generative AI pipeline.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                progress_bar = st.progress(
                    0
                )

                status_placeholder = st.empty()


                # ==========================================
                # STEP 1: RESUME PARSING
                # ==========================================

                status_placeholder.info(
                    "📄 Step 1/4: Extracting text from resume..."
                )

                progress_bar.progress(
                    15
                )

                resume_text = extract_resume_text(
                    temporary_pdf_path
                )

                resume_text = clean_whitespace(
                    resume_text
                )

                if not resume_text:

                    raise ValueError(
                        "No readable text could be extracted "
                        "from the uploaded PDF."
                    )


                # ==========================================
                # STEP 2: NLP + ML
                # ==========================================

                status_placeholder.info(
                    "🧠 Step 2/4: Running NLP, skill extraction "
                    "and machine learning matching..."
                )

                progress_bar.progress(
                    35
                )


                # ==========================================
                # STEP 3: RAG + GEMINI
                # ==========================================

                status_placeholder.info(
                    "🔍 Step 3/4: Retrieving relevant CS/IT "
                    "knowledge and generating AI insights..."
                )

                progress_bar.progress(
                    60
                )

                output = generate_ai_analysis(
                    resume_text=resume_text,
                    job_text=job_description,
                )


                # ==========================================
                # STEP 4: RESULTS
                # ==========================================

                status_placeholder.info(
                    "📊 Step 4/4: Preparing your complete "
                    "analysis dashboard..."
                )

                progress_bar.progress(
                    90
                )


                # ==========================================
                # EXTRACT OUTPUT
                # ==========================================

                if not isinstance(
                    output,
                    dict,
                ):
                    raise ValueError(
                        "The AI analysis returned an "
                        "invalid response."
                    )

                results = output.get(
                    "results",
                    {},
                )

                if not isinstance(
                    results,
                    dict,
                ):
                    results = {}

                ai_analysis = output.get(
                    "ai_analysis",
                    "",
                )

                rag_context = output.get(
                    "rag_context",
                    "",
                )


                # ==========================================
                # SCORES
                # ==========================================

                final_score = safe_score(
                    results.get(
                        "final_score",
                        0,
                    )
                )

                skill_score = safe_score(
                    results.get(
                        "skill_score",
                        0,
                    )
                )

                tfidf_score = safe_score(
                    results.get(
                        "tfidf_score",
                        0,
                    )
                )

                semantic_score = safe_score(
                    results.get(
                        "semantic_score",
                        0,
                    )
                )


                # ==========================================
                # SKILLS
                # ==========================================

                matching_skills = normalize_skill_list(
                    safe_list(
                        results.get(
                            "matching_skills",
                            [],
                        )
                    )
                )

                missing_skills = normalize_skill_list(
                    safe_list(
                        results.get(
                            "missing_skills",
                            [],
                        )
                    )
                )

                resume_skills = normalize_skill_list(
                    safe_list(
                        results.get(
                            "resume_skills",
                            [],
                        )
                    )
                )

                job_skills = normalize_skill_list(
                    safe_list(
                        results.get(
                            "job_skills",
                            [],
                        )
                    )
                )


                # ==========================================
                # FINISH PROGRESS
                # ==========================================

                progress_bar.progress(
                    100
                )

                status_placeholder.success(
                    "🎉 Analysis completed successfully!"
                )

                st.markdown(
                    "<br>",
                    unsafe_allow_html=True,
                )


                # ==========================================
                # RESULTS HEADER
                # ==========================================

                st.markdown(
                    """
                    <div class="section-header">

                        <h2>
                            🎯 Match Results
                        </h2>

                        <p>
                            A combined evaluation based on exact
                            skill matching, TF-IDF similarity and
                            semantic similarity.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                # ==========================================
                # MAIN RESULT CARD
                # ==========================================

                score_status = get_score_status(
                    final_score
                )

                score_label = get_score_label(
                    final_score
                )

                score_message = get_score_description(
                    final_score
                )

                st.markdown(
                    f"""
                    <div class="result-hero">

                        <div class="result-grid">

                            <div class="main-score">

                                <div class="main-score-value">
                                    {format_percentage(final_score)}
                                </div>

                                <div class="main-score-label">
                                    AI Match Score
                                </div>

                            </div>

                            <div>

                                <div class="result-status">
                                    {html.escape(score_status)}
                                </div>

                                <h2>
                                    {html.escape(score_label)}
                                </h2>

                                <p>
                                    {html.escape(score_message)}
                                </p>

                            </div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(
                    get_score_progress(
                        final_score
                    )
                )

                st.caption(
                    "Overall AI Match Score: "
                    f"{format_percentage(final_score)}"
                )


                # ==========================================
                # DETAILED SCORES
                # ==========================================

                st.markdown(
                    """
                    <div class="section-header">

                        <h2>
                            📊 Detailed Score Breakdown
                        </h2>

                        <p>
                            Different similarity methods contribute
                            to the final AI-powered match score.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                (
                    score_col_1,
                    score_col_2,
                    score_col_3,
                    score_col_4,
                ) = st.columns(
                    4,
                    gap="medium",
                )

                with score_col_1:

                    display_metric_card(
                        title="Overall Match",
                        score=final_score,
                        subtitle="Final weighted score",
                        icon="🎯",
                    )

                with score_col_2:

                    display_metric_card(
                        title="Skill Match",
                        score=skill_score,
                        subtitle="Exact required skill overlap",
                        icon="🧠",
                    )

                with score_col_3:

                    display_metric_card(
                        title="TF-IDF Similarity",
                        score=tfidf_score,
                        subtitle="Keyword and text similarity",
                        icon="📄",
                    )

                with score_col_4:

                    display_metric_card(
                        title="Semantic Similarity",
                        score=semantic_score,
                        subtitle="Meaning and context alignment",
                        icon="🔍",
                    )


                # ==========================================
                # SKILL ANALYSIS
                # ==========================================

                st.markdown(
                    """
                    <div class="section-header">

                        <h2>
                            🧠 Skill Gap Analysis
                        </h2>

                        <p>
                            Compare the skills detected in the resume
                            against the requirements extracted from
                            the target job description.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                skill_col_1, skill_col_2 = st.columns(
                    2,
                    gap="large",
                )

                with skill_col_1:

                    display_skill_panel(
                        title=(
                            "✅ Matching Skills "
                            f"({len(matching_skills)})"
                        ),
                        skills=matching_skills,
                        panel_class="match",
                        empty_message=(
                            "No direct matching skills were detected."
                        ),
                    )

                with skill_col_2:

                    display_skill_panel(
                        title=(
                            "⚠️ Skills to Improve "
                            f"({len(missing_skills)})"
                        ),
                        skills=missing_skills,
                        panel_class="missing",
                        empty_message=(
                            "Excellent! No major skill gaps "
                            "were detected."
                        ),
                    )


                # ==========================================
                # DETECTED SKILLS
                # ==========================================

                st.markdown(
                    """
                    <div class="section-header">

                        <h2>
                            🔎 Detected Technical Skills
                        </h2>

                        <p>
                            Skills automatically identified from the
                            resume and target job description using
                            the expanded CS/IT skills database.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                detected_col_1, detected_col_2 = st.columns(
                    2,
                    gap="large",
                )

                with detected_col_1:

                    display_skill_panel(
                        title=(
                            "📄 Resume Skills Detected "
                            f"({len(resume_skills)})"
                        ),
                        skills=resume_skills,
                        panel_class="resume",
                        empty_message=(
                            "No skills were detected in the resume."
                        ),
                    )

                with detected_col_2:

                    display_skill_panel(
                        title=(
                            "🎯 Job Skills Required "
                            f"({len(job_skills)})"
                        ),
                        skills=job_skills,
                        panel_class="job",
                        empty_message=(
                            "No skills were detected in the "
                            "job description."
                        ),
                    )


                # ==========================================
                # QUICK SUMMARY
                # ==========================================

                st.markdown(
                    """
                    <div class="section-header">

                        <h2>
                            📌 Quick Summary
                        </h2>

                        <p>
                            A high-level view of the analysis results.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                (
                    summary_col_1,
                    summary_col_2,
                    summary_col_3,
                ) = st.columns(
                    3,
                    gap="medium",
                )

                with summary_col_1:

                    st.metric(
                        label="Matching Skills",
                        value=len(
                            matching_skills
                        ),
                    )

                with summary_col_2:

                    st.metric(
                        label="Skills to Improve",
                        value=len(
                            missing_skills
                        ),
                    )

                with summary_col_3:

                    st.metric(
                        label="Job Requirements Detected",
                        value=len(
                            job_skills
                        ),
                    )


                # ==========================================
                # SKILL LIST SUMMARY
                # ==========================================

                with st.expander(
                    "📋 View Skill Lists",
                    expanded=False,
                ):

                    st.markdown(
                        "**Matching Skills**"
                    )

                    st.write(
                        format_skill_list(
                            matching_skills
                        )
                    )

                    st.markdown(
                        "**Missing Skills**"
                    )

                    st.write(
                        format_skill_list(
                            missing_skills
                        )
                    )

                    st.markdown(
                        "**All Resume Skills**"
                    )

                    st.write(
                        format_skill_list(
                            resume_skills
                        )
                    )

                    st.markdown(
                        "**All Job Skills**"
                    )

                    st.write(
                        format_skill_list(
                            job_skills
                        )
                    )


                # ==========================================
                # AI CAREER ANALYSIS
                # ==========================================

                st.markdown(
                    """
                    <div class="section-header">

                        <h2>
                            🤖 AI Career Analysis
                        </h2>

                        <p>
                            Personalized recommendations generated
                            using the actual NLP/ML scores, extracted
                            skills, skill gaps and retrieved knowledge.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="ai-analysis-header">

                        <h2>
                            ✨ Generative AI Recommendations
                        </h2>

                        <p>
                            The AI uses the calculated matching
                            results and relevant retrieved knowledge
                            to provide targeted career insights.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.container(
                    border=True
                ):

                    display_analysis_content(
                        ai_analysis
                    )


                # ==========================================
                # TECHNICAL DETAILS
                # ==========================================

                st.markdown(
                    """
                    <div class="section-header">

                        <h2>
                            🔬 Technical Details
                        </h2>

                        <p>
                            Explore the retrieved RAG knowledge and
                            extracted text used by the system.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                # ==========================================
                # RAG DETAILS
                # ==========================================

                with st.expander(
                    "📚 View Retrieved RAG Knowledge",
                    expanded=False,
                ):

                    if rag_context:

                        st.caption(
                            "Relevant knowledge retrieved from the "
                            "CS/IT knowledge base before generating "
                            "the AI analysis."
                        )

                        st.code(
                            rag_context,
                            language="text",
                        )

                    else:

                        st.info(
                            "No additional RAG context was returned."
                        )


                # ==========================================
                # EXTRACTED RESUME TEXT
                # ==========================================

                with st.expander(
                    "📄 View Extracted Resume Text",
                    expanded=False,
                ):

                    st.caption(
                        "Text extracted from the uploaded resume and "
                        "used as input for the NLP/ML pipeline."
                    )

                    st.text_area(
                        "Extracted Resume Text",
                        value=resume_text,
                        height=350,
                        disabled=True,
                        label_visibility="collapsed",
                    )


                # ==========================================
                # JOB DESCRIPTION
                # ==========================================

                with st.expander(
                    "💼 View Job Description Used",
                    expanded=False,
                ):

                    st.text_area(
                        "Job Description Used",
                        value=job_description,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                    )


                # ==========================================
                # PIPELINE INFORMATION
                # ==========================================

                with st.expander(
                    "⚙️ View Analysis Pipeline Information",
                    expanded=False,
                ):

                    st.markdown(
                        """
### AI Resume Matching Pipeline

**1. Resume Parsing**  
Extracts readable text from the uploaded PDF or TXT file.

**2. NLP Preprocessing**  
Cleans and preprocesses resume and job text.

**3. Skill Extraction**  
Detects programming, data science, artificial intelligence,
machine learning, software engineering, cloud, cybersecurity,
DevOps and other CS/IT skills from the expanded skills database.

**4. Exact Skill Matching**  
Compares required job skills against skills detected in the
candidate resume.

**5. TF-IDF Similarity**  
Measures text and keyword similarity between the resume and
job description.

**6. Semantic Similarity**  
Uses sentence embeddings to measure contextual and semantic
alignment.

**7. Advanced Match Score**  
Combines the scoring components into the final weighted
AI match score.

**8. RAG Retrieval**  
Retrieves relevant knowledge from the project knowledge base.

**9. Generative AI Analysis**  
Generates career recommendations, skill-gap analysis,
learning guidance and interview questions based on the
actual system results.
                        """
                    )


            # ==============================================
            # ERROR HANDLING
            # ==============================================

            except Exception as error:

                st.error(
                    "An error occurred during the analysis."
                )

                st.exception(
                    error
                )


            # ==============================================
            # CLEAN UP TEMPORARY FILE
            # ==============================================

            finally:

                if temporary_pdf_path:

                    try:

                        Path(
                            temporary_pdf_path
                        ).unlink(
                            missing_ok=True
                        )

                    except Exception:

                        pass


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">

        <strong>
            🤖 AI Resume & Job Match
        </strong>

        <br><br>

        Powered by NLP • Machine Learning •
        Semantic Embeddings • RAG • Generative AI

        <br><br>

        Built with Python and Streamlit

    </div>
    """,
    unsafe_allow_html=True,
)