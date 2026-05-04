import streamlit as st
import pdfplumber
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="ResumeLens · Resume Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117; color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

.hero { text-align: center; padding: 2rem 1rem 1.5rem; margin-bottom: 2rem; }
.hero-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #8b949e; font-size: 0.95rem;
    font-family: 'Space Mono', monospace; letter-spacing: 0.05em;
}
.card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;
}
.card-title {
    font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #8b949e; margin-bottom: 1rem;
}
.result-fit {
    background: linear-gradient(135deg, #0f2a1a, #1a3a2a);
    border: 2px solid #238636; border-radius: 14px;
    padding: 2rem; text-align: center; margin-bottom: 1.5rem;
}
.result-notfit {
    background: linear-gradient(135deg, #2a0f0f, #3a1a1a);
    border: 2px solid #da3633; border-radius: 14px;
    padding: 2rem; text-align: center; margin-bottom: 1.5rem;
}
.result-label { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-sub { font-size: 0.9rem; color: #8b949e; }
.score-bar-bg {
    background: #21262d; border-radius: 100px;
    height: 12px; margin: 0.5rem 0 1.5rem; overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 100px; }
.skill-chip {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 100px; font-size: 0.78rem;
    font-family: 'Space Mono', monospace; margin: 0.2rem;
}
.chip-match { background: #0f2a1a; border: 1px solid #238636; color: #aff5b4; }
.chip-missing { background: #2a0f0f; border: 1px solid #da3633; color: #ffa198; }
.suggestion-item {
    background: #1c2128; border-left: 3px solid #bc8cff;
    border-radius: 0 8px 8px 0; padding: 0.6rem 1rem;
    margin-bottom: 0.5rem; font-size: 0.88rem; color: #cdd9e5;
}
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-box {
    flex: 1; background: #1c2128; border: 1px solid #30363d;
    border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 700; }
.metric-label {
    font-size: 0.72rem; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem;
}
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: 0.6rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
textarea {
    background: #1c2128 !important; border: 1px solid #30363d !important;
    border-radius: 10px !important; color: #e6edf3 !important;
}
.warning-box {
    background: #2a1f0a; border: 1px solid #e3b341;
    border-radius: 10px; padding: 0.8rem 1rem;
    color: #e3b341; font-size: 0.88rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🔍 ResumeLens</div>
  <div class="hero-sub">// AI-POWERED RESUME SCREENING · INSTANT JOB FIT ANALYSIS</div>
</div>
""", unsafe_allow_html=True)

# ── Helper functions ───────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF with validation."""
    # Check file size
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if len(file_bytes) < 100:
        raise ValueError("The uploaded PDF appears to be empty (0 bytes). Please upload a valid resume PDF.")

    # Check it's actually a PDF
    if not file_bytes.startswith(b'%PDF'):
        raise ValueError("The uploaded file is not a valid PDF. Please upload a proper PDF file.")

    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("The PDF has no pages.")
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        raise ValueError(f"Could not read PDF: {str(e)}")

    if not text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "This usually means it's a scanned image. "
            "Please use a text-based PDF resume."
        )
    return text.strip()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.strip()

def get_match_score(jd_text: str, resume_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([clean_text(jd_text), clean_text(resume_text)])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(float(score) * 100, 1)

def extract_keywords(text: str, top_n: int = 40) -> set:
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=top_n)
    vectorizer.fit([clean_text(text)])
    return set(vectorizer.get_feature_names_out())

def get_matched_missing(jd_text: str, resume_text: str):
    jd_keywords  = extract_keywords(jd_text, top_n=40)
    resume_clean = clean_text(resume_text)
    matched = sorted({kw for kw in jd_keywords if kw in resume_clean})
    missing = sorted(jd_keywords - set(matched))
    return matched, missing

def generate_suggestions(missing_keywords: list, score: float) -> list:
    suggestions = []
    if score < 40:
        suggestions.append("Low overlap with the JD — consider rewriting your resume targeting this role specifically.")
    elif score < 55:
        suggestions.append("Moderate match — tailor your resume more to this job's requirements.")
    else:
        suggestions.append("Good match! A few targeted tweaks can push you to the top of the applicant pool.")
    if missing_keywords:
        top = ', '.join(missing_keywords[:6])
        suggestions.append(f"Add these missing keywords naturally into your resume: {top}.")
    suggestions.append("Use quantifiable achievements (e.g. 'Increased sales by 30%') rather than generic duties.")
    suggestions.append("Mirror the exact language and terminology used in the job description.")
    suggestions.append("Ensure your skills section lists all tools/technologies mentioned in the JD.")
    return suggestions

def classify(score: float):
    return ("✅ Fit", True) if score >= 55 else ("❌ Not Fit", False)

# ── Input layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="card"><div class="card-title">📋 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Job Description",
        height=280,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><div class="card-title">📄 Upload Resume (PDF only)</div>', unsafe_allow_html=True)

    # ⚠️ Warn about file type BEFORE upload
    st.markdown('<div class="warning-box">⚠️ Only PDF files are accepted. Do not upload SVG, PNG, DOCX or other formats.</div>', unsafe_allow_html=True)

    uploaded_resume = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"],          # ← strictly PDF only
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    if uploaded_resume:
        file_size = len(uploaded_resume.read())
        uploaded_resume.seek(0)
        size_kb = round(file_size / 1024, 1)

        if file_size < 100:
            st.error(f"❌ '{uploaded_resume.name}' appears to be empty ({size_kb} KB). Please upload a valid resume.")
            uploaded_resume = None
        else:
            st.success(f"✅ Uploaded: {uploaded_resume.name} ({size_kb} KB)")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Analyse button ─────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyse = st.button("🔍 Analyse Resume")

# ── Results ────────────────────────────────────────────────────────────────────
if analyse:
    if not job_description.strip():
        st.warning("⚠️ Please paste a job description!")
    elif not uploaded_resume:
        st.warning("⚠️ Please upload a valid resume PDF!")
    else:
        with st.spinner("Analysing resume..."):
            try:
                resume_text = extract_text_from_pdf(uploaded_resume)
                score            = get_match_score(job_description, resume_text)
                matched, missing = get_matched_missing(job_description, resume_text)
                label, is_fit    = classify(score)
                suggestions      = generate_suggestions(missing, score)

                st.markdown("---")

                # Verdict
                rc = "result-fit" if is_fit else "result-notfit"
                fc = "#aff5b4" if is_fit else "#ffa198"
                st.markdown(f"""
                <div class="{rc}">
                  <div class="result-label" style="color:{fc}">{label}</div>
                  <div class="result-sub">This resume is {"a strong match" if is_fit else "not a strong match"} for the job description</div>
                </div>""", unsafe_allow_html=True)

                # Metrics
                bc = "#238636" if score >= 55 else "#da3633"
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-box">
                    <div class="metric-val" style="color:{bc}">{score}%</div>
                    <div class="metric-label">Match Score</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-val" style="color:#aff5b4">{len(matched)}</div>
                    <div class="metric-label">Keywords Matched</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-val" style="color:#ffa198">{len(missing)}</div>
                    <div class="metric-label">Keywords Missing</div>
                  </div>
                </div>
                <div class="score-bar-bg">
                  <div class="score-bar-fill" style="width:{min(score,100)}%; background:{bc};"></div>
                </div>""", unsafe_allow_html=True)

                # Keywords
                kw1, kw2 = st.columns(2, gap="large")
                with kw1:
                    st.markdown("**✅ Matched Keywords**")
                    if matched:
                        st.markdown("".join([f'<span class="skill-chip chip-match">{kw}</span>' for kw in matched[:20]]), unsafe_allow_html=True)
                    else:
                        st.info("No matching keywords found.")
                with kw2:
                    st.markdown("**❌ Missing Keywords**")
                    if missing:
                        st.markdown("".join([f'<span class="skill-chip chip-missing">{kw}</span>' for kw in missing[:20]]), unsafe_allow_html=True)
                    else:
                        st.success("No missing keywords — great coverage!")

                # Suggestions
                st.markdown("<br>**💡 Suggestions to Improve Your Resume**", unsafe_allow_html=True)
                for s in suggestions:
                    st.markdown(f'<div class="suggestion-item">→ {s}</div>', unsafe_allow_html=True)

            except ValueError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")