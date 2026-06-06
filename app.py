import streamlit as st
from resume_parser import extract_text_from_pdf, clean_text
from analyzer import (
    calculate_ats_score,
    get_missing_skills,
    get_ats_feedback,
    get_strengths,
    get_score_breakdown
)
from llm_helper import (
    get_resume_suggestions,
    get_interview_questions,
    get_interview_probability
)
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")
st.subheader("Upload Resume + Job Description → Get ATS Score & Interview Questions")
st.divider()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "score" not in st.session_state:
    st.session_state.score = 0
if "resume_skills" not in st.session_state:
    st.session_state.resume_skills = []
if "jd_skills" not in st.session_state:
    st.session_state.jd_skills = []
if "missing_skills" not in st.session_state:
    st.session_state.missing_skills = []
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "strengths" not in st.session_state:
    st.session_state.strengths = []
if "breakdown" not in st.session_state:
    st.session_state.breakdown = {}

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Resume")
    uploaded_resume = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])

with col2:
    st.header("📋 Job Description")
    job_description = st.text_area(
        "Paste the Job Description here",
        height=200,
        placeholder="Paste JD here..."
    )

st.divider()

# ─────────────────────────────────────────────
# ANALYZE BUTTON
# ─────────────────────────────────────────────
if st.button("🚀 Analyze Now", use_container_width=True):

    if uploaded_resume is None:
        st.error("❌ Please upload your Resume PDF!")
    elif job_description.strip() == "":
        st.error("❌ Please paste the Job Description!")
    else:
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_resume)
            clean_resume = clean_text(resume_text)
            clean_jd = clean_text(job_description)
            score = calculate_ats_score(clean_resume, clean_jd)
            resume_skills, jd_skills, missing_skills = get_missing_skills(clean_resume, clean_jd)
            feedback = get_ats_feedback(score)
            strengths = get_strengths(resume_skills, jd_skills)
            breakdown = get_score_breakdown(clean_resume, clean_jd)

            # Save everything to session state
            st.session_state.analyzed = True
            st.session_state.resume_text = resume_text
            st.session_state.job_description = job_description
            st.session_state.score = score
            st.session_state.resume_skills = resume_skills
            st.session_state.jd_skills = jd_skills
            st.session_state.missing_skills = missing_skills
            st.session_state.feedback = feedback
            st.session_state.strengths = strengths
            st.session_state.breakdown = breakdown

# ─────────────────────────────────────────────
# SHOW RESULTS (only if analyzed)
# ─────────────────────────────────────────────
if st.session_state.analyzed:

    score         = st.session_state.score
    resume_text   = st.session_state.resume_text
    job_description = st.session_state.job_description
    resume_skills = st.session_state.resume_skills
    jd_skills     = st.session_state.jd_skills
    missing_skills= st.session_state.missing_skills
    feedback      = st.session_state.feedback
    strengths     = st.session_state.strengths
    breakdown     = st.session_state.breakdown

    st.divider()

    # METRIC CARDS
    st.header("📊 Analysis Results")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("ATS Score", f"{score}%")
    with col_m2:
        st.metric("Skills Matched", len(strengths))
    with col_m3:
        st.metric("Missing Skills", len(missing_skills))

    # GAUGE CHART
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "ATS Match Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30],  "color": "red"},
                {"range": [30, 50], "color": "orange"},
                {"range": [50, 70], "color": "yellow"},
                {"range": [70, 100],"color": "green"},
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.info(feedback)
    st.progress(score / 100)

    # SCORE BREAKDOWN
    st.divider()
    st.subheader("📈 Score Breakdown")
    for component, val in breakdown.items():
        st.write(f"**{component}**")
        st.progress(val / 100)
        st.caption(f"{val}%")

    # SKILLS SECTION
    st.divider()
    col3, col4, col5 = st.columns(3)

    with col3:
        st.subheader("✅ Your Skills")
        if resume_skills:
            for skill in resume_skills:
                st.success(skill)
        else:
            st.write("No skills found")

    with col4:
        st.subheader("📋 JD Requires")
        if jd_skills:
            for skill in jd_skills:
                st.info(skill)
        else:
            st.write("No skills found")

    with col5:
        st.subheader("❌ Missing Skills")
        if missing_skills:
            for skill in missing_skills:
                st.error(skill)
        else:
            st.success("No missing skills! 🎉")

    # STRENGTHS
    st.divider()
    st.subheader("🔥 Matching Skills (Your Strengths)")
    if strengths:
        for skill in strengths:
            st.success(skill)
    else:
        st.warning("No matching skills found")

    # DOWNLOAD REPORT
    st.divider()
    report = f"""
AI Resume Analyzer Report
==========================

ATS Score: {score}/100
Feedback: {feedback}

Score Breakdown:
{chr(10).join([f"  {k}: {v}%" for k, v in breakdown.items()])}

Your Skills:
{chr(10).join(resume_skills) if resume_skills else "None found"}

JD Required Skills:
{chr(10).join(jd_skills) if jd_skills else "None found"}

Missing Skills:
{chr(10).join(missing_skills) if missing_skills else "None - Great job!"}

Matching Skills (Strengths):
{chr(10).join(strengths) if strengths else "None found"}
"""
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="resume_report.txt",
        mime="text/plain"
    )

    # AI POWERED INSIGHTS
    st.divider()
    st.header("🤖 AI Powered Insights")

    col_ai1, col_ai2, col_ai3 = st.columns(3)

    with col_ai1:
        if st.button("🎯 Predict Interview Chances"):
            with st.spinner("Analyzing..."):
                probability = get_interview_probability(score, missing_skills)
            st.warning(probability)

    with col_ai2:
        if st.button("💡 Resume Suggestions"):
            with st.spinner("Generating Suggestions..."):
                suggestions = get_resume_suggestions(resume_text, job_description, missing_skills)
            st.write(suggestions)

    with col_ai3:
        if st.button("🎤 Interview Questions"):
            with st.spinner("Generating Questions..."):
                questions = get_interview_questions(resume_text, job_description)
            st.write(questions)

    st.divider()
    st.caption("Built using Python, Streamlit, Scikit-Learn, NLP and Groq LLM")