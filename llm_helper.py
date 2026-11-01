from openai import OpenAI
import streamlit as st

# ==============================
# Client setup
# ==============================

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL_NAME = "llama-3.3-70b-versatile"


# ==============================
# Shared helper
# ==============================

def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI response unavailable right now: {e}"


# ==============================
# Resume Suggestions
# ==============================

def get_resume_suggestions(resume_text, jd_text, missing_skills):
    skills = ", ".join(missing_skills) if missing_skills else "None"

    prompt = f"""
You are an expert HR consultant and resume coach.

Candidate Resume:
{resume_text[:2000]}

Job Description:
{jd_text[:1000]}

Missing Skills:
{skills}

Give:
1. 5 resume improvement suggestions
2. ATS optimization tips
3. Important missing keywords

Keep the response concise and practical.
"""
    return ask_llm(prompt)


# ==============================
# Interview Questions
# ==============================

def get_interview_questions(resume_text, jd_text):
    prompt = f"""
Candidate Resume:
{resume_text[:2000]}

Job Description:
{jd_text[:1000]}

Generate:
- 3 technical questions
- 2 behavioral questions
- 2 resume-based questions
- 1 tricky HR question

Also provide short answer tips.
"""
    return ask_llm(prompt)


# ==============================
# Interview Probability
# ==============================

def get_interview_probability(score, missing_skills):
    skills = ", ".join(missing_skills) if missing_skills else "None"

    prompt = f"""
ATS Score: {score}/100

Missing Skills:
{skills}

Predict:
1. Interview selection probability %
2. Main reason
3. Biggest improvement needed

Keep answer short and realistic.
"""
    return ask_llm(prompt)