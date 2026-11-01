import re
from openai import OpenAI
import streamlit as st

# ==============================
# Client setup
# ==============================

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# Tried in this order. First one your key can access gets used.
PREFERRED_MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]

# Models that are not for text chat
SKIP_KEYWORDS = ("whisper", "tts", "guard", "playai", "embed", "orpheus")


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_models():
    """Ask Groq which models this API key can actually use."""
    try:
        ids = [m.id for m in client.models.list().data]
        return [i for i in ids if not any(k in i.lower() for k in SKIP_KEYWORDS)]
    except Exception:
        return []


def get_model_order():
    available = get_available_models()
    if not available:
        return PREFERRED_MODELS
    ordered = [m for m in PREFERRED_MODELS if m in available]
    others = [m for m in available if m not in ordered]
    return ordered + others


def clean_output(text):
    # Remove <think>...</think> blocks some reasoning models add
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


# ==============================
# Shared helper
# ==============================

def ask_llm(prompt):
    last_error = None

    for model in get_model_order()[:6]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            return clean_output(response.choices[0].message.content)
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            # Only move on to the next model if this one is missing/blocked
            if "model_not_found" in msg or "404" in msg or "decommissioned" in msg or "access" in msg:
                continue
            break  # other errors (bad key, rate limit) won't be fixed by switching

    return f"AI response unavailable right now: {last_error}"


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