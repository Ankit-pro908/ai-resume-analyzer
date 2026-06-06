from openai import OpenAI

# Create Groq client
client = OpenAI(
    api_key="GROQ_API_KEY",
    base_url="https://api.groq.com/openai/v1"
)


# ==============================
# Resume Suggestions
# ==============================

def get_resume_suggestions(resume_text, jd_text, missing_skills):

    prompt = f"""
    You are an expert HR consultant and resume coach.

    Candidate Resume:
    {resume_text[:2000]}

    Job Description:
    {jd_text[:1000]}

    Missing Skills:
    {", ".join(missing_skills) if missing_skills else "None"}

    Give:
    1. 5 resume improvement suggestions
    2. ATS optimization tips
    3. Important missing keywords

    Keep the response concise and practical.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


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

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ==============================
# Interview Probability
# ==============================

def get_interview_probability(score, missing_skills):

    prompt = f"""
    ATS Score: {score}/100

    Missing Skills:
    {", ".join(missing_skills) if missing_skills else "None"}

    Predict:
    1. Interview selection probability %
    2. Main reason
    3. Biggest improvement needed

    Keep answer short and realistic.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content