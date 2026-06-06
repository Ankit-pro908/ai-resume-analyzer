from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ─────────────────────────────────────────────
# SKILL ALIASES
# ─────────────────────────────────────────────
SKILL_ALIASES = {
    "nextjs":                       ["next.js", "next js", "nextjs"],
    "nodejs":                       ["node.js", "node js", "nodejs"],
    "reactjs":                      ["react.js", "react js", "reactjs", "react"],
    "vuejs":                        ["vue.js", "vue js", "vuejs", "vue"],
    "machine learning":             ["ml", "machine learning"],
    "deep learning":                ["dl", "deep learning"],
    "natural language processing":  ["nlp", "natural language processing"],
    "computer vision":              ["cv", "computer vision"],
    "artificial intelligence":      ["ai", "artificial intelligence"],
    "javascript":                   ["js", "javascript"],
    "typescript":                   ["ts", "typescript"],
    "kubernetes":                   ["k8s", "kubernetes"],
    "amazon web services":          ["aws", "amazon web services"],
    "google cloud":                 ["gcp", "google cloud platform", "google cloud"],
    "continuous integration":       ["ci cd", "ci/cd", "continuous integration"],
    "object oriented programming":  ["oops", "oop", "object oriented"],
    "large language model":         ["llm", "large language model"],
    "retrieval augmented generation":["rag", "retrieval augmented generation"],
}

# ─────────────────────────────────────────────
# TITLE GROUPS
# ─────────────────────────────────────────────
TITLE_GROUPS = {
    "web_development": [
        "web developer", "web development intern", "frontend developer",
        "front end developer", "ui developer", "react developer",
        "full stack developer", "fullstack developer", "full-stack developer",
        "web designer", "ui ux developer", "javascript developer"
    ],
    "data": [
        "data analyst", "data analytics intern", "data analytics",
        "business analyst", "data scientist", "data engineer",
        "analytics engineer", "bi analyst", "reporting analyst"
    ],
    "cybersecurity": [
        "cybersecurity intern", "cyber security intern", "cybersecurity analyst",
        "security analyst", "ethical hacker", "penetration tester",
        "pen tester", "information security", "network security engineer",
        "security engineer", "soc analyst"
    ],
    "ai_ml": [
        "ai engineer", "machine learning engineer", "ml engineer",
        "llm engineer", "genai engineer", "deep learning engineer",
        "nlp engineer", "computer vision engineer", "data scientist",
        "ai researcher", "ml researcher"
    ],
    "backend": [
        "backend developer", "back end developer", "api developer",
        "software engineer", "software developer", "python developer",
        "java developer", "nodejs developer", "django developer",
        "flask developer", "sde", "swe"
    ],
    "devops_cloud": [
        "devops engineer", "cloud engineer", "site reliability engineer",
        "sre", "infrastructure engineer", "platform engineer",
        "aws engineer", "azure engineer", "gcp engineer"
    ],
    "android_mobile": [
        "android developer", "mobile developer", "kotlin developer",
        "flutter developer", "ios developer", "react native developer"
    ],
    "general_software": [
        "software engineer", "software developer", "sde", "swe",
        "programmer", "coder", "tech intern", "software intern",
        "it engineer", "computer science intern"
    ]
}

# ─────────────────────────────────────────────
# SKILLS BY CATEGORY
# ─────────────────────────────────────────────
SKILLS_BY_CATEGORY = {
    "web_dev": [
        "html", "css", "javascript", "react", "nodejs", "express",
        "typescript", "nextjs", "vuejs", "angular", "bootstrap",
        "tailwind", "rest api", "graphql", "jquery", "sass",
        "webpack", "vite", "figma", "responsive design", "reactjs"
    ],
    "data_science": [
        "python", "r", "pandas", "numpy", "matplotlib", "seaborn",
        "scikit-learn", "machine learning", "deep learning", "statistics",
        "data analysis", "data visualization", "jupyter", "tableau",
        "power bi", "excel", "sql", "tensorflow", "keras",
        "natural language processing", "computer vision", "feature engineering"
    ],
    "ai_ml": [
        "langchain", "rag", "vector database", "prompt engineering",
        "crewai", "hugging face", "openai", "llm", "fine tuning",
        "embeddings", "faiss", "pinecone", "llamaindex", "autogen",
        "large language model", "retrieval augmented generation",
        "stable diffusion", "generative ai"
    ],
    "backend": [
        "python", "java", "nodejs", "django", "flask", "fastapi",
        "spring boot", "express", "rest api", "graphql", "mysql",
        "postgresql", "mongodb", "redis", "docker", "kubernetes",
        "aws", "azure", "gcp", "microservices", "system design"
    ],
    "devops": [
        "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
        "aws", "azure", "gcp", "terraform", "ansible", "linux",
        "bash", "ci cd", "prometheus", "grafana", "nginx"
    ],
    "android": [
        "java", "kotlin", "android studio", "xml", "firebase",
        "rest api", "mvvm", "retrofit", "room database", "jetpack compose"
    ],
    "cybersecurity": [
        "network security", "penetration testing", "ethical hacking",
        "linux", "python", "wireshark", "metasploit", "kali linux",
        "cryptography", "firewall", "siem", "vulnerability assessment"
    ],
    "general": [
        "git", "github", "communication", "teamwork", "leadership",
        "problem solving", "agile", "scrum", "sql", "python", "java",
        "c", "c++", "data structures", "algorithms", "oops"
    ]
}

COMMON_SKILLS = list(set(
    skill for skills in SKILLS_BY_CATEGORY.values()
    for skill in skills
))

# Education levels
EDUCATION_LEVELS = {
    "phd":      4,
    "master":   3,
    "m.tech":   3,
    "mtech":    3,
    "msc":      3,
    "m.sc":     3,
    "mba":      3,
    "bachelor": 2,
    "b.tech":   2,
    "btech":    2,
    "b.e":      2,
    "be":       2,
    "bsc":      2,
    "b.sc":     2,
    "diploma":  1,
}

# Modern resume sections
IMPORTANT_SECTIONS = [
    "education",
    "experience",
    "skills",
    "projects",
    "summary"
]

# Action verbs
ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "created", "led",
    "managed", "optimized", "improved", "deployed", "automated", "analyzed",
    "delivered", "collaborated", "architected", "maintained", "integrated",
    "reduced", "increased", "launched", "trained", "researched", "solved"
]


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def normalize_text(text):
    text = text.lower()
    # Remove special characters except spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    for standard, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            text = text.replace(alias, standard)
    return text


# ─────────────────────────────────────────────
# 1. KEYWORD MATCH → 30%
# ─────────────────────────────────────────────
def calculate_keyword_score(resume_text, jd_text):
    resume_norm = normalize_text(resume_text)
    jd_norm = normalize_text(jd_text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_norm, jd_norm])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)


# ─────────────────────────────────────────────
# 2. SKILLS MATCH → 30%
# ─────────────────────────────────────────────
def extract_skills(text):
    text = normalize_text(text)
    return [skill for skill in COMMON_SKILLS if skill in text]

def calculate_skills_score(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    if not jd_skills:
        return 0
    matched = [s for s in jd_skills if s in resume_skills]
    return round((len(matched) / len(jd_skills)) * 100, 2)


# ─────────────────────────────────────────────
# 3. JOB TITLE MATCH → 15% (SMART VERSION)
# ─────────────────────────────────────────────
def find_group(title):
    """Returns which group a title belongs to"""
    for group, titles in TITLE_GROUPS.items():
        if title in titles:
            return group
    return None

def calculate_title_score(resume_text, jd_text):
    resume_norm = normalize_text(resume_text)
    jd_norm = normalize_text(jd_text)

    # Find all matching titles in JD and Resume
    jd_titles = [t for t in sum(TITLE_GROUPS.values(), []) if t in jd_norm]
    resume_titles = [t for t in sum(TITLE_GROUPS.values(), []) if t in resume_norm]

    # No title found in JD → give full marks (JD didn't specify)
    if not jd_titles:
        return 100

    # No title found in Resume
    if not resume_titles:
        return 20

    best_score = 0

    for jd_title in jd_titles:
        for resume_title in resume_titles:

            # Exact match
            if jd_title == resume_title:
                return 100

            # Same group match
            jd_group = find_group(jd_title)
            resume_group = find_group(resume_title)

            if jd_group and resume_group and jd_group == resume_group:
                best_score = max(best_score, 90)
                continue

            # Partial word match
            jd_words = set(jd_title.split())
            resume_words = set(resume_title.split())
            common_words = jd_words & resume_words

            # Remove generic words
            common_words -= {"intern", "junior", "senior", "the", "and", "for"}

            if common_words:
                best_score = max(best_score, 60)

    return best_score


# ─────────────────────────────────────────────
# 4. EDUCATION MATCH → 10%
# ─────────────────────────────────────────────
def calculate_education_score(resume_text, jd_text):
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    resume_level = 0
    jd_level = 0

    for keyword, level in EDUCATION_LEVELS.items():
        if keyword in resume_lower:
            resume_level = max(resume_level, level)
        if keyword in jd_lower:
            jd_level = max(jd_level, level)

    if jd_level == 0:
        return 100
    if resume_level >= jd_level:
        return 100
    if resume_level == jd_level - 1:
        return 50
    return 0


# ─────────────────────────────────────────────
# 5. ACTION VERBS → 10%
# ─────────────────────────────────────────────
def calculate_action_verb_score(resume_text):
    resume_lower = resume_text.lower()
    matched = [verb for verb in ACTION_VERBS if verb in resume_lower]
    return min(len(matched) * 10, 100)


# ─────────────────────────────────────────────
# 6. RESUME SECTIONS → 5%
# ─────────────────────────────────────────────
def calculate_sections_score(resume_text):
    resume_lower = resume_text.lower()
    found = [s for s in IMPORTANT_SECTIONS if s in resume_lower]
    return round((len(found) / len(IMPORTANT_SECTIONS)) * 100, 2)


# ─────────────────────────────────────────────
# FINAL WEIGHTED ATS SCORE
# ─────────────────────────────────────────────

def calculate_ats_score(resume_text, jd_text):
    keyword_score  = calculate_keyword_score(resume_text, jd_text)
    skills_score   = calculate_skills_score(resume_text, jd_text)
    title_score    = calculate_title_score(resume_text, jd_text)
    edu_score      = calculate_education_score(resume_text, jd_text)
    verb_score     = calculate_action_verb_score(resume_text)
    sections_score = calculate_sections_score(resume_text)

    final_score = (
        skills_score   * 0.40 +
        keyword_score  * 0.15 +
        title_score    * 0.20 +
        edu_score      * 0.10 +
        verb_score     * 0.10 +
        sections_score * 0.05
    )
    return round(final_score, 2)
# ─────────────────────────────────────────────
# SCORE BREAKDOWN
# ─────────────────────────────────────────────
def get_score_breakdown(resume_text, jd_text):
    return {
        "Keyword Match":   calculate_keyword_score(resume_text, jd_text),
        "Skills Match":    calculate_skills_score(resume_text, jd_text),
        "Job Title":       calculate_title_score(resume_text, jd_text),
        "Education":       calculate_education_score(resume_text, jd_text),
        "Action Verbs":    calculate_action_verb_score(resume_text),
        "Resume Sections": calculate_sections_score(resume_text)
    }


# ─────────────────────────────────────────────
# SUPPORTING FUNCTIONS
# ─────────────────────────────────────────────
def get_missing_skills(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    missing = [skill for skill in jd_skills if skill not in resume_skills]
    return resume_skills, jd_skills, missing

def get_ats_feedback(score):
    if score >= 70:
        return "🟢 Excellent! Your resume matches well with the JD."
    elif score >= 50:
        return "🟡 Good. Some improvements needed."
    elif score >= 30:
        return "🟠 Average. Work on adding more relevant keywords."
    else:
        return "🔴 Poor match. Majorly rework your resume for this JD."

def get_strengths(resume_skills, jd_skills):
    return [skill for skill in resume_skills if skill in jd_skills]