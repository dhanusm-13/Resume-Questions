from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import re, json, random
from utils import detect_resume_sections

with open("default_questions.json", "r") as f:
    DEFAULT_QUESTIONS = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
question_tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned_questions_final")
question_model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned_questions_final").to(device)
sbert = SentenceTransformer("all-MiniLM-L6-v2")

#section cleaning
def clean_section_values(sections):
    cleaned = {}

    projects = []
    for p in sections.get("projects", []):
        p = re.sub(r"^[\u2022\uf0b7\uf031-\uf04f\u25cf\d\.\-\s]+", "", p)
        p = re.sub(r"(Technologies Used:.*|tools used.*|tech stack.*|Research Publications:.*|DOI:.*)", "", p,
                   flags=re.I)
        p = re.sub(r"\[.*?\]", "", p)  # remove [anything]
        p = re.sub(r"\(.*?\)", "", p)  # remove (anything)
        p = re.sub(r"\b(20\d{2}|CGPA|%|Present).*", "", p)
        p = re.sub(r"\|.*", "", p)
        p = re.sub(r"[\.\-:]+$", "", p)
        if len(p.strip().split()) > 2:
            projects.append(p.strip())
    cleaned["projects"] = projects

    skills = []
    for s in sections.get("skills", []):
        s = s.strip()
        if s.lower() in ["technical skills", "industry.", "technical"]:
            continue
        s = re.sub(r"[^\w\s#\+\-]", "", s)  # keep C++, C#, etc.
        if 1 <= len(s.split()) <= 3:
            skills.append(s)
    cleaned["skills"] = list(dict.fromkeys(skills))[:6]

    certs = []
    for c in sections.get("certifications", []):
        c = re.sub(r"^[\u2022\uf0b7\uf031-\uf04f\u25cf\d\.\-\s]+", "", c)  # kill  and friends
        c = re.sub(r"\|?CERTIFICATE", "", c, flags=re.IGNORECASE)
        c = re.sub(r"^s:\s*", "", c, flags=re.IGNORECASE)
        c = re.sub(r"Certified by", "", c, flags=re.IGNORECASE)
        c = re.sub(r"\|.*", "", c)
        c = re.sub(r"-?Completed on.*", "", c, flags=re.IGNORECASE)
        c = re.sub(r"[\.\-:]+$", "", c)
        if len(c.strip().split()) > 1:
            certs.append(c.strip())
    cleaned["certifications"] = certs

    edu = []
    for e in sections.get("education", []):
        e = re.sub(r"\b(CGPA|%|20\d{2}|Present).*", "", e)
        e = re.sub(r"\|.*", "", e)
        e = re.sub(r"[\.\-:]+$", "", e)
        if len(e.strip().split()) > 2:
            edu.append(e.strip())
    cleaned["education"] = edu

    activities = []
    for a in sections.get("activities", []):
        a = re.sub(r"^[\u2022\uf0b7\uf031-\uf04f\u25cf\d\.\-\s]+", "", a)
        a = re.sub(r"\|.*", "", a)
        a = re.sub(r"[\.\-:]+$", "", a)
        if len(a.strip().split()) > 4:
            activities.append(a.strip())
    cleaned["activities"] = activities

    return {**sections, **cleaned}

#helper functions
def pick_one(values, fallback):
    if isinstance(values, list) and values:
        cleaned = [str(v).strip() for v in values if v and str(v).strip()]
        if cleaned:
            return random.choice(cleaned)
    elif isinstance(values, str) and values.strip():
        return values.strip()
    return fallback


def fill_placeholders(template, resume_data):
    replacements = {
        "project": pick_one(resume_data.get("projects", []), "a project"),
        "skill": pick_one(resume_data.get("skills", []), "a skill"),
        "certification": pick_one(resume_data.get("certifications", []), "a certification"),
        "education": pick_one(resume_data.get("education", []), "an academic program"),
    }

    if "your education" in replacements["education"].lower():
        replacements["education"] = "your degree"
    if "a certification" in replacements["certification"].lower():
        replacements["certification"] = "a professional certification"
    if replacements["skill"].lower() in ["technical", "technical skills"]:
        replacements["skill"] = "Python"

    try:
        return template.format(**replacements)
    except Exception:
        return template

def polish_question(text: str, resume_summary: str = "") -> str:
    text = re.sub(r'\s+', ' ', text).strip()

    # remove fillers
    vague_patterns = [
        "what does this statement mean", "candidate's resume",
        "your resume", "this resume", "the resume",
        "mca student", "bca student", "defined yourself as"
    ]
    if any(v in text.lower() for v in vague_patterns):
        return ""

    # fix encoding issues
    text = (text.replace("â€™", "'")
                 .replace("â€œ", '"')
                 .replace("â€�", '"'))

    # dedup like "C C# Java Python"
    text = re.sub(r"(C\s+){2,}", "C ", text)

    # shorten overlong quoted fragments
    text = re.sub(r"'([^']+)'", lambda m: "'" + " ".join(m.group(1).split()[:3]) + "'", text)

    # kill tech stack junk
    text = re.sub(r"(Technologies Used|Tools Used|Tech Stack):.*?(?=\?|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?\]", "", text)  # remove [anything]

    # trim overly long questions
    if len(text.split()) > 25:
        text = " ".join(text.split()[:20]) + " ...?"

    if not text.endswith("?"):
        text = text.rstrip(".") + "?"

    if resume_summary and any(w.lower() in text.lower() for w in resume_summary.split()[:5]):
        return ""

    return text

def classify_question(q):
    ql = q.lower()

    #Technical first
    if any(skill in ql for skill in ["python", "java", "sql", "css", "html", "c++", "c#"]):
        return "Technical"

    # Project detection
    if any(word in ql for word in ["project", "portfolio", "analysis", "application", "manager", "prediction"]):
        return "Project"

    # Education/Certification
    if any(k in ql for k in ["university", "college", "degree", "certification",
                             "course", "training", "exam", "study", "subject"]):
        return "Education/Certification"

    return "General/Behavioral"

def deduplicate_and_polish(questions):
    if not questions:
        return []
    embeddings = sbert.encode(questions, convert_to_tensor=True)
    keep = [0]
    for i in range(1, len(questions)):
        scores = util.cos_sim(embeddings[i], embeddings[keep])
        if all(score.item() < 0.85 for score in scores[0]):
            keep.append(i)
    cleaned = [polish_question(questions[i]) for i in keep]
    return [q for q in cleaned if q]

def generate_project_questions(projects, num=2):
    questions = []
    for proj in projects[:3]:  # limit to 3 projects
        templates = random.sample(DEFAULT_QUESTIONS["Project"], k=num)
        for t in templates:
            questions.append(fill_placeholders(t, {"projects": [proj]}))
    return questions

def generate_certification_questions(certs, num=1):
    questions = []
    for cert in certs[:3]:
        templates = random.sample(DEFAULT_QUESTIONS["Education/Certification"], k=num)
        for t in templates:
            questions.append(fill_placeholders(t, {"certifications": [cert]}))
    return questions

def generate_skill_questions(skills, num=1):
    questions = []
    for skill in skills[:5]:
        templates = random.sample(DEFAULT_QUESTIONS["Technical"], k=num)
        for t in templates:
            questions.append(fill_placeholders(t, {"skills": [skill]}))
    return questions

def generate_education_questions(edu, num=1):
    questions = []
    for e in edu[:2]:
        templates = random.sample(DEFAULT_QUESTIONS["Education/Certification"], k=num)
        for t in templates:
            questions.append(fill_placeholders(t, {"education": [e]}))
    return questions

# ---------------- Main Generator ---------------- #
def generate_questions_from_resume(resume_text):
    sections = detect_resume_sections(resume_text)
    sections = clean_section_values(sections)  # 🔧 clean noisy inputs
    questions = []

    # Rule-based generation first
    if sections.get("projects"):
        questions.extend(generate_project_questions(sections["projects"]))

    if sections.get("certifications"):
        questions.extend(generate_certification_questions(sections["certifications"]))

    if sections.get("skills"):
        questions.extend(generate_skill_questions(sections["skills"]))

    if sections.get("education"):
        questions.extend(generate_education_questions(sections["education"]))

    # Use T5 only for summary + experience
    priority = ["experience", "summary"]

    exp_section = " ".join(sections.get("experience", [])) if isinstance(sections.get("experience", []), list) else sections.get("experience", "")
    if len(exp_section.split()) < 20 and "experience" in priority:
        priority.remove("experience")

    for sec in priority:
        content = sections.get(sec, "")
        if isinstance(content, list):
            content = " ".join(content)
        content = content.strip()
        if not content:
            continue
        if len(content.split()) > 60:
            content = " ".join(content.split()[:60])

        prompt = f"Generate 5 interview questions based on {sec} section:\n{content}"
        inputs = question_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)

        with torch.no_grad():
            outputs = question_model.generate(
                inputs["input_ids"],
                max_length=256,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                early_stopping=True
            )
        decoded = question_tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_qs = re.split(r'\n|(?<=[?])\s+', decoded)

        for q in raw_qs:
            clean = q.strip().lstrip("1234567890). ").strip()
            if clean and len(clean.split()) > 4 and '?' in clean:
                clean = polish_question(clean)
                if clean:
                    questions.append(clean)

    # Deduplicate + polish
    questions = deduplicate_and_polish(questions)

    # Categorize
    final_questions = {"Project": [], "Education/Certification": [], "Technical": [], "General/Behavioral": []}
    for q in questions:
        cat = classify_question(q)
        final_questions[cat].append(q)

    # Ensure minimum coverage using defaults
    for cat, templates in DEFAULT_QUESTIONS.items():
        if len(final_questions[cat]) < 2:
            needed = 2 - len(final_questions[cat])
            sampled = random.sample(templates, k=min(needed, len(templates)))
            final_questions[cat].extend([fill_placeholders(t, sections) for t in sampled])

    #Flatten output
    output = []
    for cat, qs in final_questions.items():
        for q in qs:
            output.append({"question": q, "category": cat})

    return output

