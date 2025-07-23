from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
question_tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned_questions_final")
question_model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned_questions_final").to(device)
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Section splitting and normalization
# def split_into_sections(text):
#     section_titles = ["summary", "objective", "skills", "experience", "education", "projects", "certifications", "certificates"]
#     sections = {}
#     current = "general"
#     sections[current] = []
#
#     section_title_pattern = re.compile(r'^\s*(' + '|'.join(section_titles) + r')\s*[:\s]*$', re.IGNORECASE)
#     for line in text.split('\n'):
#         stripped = line.strip()
#         if section_title_pattern.match(stripped):
#             current = stripped.lower().replace(":", "").strip()
#             sections[current] = []
#         else:
#             sections[current].append(line)
#     return {k: "\n".join(v).strip() for k, v in sections.items()}
def split_into_sections(text):
    section_titles = [
        "summary", "objective", "skills", "experience", "education", "projects",
        "certifications", "certification", "certificates", "awards", "publications", "achievements"
    ]
    section_aliases = {
        "certification": "certifications",
        "certificates": "certifications"
    }

    sections = {}
    current = "general"
    sections[current] = []

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        normalized = re.sub(r'[^a-z]', '', lower)

        matched = False
        for title in section_titles:
            norm_title = re.sub(r'[^a-z]', '', title.lower())
            if normalized == norm_title or normalized.startswith(norm_title):
                mapped = section_aliases.get(title.lower(), title.lower())
                current = mapped
                if current not in sections:
                    sections[current] = []
                matched = True
                break

        if not matched:
            if current not in sections:
                sections[current] = []
            sections[current].append(stripped)

    # Join each section’s content

    print("==== DEBUG: All Split Sections ====")
    for k, v in sections.items():
        print(f"[{k.upper()}] => {v[:300]}")
    print("==== DEBUG: Raw Cert Text ====\n", sections.get("certifications", ""))

    print("===================================")
    return {key: "\n".join(value).strip() for key, value in sections.items()}
    #return {k: "\n".join(v).strip() for k, v in sections.items()}




def detect_resume_sections(text):
    sections = split_into_sections(text)

    # Normalize certificate section
    if "certificates" in sections and "certifications" not in sections:
        sections["certifications"] = sections["certificates"]

    return {
        "summary": sections.get("summary", ""),
        "skills": sections.get("skills", ""),
        "projects": sections.get("projects", ""),
        "experience": sections.get("experience", ""),
        "certifications": sections.get("certifications", ""),
        "education": sections.get("education", "")
    }

# Question classification
def classify_question(q):
    q = q.lower()
    if any(k in q for k in ["how did you", "describe a time", "challenge", "handle", "conflict", "situation"]):
        return "Behavioral"
    elif any(k in q for k in ["python", "sql", "tools", "framework", "deployment", "project", "technical", "aws", "ml"]):
        return "Technical"
    else:
        return "General"

# Deduplicate similar questions
def deduplicate_questions(questions):
    if not questions:
        return []
    embeddings = sbert.encode(questions, convert_to_tensor=True)
    keep = [0]
    for i in range(1, len(questions)):
        scores = util.cos_sim(embeddings[i], embeddings[keep])
        if all(score.item() < 0.85 for score in scores[0]):
            keep.append(i)
    return [questions[i] for i in keep]

# Extract course names from education
def extract_course_names(education_text):
    lines = education_text.split("\n")
    course_names = []

    # Patterns that match course names followed by institute/dates
    patterns = [
        r"(MCA|Master of Computer Applications?)\b",
        r"(BCA|Bachelor of Computer Applications?)\b",
        r"\b(B\.?E\.?|Bachelor of Engineering)\b",
        r"\b(B\.?Tech|Bachelor of Technology)\b",
        r"\b(M\.?Tech|Master of Technology)\b",
        r"\b(MBA|Master of Business Administration)\b",
        r"\b(BBA|Bachelor of Business Administration)\b",
        r"\b(MSc|Master of Science)\b",
        r"\b(BSc|Bachelor of Science)\b",
        r"\b(PUC|Pre-University Course)\b",
        r"\b(SSLC|Secondary School Leaving Certificate)\b"
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are clearly dates or percentages
        if re.match(r"^(20\d{2}|19\d{2})", line) or '%' in line:
            continue

        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                course = match.group(0).strip()
                # Clean up any trailing special characters
                course = re.sub(r'[^a-zA-Z\s]', '', course)
                # Proper capitalization
                course = ' '.join(word.capitalize() for word in course.split())

                if course and course not in course_names:
                    course_names.append(course)
                    break

    return course_names[:5]  # Return first 5 matches

#extract certificates title
def extract_certification_titles(cert_text):
    keywords = ["certification", "certificate", "certified", "aws", "mysql", "java", "sql", "training", "linkedin", "cloud", "programming", "course"]
    certs = []
    for line in cert_text.splitlines():
        line = line.strip()
        if not line or len(line) < 5:
            continue

        # Normalize line for checking
        lower_line = line.lower()

        # Check for relevant keywords in content
        if any(kw in lower_line for kw in keywords):
            # Remove common leading bullets or artifacts
            clean = re.sub(r'^[oO\-–●•\d.]+\s*', '', line)
            # Inside extract_certification_titles()
            clean = re.sub(r'[-|].*$', '', clean).strip()  # Cut off after '-' or '|' to keep main name

            # Remove trailing credential IDs, links, etc.
            clean = re.split(r'\bissued\b|credential|of comple|certificat(e|ion) id|id\b|link|www\.|https?:', clean, flags=re.IGNORECASE)[0]
            clean = re.sub(r'[^a-zA-Z0-9\s\-()&]', '', clean).strip()

            # Title case & deduplicate
            if 5 < len(clean) < 100:
                title_cased = clean.title()
                if title_cased not in certs:
                    certs.append(title_cased)

    return certs[:3]  # top 3 max


#get sections
def get_section(sections, *keys):
    for key in keys:
        if key in sections and sections[key].strip():
            return sections[key]
    return ""

# Generate from certs
def generate_questions_from_certifications(cert_names):
    questions = []
    for cert in cert_names:
        name = cert.strip().title()
        lower = name.lower()
        if "aws" in lower:
            questions.append(f"How did the '{name}' course help you understand cloud services?")
            questions.append(f"Can you explain a deployment you performed after completing '{name}'?")
        elif "mysql" in lower or "sql" in lower:
            questions.append(f"Which SQL concepts did you apply from '{name}'?")
            questions.append(f"What kind of queries or projects were influenced by your '{name}' training?")
        elif "java" in lower:
            questions.append(f"What Java concepts did you strengthen through '{name}'?")
            questions.append(f"How did '{name}' enhance your programming confidence?")
        elif "web" in lower or "website" in lower:
            questions.append(f"What tools or frameworks did you explore in '{name}'?")
            questions.append(f"How did the '{name}' certification improve your front-end development skills?")
        else:
            questions.append(f"What did you gain from completing '{name}'?")
            questions.append(f"Have you applied your learnings from '{name}' to real-world projects?")
    return questions[:4]

# Final generator
def generate_questions_from_resume(resume_text, chunk_size=500):
    sections = detect_resume_sections(resume_text)
    #preview text
    print("==== DEBUG: All Split Sections ====")
    for k, v in sections.items():
        print(f"[{k.upper()}] =>", v[:300])  # preview first 300 chars
    questions = []
    section_prompts = {

    "projects": "Generate 5 detailed technical interview questions focusing on specific tools, technologies, project outcomes, and real-world implementation mentioned in the projects section.\n",
    "certifications": "Generate 5 targeted questions about each named certification, its key learnings, and how the candidate applied them.\n",
    "skills": "Generate 5 precise technical interview questions about listed programming languages, tools, or frameworks.\n",
    "experience": "Generate 5 behavioral and technical interview questions focusing on actual job tasks, achievements, and technologies used.\n",
    "summary": "Generate 3 general but meaningful questions that reflect the candidate’s profile, strengths, and career goals.\n",
    "education": "Generate 3 academic-related questions focusing on the candidate's course titles or specializations.\n"

    }

    priority = ["experience", "projects", "certifications", "skills", "education", "summary"]
    if len(sections.get("experience", "").split()) < 20:
        priority.remove("experience")
        priority.insert(0, "projects")

    for sec in priority:
        content = sections.get(sec, "").strip()
        if not content:
            continue
        prompt = section_prompts.get(sec, "") + content
        inputs = question_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            outputs = question_model.generate(
                inputs["input_ids"], max_length=256, num_beams=4,
                early_stopping=True, do_sample=True, top_p=0.9
            )

        decoded = question_tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_qs = re.split(r'\n|(?<=[?])\s+', decoded)
        for q in raw_qs:
            clean = q.strip().lstrip("1234567890). ").strip()
            vague_patterns = [
                "monitoring certification", "creating passwords", "resume", "you are",
                "how is it relevant", "why did you choose this course", "what did you learn during this course"
            ]

            if (
                    clean
                    and len(clean.split()) > 4
                    and not any(phrase in clean.lower() for phrase in vague_patterns)
            ):
                questions.append(clean)

        if len(questions) >= 8:
            break

    # Deduplicate
    unique_questions = deduplicate_questions(questions)[:10]
    final = [{"category": classify_question(q), "question": q} for q in unique_questions]

    # Generate questions specifically for listed skills
    skills_text = sections.get("skills", "")
    skills_list = re.findall(
        r'\b(Java|Python|SQL|C\+\+|C#|JavaScript|HTML|CSS|MySQL|PHP|React|Node\.js|Django|Flask|MongoDB|AWS|Git|Linux|Docker|Kubernetes)\b',
        skills_text, re.IGNORECASE)

    skills_list = list(set([skill.title() for skill in skills_list]))[:5]  # Unique and limited to top 5
    for skill in skills_list:
        skill_qs = [
            f"How have you applied {skill} in real-world projects?",
            f"What are some challenges you've faced while working with {skill}?",
            f"How do you keep your {skill} skills up to date?",
            f"What projects best demonstrate your proficiency in {skill}?"
        ]
        for q in skill_qs:
            if all(q.lower() not in f["question"].lower() for f in final):
                final.append({"category": "Technical", "question": q})

    # Education fallback
    course_names = extract_course_names(sections.get("education", ""))
    course = course_names[0] if course_names else None
    if course and not any("study" in q["question"].lower() for q in final):
        final.append({"category": "General", "question": f"What did you study in {course}, and how is it relevant to this job?"})

    # Certification fallback
    cert_text = get_section(sections, "certifications", "certification","")
    cert_names = extract_certification_titles(cert_text)
    print("==== DEBUG: Raw Cert Text ====")
    print(sections.get("certifications", ""))
    print("Cert Names Found:", cert_names)

    if cert_names:
        cert_qs = generate_questions_from_certifications(cert_names)
        for q in cert_qs:
            if all(q.lower() not in f["question"].lower() for f in final):
                final.append({"category": classify_question(q), "question": q})

    return final
