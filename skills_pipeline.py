import re

# Known Skills Dictionary
known_skills = [
    # Programming Languages
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "Go",
    "Ruby", "PHP", "R", "Kotlin", "Swift",

    # Web / Frameworks
    "Django", "Flask", "Spring", "React", "Angular", "Vue", "Node.js", "Express",

    # Databases
    "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis",

    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD", "Jenkins",

    # Data Science / ML / AI
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Matplotlib",
    "Seaborn", "Hadoop", "Spark", "NLTK", "OpenCV",

    # Tools & Misc
    "Git", "Linux", "Tableau", "Power BI", "Excel"
]

# Normalize Section
def normalize_section(section):
    """
    Converts list or string section into a clean string.
    """
    if isinstance(section, list):
        return " ".join(str(s) for s in section if s).strip()
    elif isinstance(section, str):
        return section.strip()
    return ""

# Clean Skill Text
def get_clean_skills_text(sections):
    """
    Extracts and cleans the 'skills' section from resume sections.
    """
    raw_text = normalize_section(sections.get("skills", ""))

    # Remove common noise words
    noise = ["technical skills", "skills", "industry", "experience",
             "proficient in", "knowledge of"]
    for n in noise:
        raw_text = raw_text.replace(n, "")

    return raw_text.strip()

# Extract Skills with Fuzzy Matching
def extract_skills_from_text(text):
    """
    Extracts skills from free text using known_skills dictionary
    with exact + fuzzy matching (safe).
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    found_skills = set()
    for skill in known_skills:
        s = skill.lower()

        # Exact word/phrase match
        if re.search(rf"\b{s}\b", text):
            found_skills.add(skill)

        # Fuzzy / partial match but safe
        elif len(s) > 2 and s in text:
            found_skills.add(skill)

    cleaned = [s for s in found_skills if len(s) > 1 or s.upper() in {"C", "R", "Go"}]

    return sorted(set(cleaned))


# Pipeline Function
def extract_skills_from_sections(sections):
    """
    Full pipeline: clean skills text + extract skills.
    """
    skill_text = get_clean_skills_text(sections)
    return extract_skills_from_text(skill_text)

def unify_skills(parsed_skills, text):
    fuzzy = extract_skills_from_text(text)
    return sorted(set(parsed_skills) | set(fuzzy))

