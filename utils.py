import re
import docx
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.high_level import extract_text as extract_pdf_text
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- TEXT EXTRACTION ---
def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# def extract_text(file_path):
#     ext = file_path.lower().split('.')[-1]
#     text = ""
#     if ext == 'pdf':
#         text = extract_pdf_text(file_path)
#     elif ext in ['doc', 'docx']:
#         text = extract_docx_text(file_path)
#
#     # Normalize newlines and whitespace
#     text = text.replace('\r', '\n')
#     text = re.sub(r'\n+', '\n', text)
#     return text.strip()

def extract_text(file_path):
    ext = file_path.lower().split('.')[-1]
    text = ""

    try:
        if ext == 'pdf':
            text = extract_pdf_text(file_path)
        elif ext in ['doc', 'docx']:
            text = extract_docx_text(file_path)
        else:
            raise ValueError("Unsupported file type.")
    except PDFSyntaxError:
        raise PDFSyntaxError("The uploaded file is not a valid PDF.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text: {str(e)}")

    # Normalize newlines and whitespace
    text = text.replace('\r', '\n')
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

# --- KNOWN SKILLS LIST ---
known_skills = [
    "python", "java", "c++", "sql", "html", "css", "javascript", "react", "angular", "node",
    "ux/ui", "node js", "angular", "mongo db", "aws", "azure", "linux", "git", "github",
    "tensorflow", "keras", "pytorch", "pandas", "numpy", "scikit-learn", "flask", "django",
    "php", "mysql",'c'
]

# --- SKILL EXTRACTION ---
def extract_skills_from_text(resume_text):
    doc = nlp(resume_text.lower())
    skills = set()

    # Regex to match percentage values like 80.04% or 77.8
    percentage_pattern = re.compile(r'^\d{2,3}\.\d{1,2}%?$')

    for token in doc:
        if token.text in known_skills and not percentage_pattern.match(token.text):
            skills.add(token.text)

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        if chunk_text in known_skills and not percentage_pattern.match(chunk_text):
            skills.add(chunk_text)

    return sorted(skills)

# --- CLEAN SKILLS SECTION ---
def clean_skills_section(section_text):
    lines = section_text.splitlines()
    cleaned_lines = []

    # Regex to match percentage lines (e.g., 80.04% or 77.8%)
    percentage_pattern = re.compile(r'^\s*\d{2,3}\.\d{1,2}%.*$')

    for line in lines:
        if not percentage_pattern.match(line.strip()):
            cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines).strip()

# --- SECTION SPLITTING ---
def split_into_sections(text):
    section_patterns = {
        "summary": r"(summary|profile|professional summary)",
        "skills": r"(skills|technical skills|key skills)",
        "experience": r"(experience|work experience|employment history|professional experience)",
        "education": r"(education|academic background|qualifications)",
        "projects": r"(projects|academic projects|personal projects)",
        "certifications": r"(CERTIFICATIONS|certifications|certificates|certification)",
        "contact": r"(\b\d{10}\b|\b[\w\.-]+@[\w\.-]+\.\w+\b|\blinkedin\.com\b|\bgithub\.com\b)"
    }

    # Normalize the text
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]

    sections = {}
    current_section = None

    for line in lines:
        lower_line = line.lower()
        found_section = None

        for section, pattern in section_patterns.items():
            if re.match(rf"^{pattern}\b", lower_line, re.IGNORECASE):
                print(f"[DEBUG] Found section: {section} --> {line}")
                found_section = section
                current_section = section
                sections[current_section] = []
                break

        if current_section and not found_section:
            sections[current_section].append(line)

    for section in sections:
        sections[section] = "\n".join(sections[section]).strip()

    return sections

# --- EXAMPLE USAGE ---
# (Use this in your main script after loading text from a file)

# resume_text = extract_text("path_to_resume.pdf")
# sections = split_into_sections(resume_text)

# if "skills" in sections:
#     cleaned_skill_text = clean_skills_section(sections["skills"])
#     extracted_skills = extract_skills_from_text(cleaned_skill_text)
# else:
#     extracted_skills = []

# print("Extracted Skills:", extracted_skills)
