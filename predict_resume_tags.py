from transformers import BertTokenizer, BertForSequenceClassification
from utils import split_into_sections
import torch
import joblib


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_path = "resume_bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load multi-label binarizer
mlb = joblib.load("label_binarizer.pkl")

# Suggestion mapping
label_suggestions = {
    'missing_metrics': "Add quantifiable results (e.g., 'reduced processing time by 30%').",
    'missing_skills': "Include relevant technical skills like Python, SQL, or Docker.",
    'format_issue': "Ensure consistent formatting and spacing across all sections.",
    'outdated_stack': "Mention modern tools or frameworks you've used recently.",
    'generic_summary': "Make your summary more specific and tailored to your achievements."
}


def suggest_resume_improvements(resume_text):
    sections = split_into_sections(resume_text)
    suggestions = []

    # Summary Checks
    summary = sections.get("summary", "")
    if len(summary.split()) < 30:
        suggestions.append("Your summary is too short. Aim for 2–3 impactful lines.")
    if any(word in summary.lower() for word in ["hardworking", "team player", "passionate"]):
        suggestions.append("Avoid generic terms in your summary. Be specific about your achievements.")

    # Skills Checks
    skills = sections.get("skills", "")
    if len(skills.split(',')) < 5:
        suggestions.append("List at least 5–10 relevant skills to strengthen your profile.")
    missing_tools = [tool for tool in [ "aws", "git", "sql", "react","python","java","mysql"] if tool not in skills.lower()]
    if missing_tools:
        suggestions.append(f"Consider adding modern tools: {', '.join(missing_tools[:3])}.")

    # Experience Checks
    experience = sections.get("experience", "")
    if not any(char in experience for char in "%1234567890"):
        suggestions.append("Include quantifiable results in your experience section (e.g., 'Reduced processing time by 30%').")
    if len(experience.strip().split('.')) < 3:
        suggestions.append("Break your experience into multiple bullet points or sentences.")

    # Education Checks
    education = sections.get("education", "")
    if not any(kw in education.lower() for kw in ["bachelor", "master", "phd", "degree"]):
        suggestions.append("Mention your degree explicitly in the education section.")
    if not any(year in education for year in ["2020", "2021", "2022", "2023", "2024", "2025"]):
        suggestions.append("Include your graduation year for context.")

    # Certifications Checks
    sections.get("certifications", "")
    if not any():
        suggestions.append("Mention the certifications you have done")


    # Formatting
    if not any(h in resume_text.lower() for h in ["experience", "skills", "education", "summary"]):
        suggestions.append("Your resume is missing clear section headings.")

    return suggestions