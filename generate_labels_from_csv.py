import pandas as pd
import re
from utils import split_into_sections, extract_skills_from_text  # Make sure these are available

# --- Config ---
KNOWN_SKILLS = {
    "python", "sql", "java", "aws", "azure", "tensorflow", "pytorch", "html",
    "css", "javascript", "git", "react", "docker", "kubernetes"
}
RECENT_TOOLS = {"aws", "gpt", "mlops", "transformers", "cloud", "docker", "kubernetes"}
WEAK_SUMMARY_WORDS = {"hardworking", "motivated", "passionate", "team player"}


# --- Rule Engine ---
def label_resume(text):
    labels = set()
    text = text.replace('\n', ' ')
    sections = split_into_sections(text)

    # 1. Missing Metrics
    if not re.search(r'\d+%|increased|decreased|reduced|saved|improved', text.lower()):
        labels.add("missing_metrics")

    # 2. Missing Skills
    detected_skills = extract_skills_from_text(text)
    if len(set(detected_skills).intersection(KNOWN_SKILLS)) < 3:
        labels.add("missing_skills")

    # 3. Format Issues
    if len(re.findall(r'\n{2,}', text)) > 5 or not any(
            h in text.lower() for h in ["experience", "skills", "education"]):
        labels.add("format_issue")

    # 4. Outdated Stack
    if not any(tool in text.lower() for tool in RECENT_TOOLS):
        labels.add("outdated_stack")

    # 5. Generic Summary
    summary = sections.get("summary", "")
    if any(word in summary.lower() for word in WEAK_SUMMARY_WORDS) or len(summary.split()) < 20:
        labels.add("generic_summary")

    return list(labels)


# --- Main Execution ---
if __name__ == "__main__":
    df = pd.read_csv("UpdatedResumeDataSet.csv")

    if "Resume" not in df.columns:
        raise ValueError("Expected column 'Resume' not found. Please rename your resume column.")

    df["labels"] = df["Resume"].apply(label_resume)

    # Save labeled dataset
    df.to_csv("labeled_resumes_for_bert.csv", index=False)
    # print("✅ Labeled dataset saved as 'labeled_resumes_for_bert.csv'")

import pandas as pd
from utils import split_into_sections, extract_skills_from_text  # must exist
import re

# CONFIG
KNOWN_SKILLS = {
    "python", "sql", "java", "aws", "azure", "tensorflow", "pytorch", "html",
    "css", "javascript", "git", "react", "docker", "kubernetes"
}
RECENT_TOOLS = {"aws", "gpt", "mlops", "transformers", "cloud", "kubernetes", "docker"}
WEAK_SUMMARY_WORDS = {"hardworking", "motivated", "passionate", "team player"}

def label_resume(text):
    labels = set()
    sections = split_into_sections(text)

    # 1. Missing Metrics
    if not re.search(r'\d+%|increased|decreased|reduced|saved|improved', text.lower()):
        labels.add("missing_metrics")

    # 2. Missing Skills
    skills = extract_skills_from_text(text)
    if len(set(skills).intersection(KNOWN_SKILLS)) < 3:
        labels.add("missing_skills")

    # 3. Format Issues
    if len(re.findall(r'\n{2,}', text)) > 5 or not any(h in text.lower() for h in ["experience", "skills", "education"]):
        labels.add("format_issue")

    # 4. Outdated Stack
    if not any(tool in text.lower() for tool in RECENT_TOOLS):
        labels.add("outdated_stack")

    # 5. Generic Summary
    summary = sections.get("summary", "")
    if any(word in summary.lower() for word in WEAK_SUMMARY_WORDS) or len(summary.split()) < 20:
        labels.add("generic_summary")

    return list(labels)

# Load CSV and label
df = pd.read_csv("UpdatedResumeDataSet.csv")
df['labels'] = df['Resume'].apply(label_resume)
df = df[['Resume', 'labels']]
df.to_csv("auto_labeled_resume_dataset.csv", index=False)
print("✅ Saved to auto_labeled_resume_dataset.csv")
