from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os, sqlite3
import logging
from utils import extract_text, extract_skills_from_text, split_into_sections,clean_skills_section
import torch
import spacy
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from question_generator import generate_questions_from_resume
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import joblib
import logging

# Set the logging level for pdfminer to WARNING to hide the debug messages
logging.getLogger('pdfminer').setLevel(logging.WARNING)

# INIT APP
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Load spaCy for skill extraction
nlp = spacy.load("en_core_web_sm")

# Sentence-BERT model for similarity
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load BERT model for resume improvements
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_path = "resume_bert_model"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_model.to(device)
bert_model.eval()

mlb = MultiLabelBinarizer()
try:
    mlb = joblib.load("label_binarizer.pkl")  # ✅ safer loading if you saved it
except:
    mlb.classes_ = ['missing_metrics', 'missing_skills', 'format_issue', 'outdated_stack', 'generic_summary']

# Skill List (Removed docker and kubernetes as requested previously)
known_skills = ["python", "java", "c++", "sql", "html", "css", "javascript", "react", "angular", "node",
                "aws", "azure", "linux", "git", "github", "tensorflow", "keras",
                "pytorch", "pandas", "numpy", "scikit-learn", "flask", "django"]




# Load Local T5 Model for LLM Feedback
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_model.to(device)

# def get_t5_feedback(resume_text):
#     # Remove extra whitespace and special characters
#     cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,%-]', '', resume_text)
#     truncated_text = cleaned_text[:500]  # Limit for better model behavior
#     prompt = f"suggest improvements for this resume: {truncated_text}"
#
#     inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
#
#     with torch.no_grad():
#         outputs = t5_model.generate(inputs['input_ids'], max_length=100, num_beams=4, early_stopping=True)
#
#     suggestion_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     suggestions = suggestion_text.split('. ')
#
#     # Only include non-empty and non-raw echoes
#     return [{"type": "insights", "message": s.strip()} for s in suggestions if s.strip() and len(s.strip()) < 200]

# def split_into_sections(resume_text):
#     sections = {}
#     current_section = None
#     lines = resume_text.split('\n')
#     for line in lines:
#         line = line.strip()
#         if line.lower() in ['summary', 'technical skills', 'education', 'projects', 'experience', 'certifications']:
#             current_section = line.lower()
#             sections[current_section] = []
#         elif current_section and line:
#             sections[current_section].append(line)
#     return {k: ' '.join(v) for k, v in sections.items()}

#feedback
def get_t5_feedback(resume_text):
    # Split resume into sections
    sections = split_into_sections(resume_text)
    relevant_sections = [
        sections.get("summary", ""),
        sections.get("experience", ""),
        sections.get("projects", ""),
        sections.get("education", ""),
        sections.get("technical skills", "")
    ]
    cleaned_text = " ".join([sec for sec in relevant_sections if sec]).strip()

    # Clean input: remove grades, years, headers, lists, and sensitive info
    cleaned_text = re.sub(
        r'(\b\d{10}\b|\b[\w\.-]+@[\w\.-]+\.\w+\b|\blinkedin\.com\b|\bgithub\.com\b|' +
        r'\b(summary|technical skills|education|projects|experience|certifications|front\s*-\s*end|databases|tools)\b|' +
        r'\b[\w\s,-]+(,[\w\s,-]+)+|' +  # Lists like "Python,Java,SQL"
        r'\b\d{4}-\d{4}\b|\b\d{1,2}\.\d{1,2}\b|\b(cgpa|sgpa|percentage|%)\b|' +  # Grades and CGPA
        r'\b\d{4}\b|\[\s*technologies\s*used\s*:.*?\]|\bportfolio\b)',  # Years and project titles
        '', cleaned_text, flags=re.IGNORECASE
    )
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    truncated_text = cleaned_text[:500]  # Limit input length

    # Specific prompt for resume improvement
    prompt = (
        "Generate 2-4 concise, actionable suggestions to improve this resume for software development roles. "
        "Focus on enhancing technical skills, projects, and measurable outcomes. "
        "Provide declarative suggestions, not questions. "
        "Do not repeat raw resume content like grades, years, section headers, or skill lists: "
        f"{truncated_text}"
    )

    try:
        t5_tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned_questions_final")
        t5_model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned_questions_final").to(device)
        inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs['input_ids'],
                max_length=150,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1
            )
        suggestion_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        suggestions = [s.strip() for s in suggestion_text.split('. ') if s.strip()]

        # Filter out question-like outputs and invalid suggestions
        invalid_patterns = [
            r'^(what|how|why|can|describe|explain)\b.*\?$',  # Questions
            r'^\s*(summary|technical skills|education|projects|experience|certifications|front\s*-\s*end|databases|tools)\b',
            r'^\s*[\w\s,-]+(,[\w\s,-]+)+$',  # Lists like "Python,Java,SQL"
            r'^\s*\d+\s*$',  # Lone numbers
            r'^\s*[\d\.\s%-]+$',  # Grades or percentages
            r'^\s*\[.*\]$',  # Bracketed content
            r'^\s*\w+\s*:\s*$',  # Headers like "Front - End Technologies:"
            r'\b(cgpa|sgpa|percentage|portfolio|skills|technologies)\b',  # Forbidden keywords
            r'\b\d{4}-\d{4}\b|\b\d{4}\b'  # Years
        ]
        filtered_suggestions = [
            s for s in suggestions
            if (len(s) > 20 and len(s) < 200 and
                not any(re.match(p, s, re.IGNORECASE) for p in invalid_patterns) and
                all(word not in s.lower() for word in
                    ["cgpa", "sgpa", "percentage", "skills", "technologies", "portfolio", "what", "how", "why"]) and
                not s.endswith('?'))
        ]

        # Fallback suggestions if none are valid
        if not filtered_suggestions:
            filtered_suggestions = [
                "Highlight specific projects to demonstrate technical expertise, such as 'Developed a web app using Python and SQL'.",
                "Quantify achievements in your experience section, e.g., 'Reduced database query time by 20%'.",
                "Add modern tools like Git or AWS to align with software development roles."
            ]

        return [{"type": "Insights", "message": f"• {s}"} for s in filtered_suggestions[:4]]
    except Exception as e:
        logger.error(f"T5 feedback generation failed: {str(e)}")
        return [{"type": "Insights", "message": "• Unable to generate insights due to processing error."}]
# advance style check
def advanced_style_checks(text):
    issues = []

    # Passive voice (basic regex-based)
    passive_patterns = [
        r'\b(is|was|were|are|been|being)\s+\w+ed\b',
        r'\b(get|got|gets)\s+\w+ed\b'
    ]
    for pattern in passive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append("Avoid passive voice; use active verbs to clarify ownership of results.")

    # Repeated words (simple consecutive repetition)
    repeated = re.findall(r'\b(\w+)\s+\1\b', text.lower())
    if repeated:
        issues.append(f"Repeated words detected (e.g., '{', '.join(set(repeated))}'). Try to vary phrasing.")

    # Long paragraphs (not bullet-friendly)
    paragraphs = [p for p in text.split('\n') if len(p.strip()) > 120]
    if len(paragraphs) > 0:
        issues.append("Some sections have long paragraphs. Break them into bullet points for better readability.")

    # Generic phrases
    weak_phrases = ["responsible for", "worked on", "involved in", "participated in", "helped with"]
    for phrase in weak_phrases:
        if phrase in text.lower():
            issues.append(f"Phrase like '{phrase}' is generic. Use more specific action verbs (e.g., 'led', 'built').")
            break

    return [{"type": "style", "message": msg} for msg in issues]


#scoring function
def score_section(name, text):
    score = 0
    feedback = []

    # Metrics check
    if any(term in text.lower() for term in ['%', 'increased', 'decreased', 'reduced', 'grew', 'growth']):
        score += 10
    else:
        feedback.append(f"{name.title()} section lacks quantifiable metrics.")

    # Clarity check: average sentence length
    sentences = [s for s in text.split('.') if len(s.strip()) > 0]
    avg_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    if avg_len > 25:
        feedback.append(f"{name.title()} section could be more concise.")
    else:
        score += 10

    # Recency check for tools in experience/skills (Removed docker and kubernetes)
    recent_keywords = ['transformers', 'gpt', 'cloud', 'k8s', 'mlops']
    if name in ['skills', 'experience'] and any(k in text.lower() for k in recent_keywords):
        score += 10
    else:
        feedback.append(f"{name.title()} section may lack recent technologies.")

    return score, feedback

# paraphrase
def paraphrase_tip(tip):
    prompt = f"Paraphrase this suggestion professionally: {tip}"
    inputs = t5_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = t5_model.generate(inputs['input_ids'], max_length=64)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)



# Resume Improvement Prediction
def suggest_resume_improvements(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    threshold = 0.5
    predicted_labels = [mlb.classes_[i] for i, p in enumerate(probs) if p > threshold]

    # Rule-based suggestions per label
    label_to_tip = {
        "missing_metrics": "Add quantifiable achievements (e.g., 'reduced processing time by 30%').",
        "missing_skills": "Include important technical skills like Python, SQL, or relevant frameworks.",
        "format_issue": "Ensure consistent formatting — align headers, spacing, and bullet points.",
        "outdated_stack": "Update your tech stack to include modern tools like React, AWS, etc.",
        "generic_summary": "Make your summary more specific — highlight major accomplishments and goals."
    }

    suggestions = [label_to_tip[label] for label in predicted_labels if label in label_to_tip]

    return {
        "labels": predicted_labels,
        "suggestions": suggestions
    }

# evaluate section
# def evaluate_section(name, text):
#     tips = []
#
#     # Summary
#     if name == 'summary':
#         if len(text.split()) < 30:
#             tips.append("Expand your summary to give a more complete professional overview.")
#         if any(word in text.lower() for word in ['hardworking', 'team player', 'passionate']):
#             tips.append("Avoid generic terms in your summary. Be specific about your achievements.")
#
#     # Skills
#     if name == 'skills':
#         # Removed docker and kubernetes from here too
#         missing_modern_tools = [tool for tool in ["aws", "git"] if tool not in text.lower()]
#         if missing_modern_tools:
#             tips.append(f"Consider adding modern tools like: {', '.join(missing_modern_tools)}.")
#
#     # Experience
#     if name == 'experience':
#         if not any(char in text for char in "%1234567890"):
#             tips.append("Add measurable results or metrics (e.g., 'Improved efficiency by 25%').")
#         if len(text.split('.')) < 3:
#             tips.append("Break down your experience into clear bullet points.")
#
#     # Education
#     if name == 'education':
#         if not any(keyword in text.lower() for keyword in ['bachelor', 'master', 'degree']):
#             tips.append("Include your degree and university name.")
#         if not any(year in text for year in ['2020', '2021', '2022', '2023', '2024']):
#             tips.append("Consider adding graduation year to show recency.")
#
#     return tips

# app.py
def evaluate_section(name, text, skills=None, education=None):
    tips = []
    skills = skills or []
    education = education or ""

    if name == 'summary':
        if len(text.split()) < 30:
            tips.append(
                "Expand your summary with specific achievements and career goals (e.g., 'Aspiring software engineer with experience in Python web development').")
        if any(word in text.lower() for word in ['hardworking', 'team player', 'passionate']):
            tips.append("Replace generic terms in your summary with concrete accomplishments.")

    if name == 'skills':
        modern_tools = ["aws", "git", "react", "django"]  # Tailored for MCA student
        missing_tools = [tool for tool in modern_tools if tool not in text.lower() and tool not in skills]
        if missing_tools:
            tips.append(
                f"Add relevant modern tools like {', '.join(missing_tools)} to align with software development roles.")

    if name == 'experience':
        if not any(char in text for char in "%1234567890"):
            tips.append("Include measurable results (e.g., 'Developed a web app used by 100+ users').")
        if len(text.split('.')) < 3:
            tips.append("Use bullet points to clearly outline your responsibilities and achievements.")

    if name == 'education':
        if "mca" in education.lower():
            tips.append(
                "Highlight MCA coursework relevant to software development (e.g., 'Studied advanced algorithms and database systems').")
        if not any(year in text for year in ['2020', '2021', '2022', '2023', '2024', '2025']):
            tips.append("Add graduation year to show recency (e.g., 'MCA, Expected 2025').")

    return tips


# rank feedback section
def rank_feedback(feedback_list):
    priority_map = {
        #"missing_metrics": 5,
        "missing_skills": 5,
        "format_issue": 4,
        "outdated_stack": 3,
        "generic_summary": 3,
        "grammar": 2,
        "llm": 3,
        "summary": 3,
        "skills": 4,
        "experience": 5,
        "education": 3,
        "format": 4,
        "improvement": 5,
        "strength": 1  # lower priority
    }

    # Attach a score to each item
    for fb in feedback_list:
        fb["priority"] = priority_map.get(fb.get("type", "").lower(), 2)

    # Sort descending by priority
    sorted_feedback = sorted(feedback_list, key=lambda x: -x["priority"])
    return sorted_feedback


# Similarity Scoring
def compute_similarity(resume_text, job_description):
    embeddings = sbert_model.encode([resume_text, job_description], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(score * 100, 2)

# SQLite Functions
def init_db():
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT, option TEXT, score REAL, skills TEXT, feedback TEXT, timestamp TEXT
    )''')
    conn.commit()
    conn.close()

def save_result(filename, option, score, skills, feedback):
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    c.execute('''INSERT INTO analysis_results
                 (filename, option, score, skills, feedback, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (filename, option, score, ", ".join(skills) if isinstance(skills, list) else skills,
               feedback, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
#deduplication
# def deduplicate_feedback(feedback_list):
#     if not feedback_list:
#         return []
#     messages = [f["message"] for f in feedback_list]
#     embeddings = sbert_model.encode(messages, convert_to_tensor=True)
#     keep = [0]
#     for i in range(1, len(messages)):
#         scores = util.pytorch_cos_sim(embeddings[i], embeddings[keep])
#         if all(score.item() < 0.85 for score in scores[0]):  # Threshold for similarity
#             keep.append(i)
#     deduped = [feedback_list[i] for i in keep]
#     # Consolidate similar suggestions
#     consolidated = []
#     seen = set()
#     for fb in deduped:
#         if fb["type"] in ["summary", "suggestion"] and "summary" in fb["message"].lower():
#             if "summary" not in seen:
#                 consolidated.append({
#                     "type": "summary",
#                     "message": "Enhance your summary with specific, quantifiable achievements (e.g., 'developed a web app used by 100+ users') and clear career goals."
#                 })
#                 seen.add("summary")
#         else:
#             consolidated.append(fb)
#     return consolidated
#

# app.py
# app.py
def deduplicate_feedback(feedback_list):
    if not feedback_list:
        return []
    messages = [f["message"] for f in feedback_list]
    embeddings = sbert_model.encode(messages, convert_to_tensor=True)
    keep = [0]
    for i in range(1, len(messages)):
        scores = util.pytorch_cos_sim(embeddings[i], embeddings[keep])
        if all(score.item() < 0.85 for score in scores[0]):  # Threshold for similarity
            keep.append(i)
    deduped = [feedback_list[i] for i in keep]

    # Consolidate overlapping suggestions
    consolidated = []
    seen = set()
    for fb in deduped:
        msg_lower = fb["message"].lower()
        if fb["type"] in ["Summary", "Suggestion", "Insights"] and "summary" in msg_lower:
            if "summary" not in seen:
                consolidated.append({
                    "type": "Summary",
                    "message": "• Enhance your summary with specific, quantifiable achievements (e.g., 'Developed a web app used by 100+ users') and clear career goals."
                })
                seen.add("summary")
        elif fb["type"] in ["Skills", "Suggestion", "Insights"] and any(
                tool in msg_lower for tool in ["aws", "git", "react", "django"]):
            if "tools" not in seen:
                consolidated.append({
                    "type": "Skills",
                    "message": "• Add modern tools like AWS, Git, React, or Django to align with software development roles."
                })
                seen.add("tools")
        else:
            fb["message"] = f"• {fb['message'].lstrip('• ').strip()}"
            consolidated.append(fb)
    return consolidated

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])

# def analyze():
#     option = request.form.get('option')
#     file = request.files['resume']
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)
#
#     resume_text = extract_text(file_path)
#     print("\n=== Raw Resume Text ===\n", resume_text[:2000])
#
#     # ✅ Split resume into logical sections
#     resume_sections = split_into_sections(resume_text)
#     print("\n=== Section Split Output ===")
#     for k, v in resume_sections.items():
#         print(f"{k.upper()}:\n{v[:300]}\n---")
#     section_titles = ['summary', 'skills', 'experience', 'education']
#     resume_sections = {k.lower(): v for k, v in resume_sections.items() if k.lower() in section_titles}
#
#     # ✅ Initialize section score accumulators
#     section_scores = {}
#     section_feedback = []
#     total_section_score = 0
#
#     # ✅ Score each section
#     for section, text in resume_sections.items():
#         score, feedback = score_section(section, text)
#         section_scores[section] = score
#         total_section_score += score
#         section_feedback.extend([{"type": section, "message": msg} for msg in feedback])
#
#     # 🧠 (Optional) Print debug info
#     print("SECTION SPLIT:", resume_sections)
#     print("SECTION SCORES:", section_scores)
#
#     response_data = {}
#
#     if option == 'analysis':
#         # Extract skills
#         # Clean and extract skills only from the "skills" section
#         if "skills" in resume_sections:
#             cleaned_skill_text = clean_skills_section(resume_sections["skills"])
#             skills_found = extract_skills_from_text(cleaned_skill_text)
#             resume_sections["skills"] = cleaned_skill_text  # ✅ Ensure the preview doesn't show percentages
#         else:
#             skills_found = []
#         num_skills = len(skills_found)
#
#         # Modern skill scoring (Removed docker and kubernetes)
#         modern_tools = {"aws", "azure", "tensorflow", "pytorch"}
#         modern_score = sum(1 for skill in skills_found if skill in modern_tools) * 3
#         skill_score = min(num_skills * 1.5 + modern_score, 40)
#
#         # Metrics & Format
#         has_metrics = any(word in resume_text.lower() for word in ['%', 'percent', 'increased', 'reduced', 'improved'])
#         metrics_score = 30 if has_metrics else 10
#         has_headings = any(h in resume_text.lower() for h in ['experience', 'skills', 'education'])
#         format_score = 30 if has_headings else 15
#
#         base_score = skill_score + metrics_score + format_score
#
#         # Section Analysis
#         section_feedback = []
#         for section, content in resume_sections.items():
#             tips = evaluate_section(section, content)
#             for tip in tips:
#                 section_feedback.append({"type": section.capitalize(), "message": tip})
#         section_score = total_section_score  # from earlier scoring loop
#
#         # BERT Feedback
#         improvement_suggestions = suggest_resume_improvements(resume_text)
#         bert_score = min(len(improvement_suggestions) * 10, 60)  # 3 issues = 30 score, cap at 60
#
#         # Hybrid Score
#         hybrid_score = round((0.6 * bert_score) + (0.4 * section_score), 2)
#
#         # Final Feedback
#         # feedback = [
#         #         {"type": "Strength", "message": f"Detected skills: {', '.join(skills_found)}"}
#         #     ]
#         #     feedback += [
#         #         {"type": "Suggestion", "message": tip}
#         #         for tip in improvement_suggestions.get("suggestions", [])
#         #     ]
#         #     feedback += [
#         #         {"type": sec.capitalize(), "message": tip}
#         #         for sec, tips in [(sec, evaluate_section(sec, content)) for sec, content in resume_sections.items()]
#         #         for tip in tips
#         #     ]
#         #     if not has_metrics:
#         #         feedback.append({"type": "Improvement", "message": "Add quantifiable results (e.g., 'reduced costs by 20%')."})
#         #     if not has_headings:
#         #         feedback.append({"type": "Format", "message": "Add clear section headings like 'Experience', 'Skills', 'Education'."})
#         #     feedback += get_t5_feedback(resume_text)
#         #     feedback += [
#         #         {"type": "Style", "message": msg["message"]}
#         #         for msg in advanced_style_checks(resume_text)
#         #     ]
#         #     feedback = deduplicate_feedback(feedback)
#         #     feedback = rank_feedback(feedback)
#         #
#         #     # Format for display
#         #     formatted_feedback = [
#         #         {"type": fb["type"], "message": f"• {fb['message']}"}
#         #         for fb in feedback
#         #     ]
#         #
#         #     # Update summary_report
#         #     summary_report = {
#         #         "ATS Score": f"{hybrid_score}%",
#         #         "Top Skills": skills_found[:5] if skills_found else ["Not detected"],
#         #         "Resume Highlights": [],
#         #         "Suggestions Summary": [f["message"].lstrip("• ") for f in feedback if f["type"] not in ("Strength", "Insights")]
#         #     }
#         feedback = [
#             {"type": "Strength", "message": f"Detected skills: {', '.join(skills_found)}"}
#         ]
#         feedback += [
#             {"type": "Suggestion", "message": tip}
#             for tip in improvement_suggestions.get("suggestions", [])
#         ]
#         feedback += [
#             {"type": sec.capitalize(), "message": tip}
#             for sec, tips in [(sec, evaluate_section(sec, content)) for sec, content in resume_sections.items()]
#             for tip in tips
#         ]
#         if not has_metrics:
#             feedback.append(
#                 {"type": "Improvement", "message": "Add quantifiable results (e.g., 'reduced costs by 20%')."})
#         if not has_headings:
#             feedback.append(
#                 {"type": "Format", "message": "Add clear section headings like 'Experience', 'Skills', 'Education'."})
#         feedback += get_t5_feedback(resume_text)
#         feedback += [
#             {"type": "Style", "message": msg["message"]}
#             for msg in advanced_style_checks(resume_text)
#         ]
#         feedback = deduplicate_feedback(feedback)
#         feedback = rank_feedback(feedback)
#
#         # Format for display
#         formatted_feedback = [
#             {"type": fb["type"], "message": f"• {fb['message']}"}
#             for fb in feedback
#         ]
#
#         # Update summary_report
#         summary_report = {
#             "ATS Score": f"{hybrid_score}%",
#             "Top Skills": skills_found[:5] if skills_found else ["Not detected"],
#             "Resume Highlights": [],
#             "Suggestions Summary": [f["message"].lstrip("• ") for f in feedback if
#                                     f["type"] not in ("Strength", "Insights")]
#         }
#
#         # Add highlights based on actual conditions
#         if any(tool in resume_text.lower() for tool in ['aws']): # Removed docker and kubernetes
#             summary_report["Resume Highlights"].append("Shows familiarity with modern DevOps tools.")
#         else:
#             summary_report["Resume Highlights"].append("Consider adding modern tools like AWS.") # Removed docker and kubernetes
#
#         if has_metrics:
#             summary_report["Resume Highlights"].append("Includes measurable achievements (e.g., 20% improvement).")
#         else:
#             summary_report["Resume Highlights"].append("Lacks quantified results. Try adding metrics or percentages.")
#
#         if has_headings:
#             summary_report["Resume Highlights"].append("Structured with clear section headings.")
#         else:
#             summary_report["Resume Highlights"].append("Missing section headers like Experience or Skills.")
#
#         analysis = {
#             "matchScore": hybrid_score,
#             "bertScore": bert_score,
#             "sectionScore": section_score,
#             "skillsFound": num_skills,
#             "recommendation": "Strong" if hybrid_score > 75 else "Moderate" if hybrid_score > 50 else "Weak",
#             "feedback": feedback, # Already ranked and comprehensive
#             "summaryReport": summary_report
#
#         }
#
#         save_result(
#             filename=file.filename,
#             option="analysis",
#             score=hybrid_score,
#             skills=skills_found,
#             feedback="; ".join([f"{f['type']}: {f['message']}" for f in analysis["feedback"]])
#         )
#
#         response_data = {"analysisResults": analysis}

@app.route('/analyze', methods=['POST'])
def analyze():
    option = request.form.get('option')
    file = request.files['resume']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    resume_text = extract_text(file_path)
    print("\n=== Raw Resume Text ===\n", resume_text[:2000])

    # Split resume into logical sections
    resume_sections = split_into_sections(resume_text)
    print("\n=== Section Split Output ===")
    for k, v in resume_sections.items():
        print(f"{k.upper()}:\n{v[:300]}\n---")
    section_titles = ['summary', 'skills', 'experience', 'education']
    resume_sections = {k.lower(): v for k, v in resume_sections.items() if k.lower() in section_titles}

    # Initialize section score accumulators
    section_scores = {}
    section_feedback = []
    total_section_score = 0

    # Score each section
    for section, text in resume_sections.items():
        score, feedback = score_section(section, text)
        section_scores[section] = score
        total_section_score += score
        section_feedback.extend([{"type": section, "message": msg} for msg in feedback])

    print("SECTION SPLIT:", resume_sections)
    print("SECTION SCORES:", section_scores)

    response_data = {}

    if option == 'analysis':
        # Extract skills
        if "skills" in resume_sections:
            cleaned_skill_text = clean_skills_section(resume_sections["skills"])
            skills_found = extract_skills_from_text(cleaned_skill_text)
            resume_sections["skills"] = cleaned_skill_text
        else:
            skills_found = []
        num_skills = len(skills_found)

        # Modern skill scoring
        modern_tools = {"aws", "azure", "tensorflow", "pytorch"}
        modern_score = sum(1 for skill in skills_found if skill in modern_tools) * 3
        skill_score = min(num_skills * 1.5 + modern_score, 40)

        # Metrics & Format
        has_metrics = any(word in resume_text.lower() for word in ['%', 'percent', 'increased', 'reduced', 'improved'])
        metrics_score = 30 if has_metrics else 10
        has_headings = any(h in resume_text.lower() for h in ['experience', 'skills', 'education'])
        format_score = 30 if has_headings else 15

        base_score = skill_score + metrics_score + format_score

        # Section Analysis
        section_feedback = []
        for section, content in resume_sections.items():
            tips = evaluate_section(section, content, skills=skills_found, education=resume_sections.get('education', ''))
            section_feedback.extend([{"type": section.capitalize(), "message": tip} for tip in tips])
        section_score = total_section_score

        # BERT Feedback
        improvement_suggestions = suggest_resume_improvements(resume_text)
        bert_score = min(len(improvement_suggestions) * 10, 60)

        # Hybrid Score
        hybrid_score = round((0.6 * bert_score) + (0.4 * section_score), 2)

        # Final Feedback
        # app.py
        # app.py
        # Final Feedback
        feedback = [{"type": "Strength", "message": f"• Detected skills: {', '.join(skills_found)}"}]
        feedback += [{"type": "Suggestion", "message": f"• {tip}"} for tip in
                     improvement_suggestions.get("suggestions", [])]
        feedback += section_feedback
        if not has_metrics:
            feedback.append(
                {"type": "Improvement", "message": "• Add quantifiable results (e.g., 'reduced costs by 20%')."})
        if not has_headings:
            feedback.append(
                {"type": "Format", "message": "• Add clear section headings like 'Experience', 'Skills', 'Education'."})
        feedback += get_t5_feedback(resume_text)
        feedback += [{"type": "Style", "message": f"• {msg['message']}"} for msg in advanced_style_checks(resume_text)]

        # Deduplicate and consolidate feedback
        feedback = deduplicate_feedback(feedback)
        feedback = rank_feedback(feedback)
        # Build Summary Report
        summary_report = {
            "ATS Score": f"{hybrid_score}%",
            "Top Skills": skills_found[:5] if skills_found else ["Not detected"],
            "Resume Highlights": [],
            "Suggestions Summary": [f["message"] for f in feedback if f["type"] not in ("strength", "llm")]
        }

        if any(tool in resume_text.lower() for tool in ['aws']):
            summary_report["Resume Highlights"].append("Shows familiarity with modern DevOps tools.")
        else:
            summary_report["Resume Highlights"].append("Consider adding modern tools like AWS.")

        if has_metrics:
            summary_report["Resume Highlights"].append("Includes measurable achievements (e.g., 20% improvement).")
        else:
            summary_report["Resume Highlights"].append("Lacks quantified results. Try adding metrics or percentages.")

        if has_headings:
            summary_report["Resume Highlights"].append("Structured with clear section headings.")
        else:
            summary_report["Resume Highlights"].append("Missing section headers like Experience or Skills.")

        analysis = {
            "matchScore": hybrid_score,
            "bertScore": bert_score,
            "sectionScore": section_score,
            "skillsFound": num_skills,
            "recommendation": "Strong" if hybrid_score > 75 else "Moderate" if hybrid_score > 50 else "Weak",
            "feedback": feedback,
            "summaryReport": summary_report
        }

        save_result(
            filename=file.filename,
            option="analysis",
            score=hybrid_score,
            skills=skills_found,
            feedback="; ".join([f"{f['type']}: {f['message']}" for f in analysis["feedback"]])
        )

        response_data = {"analysisResults": analysis}

    elif option == 'questions':

        #questions = generate_questions(resume_text)
        # resume_sections = split_into_sections(resume_text)
        # questions = generate_questions_from_resume_sections(resume_sections)
        # response_data = {"questionsResults": questions}
        questions = generate_questions_from_resume(resume_text)
        print("Debug: Questions generated:",questions)
        response_data = {"questionsResults": questions}

    elif option == 'modification':
        result = suggest_resume_improvements(resume_text)

        response_data = {
            "modificationResults": [

                                       {"type": "Suggestion", "message": tip} for tip in result["suggestions"]
                                   ]
        }



    elif option == 'matching':
        job_title = request.form.get('job_title')
        job_desc = request.form.get('job_description')
        similarity = compute_similarity(resume_text, job_desc)
        response_data = {
            "matchingResults": [
                {"type": "Match Score", "message": f"{similarity}% similarity between resume and job description."},
                {"type": "Summary", "message": "This is a semantic match, not just keyword overlap."},
                {"type": "Tip", "message": "Ensure your resume mirrors key terms used in the job description."}
            ]
        }
        save_result(file.filename, option, similarity, "", "; ".join(
            [f"{m['type']}: {m['message']}" for m in response_data["matchingResults"]]))

    else:
        response_data = {"error": "Invalid option selected"}



    return jsonify(response_data)

@app.route('/predict-category', methods=['POST'])
def predict_category():
    file = request.files['resume']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    resume_text = extract_text(file_path)

    inputs = bert_tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    predicted = [mlb.classes_[i] for i, p in enumerate(probs) if p > 0.5]
    return jsonify({"categories": predicted})

def rank_feedback(feedback_list):
    priority_map = {
        "missing_metrics": 5,
        "missing_skills": 5,
        "format_issue": 4,
        "outdated_stack": 3,
        "generic_summary": 3,
        "grammar": 2,
        "llm": 3,
        "summary": 3,
        "skills": 4,
        "experience": 5,
        "education": 3,
        "format": 4,
        "improvement": 5,
        "strength": 1  # lower priority
    }

    # Attach a score to each item
    for fb in feedback_list:
        fb["priority"] = priority_map.get(fb.get("type", "").lower(), 2)

    # Sort descending by priority
    sorted_feedback = sorted(feedback_list, key=lambda x: -x["priority"])
    return sorted_feedback

# utilize function
def score_section(section_name, text):
    score = 0
    feedback = []

    # Metric presence
    if re.search(r'\d+%|\$|years|reduced|increased|grew|saved|cut', text, re.I):
        score += 10
    else:
        feedback.append(f"{section_name.capitalize()}: No metrics or quantified results found.")

    # Sentence structure
    sentences = re.split(r'\.|\n', text)
    avg_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
    if 8 <= avg_length <= 25:
        score += 10
    else:
        feedback.append(f"{section_name.capitalize()}: Sentences might be too short or too long.")

    # Recency & relevance
    recent_years = re.findall(r'20\d{2}', text)
    if any(int(y) >= 2019 for y in recent_years):
        score += 5
    else:
        feedback.append(f"{section_name.capitalize()}: Consider including recent experiences or education.")

    modern_keywords = [ 'aws', 'tensorflow', 'pytorch','aws',''] # Removed docker and kubernetes
    if any(word in text.lower() for word in modern_keywords):
        score += 5
    else:
        feedback.append(f"{section_name.capitalize()}: Modern tools or frameworks not mentioned.")

    return min(score, 30), feedback

# resume score section
def score_resume_sections(sections):
    scores = {
        'summary': 0,
        'skills': 0,
        'experience': 0,
        'education': 0
    }

    feedback = []

    # Summary
    summary = sections.get('summary', '')
    if len(summary.split()) > 20:
        scores['summary'] = 20
        if any(word in summary.lower() for word in ['experienced', 'proven', 'results', 'track record']):
            scores['summary'] += 5
        else:
            feedback.append({"type": "summary", "message": "Your summary could include more impact keywords."})
    else:
        feedback.append({"type": "summary", "message": "Your summary is too short. Aim for 2–3 impactful lines."})

    # Skills
    skills = sections.get('skills', '')
    skill_count = len(skills.split(','))  # crude split
    if skill_count >= 6:
        scores['skills'] = 25
    elif skill_count >= 3:
        scores['skills'] = 15
        feedback.append({"type": "skills", "message": "Add more relevant skills to strengthen your profile."})
    else:
        scores['skills'] = 5
        feedback.append({"type": "skills", "message": "Too few skills listed. Add 5–10 relevant skills."})

    # Experience
    experience = sections.get('experience', '')
    if any(x in experience.lower() for x in ['led', 'built', 'developed', 'managed', 'improved']):
        scores['experience'] = 25
    elif len(experience.strip().split()) > 50:
        scores['experience'] = 20
        feedback.append({"type": "experience", "message": "Include action verbs like 'led', 'managed', etc. for more impact."})
    else:
        scores['experience'] = 10
        feedback.append({"type": "experience", "message": "Expand your experience section with more detail and results."})

    # Education
    education = sections.get('education', '')
    if any(x in education.lower() for x in ['bachelor', 'master', 'phd']):
        scores['education'] = 25
    else:
        scores['education'] = 10
        feedback.append({"type": "education", "message": "Mention your degree clearly (e.g., B.Sc, M.Tech)."})

    total = sum(scores.values())
    return total, scores, feedback

#new analyze_resume

# def analyze_resume():
#     try:
#         # Log incoming request
#         logger.debug("Received request to /analyze")
#         logger.debug(f"Form data: {request.form}")
#         logger.debug(f"Files: {request.files}")
#
#         # Check for required fields
#         if 'resume' not in request.files:
#             logger.error("No resume file provided")
#             return jsonify({"error": "No resume file provided"}), 400
#         if 'option' not in request.form:
#             logger.error("No option provided")
#             return jsonify({"error": "No option provided"}), 400
#
#         resume_file = request.files['resume']
#         option = request.form['option']
#         job_title = request.form.get('job_title', '')
#         job_description = request.form.get('job_description', '')
#
#         logger.debug(f"Option: {option}, Job Title: {job_title}, Job Description: {job_description[:50]}...")
#
#         # Placeholder for file processing (e.g., PDF/DOCX parsing)
#         # Add your AI processing logic here
#         # Example: Read PDF or DOCX, process with AI, return results
#         result = {
#             "analysisResults": {
#                 "matchScore": 85,
#                 "skillsFound": 12,
#                 "recommendation": "Strong",
#                 "feedback": [
#                     {"type": "strength", "message": "Strong Python skills."},
#                     {"type": "improvement", "message": "Add metrics."}
#                 ]
#             },
#             "questionsResults": [
#                 {"category": "AI", "question": "Describe your ML experience."}
#             ],
#             "modificationResults": [
#                 {"type": "Format", "message": "Use consistent bullet points."}
#             ],
#             "matchingResults": [
#                 {"type": "Strong Match", "message": "85% match with job requirements."}
#             ],
#             "matchScore": 85  # Explicit match score
#         }
#
#         logger.debug("Analysis completed successfully")
#         return jsonify(result)
#
#     except Exception as e:
#         logger.error(f"Error in /analyze: {str(e)}", exc_info=True)
#         return jsonify({"error": "Internal server error"}), 500

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    data = request.json
    pdf = generate_pdf_report(data)
    return send_file(pdf, as_attachment=True, download_name="resume_analysis.pdf", mimetype='application/pdf')

def generate_pdf_report(data, filename="resume_report.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AI Resume Analysis Report")
    y -= 30

    c.setFont("Helvetica", 12)
    for key, value in data.items():
        if key == "resumeText":
            continue  # ✅ Skip raw resume text

        if isinstance(value, list):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, f"{key}:")
            y -= 20
            c.setFont("Helvetica", 10)
            for item in value:
                text = f"• {item.get('type', '')}: {item.get('message', '')}"
                c.drawString(70, y, text)
                y -= 15
                if y < 100:
                    c.showPage()
                    y = height - 50
        else:
            c.setFont("Helvetica", 10)
            c.drawString(50, y, f"{key}: {value}")
            y -= 20
            if y < 100:
                c.showPage()
                y = height - 50

    c.save()
    buffer.seek(0)
    return buffer


# Entry Point
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0',debug=True, port=5000)