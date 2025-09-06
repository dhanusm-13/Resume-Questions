from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import re
import json
import logging
import io
import torch

from skills_pipeline import extract_skills_from_sections
from utils import bert_tokenizer, bert_model, device, mlb
from utils import (
    extract_text, extract_skills_from_text, clean_skills_section, detect_resume_sections,
    get_t5_feedback, advanced_style_checks, suggest_resume_improvements,
    evaluate_section, deduplicate_feedback, rank_feedback, compute_similarity,
    init_db, save_result, generate_pdf, score_section, score_resume_sections,
    check_ats_formatting, check_language, detect_red_flags, extract_soft_skills, generate_insights,
    compute_section_match,clean_text
)
from question_generator import generate_questions_from_resume


# INIT APP
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    option = request.form.get('option')
    print("== Option selected:", option)

    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
    file = request.files['resume']

    if not file.filename.lower().endswith(('.pdf', '.docx')):
        return jsonify({"error": "Only PDF or DOCX files are supported for ATS compatibility"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    resume_text = extract_text(file_path)
    logger.debug(f"Raw Resume Text: {resume_text[:2000]}")

    resume_sections = detect_resume_sections(resume_text)
    print("== SECTIONS DETECTED ==")
    for key, value in resume_sections.items():
        if isinstance(value, list):
            # join lists into readable string
            display_value = ", ".join(value[:8])  # show first 8 items
        else:
            display_value = str(value)[:100]  # keep 100 chars for long text
        print(f"-> {key}: {display_value}")

    section_titles = ['summary', 'skills', 'experience', 'education', 'projects', 'certifications']
    resume_sections = {k.lower(): v for k, v in resume_sections.items() if k.lower() in section_titles}

    # ----------- CLEANING PHASE -----------
    skills_entry = resume_sections.get("skills", "")
    if isinstance(skills_entry, list):
        skills_text = " ".join([
            v for v in skills_entry
            if len(v.split()) <= 4  # keep short tokens like "Python", "SQL"
               and not any(bad in v.lower() for bad in ["experience", "summary", "project", "industry"])
        ])
    else:
        skills_text = str(skills_entry)

    resume_sections["skills"] = skills_text.strip()
    # Certifications cleanup (remove unrelated stuff)
    UNWANTED_TOKENS = [
        "languages known", "extra-curricular", "academic seminar",
        "team sports", "professional working proficiency"
    ]
    certs_entry = resume_sections.get("certifications", "")
    if isinstance(certs_entry, list):
        certs_text = " ".join([
            v for v in certs_entry
            if not any(bad in v.lower() for bad in UNWANTED_TOKENS)
        ])
    else:
        certs_text = str(certs_entry)
    resume_sections["certifications"] = certs_text

    edu_entry = resume_sections.get("education", "")
    if isinstance(edu_entry, list):
        edu_text = " ".join(edu_entry)
    else:
        edu_text = str(edu_entry)

    edu_items = re.split(r'(?i)(?=bachelor|master|ph\.?d|pre university|high school|diploma)', edu_text)

    # Clean and filter
    edu_items = [e.strip() for e in edu_items if e.strip()]

    # Save back
    resume_sections["education"] = edu_items  # structured list
    resume_sections["education_text"] = " ".join(edu_items)

    # SKILL EXTRACTION
    skills_found = extract_skills_from_sections({"skills": skills_text}) or []

    response_data = {}

    if option == 'analysis':
        #  Extract Skills
        skills_found = extract_skills_from_text(resume_text)  # ✅ use clean extractor
        num_skills = len(skills_found)
        logger.debug(f"Skills Found: {skills_found}")
        logger.debug(f"Number of skills found: {num_skills}")

        # Extract soft skills
        soft_skills_found = extract_soft_skills(resume_text)

        # Score sections
        total_score, section_scores, section_feedback = score_resume_sections(resume_sections)

        #AI / BERT Feedback
        improvement_suggestions = suggest_resume_improvements(
            resume_text, detected_skills=skills_found, resume_sections=resume_sections
        )
        bert_score = min(len(improvement_suggestions.get("suggestions", [])) * 10, 60)

        # ---- Hybrid ATS Score ----
        hybrid_score = round(0.5 * bert_score + 0.5 * total_score, 2)

        feedback = []

        # Technical skills feedback
        if skills_found:
            feedback.append({
                "type": "Strength",
                "message": f"Detected skills: {', '.join(skills_found)}"
            })
        else:
            feedback.append({
                "type": "Suggestion",
                "message": "No technical skills were clearly detected. Consider listing tools or languages explicitly, such as Python, SQL, or React."
            })

        # AI suggestions
        feedback += [{"type": "Suggestion", "message": f"{tip}"} for tip in
                     improvement_suggestions.get("suggestions", [])]

        # Section feedback
        feedback += section_feedback

        # Formatting / language / red flags
        feedback += check_ats_formatting(resume_text, resume_sections)
        feedback += check_language(resume_text)
        feedback += detect_red_flags(resume_sections)

        # Soft skills
        if soft_skills_found:
            feedback.append({
                "type": "Strength",
                "message": f"Soft skills identified: {', '.join(soft_skills_found)}"
            })

        # Extra insights
        feedback += generate_insights(resume_sections)

        # Extra suggestion rules
        if not re.search(r'\d+%|\$|increased|reduced', resume_text, re.IGNORECASE):
            feedback.append({
                "type": "Suggestion",
                "message": "Add measurable results (e.g., 'Improved efficiency by 20%')."
            })

        if not any(word in resume_text.lower() for word in ["developed", "led", "implemented", "designed"]):
            feedback.append({
                "type": "Suggestion",
                "message": "Use strong action verbs like 'Developed', 'Led', 'Designed' to start bullet points."
            })

        if resume_text.count("\n") < 10:
            feedback.append({
                "type": "Suggestion",
                "message": "Break content into clear sections with bullet points for better readability."
            })

        # CLEAN FEEDBACK
        feedback = [f for f in feedback if f.get("message", "").strip()]
        feedback = deduplicate_feedback(feedback)
        feedback = rank_feedback(feedback)

        # Grouped feedback
        grouped_feedback = {
            "ATS Optimization": [f["message"].strip() for f in feedback if f["type"] == "ATS Optimization"],
            "Insights": [f["message"].strip() for f in feedback if f["type"] == "Insights"],
            "Suggestions": [f["message"].strip() for f in feedback if f["type"] == "Suggestion"],
            "Strengths": [f["message"].strip() for f in feedback if f["type"] == "Strength"],
            "Style": [f["message"].strip() for f in feedback if f["type"] == "Style"],
            "Red Flags": [f["message"].strip() for f in feedback if f["type"] == "Red Flag"],
        }
        grouped_feedback = {k: v for k, v in grouped_feedback.items() if v}

        # Summary Report
        summary_report = {
            "ATS Score": f"{hybrid_score}%",
            "Top Skills": skills_found[:5] if skills_found else ["None detected"],
            "Resume Highlights": [],
            "Suggestions Summary": [f["message"].strip() for f in feedback if f["type"] not in ("Strength", "Insights")]
        }

        if any(tool in resume_text.lower() for tool in ['aws', 'pytorch', 'tensorflow']):
            summary_report["Resume Highlights"].append("Includes modern tools relevant to software roles.")
        else:
            summary_report["Resume Highlights"].append("Consider adding modern tools like AWS or TensorFlow.")

        if re.search(r'\d+%|\$|increased|reduced', resume_text, re.IGNORECASE):
            summary_report["Resume Highlights"].append("Contains measurable achievements.")
        else:
            summary_report["Resume Highlights"].append("Add quantifiable results (e.g., 'reduced costs by 20%').")

        if any(h in resume_text.lower() for h in ['experience', 'skills', 'education']):
            summary_report["Resume Highlights"].append("Structured with clear section headings.")
        else:
            summary_report["Resume Highlights"].append("Add section headers like 'Skills' or 'Experience'.")

        if hybrid_score > 85:
            recommendation = "Tier 1 – Highly Recommend"
        elif hybrid_score > 70:
            recommendation = "Tier 2 – Consider"
        else:
            recommendation = "Tier 3 – Needs Improvement"

        summary_report["Resume Tier"] = recommendation

        # Final Response
        analysis = {
            "matchScore": hybrid_score,
            "bertScore": bert_score,
            "sectionScore": total_score,
            "skillsFound": num_skills,
            "recommendation": recommendation,
            "feedback": feedback,
            "feedbackGrouped": grouped_feedback,
            "summaryReport": summary_report
        }

        response_data = {"analysisResults": analysis}
        logger.debug(f"Analysis response: {response_data}")

    elif option == 'questions':
        questions = generate_questions_from_resume(resume_text)
        logger.debug(f"Questions generated: {questions}")
        response_data = {"questionsResults": questions}

    elif option == 'modification':
        job_title = request.form.get('job_title', '').strip()
        job_desc = request.form.get('job_description', '').strip()

        result = suggest_resume_improvements(
            resume_text,
            detected_skills=skills_found,
            resume_sections=resume_sections,
            job_description=job_desc,
            job_title=job_title
        )

        modification_results = []
        counter = 1
        for sec in result["sections"]:
            lines = [f"{counter}. {sec.get('title', '')}"]

            if sec.get("cv_has"):
                lines.append("Your CV lists:")
                lines.extend([f"- {skill}" for skill in sec["cv_has"]])

            if sec.get("jd_has"):
                lines.append("JD expects:")
                lines.extend([f"- {skill}" for skill in sec["jd_has"]])

            if sec.get("missing"):
                lines.append("Missing:")
                lines.extend([f"- {skill}" for skill in sec["missing"]])

            if sec.get("suggestion"):
                lines.append(f"Suggestion: {sec['suggestion']}")

            if sec.get("tips"):
                lines.append("Tips:")
                lines.extend([f"- {tip}" for tip in sec["tips"]])

            modification_results.append({
                "type": "Suggestion",
                "message": "\n".join(lines)
            })
            counter += 1

        response_data = {"modificationResults": modification_results}
        logger.debug(f"Modification response: {response_data}")

    elif option == 'matching':
        job_title = request.form.get('job_title')
        job_desc = request.form.get('job_description')
        overall_similarity = compute_similarity(resume_text, job_desc)
        section_match_scores = compute_section_match(resume_sections, job_desc)
        jd_skills = set(extract_skills_from_sections({"skills": job_desc}))
        resume_skills = set(extract_skills_from_sections({"skills": resume_text}))
        matched_skills = jd_skills & resume_skills
        missing_skills = jd_skills - resume_skills

        exp_score = section_match_scores.get("experience")
        if exp_score is None and "projects" in resume_sections:
            exp_score = section_match_scores.get("projects", 0)
            exp_message = f"{exp_score:.2f}% - Calculated from your Projects section since no formal Experience section was found."
        elif exp_score is None:
            exp_score = 0
            exp_message = "Experience section not found — add internships, projects, or roles to improve this score."
        else:
            exp_message = f"{exp_score:.2f}%"

        # Education and certification match
        edu_score = section_match_scores.get("education", 0)
        cert_score = section_match_scores.get("certifications", 0)

        # Skills score
        skills_score = (len(matched_skills) / max(1, len(jd_skills))) * 100 if jd_skills else 0

        # Detect fresher role
        fresher_keywords = ["fresher", "graduate", "entry-level", "0-1 years"]
        is_fresher_role = any(keyword in job_desc.lower() for keyword in fresher_keywords)

        # Weighting
        if is_fresher_role:
            weights = {"skills": 0.5, "experience": 0.2, "education": 0.2, "certifications": 0.1}
        else:
            weights = {"skills": 0.4, "experience": 0.35, "education": 0.15, "certifications": 0.1}

        # Final weighted score
        final_score = round(
            (skills_score * weights["skills"]) +
            (exp_score * weights["experience"]) +
            (edu_score * weights["education"]) +
            (cert_score * weights["certifications"]),
            2
        )

        # labels for tiers
        tier = "Excellent Match" if final_score >= 85 else "Good Match" if final_score >= 70 else "Needs Improvement"
        # High priority vs other missing skills
        high_priority_skills = sorted(list(missing_skills))[:3]
        other_missing_skills = sorted(list(missing_skills))[3:]
        # Build tips / next steps
        next_steps = []

        if high_priority_skills:
            next_steps.append(f"Gain hands-on experience with {', '.join(high_priority_skills)}.")
        if exp_score < 40 and not is_fresher_role:
            next_steps.append("Build more professional experience or highlight relevant past projects.")
        if cert_score < 50:
            next_steps.append("Earn certifications in key areas to strengthen your profile.")

        # Recommendations
        if final_score >= 75:
            recommendation = "Strong match — You should definitely apply for this role."
        elif final_score >= 60:
            recommendation = f"Moderate match — You can apply, but tailor your resume. Focus on {', '.join(high_priority_skills)}."
        else:
            recommendation = f"Low match — Consider improving your resume first. High priority skills: {', '.join(high_priority_skills)}."

        # Add experience gap note for experienced roles
        if not is_fresher_role and exp_score < 40:
            recommendation += " Note: Your experience level appears below typical requirements for this role."

        # Format matching results
        matching_results = [
            {"type": "Overall Similarity", "message": f"{overall_similarity:.2f}%"},
            {"type": "Final Match Score", "message": f"{final_score:.2f}% ({tier})"},
            {"type": "Skills Matched", "message": f"{', '.join(sorted(matched_skills)) or 'None'}"},
            {"type": "High Priority Missing Skills", "message": f"{', '.join(high_priority_skills) or 'None'}"},
            {"type": "Other Missing Skills", "message": f"{', '.join(other_missing_skills) or 'None'}"},
            {"type": "Experience Match", "message": exp_message},
            {"type": "Education Match", "message": f"{edu_score:.2f}%"},
            {"type": "Certification Match", "message": f"{cert_score:.2f}%"},
            {"type": "Recommendation", "message": recommendation},
            {"type": "Next Steps",
             "message": " • ".join(next_steps) if next_steps else "No additional steps recommended."}
        ]
        response_data = {"matchingResults": matching_results}
        logger.debug(f"Matching response: {response_data}")
    else:
        response_data = {"error": "Invalid option selected"}

    logger.debug(f"Final response data: {response_data}")
    save_result(
        file.filename,
        option,
        response_data.get("analysisResults", {}).get("matchScore", 0),
        skills_found,
        "; ".join([
            f"{m['type']}: {m['message']}" for m in (
                    response_data.get("matchingResults") or
                    response_data.get("modificationResults") or
                    response_data.get("questionsResults") or []
            ) if 'type' in m and 'message' in m
        ])
    )
    print("=== FINAL RESPONSE JSON ===")
    print(json.dumps(response_data, indent=2))

    return jsonify(response_data)
2
@app.route('/predict-category', methods=['POST'])
def predict_category():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
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


@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    try:
        data = request.get_json()
        pdf_bytes = generate_pdf(data)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="resume_analysis.pdf"
        )
    except Exception as e:
        print("PDF generation failed:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', debug=True, port=5000)

