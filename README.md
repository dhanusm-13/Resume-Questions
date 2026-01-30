Project Title
🤖 AI-Powered Resume Screening & Interview Question Generator
This is an intelligent system that analyzes resumes and automatically generates personalized interview questions. By combining Natural Language Processing (NLP), Machine Learning, and Deep Learning, the project helps recruiters and job seekers streamline the hiring process through automated parsing and intelligent question synthesis.

🚀 Key Features
📂 Resume Parsing – Automatically extracts skills, education, projects, and certifications from uploads.
🧠 AI Question Generation – Uses fine-tuned T5 and BERT models to create role-specific interview questions.
✅ ATS Compatibility Check – Analyzes resumes for missing sections or formatting issues to improve ranking.
📊 Reports & Analytics – Provides scoring reports and actionable suggestions for resume improvement.
⚡ Web App Interface – Provides a seamless user experience built with the Flask framework.

Component,Technology
Backend,Flask (Python)
ML/DL Models,"T5, BERT, SBERT"
Frameworks,"TensorFlow / PyTorch, Hugging Face Transformers"
Frontend,"HTML, CSS, JavaScript"
Database,SQLite

📂 Project Structure
Resume-Questions/
├── app.py                      # Flask app entry point
├── inference.py                # Inference scripts
├── question_generator.py       # T5-based question generation
├── skills_pipeline.py          # Skill extraction pipeline
├── train_t5_finetune.py        # Training script for T5 model
├── train_bert_resume_model.py  # Training script for BERT
├── utils.py                    # Utility functions
├── static/                     # CSS & JS files
├── templates/                  # HTML templates
├── models/                     # Pretrained / fine-tuned models
│   ├── resume_bert_model/      # BERT model files
│   └── t5_finetuned_questions/ # Fine-tuned T5 model files
├── requirements.txt            # Python dependencies
└── default_questions.json      # Base questions

🖥 System Requirements
Python 3.8 – 3.11
RAM: 8 GB (Recommended for running Transformers models)
Storage: Sufficient space for downloading pre-trained models from Hugging Face

📌 Installation & Setup
1️⃣ Create Virtual Environment
Bash
python -m venv .venv
Windows: .venv\Scripts\activate
Linux/Mac: source .venv/bin/activate

2️⃣ Install Dependencies
Bash
pip install -r requirements.txt

3️⃣ Run the Application
Bash
python app.py
Open your browser at: http://127.0.0.1:5000🧠 

System Workflow
Resume Upload: User uploads a PDF or DOCX resume via the Flask web interface.
Information Extraction: The BERT-based parser identifies key sections (Skills, Education, Experience).
Semantic Analysis: SBERT matches extracted skills against job requirements or categories.
Question Synthesis: The Fine-tuned T5 model processes the extracted text to generate tailored interview questions.
Output: The system displays a comprehensive report, including the generated questions and an ATS compatibility score.

🔮 Future Enhancements
Multi-format support: Enhancing support for various document layouts.

Real-time Feedback: Integration with a live chat interface for candidates.

Automated Scoring: Enhanced deep learning models for candidate ranking.
👨‍💻 Developer
Name: Dhananjaya S M

Project: 🤖 AI-Powered Resume Screening & Interview Question Generator 
📜 LicenseThis project is licensed under the MIT License.
