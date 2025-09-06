// Initialize DOM element references
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('resumeFile');
const fileInfo = document.getElementById('fileInfo');
const optionsCard = document.getElementById('optionsCard');
const jobDescCard = document.getElementById('jobDescCard');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.getElementById('btnText');
const resultsContainer = document.getElementById('results');
const loadingIndicator = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const jobTitleInput = document.getElementById('jobTitle');
const jobDescriptionInput = document.getElementById('jobDescription');
const themeToggle = document.getElementById('themeToggle');

let selectedFile = null;
let selectedOption = null;

// Initialize window.report to store all results, preserving previous data
window.report = window.report || {
    analysisResults: null,
    questionsResults: [],
    modificationResults: [],
    matchingResults: []
};

// Initialize theme based on localStorage or system preference
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const defaultTheme = savedTheme || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', defaultTheme);
    themeToggle.textContent = defaultTheme === 'dark' ? 'Light' : 'Dark';
}

function showLoading(message = 'Processing...') {
    const loadingIndicator = document.getElementById('loading');
    if (!loadingIndicator) return; // if missing, exit

    // Restore the spinner HTML if somehow removed
    if (!document.getElementById('loadingText')) {
        loadingIndicator.innerHTML = `
            <div class="spinner"></div>
            <p id="loadingText"></p>
            <small>This may take a few moments</small>
        `;
    }

    loadingIndicator.style.display = 'block';
    document.getElementById('loadingText').textContent = message;
}

function hideLoading() {
    const loadingIndicator = document.getElementById('loading');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
        loadingIndicator.textContent = '';
    }
}

// Attach event listener for Download PDF button
document.getElementById('downloadPdfBtn').addEventListener('click', downloadPDF);

// Handle theme toggle button click
themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    themeToggle.textContent = newTheme === 'dark' ? 'Light' : 'Dark';
});

function resetResultsView() {
    ['analysisResults', 'questionsResults', 'modificationResults', 'matchingResults']
        .forEach(id => document.getElementById(id).style.display = 'none');
    resultsContainer.style.display = 'none';
}

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    const allowedTypes = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];

    if (!allowedTypes.includes(file.type)) {
        showError('Please select a PDF or Word document.');
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        showError('File size exceeds 5MB limit.');
        return;
    }

    selectedFile = file;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    fileInfo.classList.add('show');

    optionsCard.style.display = 'block';
    optionsCard.scrollIntoView({ behavior: 'smooth' });

    resetResultsView();
    jobDescCard.style.display = 'none';
    analyzeBtn.style.display = 'none';

    document.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));
    selectedOption = null;

    jobTitleInput.value = '';
    jobDescriptionInput.value = '';

    updateUI();
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.color = 'var(--error-color)';
    errorDiv.style.marginTop = '10px';
    errorDiv.textContent = message;
    uploadArea.appendChild(errorDiv);
    setTimeout(() => errorDiv.remove(), 3000);
}

function checkJobInputs() {
    const jobTitle = jobTitleInput ? jobTitleInput.value.trim() : "";
    const jobDescription = jobDescriptionInput ? jobDescriptionInput.value.trim() : "";
    analyzeBtn.style.display = (jobTitle && jobDescription) ? 'block' : 'none';
}

document.querySelectorAll('.option-card').forEach(card => {
    card.addEventListener('click', function () {
        document.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));
        this.classList.add('selected');
        selectedOption = this.dataset.option;

        resetResultsView();

        const jobTitleField = document.getElementById('jobTitle').parentElement;
        const jobDescriptionField = document.getElementById('jobDescription').parentElement;

        if (selectedOption === 'matching' || selectedOption === 'modification') {
            optionsCard.style.display = 'none';
            jobDescCard.style.display = 'block';
            jobDescCard.scrollIntoView({ behavior: 'smooth' });
            jobTitleField.style.display = 'block';
            jobDescriptionField.style.display = 'block';
            analyzeBtn.style.display = 'none';
            const buttonTextMap = {
                matching: 'Match with Job',
                modification: '✏ Get Suggestions',
            };
            btnText.textContent = buttonTextMap[selectedOption] || 'Process Request';
        } else if (selectedOption === 'analysis' || selectedOption === 'questions') {
            optionsCard.style.display = 'block';
            jobDescCard.style.display = 'none';
            analyzeBtn.style.display = 'none';
            processRequest();
        } else {
            optionsCard.style.display = 'block';
            jobDescCard.style.display = 'none';
            analyzeBtn.style.display = 'none';
        }

        checkJobInputs();
    });
});

jobTitleInput.addEventListener('input', checkJobInputs);
jobDescriptionInput.addEventListener('input', checkJobInputs);

analyzeBtn.addEventListener('click', processRequest);

function deduplicateArray(arr) {
    const seen = new Set();
    return arr.filter(item => {
        const val = `${item.type}|${item.message}`;
        if (seen.has(val)) return false;
        seen.add(val);
        return true;
    });
}


async function processRequest() {
    const formData = new FormData();
    const fileInput = document.getElementById('resumeFile');

    if (!fileInput || fileInput.files.length === 0) {
        showError('Please upload a resume file.');
        return;
    }

    // ✅ Fix: don't redefine selectedOption — use the global one
    formData.append("resume", fileInput.files[0]);
    formData.append("option", selectedOption);  // <-- correct usage

    let jobTitleValue = '';
    let jobDescValue = '';

    if (selectedOption === 'matching' || selectedOption === 'modification') {
        const jobTitleInput = document.getElementById('jobTitle');
        const jobDescInput = document.getElementById('jobDescription');
        jobTitleValue = jobTitleInput ? jobTitleInput.value.trim() : '';
        jobDescValue = jobDescInput ? jobDescInput.value.trim() : '';

        if (!jobTitleValue || !jobDescValue) {
            showError('Please provide a job title and description for matching/modification.');
            return;
        }

        formData.append('job_title', jobTitleValue);
        formData.append('job_description', jobDescValue);
    }

    if (!selectedOption) {
        showError('Please select an analysis option.');
        return;
    }
    formData.append('option', selectedOption);

    try {
        showLoading('Processing request...');

        const response = await fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const newData = await response.json();
        console.log('✅ Full response from /analyze:', newData);

        // Initialize window.report if not already set
        window.report = window.report || {
            analysisResults: null,
            questionsResults: [],
            modificationResults: [],
            matchingResults: []
        };

        // Accumulate results
        if (newData.analysisResults) {
            if (newData.analysisResults.feedback) {
                newData.analysisResults.feedback = deduplicateArray(newData.analysisResults.feedback);
            }
            window.report.analysisResults = newData.analysisResults;
        }

        if (newData.questionsResults && Array.isArray(newData.questionsResults)) {
            // Append only if new questions are not already present
            newData.questionsResults.forEach(newQ => {
                const exists = window.report.questionsResults.some(q => q.question === newQ.question);
                if (!exists) {
                    window.report.questionsResults.push(newQ);
                }
            });
        }



        if (newData.modificationResults) {
            window.report.modificationResults = deduplicateArray(
                window.report.modificationResults.concat(newData.modificationResults || [])
            );
        }

        if (newData.matchingResults) {
            window.report.matchingResults = deduplicateArray(
                window.report.matchingResults.concat(newData.matchingResults || [])
            );
        }

        // Warnings for missing results based on selected option
        if (selectedOption === 'questions' &&
            (!Array.isArray(newData.questionsResults) || newData.questionsResults.length === 0)) {
            console.warn("⚠️ No questionsResults received for option:", selectedOption);
        }
        if (selectedOption === 'matching' &&
            (!Array.isArray(newData.matchingResults) || newData.matchingResults.length === 0)) {
            console.warn("⚠️ No matchingResults received for option:", selectedOption);
        }
        if (selectedOption === 'modification' &&
            (!Array.isArray(newData.modificationResults) || newData.modificationResults.length === 0)) {
            console.warn("⚠️ No modificationResults received for option:", selectedOption);
        }

        console.log('✅ Final window.report:', window.report);
        displayResults(newData);

    } catch (error) {
        console.error('❌ Processing failed:', error);
        showError('Processing failed. Please try again.');
    } finally {
        hideLoading();
    }
}


function displayResults(data) {
    if (data.resumeText) {
        delete data.resumeText;
    }

    resetResultsView();

    const resultsTitle = document.getElementById('resultsTitle');
    console.log('🧭 Selected option is:', selectedOption);

    switch (selectedOption) {
        case 'analysis':
            resultsTitle.textContent = 'Resume Analysis Results';
            displayAnalysisResults(data);
            break;
        case 'questions':
            console.log('📢 Triggering displayQuestionsResults()');
            resultsTitle.textContent = 'Interview Questions';
            displayQuestionsResults(data);
            break;

        case 'modification':
            resultsTitle.textContent = '✏ Improvement Suggestions';
            displayModificationResults(data);
            break;
        case 'matching':
            resultsTitle.textContent = 'Job Matching Results';
            displayMatchingResults(data);
            break;
    }

    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth' });

    document.getElementById('downloadPdfBtn').style.display = 'inline-block';
}

//function displayAnalysisResults(data) {
//    const analysisData = data.analysisResults || {
//        matchScore: 85,
//        skillsFound: 12,
//        recommendation: "Strong",
//        feedback: [
//            { type: "strength", message: "Your technical skills align well with industry standards, particularly in Python and machine learning." },
//            { type: "improvement", message: "Consider adding more specific metrics to quantify your achievements (e.g., 'improved efficiency by 30%')." },
//            { type: "suggestion", message: "Include more recent projects to showcase current technology expertise." }
//        ]
//    };
//
//    document.getElementById('ResumeScore').textContent = analysisData.matchScore + '%';
//    document.getElementById('skillsFound').textContent = analysisData.skillsFound;
//    document.getElementById('recommendation').textContent = analysisData.recommendation;
//
//    const feedbackList = document.getElementById('feedbackList');
//    feedbackList.innerHTML = analysisData.feedback
//        .filter(item => item.message.length < 300 && !item.message.includes("Objective") && !item.message.match(/\bexperience\b.*\d{4}/i))
//        .map(item => `
//            <div class="feedback-item">
//                <strong>${item.type.charAt(0).toUpperCase() + item.type.slice(1)}:</strong> ${item.message}
//            </div>
//        `).join('');
//
//    document.getElementById('analysisResults').style.display = 'block';
//}

function displayAnalysisResults(data) {
    const analysisData = data.analysisResults || {};

    // Basic stats
    document.getElementById('ResumeScore').textContent = (analysisData.matchScore || 0) + '%';
    document.getElementById('skillsFound').textContent = analysisData.skillsFound || 0;
    document.getElementById('recommendation').textContent = analysisData.recommendation || "N/A";

    const feedbackList = document.getElementById('feedbackList');
    feedbackList.innerHTML = "";

    // Use grouped feedback if available
    const displayOrder = ["ATS Optimization", "Insights", "Red Flags", "Strengths", "Suggestions", "Style"];

    if (analysisData.feedbackGrouped) {
        displayOrder.forEach(category => {
            const messages = analysisData.feedbackGrouped[category];
            if (!messages || messages.length === 0) return;

            const heading = document.createElement('h4');
            heading.textContent = category;
            heading.className = 'feedback-category';
            feedbackList.appendChild(heading);

            // ✅ Special handling for Red Flags
            if (category === "Red Flags" && messages.length > 1) {
                const ul = document.createElement('ul');
                ul.className = 'feedback-list';
                messages.forEach(msg => {
                    if (msg && msg.trim()) {
                        const li = document.createElement('li');
                        li.textContent = msg.trim();
                        ul.appendChild(li);
                    }
                });
                feedbackList.appendChild(ul);
            } else {
                // Normal handling for all other categories OR single red flag
                messages.forEach(msg => {
                    if (msg && msg.trim().length > 3 && msg.length < 300) {
                        const itemDiv = document.createElement('div');
                        itemDiv.className = 'feedback-item';
                        itemDiv.textContent = msg.trim();
                        feedbackList.appendChild(itemDiv);
                    }
                });
            }
        });
    }

    // Fallback if grouping not available
    else if (analysisData.feedback) {
        analysisData.feedback.forEach(item => {
            if (item.message && item.message.trim().length > 5) {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'feedback-item';
                itemDiv.textContent = `${item.type}: ${item.message.trim()}`;
                feedbackList.appendChild(itemDiv);
            }
        });
    }

    document.getElementById('analysisResults').style.display = 'block';
}



//function displayQuestionsResults(data) {
//    const questionsData = data.questionsResults || [];
//
//    const questionsList = document.getElementById('questionsList');
//    questionsList.innerHTML = '';
//
//    questionsData.forEach(item => {
//        let questionText = '';
//
//        if (typeof item === 'object' && item.question) {
//            if (typeof item.question === 'string') {
//                questionText = item.question;
//            } else if (typeof item.question === 'object' && item.question.question) {
//                questionText = item.question.question;
//            }
//        }
//
//        const questionDiv = document.createElement('div');
//        questionDiv.className = 'question-item';
//        questionDiv.innerHTML = `
//            <div class="question-category">${item.category || 'AI'}</div>
//            <div>${questionText || '[No question found]'}</div>
//        `;
//        questionsList.appendChild(questionDiv);
//    });
//
//    document.getElementById('questionsResults').style.display = 'block';
//}

// Display generated interview questions
function displayQuestionsResults(data) {
    const questionsData = data.questionsResults || [];

    // Group questions by category
    const grouped = {};
    questionsData.forEach(item => {
        const category = item.category || 'General';
        let questionText = '';

        if (typeof item === 'object' && item.question) {
            if (typeof item.question === 'string') {
                questionText = item.question;
            } else if (typeof item.question === 'object' && item.question.question) {
                questionText = item.question.question;
            }
        } else {
            questionText = String(item);
        }

        if (!grouped[category]) grouped[category] = [];
        grouped[category].push(questionText || '[No question found]');
    });

    const questionsList = document.getElementById('questionsList');
    questionsList.innerHTML = '';

    // Render each category block
    Object.keys(grouped).forEach(cat => {
        const catBlock = document.createElement('div');
        catBlock.className = 'question-category-block';
        catBlock.innerHTML = `<h3 style="margin-top:15px; color:var(--question-category);">${cat} Questions</h3>`;

        grouped[cat].forEach(q => {
            const qDiv = document.createElement('div');
            qDiv.className = 'question-item';
            qDiv.innerHTML = `<div>${q}</div>`;
            catBlock.appendChild(qDiv);
        });

        questionsList.appendChild(catBlock);
    });

    document.getElementById('questionsResults').style.display = 'block';
}

function displayModificationResults(data) {
    const modificationsData = data.modificationResults || [];

    const modificationList = document.getElementById('modificationList');

    modificationList.innerHTML = modificationsData.map(item => {
        let formattedMessage = item.message
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // ✅ Make numbered headings bold (e.g., "1. Something here")
        formattedMessage = formattedMessage.replace(
            /^(\d+\.\s.*)$/m,
            '<strong>$1</strong>'
        );

        // ✅ Also bold any numbered headings inside the text
        formattedMessage = formattedMessage.replace(
            /\n(\d+\.\s.*)/g,
            '\n<strong>$1</strong>'
        );

        // Convert "- " bullets into HTML list items
        if (formattedMessage.includes('- ')) {
            formattedMessage = formattedMessage.replace(/\n- /g, '\n• ');
            const listItems = formattedMessage
                .split('\n')
                .map(line => {
                    if (line.trim().startsWith('•')) {
                        return `<li>${line.trim().slice(1).trim()}</li>`;
                    }
                    return `<p>${line.trim()}</p>`;
                })
                .join('');

            formattedMessage = listItems.replace(/(<p><\/p>)+/g, '');
            formattedMessage = formattedMessage.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
        } else {
            formattedMessage = formattedMessage.replace(/\n/g, '<br>');
        }

        return `
            <div class="modification-item">
                ${formattedMessage}
            </div>
        `;
    }).join('');

    document.getElementById('modificationResults').style.display = 'block';
}


function displayMatchingResults(data) {
    const matchingData = data.matchingResults || [
        { type: "Strong Match", message: "Your Python and machine learning experience directly aligns with the job requirements." },
        { type: "Partial Match", message: "You have some experience with cloud platforms, but AWS certification would strengthen your profile." },
        { type: "Missing Skills", message: "Consider gaining experience with Docker and Kubernetes as mentioned in the job description." },
        { type: "Recommendation", message: "Highlight your project management experience more prominently to match the leadership aspects of this role." }
    ];

    let percentMatch = 0;
    if (data.matchScore) {
        percentMatch = parseFloat(data.matchScore) || 0;
    } else if (matchingData.length > 0) {
        for (const item of matchingData) {
            const scoreMatch = item.message.match(/\d+\.\d+/);
            if (scoreMatch) {
                percentMatch = parseFloat(scoreMatch[0]);
                break;
            }
        }
    }
    percentMatch = percentMatch || 50;
    updateMatchScore(percentMatch);

    const matchingList = document.getElementById('matchingList');
    matchingList.innerHTML = matchingData.map(item => {
        let content = "";

        // Style Recommendation Box
        if (item.type === "Recommendation") {
            let tierClass = "";
            if (item.message.includes("Excellent")) tierClass = "excellent";
            else if (item.message.includes("Good")) tierClass = "good";
            else tierClass = "needs-improvement";

            content = `
                <div class="matching-item recommendation ${tierClass}">
                    <strong>${item.type}:</strong> ${item.message}
                </div>
            `;
        }
        // Style Next Steps as Bullet List
        else if (item.type === "Next Steps") {
            let steps = item.message.split("•").map(s => s.trim()).filter(s => s);
            content = `
                <div class="matching-item next-steps">
                    <strong>${item.type}:</strong>
                    <ul>${steps.map(s => `<li>${s}</li>`).join('')}</ul>
                </div>
            `;
        }
        // Highlight High Priority Missing Skills in Red
        else if (item.type === "High Priority Missing Skills") {
            content = `
                <div class="matching-item high-priority">
                    <strong>${item.type}:</strong> ${item.message}
                </div>
            `;
        }
        // Default Style
        else {
            content = `
                <div class="matching-item">
                    <strong>${item.type}:</strong> ${item.message}
                </div>
            `;
        }

        return content;
    }).join('');

    const matchScoreCard = document.querySelector('.match-score-card');
    if (matchScoreCard) {
        matchScoreCard.style.display = 'block';
    }

    document.getElementById('matchingResults').style.display = 'block';
}

function updateMatchScore(percent) {
    const circle = document.getElementById("progressPath");
    const scoreText = document.getElementById("matchScoreText");

    if (!circle || !scoreText) {
        console.error('Match score elements missing:', {
            progressPath: !!circle,
            matchScoreText: !!scoreText
        });
        return;
    }

    const value = Math.min(100, Math.max(0, percent));
    const dashArray = `${(value * 100) / 100}, 100`;

    circle.setAttribute("stroke-dasharray", dashArray);
    scoreText.textContent = `${Math.round(value)}%`;
    scoreText.classList.add('score-value');
}

function downloadPDF() {
    console.log("Current window.report:", window.report);
    if (!window.report || Object.keys(window.report).every(key => !window.report[key] && (!Array.isArray(window.report[key]) || window.report[key].length === 0))) {
        alert("⚠️ Please run an analysis (e.g., resume analysis, matching, etc.) before downloading.");
        return;
    }

    fetch("/download-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(window.report),
    })
    .then((res) => {
        console.log("Response status:", res.status);
        if (!res.ok) {
            return res.json().then(err => { throw new Error(err.error || 'Failed to fetch PDF'); });
        }
        return res.blob();
    })
    .then((blob) => {
        console.log("Received blob size:", blob.size);
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "resume_analysis.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    })
    .catch((err) => {
        console.error("❌ PDF download failed:", err);
        alert("PDF generation failed. Check console for errors.");
    });
}

const featureInstructionsMap = {
    analysis: [
        "Upload your resume (PDF or DOCX) in the file upload area.",
        "The system will analyze skills, experience, and structure.",
        "Click 'Analyze Resume' Get feedback and scoring instantly."
    ],
    matching: [
        "Upload your resume.",
        "Paste a job description you’re targeting.",
        "This will evaluate fit between your profile and job requirements."
    ],
    modification: [
        "Upload your resume.",
        "Get actionable insights to optimize your resume for ATS systems."
    ],
    questions: [
        "Upload your resume.",
        "Receive tailored interview questions.",
        "Use them to prepare effectively for upcoming interviews."
    ]
};

const cards = document.querySelectorAll('.feature-card');
const list = document.getElementById('featureInstructionContent');

const defaultOption = "analysis";
const defaultInstructions = featureInstructionsMap[defaultOption] || [];
list.innerHTML = defaultInstructions.map(i => `<li>${i}</li>`).join('');

cards.forEach(card => {
    if (card.dataset.option === defaultOption) {
        card.classList.add('selected');
    } else {
        card.classList.remove('selected');
    }
});

cards.forEach(card => {
    card.addEventListener('click', () => {
        cards.forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');

        const option = card.dataset.option;
        const instructions = featureInstructionsMap[option] || ["No instructions available."];
        list.innerHTML = instructions.map(i => `<li>${i}</li>`).join('');

        selectedOption = option;
        resetResultsView();
        updateUI();
    });
});

const featureCards = document.querySelectorAll('.feature-card');
const optionCards = document.querySelectorAll('.option-card');
const optionsCardContainer = document.getElementById('optionsCard');
const uploadInput = document.getElementById('resumeFile');
const featureSections = document.querySelectorAll('.feature-section');

let selectedFeature = 'analysis';
let resumeUploaded = false;

function updateUI() {
    if (!resumeUploaded) {
        optionsCardContainer.style.display = 'none';
        jobDescCard.style.display = 'none';
        analyzeBtn.style.display = 'none';
        return;
    }

    if (selectedFeature === 'matching' || selectedFeature === 'modification') {
        optionsCardContainer.style.display = 'none';
        jobDescCard.style.display = 'block';
        checkJobInputs();
    } else {
        optionsCardContainer.style.display = 'block';
        jobDescCard.style.display = 'none';
        analyzeBtn.style.display = 'none';
    }

    optionCards.forEach(card => {
        card.style.display = card.dataset.option === selectedFeature ? 'flex' : 'none';
    });

    featureSections.forEach(section => {
        section.classList.remove('active');
        if (section.id === selectedFeature + 'Section') {
            section.classList.add('active');
        }
    });
}

featureCards.forEach(card => {
    card.addEventListener('click', () => {
        selectedFeature = card.dataset.option;
        updateUI();
    });
});

uploadInput.addEventListener('change', () => {
    resumeUploaded = true;
    updateUI();
});

document.addEventListener('DOMContentLoaded', initializeTheme);
