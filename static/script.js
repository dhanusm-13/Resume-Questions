////// Constants & DOM Elements
////const uploadArea = document.getElementById('uploadArea');
////const fileInput = document.getElementById('resumeFile');
////const fileInfo = document.getElementById('fileInfo');
////const optionsCard = document.getElementById('optionsCard');
////const jobDescCard = document.getElementById('jobDescCard');
////const analyzeBtn = document.getElementById('analyzeBtn');
////const btnText = document.getElementById('btnText');
////const resultsContainer = document.getElementById('results');
////const loadingIndicator = document.getElementById('loading');
////const loadingText = document.getElementById('loadingText');
////
////let selectedFile = null;
////let selectedOption = null;
////
////// Utility function to reset/hide all result sections
////function resetResultsView() {
////    ['analysisResults', 'questionsResults', 'modificationResults', 'matchingResults']
////        .forEach(id => document.getElementById(id).style.display = 'none');
////    resultsContainer.style.display = 'none';
////}
////
////// 1. File upload handling
////
////uploadArea.addEventListener('dragover', (e) => {
////    e.preventDefault();
////    uploadArea.classList.add('drag-over');
////});
////
////uploadArea.addEventListener('dragleave', () => {
////    uploadArea.classList.remove('drag-over');
////});
////
////uploadArea.addEventListener('drop', (e) => {
////    e.preventDefault();
////    uploadArea.classList.remove('drag-over');
////    const files = e.dataTransfer.files;
////    if (files.length > 0) {
////        handleFileSelect(files[0]);
////    }
////});
////
////fileInput.addEventListener('change', (e) => {
////    if (e.target.files.length > 0) {
////        handleFileSelect(e.target.files[0]);
////    }
////});
////
////function handleFileSelect(file) {
////    const allowedTypes = [
////        'application/pdf',
////        'application/msword',
////        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
////    ];
////
////    if (!allowedTypes.includes(file.type)) {
////        alert('Please select a PDF or Word document.');
////        return;
////    }
////
////    // Optional: Check max file size 2MB
////    if (file.size > 5 * 1024 * 1024) {
////        alert('File size exceeds 5MB limit.');
////        return;
////    }
////
////    selectedFile = file;
////    document.getElementById('fileName').textContent = file.name;
////    document.getElementById('fileSize').textContent = `${(file.size / 1024 / 1024).toFixed(5)} MB`;
////    fileInfo.classList.add('show');
////
////    // Show options card after file upload
////    optionsCard.style.display = 'block';
////    optionsCard.scrollIntoView({ behavior: 'smooth' });
////
////    // Reset results and job description card when a new file is uploaded
////    resetResultsView();
////    jobDescCard.style.display = 'none';
////
////    document.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));
////    selectedOption = null;
////
////    // Reset job description fields
////    document.getElementById('jobTitle').value = '';
////    document.getElementById('jobDescription').value = '';
////}
////
////// 2. Option card selection handling
////
////document.querySelectorAll('.option-card').forEach(card => {
////    card.addEventListener('click', function () {
////        // Remove selected class from all cards
////        document.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));
////
////        // Add selected class to clicked card
////        this.classList.add('selected');
////        selectedOption = this.dataset.option;
////
////        // Reset all results view on option change
////        resetResultsView();
////
////        // Get references to job description fields and the analyze button
////        const jobTitleField = document.getElementById('jobTitle').parentElement;
////        const jobDescriptionField = document.getElementById('jobDescription').parentElement;
////
////        // Show/hide job description card and analyze button based on selected option
////        if (selectedOption === 'matching' || selectedOption === 'modification') {
////            jobDescCard.style.display = 'block';
////            jobDescCard.scrollIntoView({ behavior: 'smooth' });
////
////            jobTitleField.style.display = 'block';
////            jobDescriptionField.style.display = 'block';
////            analyzeBtn.style.display = 'block';
////
////            // Update button text based on option using a map for cleaner code
////            const buttonTextMap = {
////                matching: ' Match with Job',
////                modification: '✏ Get Suggestions',
////            };
////            btnText.textContent = buttonTextMap[selectedOption] || 'Analyze';
////        } else if (selectedOption === 'analysis' || selectedOption === 'questions') {
////            // Hide job description and analyze button
////            jobDescCard.style.display = 'none';
////            analyzeBtn.style.display = 'none';
////
////            // Immediately trigger processing for these options
////            processRequest();
////        } else {
////            // Default fallback: hide job desc and analyze button
////            jobDescCard.style.display = 'none';
////            analyzeBtn.style.display = 'none';
////        }
////    });
////});
////
////// 3. Process request to backend
////
////async function processRequest() {
////    if (!selectedFile) {
////        alert('Please upload your resume first.');
////        return;
////    }
////
////    if (!selectedOption) {
////        alert('Please select a service option.');
////        return;
////    }
////
////    const formData = new FormData();
////    formData.append('resume', selectedFile);
////    formData.append('option', selectedOption);
////
////    let jobTitle = '';
////    let jobDescription = '';
////
////    if (selectedOption === 'matching' || selectedOption === 'modification') {
////        jobTitle = document.getElementById('jobTitle').value.trim();
////        jobDescription = document.getElementById('jobDescription').value.trim();
////
////        if (!jobTitle || !jobDescription) {
////            alert('Please fill in the job title and description.');
////            return;
////        }
////        formData.append('job_title', jobTitle);
////        formData.append('job_description', jobDescription);
////    }
////
////    // Show loading UI
////    loadingIndicator.style.display = 'block';
////    resultsContainer.style.display = 'none';
////    analyzeBtn.disabled = true;
////
////    // Update loading text based on option
////    const loadingTexts = {
////        analysis: 'Analyzing your resume with AI...',
////        questions: 'Generating personalized interview questions...',
////        modification: 'Creating improvement suggestions...',
////        matching: 'Matching your resume with job requirements...',
////    };
////    loadingText.textContent = loadingTexts[selectedOption] || 'Processing...';
////
////    try {
////        const response = await fetch('http://127.0.0.1:5000/analyze', {
////            method: 'POST',
////            body: formData,
////        });
////
////        if (!response.ok) {
////            throw new Error(`HTTP error! status: ${response.status}`);
////        }
////
////        const data = await response.json();
////        console.log('Backend response:', data);
////
////        displayResults(data);
////
////    } catch (error) {
////        console.error('Processing failed:', error);
////        alert('Processing failed. Please try again. Check console for details.');
////    } finally {
////        loadingIndicator.style.display = 'none';
////        analyzeBtn.disabled = false;
////    }
////}
////
////// 4. Display results dispatcher
////
////function displayResults(data) {
////    // 🔒 Ensure resume text is not shown in frontend
////    if (data.resumeText) {
////        delete data.resumeText;
////    }
////
////    resetResultsView();
////
////    const resultsTitle = document.getElementById('resultsTitle');
////
////    switch (selectedOption) {
////        case 'analysis':
////            resultsTitle.textContent = ' Resume Analysis Results';
////            displayAnalysisResults(data);
////            break;
////        case 'questions':
////            resultsTitle.textContent = ' Interview Questions';
////            displayQuestionsResults(data);
////            break;
////        case 'modification':
////            resultsTitle.textContent = '✏ Improvement Suggestions';
////            displayModificationResults(data);
////            break;
////        case 'matching':
////            resultsTitle.textContent = ' Job Matching Results';
////            displayMatchingResults(data);
////            break;
////    }
////
////    resultsContainer.style.display = 'block';
////    resultsContainer.scrollIntoView({ behavior: 'smooth' });
////
////    // Show the download button
////    document.getElementById('downloadPdfBtn').style.display = 'inline-block';
////}
////
////
////
////// 5. Specific result display handlers
////
////function displayAnalysisResults(data) {
////    const analysisData = data.analysisResults || {
////        matchScore: 85,
////        skillsFound: 12,
////        recommendation: "Strong",
////        feedback: [
////            { type: "strength", message: "Your technical skills align well with industry standards, particularly in Python and machine learning." },
////            { type: "improvement", message: "Consider adding more specific metrics to quantify your achievements (e.g., 'improved efficiency by 30%')." },
////            { type: "suggestion", message: "Include more recent projects to showcase current technology expertise." }
////        ]
////    };
////
////    document.getElementById('ResumeScore').textContent = analysisData.matchScore + '%';
////    document.getElementById('skillsFound').textContent = analysisData.skillsFound;
////    document.getElementById('recommendation').textContent = analysisData.recommendation;
////
////    const feedbackList = document.getElementById('feedbackList');
////    feedbackList.innerHTML = analysisData.feedback
////      .filter(item => item.message.length < 300 && !item.message.includes("Objective") && !item.message.match(/\bexperience\b.*\d{4}/i))
////      .map(item => `
////          <div class="feedback-item">
////              <strong>${item.type.charAt(0).toUpperCase() + item.type.slice(1)}:</strong> ${item.message}
////          </div>
////  ` ).join('');
////
////    document.getElementById('analysisResults').style.display = 'block';
////}
////
////function displayQuestionsResults(data) {
////    const questionsData = data.questionsResults || [];
////
////    const questionsList = document.getElementById('questionsList');
////    questionsList.innerHTML = '';
////
////    questionsData.forEach(item => {
////        let questionText = '';
////
////        // 🛠 Check and extract nested text safely
////        if (typeof item === 'object' && item.question) {
////            if (typeof item.question === 'string') {
////                questionText = item.question;
////            } else if (typeof item.question === 'object' && item.question.question) {
////                questionText = item.question.question;
////            }
////        }
////
////        const questionDiv = document.createElement('div');
////        questionDiv.className = 'question-item';
////        questionDiv.innerHTML = `
////            <div class="question-category">${item.category || 'AI'}</div>
////            <div>${questionText || '[No question found]'}</div>
////        `;
////        questionsList.appendChild(questionDiv);
////    });
////
////    document.getElementById('questionsResults').style.display = 'block';
////}
////
////function displayModificationResults(data) {
////    const modificationsData = data.modificationResults || [
////        { type: "Format", message: "Use consistent bullet points and ensure proper spacing between sections." },
////        { type: "Content", message: "Add quantifiable achievements (e.g., 'Increased sales by 25%' instead of 'Increased sales')." },
////        { type: "Keywords", message: "Include more industry-specific keywords like 'Agile', 'Scrum', 'CI/CD' for better ATS compatibility." },
////        { type: "Structure", message: "Consider adding a 'Key Achievements' section to highlight your most significant accomplishments." },
////        { type: "Skills", message: "Group technical skills by category (Programming Languages, Frameworks, Tools) for better readability." }
////    ];
////
////    const modificationList = document.getElementById('modificationList');
////    modificationList.innerHTML = modificationsData.map(item => `
////        <div class="modification-item">
////            <strong>${item.type}:</strong> ${item.message}
////        </div>
////    `).join('');
////
////    document.getElementById('modificationResults').style.display = 'block';
////}
////
////function displayMatchingResults(data) {
////    const matchingData = data.matchingResults || [
////        { type: "Strong Match", message: "Your Python and machine learning experience directly aligns with the job requirements." },
////        { type: "Partial Match", message: "You have some experience with cloud platforms, but AWS certification would strengthen your profile." },
////        { type: "Missing Skills", message: "Consider gaining experience with Docker and Kubernetes as mentioned in the job description." },
////        { type: "Recommendation", message: "Highlight your project management experience more prominently to match the leadership aspects of this role." }
////    ];
////
////    // Extract and update match score if possible
////    if (matchingData.length > 0) {
////        const firstMessage = matchingData[0].message;
////        const matchScoreMatch = firstMessage.match(/[\d.]+/);
////        const percentMatch = matchScoreMatch ? parseFloat(matchScoreMatch[0]) : 0;
////        updateMatchScore(percentMatch);
////    } else {
////        updateMatchScore(0);
////    }
////
////    const matchingList = document.getElementById('matchingList');
////    matchingList.innerHTML = matchingData.map(item => `
////        <div class="matching-item">
////            <strong>${item.type}:</strong> ${item.message}
////        </div>
////    `).join('');
////
////    document.getElementById('matchingResults').style.display = 'block';
////}
////
////// Utility to update the match score progress bar or display
//////function updateMatchScore(percent) {
////    //const progressBar = document.getElementById('matchProgressBar');
////    //const progressText = document.getElementById('matchProgressText');
////
////    //if (progressBar && progressText) {
////        //progressBar.style.width = percent + '%';
////        //progressText.textContent = `${percent}%`;
////    //}
////function updateMatchScore(percent) {
////  const circle = document.getElementById("progressPath");
////  const scoreText = document.getElementById("matchScoreText");
////
////  if (!circle || !scoreText) return;
////
////  const value = Math.min(100, Math.max(0, percent));
////  const dashArray = `${(value * 100) / 100}, 100`;
////
////  circle.setAttribute("stroke-dasharray", dashArray);
////  scoreText.textContent = `${Math.round(value)}%`;
////}
////
//////to download the report
////function downloadPDF() {
////  const analysis = {
////    matchScore: document.getElementById('ResumeScore').textContent,
////    skillsFound: document.getElementById('skillsFound').textContent,
////    recommendation: document.getElementById('recommendation').textContent,
////    feedback: []
////  };
////
////  document.querySelectorAll('#feedbackList .feedback-item').forEach(item => {
////    const parts = item.textContent.split(':');
////    analysis.feedback.push({
////      type: parts[0].trim(),
////      message: parts.slice(1).join(':').trim()
////    });
////  });
////
////  fetch('/download-pdf', {
////    method: 'POST',
////    headers: { 'Content-Type': 'application/json' },
////    body: JSON.stringify(analysis)
////  })
////  .then(res => res.blob())
////  .then(blob => {
////    const url = window.URL.createObjectURL(blob);
////    const a = document.createElement('a');
////    a.href = url;
////    a.download = "resume_analysis.pdf";
////    document.body.appendChild(a);
////    a.click();
////    a.remove();
////  });
////}
////
//
//// Initialize DOM element references
//const uploadArea = document.getElementById('uploadArea');
//const fileInput = document.getElementById('resumeFile');
//const fileInfo = document.getElementById('fileInfo');
//const optionsCard = document.getElementById('optionsCard');
//const jobDescCard = document.getElementById('jobDescCard');
//const analyzeBtn = document.getElementById('analyzeBtn');
//const btnText = document.getElementById('btnText');
//const resultsContainer = document.getElementById('results');
//const loadingIndicator = document.getElementById('loading');
//const loadingText = document.getElementById('loadingText');
//const jobTitleInput = document.getElementById('jobTitle');
//const jobDescriptionInput = document.getElementById('jobDescription');
//
//let selectedFile = null;
//let selectedOption = null;
//
//
//// Initialize theme based on localStorage or system preference
//function initializeTheme() {
//    const savedTheme = localStorage.getItem('theme');
//    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
//    const defaultTheme = savedTheme || (prefersDark ? 'dark' : 'light');
//    document.documentElement.setAttribute('data-theme', defaultTheme);
//    themeToggle.textContent = defaultTheme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme';
//}
//
//// Handle theme toggle button click
//themeToggle.addEventListener('click', () => {
//    const currentTheme = document.documentElement.getAttribute('data-theme');
//    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
//    document.documentElement.setAttribute('data-theme', newTheme);
//    localStorage.setItem('theme', newTheme);
//    themeToggle.textContent = newTheme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme';
//});
//
//// Hide all result sections to reset the view
//function resetResultsView() {
//    ['analysisResults', 'questionsResults', 'modificationResults', 'matchingResults']
//        .forEach(id => document.getElementById(id).style.display = 'none');
//    resultsContainer.style.display = 'none';
//}
//
//// Handle drag-over event for resume upload area
//uploadArea.addEventListener('dragover', (e) => {
//    e.preventDefault();
//    uploadArea.classList.add('drag-over');
//});
//
//// Handle drag-leave event for resume upload area
//uploadArea.addEventListener('dragleave', () => {
//    uploadArea.classList.remove('drag-over');
//});
//
//// Handle file drop event for resume upload
//uploadArea.addEventListener('drop', (e) => {
//    e.preventDefault();
//    uploadArea.classList.remove('drag-over');
//    const files = e.dataTransfer.files;
//    if (files.length > 0) {
//        handleFileSelect(files[0]);
//    }
//});
//
//// Handle file selection via file input
//fileInput.addEventListener('change', (e) => {
//    if (e.target.files.length > 0) {
//        handleFileSelect(e.target.files[0]);
//    }
//});
//
//// Process selected resume file and update UI
//function handleFileSelect(file) {
//    const allowedTypes = [
//        'application/pdf',
//        'application/msword',
//        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
//    ];
//
//    if (!allowedTypes.includes(file.type)) {
//        showError('Please select a PDF or Word document.');
//        return;
//    }
//
//    if (file.size > 5 * 1024 * 1024) {
//        showError('File size exceeds 5MB limit.');
//        return;
//    }
//
//    selectedFile = file;
//    document.getElementById('fileName').textContent = file.name;
//    document.getElementById('fileSize').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
//    fileInfo.classList.add('show');
//
//    optionsCard.style.display = 'block';
//    optionsCard.scrollIntoView({ behavior: 'smooth' });
//
//    resetResultsView();
//    jobDescCard.style.display = 'none';
//    analyzeBtn.style.display = 'none';
//
//    document.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));
//    selectedOption = null;
//
//    jobTitleInput.value = '';
//    jobDescriptionInput.value = '';
//
//    updateUI();
//}
//
//// Display error messages in the upload area
//function showError(message) {
//    const errorDiv = document.createElement('div');
//    errorDiv.className = 'error-message';
//    errorDiv.style.color = '#ff4444';
//    errorDiv.style.marginTop = '10px';
//    errorDiv.textContent = message;
//    uploadArea.appendChild(errorDiv);
//    setTimeout(() => errorDiv.remove(), 3000);
//}
//
//// Check if job title and description are filled to show the Process Request button
//function checkJobInputs() {
//    const jobTitle = jobTitleInput.value.trim();
//    const jobDescription = jobDescriptionInput.value.trim();
//    analyzeBtn.style.display = (jobTitle && jobDescription) ? 'block' : 'none';
//}
//
//// Handle option card clicks to select a service
//document.querySelectorAll('.option-card').forEach(card => {
//    card.addEventListener('click', function () {
//        document.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));
//        this.classList.add('selected');
//        selectedOption = this.dataset.option;
//
//        resetResultsView();
//
//        const jobTitleField = document.getElementById('jobTitle').parentElement;
//        const jobDescriptionField = document.getElementById('jobDescription').parentElement;
//
//        if (selectedOption === 'matching' || selectedOption === 'modification') {
//            optionsCard.style.display = 'none';
//            jobDescCard.style.display = 'block';
//            jobDescCard.scrollIntoView({ behavior: 'smooth' });
//            jobTitleField.style.display = 'block';
//            jobDescriptionField.style.display = 'block';
//            analyzeBtn.style.display = 'none';
//            const buttonTextMap = {
//                matching: 'Match with Job',
//                modification: '✏ Get Suggestions',
//            };
//            btnText.textContent = buttonTextMap[selectedOption] || 'Process Request';
//        } else if (selectedOption === 'analysis' || selectedOption === 'questions') {
//            optionsCard.style.display = 'block';
//            jobDescCard.style.display = 'none';
//            analyzeBtn.style.display = 'none';
//            processRequest();
//        } else {
//            optionsCard.style.display = 'block';
//            jobDescCard.style.display = 'none';
//            analyzeBtn.style.display = 'none';
//        }
//
//        checkJobInputs();
//    });
//});
//
//// Monitor job title input for changes
//jobTitleInput.addEventListener('input', checkJobInputs);
//
//// Monitor job description input for changes
//jobDescriptionInput.addEventListener('input', checkJobInputs);
//
//// Handle Process Request button click to initiate analysis
//analyzeBtn.addEventListener('click', processRequest);
//
//// Send resume and selected option to backend for processing
//async function processRequest() {
//    if (!selectedFile) {
//        showError('Please upload your resume first.');
//        return;
//    }
//
//    if (!selectedOption) {
//        showError('Please select a service option.');
//        return;
//    }
//
//    const formData = new FormData();
//    formData.append('resume', selectedFile);
//    formData.append('option', selectedOption);
//
//    let jobTitle = '';
//    let jobDescription = '';
//
//    if (selectedOption === 'matching' || selectedOption === 'modification') {
//        jobTitle = document.getElementById('jobTitle').value.trim();
//        jobDescription = document.getElementById('jobDescription').value.trim();
//
//        if (!jobTitle || !jobDescription) {
//            showError('Please fill in the job title and description.');
//            return;
//        }
//        formData.append('job_title', jobTitle);
//        formData.append('job_description', jobDescription);
//    }
//
//    loadingIndicator.style.display = 'block';
//    resultsContainer.style.display = 'none';
//    analyzeBtn.disabled = true;
//
//    const loadingTexts = {
//        analysis: 'Analyzing your resume with AI...',
//        questions: 'Generating personalized interview questions...',
//        modification: 'Creating improvement suggestions...',
//        matching: 'Matching your resume with job requirements...',
//    };
//    loadingText.textContent = loadingTexts[selectedOption] || 'Processing...';
//
//    try {
//        const response = await fetch('http://127.0.0.1:5000/analyze', {
//            method: 'POST',
//            body: formData,
//        });
//
//        if (!response.ok) {
//            throw new Error(`HTTP error! status: ${response.status}`);
//        }
//
//        const data = await response.json();
//        console.log('Backend response:', data);
//
//        displayResults(data);
//
//    } catch (error) {
//        console.error('Processing failed:', error);
//        showError('Processing failed. Please try again.');
//    } finally {
//        loadingIndicator.style.display = 'none';
//        analyzeBtn.disabled = false;
//    }
//}
//
//// Display results based on the selected option
//function displayResults(data) {
//    if (data.resumeText) {
//        delete data.resumeText;
//    }
//
//    resetResultsView();
//
//    const resultsTitle = document.getElementById('resultsTitle');
//
//    switch (selectedOption) {
//        case 'analysis':
//            resultsTitle.textContent = 'Resume Analysis Results';
//            displayAnalysisResults(data);
//            break;
//        case 'questions':
//            resultsTitle.textContent = 'Interview Questions';
//            displayQuestionsResults(data);
//            break;
//        case 'modification':
//            resultsTitle.textContent = '✏ Improvement Suggestions';
//            displayModificationResults(data);
//            break;
//        case 'matching':
//            resultsTitle.textContent = 'Job Matching Results';
//            displayMatchingResults(data);
//            break;
//    }
//
//    resultsContainer.style.display = 'block';
//    resultsContainer.scrollIntoView({ behavior: 'smooth' });
//
//    document.getElementById('downloadPdfBtn').style.display = 'inline-block';
//}
//
//// Display resume analysis results
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
//
//// Display generated interview questions
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
//
//// Display resume improvement suggestions
//function displayModificationResults(data) {
//    const modificationsData = data.modificationResults || [
//        { type: "Format", message: "Use consistent bullet points and ensure proper spacing between sections." },
//        { type: "Content", message: "Add quantifiable achievements (e.g., 'Increased sales by 25%' instead of 'Increased sales')." },
//        { type: "Keywords", message: "Include more industry-specific keywords like 'Agile', 'Scrum', 'CI/CD' for better ATS compatibility." },
//        { type: "Structure", message: "Consider adding a 'Key Achievements' section to highlight your most significant accomplishments." },
//        { type: "Skills", message: "Group technical skills by category (Programming Languages, Frameworks, Tools) for better readability." }
//    ];
//
//    const modificationList = document.getElementById('modificationList');
//    modificationList.innerHTML = modificationsData.map(item => `
//        <div class="modification-item">
//            <strong>${item.type}:</strong> ${item.message}
//        </div>
//    `).join('');
//
//    document.getElementById('modificationResults').style.display = 'block';
//}
//
//// Display job matching analysis results
//function displayMatchingResults(data) {
//    const matchingData = data.matchingResults || [
//        { type: "Strong Match", message: "Your Python and machine learning experience directly aligns with the job requirements." },
//        { type: "Partial Match", message: "You have some experience with cloud platforms, but AWS certification would strengthen your profile." },
//        { type: "Missing Skills", message: "Consider gaining experience with Docker and Kubernetes as mentioned in the job description." },
//        { type: "Recommendation", message: "Highlight your project management experience more prominently to match the leadership aspects of this role." }
//    ];
//
//    if (matchingData.length > 0) {
//        const firstMessage = matchingData[0].message;
//        const matchScoreMatch = firstMessage.match(/[\d.]+/);
//        const percentMatch = matchScoreMatch ? parseFloat(matchScoreMatch[0]) : 0;
//        updateMatchScore(percentMatch);
//    } else {
//        updateMatchScore(0);
//    }
//
//    const matchingList = document.getElementById('matchingList');
//    matchingList.innerHTML = matchingData.map(item => `
//        <div class="matching-item">
//            <strong>${item.type}:</strong> ${item.message}
//        </div>
//    `).join('');
//
//    document.getElementById('matchingResults').style.display = 'block';
//}
//
//// Update the circular progress bar for match score
//function updateMatchScore(percent) {
//    const circle = document.getElementById("progressPath");
//    const scoreText = document.getElementById("matchScoreText");
//
//    if (!circle || !scoreText) return;
//
//    const value = Math.min(100, Math.max(0, percent));
//    const dashArray = `${(value * 100) / 100}, 100`;
//
//    circle.setAttribute("stroke-dasharray", dashArray);
//    scoreText.textContent = `${Math.round(value)}%`;
//}
//
//// Generate and download PDF report of results
//function downloadPDF() {
//    const analysis = {
//        matchScore: document.getElementById('ResumeScore').textContent,
//        skillsFound: document.getElementById('skillsFound').textContent,
//        recommendation: document.getElementById('recommendation').textContent,
//        feedback: []
//    };
//
//    document.querySelectorAll('#feedbackList .feedback-item').forEach(item => {
//        const parts = item.textContent.split(':');
//        analysis.feedback.push({
//            type: parts[0].trim(),
//            message: parts.slice(1).join(':').trim()
//        });
//    });
//
//    fetch('/download-pdf', {
//        method: 'POST',
//        headers: { 'Content-Type': 'application/json' },
//        body: JSON.stringify(analysis)
//    })
//    .then(res => res.blob())
//    .then(blob => {
//        const url = window.URL.createObjectURL(blob);
//        const a = document.createElement('a');
//        a.href = url;
//        a.download = "resume_analysis.pdf";
//        document.body.appendChild(a);
//        a.click();
//        a.remove();
//    });
//}
//
//// Define instructions for each feature
//const featureInstructionsMap = {
//    analysis: [
//        "Upload your resume (PDF or DOCX) in the file upload area.",
//        "The system will analyze skills, experience, and structure.",
//        "Click 'Analyze Resume' Get feedback and scoring instantly."
//    ],
//    matching: [
//        "Upload your resume.",
//        "Paste a job description you’re targeting.",
//        "This will evaluate fit between your profile and job requirements."
//    ],
//    modification: [
//        "Upload your resume.",
//        "Get actionable insights to optimize your resume for ATS systems."
//    ],
//    questions: [
//        "Upload your resume.",
//        "Receive tailored interview questions.",
//        "Use them to prepare effectively for upcoming interviews."
//    ]
//};
//
//// Initialize feature instructions and default selection
//const cards = document.querySelectorAll('.feature-card');
//const list = document.getElementById('featureInstructionContent');
//
//const defaultOption = "analysis";
//const defaultInstructions = featureInstructionsMap[defaultOption] || [];
//list.innerHTML = defaultInstructions.map(i => `<li>${i}</li>`).join('');
//
//cards.forEach(card => {
//    if (card.dataset.option === defaultOption) {
//        card.classList.add('selected');
//    } else {
//        card.classList.remove('selected');
//    }
//});
//
//// Handle feature card clicks to update instructions and UI
//cards.forEach(card => {
//    card.addEventListener('click', () => {
//        cards.forEach(c => c.classList.remove('selected'));
//        card.classList.add('selected');
//
//        const option = card.dataset.option;
//        const instructions = featureInstructionsMap[option] || ["No instructions available."];
//        list.innerHTML = instructions.map(i => `<li>${i}</li>`).join('');
//
//        selectedOption = option;
//        resetResultsView();
//        updateUI();
//    });
//});
//
//// Initialize DOM references for feature cards and UI state
//const featureCards = document.querySelectorAll('.feature-card');
//const optionCards = document.querySelectorAll('.option-card');
//const optionsCardContainer = document.getElementById('optionsCard');
//const uploadInput = document.getElementById('resumeFile');
//const featureSections = document.querySelectorAll('.feature-section');
//
//let selectedFeature = 'analysis';
//let resumeUploaded = false;
//
//// Update UI based on selected feature and upload state
//function updateUI() {
//    if (!resumeUploaded) {
//        optionsCardContainer.style.display = 'none';
//        jobDescCard.style.display = 'none';
//        analyzeBtn.style.display = 'none';
//        return;
//    }
//
//    if (selectedFeature === 'matching' || selectedFeature === 'modification') {
//        optionsCardContainer.style.display = 'none';
//        jobDescCard.style.display = 'block';
//        checkJobInputs();
//    } else {
//        optionsCardContainer.style.display = 'block';
//        jobDescCard.style.display = 'none';
//        analyzeBtn.style.display = 'none';
//    }
//
//    optionCards.forEach(card => {
//        card.style.display = card.dataset.option === selectedFeature ? 'flex' : 'none';
//    });
//
//    featureSections.forEach(section => {
//        section.classList.remove('active');
//        if (section.id === selectedFeature + 'Section') {
//            section.classList.add('active');
//        }
//    });
//}
//
//// Handle feature card clicks to update selected feature
//featureCards.forEach(card => {
//    card.addEventListener('click', () => {
//        selectedFeature = card.dataset.option;
//        updateUI();
//    });
//});
//
//// Handle resume upload to update UI state
//uploadInput.addEventListener('change', () => {
//    resumeUploaded = true;
//    updateUI();
//});

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

// Initialize theme based on localStorage or system preference
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const defaultTheme = savedTheme || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', defaultTheme);
    themeToggle.textContent = defaultTheme === 'dark' ? 'Light' : 'Dark';
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

// Hide all result sections to reset the view
function resetResultsView() {
    ['analysisResults', 'questionsResults', 'modificationResults', 'matchingResults']
        .forEach(id => document.getElementById(id).style.display = 'none');
    resultsContainer.style.display = 'none';
}

// Handle drag-over event for resume upload area
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

// Handle drag-leave event for resume upload area
uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

// Handle file drop event for resume upload
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// Handle file selection via file input
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Process selected resume file and update UI
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

// Display error messages in the upload area
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.color = 'var(--error-color)';
    errorDiv.style.marginTop = '10px';
    errorDiv.textContent = message;
    uploadArea.appendChild(errorDiv);
    setTimeout(() => errorDiv.remove(), 3000);
}

// Check if job title and description are filled to show the Process Request button
function checkJobInputs() {
    const jobTitle = jobTitleInput.value.trim();
    const jobDescription = jobDescriptionInput.value.trim();
    analyzeBtn.style.display = (jobTitle && jobDescription) ? 'block' : 'none';
}

// Handle option card clicks to select a service
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

// Monitor job title input for changes
jobTitleInput.addEventListener('input', checkJobInputs);

// Monitor job description input for changes
jobDescriptionInput.addEventListener('input', checkJobInputs);

// Handle Process Request button click to initiate analysis
analyzeBtn.addEventListener('click', processRequest);

// Send resume and selected option to backend for processing
async function processRequest() {
    if (!selectedFile) {
        showError('Please upload your resume first.');
        return;
    }

    if (!selectedOption) {
        showError('Please select a service option.');
        return;
    }

    const formData = new FormData();
    formData.append('resume', selectedFile);
    formData.append('option', selectedOption);

    let jobTitle = '';
    let jobDescription = '';

    if (selectedOption === 'matching' || selectedOption === 'modification') {
        jobTitle = document.getElementById('jobTitle').value.trim();
        jobDescription = document.getElementById('jobDescription').value.trim();

        if (!jobTitle || !jobDescription) {
            showError('Please fill in the job title and description.');
            return;
        }
        formData.append('job_title', jobTitle);
        formData.append('job_description', jobDescription);
    }

    loadingIndicator.style.display = 'block';
    resultsContainer.style.display = 'none';
    analyzeBtn.disabled = true;

    const loadingTexts = {
        analysis: 'Analyzing your resume with AI...',
        questions: 'Generating personalized interview questions...',
        modification: 'Creating improvement suggestions...',
        matching: 'Matching your resume with job requirements...',
    };
    loadingText.textContent = loadingTexts[selectedOption] || 'Processing...';

    try {
        const response = await fetch('http://127.0.0.1:5000/analyze', {
        /* fetch('http://192.168.1.5:5000/analyze', { */
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Backend response:', data);

        displayResults(data);

    } catch (error) {
        console.error('Processing failed:', error);
        showError('Processing failed. Please try again.');
    } finally {
        loadingIndicator.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display results based on the selected option
function displayResults(data) {
    if (data.resumeText) {
        delete data.resumeText;
    }

    resetResultsView();

    const resultsTitle = document.getElementById('resultsTitle');

    switch (selectedOption) {
        case 'analysis':
            resultsTitle.textContent = 'Resume Analysis Results';
            displayAnalysisResults(data);
            break;
        case 'questions':
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

// Display resume analysis results
function displayAnalysisResults(data) {
    const analysisData = data.analysisResults || {
        matchScore: 85,
        skillsFound: 12,
        recommendation: "Strong",
        feedback: [
            { type: "strength", message: "Your technical skills align well with industry standards, particularly in Python and machine learning." },
            { type: "improvement", message: "Consider adding more specific metrics to quantify your achievements (e.g., 'improved efficiency by 30%')." },
            { type: "suggestion", message: "Include more recent projects to showcase current technology expertise." }
        ]
    };

    document.getElementById('ResumeScore').textContent = analysisData.matchScore + '%';
    document.getElementById('skillsFound').textContent = analysisData.skillsFound;
    document.getElementById('recommendation').textContent = analysisData.recommendation;

    const feedbackList = document.getElementById('feedbackList');
    feedbackList.innerHTML = analysisData.feedback
        .filter(item => item.message.length < 300 && !item.message.includes("Objective") && !item.message.match(/\bexperience\b.*\d{4}/i))
        .map(item => `
            <div class="feedback-item">
                <strong>${item.type.charAt(0).toUpperCase() + item.type.slice(1)}:</strong> ${item.message}
            </div>
        `).join('');

    document.getElementById('analysisResults').style.display = 'block';
}

// Display generated interview questions
function displayQuestionsResults(data) {
    const questionsData = data.questionsResults || [];

    const questionsList = document.getElementById('questionsList');
    questionsList.innerHTML = '';

    questionsData.forEach(item => {
        let questionText = '';

        if (typeof item === 'object' && item.question) {
            if (typeof item.question === 'string') {
                questionText = item.question;
            } else if (typeof item.question === 'object' && item.question.question) {
                questionText = item.question.question;
            }
        }

        const questionDiv = document.createElement('div');
        questionDiv.className = 'question-item';
        questionDiv.innerHTML = `
            <div class="question-category">${item.category || 'AI'}</div>
            <div>${questionText || '[No question found]'}</div>
        `;
        questionsList.appendChild(questionDiv);
    });

    document.getElementById('questionsResults').style.display = 'block';
}

// Display resume improvement suggestions
function displayModificationResults(data) {
    const modificationsData = data.modificationResults || [
        { type: "Format", message: "Use consistent bullet points and ensure proper spacing between sections." },
        { type: "Content", message: "Add quantifiable achievements (e.g., 'Increased sales by 25%' instead of 'Increased sales')." },
        { type: "Keywords", message: "Include more industry-specific keywords like 'Agile', 'Scrum', 'CI/CD' for better ATS compatibility." },
        { type: "Structure", message: "Consider adding a 'Key Achievements' section to highlight your most significant accomplishments." },
        { type: "Skills", message: "Group technical skills by category (Programming Languages, Frameworks, Tools) for better readability." }
    ];

    const modificationList = document.getElementById('modificationList');
    modificationList.innerHTML = modificationsData.map(item => `
        <div class="modification-item">
            <strong>${item.type}:</strong> ${item.message}
        </div>
    `).join('');

    document.getElementById('modificationResults').style.display = 'block';
}

// Display job matching analysis results
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
        // Search all messages for a numeric score
        for (const item of matchingData) {
            const scoreMatch = item.message.match(/[\d.]+/);
            if (scoreMatch) {
                percentMatch = parseFloat(scoreMatch[0]);
                break;
            }
        }
    }
    // Fallback to a default score if none found
    percentMatch = percentMatch || 50; // Default to 50% if no score is found
    updateMatchScore(percentMatch);

    const matchingList = document.getElementById('matchingList');
    matchingList.innerHTML = matchingData.map(item => `
        <div class="matching-item">
            <strong>${item.type}:</strong> ${item.message}
        </div>
    `).join('');

    // Ensure match score is displayed in the match-score-card
    const matchScoreCard = document.querySelector('.match-score-card');
    if (matchScoreCard) {
        matchScoreCard.style.display = 'block';
    }

    document.getElementById('matchingResults').style.display = 'block';
}

// Update the circular progress bar for match score
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
    scoreText.classList.add('score-value'); // Ensure match score is captured in PDF
}
// Generate and download PDF report of all analysis results
function downloadPDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    let yOffset = 10;

    // Add title
    doc.setFontSize(16);
    doc.setTextColor(0, 0, 0);
    doc.text('Resume Analysis Report', 10, yOffset);
    yOffset += 10;

    // Helper function to add section content
    function addSection(selector, title) {
        const section = document.querySelector(selector);
        if (section) {
            doc.setFontSize(12);
            doc.setFont('helvetica', 'bold');
            doc.text(title, 10, yOffset);
            yOffset += 7;

            const elements = section.querySelectorAll('p, li, .score-value, .score-label, .question-category, .feedback-item, .modification-item, .matching-item');
            elements.forEach((el) => {
                let text = el.textContent.trim();
                if (text) {
                    if (el.classList.contains('score-value')) {
                        doc.setFontSize(14);
                        doc.setFont('helvetica', 'bold');
                    } else if (el.classList.contains('question-category')) {
                        doc.setFontSize(10);
                        doc.setFont('helvetica', 'bold');
                    } else if (el.classList.contains('feedback-item') || el.classList.contains('modification-item') || el.classList.contains('matching-item')) {
                        doc.setFontSize(10);
                        doc.setFont('helvetica', 'normal');
                        const parts = text.split(':');
                        if (parts.length > 1) {
                            doc.setFont('helvetica', 'bold');
                            doc.text(parts[0].trim() + ':', 10, yOffset);
                            doc.setFont('helvetica', 'normal');
                            const messageLines = doc.splitTextToSize(parts.slice(1).join(':').trim(), 180);
                            doc.text(messageLines, 15, yOffset);
                            yOffset += messageLines.length * 7;
                        } else {
                            const lines = doc.splitTextToSize(text, 180);
                            doc.text(lines, 10, yOffset);
                            yOffset += lines.length * 7;
                        }
                        return;
                    } else {
                        doc.setFontSize(10);
                        doc.setFont('helvetica', 'normal');
                    }
                    const lines = doc.splitTextToSize(text, 180);
                    doc.text(lines, 10, yOffset);
                    yOffset += lines.length * 7;
                }
            });
            yOffset += 5; // Space between sections
        }
    }

    // Add all sections
    addSection('.score-section', 'Scores');
    addSection('.questions-section', 'Interview Questions');
    addSection('.feedback-section', 'Feedback');
    addSection('.modification-section', 'Improvement Suggestions');
    addSection('.matching-section', 'Job Matching Results');

    // Save the PDF
    doc.save('resume_analysis.pdf');
}

// Define instructions for each feature
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

// Initialize feature instructions and default selection
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

// Handle feature card clicks to update instructions and UI
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

// Initialize DOM references for feature cards and UI state
const featureCards = document.querySelectorAll('.feature-card');
const optionCards = document.querySelectorAll('.option-card');
const optionsCardContainer = document.getElementById('optionsCard');
const uploadInput = document.getElementById('resumeFile');
const featureSections = document.querySelectorAll('.feature-section');

let selectedFeature = 'analysis';
let resumeUploaded = false;

// Update UI based on selected feature and upload state
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

// Handle feature card clicks to update selected feature
featureCards.forEach(card => {
    card.addEventListener('click', () => {
        selectedFeature = card.dataset.option;
        updateUI();
    });
});

// Handle resume upload to update UI state
uploadInput.addEventListener('change', () => {
    resumeUploaded = true;
    updateUI();
});

document.addEventListener('DOMContentLoaded', initializeTheme);