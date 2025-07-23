# import re
#
#
# def extract_course_names(education_text):
#     lines = education_text.split("\n")
#     course_names = []
#
#     course_patterns = [
#         # Exact matches first
#         r"(master\s+of\s+computer\s+application)",
#         r"(bachelor\s+of\s+computer\s+application)",
#
#         # General patterns
#         r"(master\s+of\s+[^-@\n]+)",
#         r"(bachelor\s+of\s+[^-@\n]+)",
#         r"(b\.?e\.?\s*(?:in)?\s*[^-@\n]+)",
#         r"(btech\s*(?:in)?\s*[^-@\n]+)",
#         r"(mtech\s*(?:in)?\s*[^-@\n]+)",
#         r"(msc\s*(?:in)?\s*[^-@\n]+)",
#         r"(bsc\s*(?:in)?\s*[^-@\n]+)",
#         r"(mca\s*(?:in)?\s*[^-@\n]+)",
#
#         # Standalone abbreviations
#         r"\b(mca)\b",
#         r"\b(bca)\b",
#         r"\b(pgdm)\b",
#         r"\b(mba)\b",
#         r"\b(phd)\b"
#     ]
#
#     compiled_patterns = [re.compile(p, re.IGNORECASE) for p in course_patterns]
#
#     for line in lines:
#         cleaned_line = line.strip()
#         if not cleaned_line:
#             continue
#
#         for pattern in compiled_patterns:
#             match = pattern.search(cleaned_line)
#             if match:
#                 course_candidate = match.group(0).strip()
#
#                 # Clean up any trailing special characters
#                 course_candidate = re.sub(r'[\s,.-]+$', '', course_candidate)
#
#                 # Capitalize properly
#                 final_course_name = ' '.join(word.capitalize() for word in course_candidate.split())
#
#                 if 4 < len(final_course_name) < 100 and final_course_name not in course_names:
#                     course_names.append(final_course_name)
#                     break
#
#     return course_names[:2]
#
#
# education_text = """
# Acharya Institute Of Technology – MCA
# 2023-2025
# 8.1 – SGPA(1st sem)
#
# Ramaiah Institute of Business Studies, Bangalore – BCA
# October 2020 – September 2023
# 7.83 - S.G.P.A
#
# ST. Theresa PU College, Bangalore  - PUC
# April 2019 - May 2020 (PCMB)
# 64.33%
#
# TRIVENI MEMORIAL HIGH SCHOOL, Bangalore - SSLC
# May 2017 - April 2018
# 74.72%
# """
#
# courses = extract_course_names(education_text)
# print(courses)

# import re
#
# def split_into_sections(text):
#     section_titles = [
#         "summary", "objective", "skills", "experience", "education",
#         "projects", "certifications", "certification", "certificates",
#         "awards", "publications", "achievements"
#     ]
#     section_aliases = {
#         "certification": "certifications",
#         "certificates": "certifications",
#         "certs": "certifications"
#     }
#
#     sections = {}
#     current_section = "general"
#     sections[current_section] = []
#
#     lines = text.splitlines()
#     for line in lines:
#         stripped = line.strip()
#         lower = stripped.lower()
#         normalized = re.sub(r'[^a-z]', '', lower)
#
#         matched = False
#         for title in section_titles:
#             norm_title = re.sub(r'[^a-z]', '', title.lower())
#             if normalized == norm_title or normalized.startswith(norm_title):
#                 mapped_title = section_aliases.get(title.lower(), title.lower())
#                 current_section = mapped_title
#                 if current_section not in sections:
#                     sections[current_section] = []
#                 matched = True
#                 break
#
#         if not matched:
#             if current_section not in sections:
#                 sections[current_section] = []
#             sections[current_section].append(stripped)
#
#     return {key: "\n".join(val).strip() for key, val in sections.items()}
#
#
# def extract_certification_titles(cert_text):
#     keywords = ["certification", "certificate", "certified", "aws", "linkedin", "course", "udemy", "coursera"]
#     certs = []
#     for line in cert_text.splitlines():
#         clean = re.sub(r'[^a-zA-Z0-9\s\-()]', '', line).strip()
#         if any(k in line.lower() for k in keywords) and 4 < len(clean) < 100 and "issued" not in line.lower():
#             clean = re.sub(r'^[oO\-–●]+\s*', '', clean)  # remove bullet points
#             certs.append(clean.title())
#     return list(dict.fromkeys(certs))[:3]  # Top 3
#
#
# # ==== 🔍 Paste your resume text here ====
# sample_resume = """
# CERTIFICATES
# o  Introduction to Java Programming - Programming Hub-2024 | CERTIFICATE
# o  Building a Website Certificate - Programming Hub-2024 |CERTIFICATE
# o  MySQL Essential Training - LinkedIn Learning 2025 |CERTIFICATE
#
# """
#
# # === Run Extraction ===
# sections = split_into_sections(sample_resume)
# print("==== DEBUG: All Split Sections ====")
# for key, val in sections.items():
#     print(f"[{key.upper()}] => {val[:150]}...")  # trimmed for view
#
# cert_section = sections.get("certifications", "")
# print("\n==== DEBUG: Raw Cert Text ====\n", cert_section)
#
# cert_names = extract_certification_titles(cert_section)
# print("\nCert Names Found:", cert_names)

import re


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
        r"\b(BSc|Bachelor of Science)\b"

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


# Test with your resume text
education_text = """
EDUCATION   
Master of Computer Application - AIT
Acharya Institute Of Technology – MCA 
2023-2025 
8.1 – SGPA(1st sem) 
Ramaiah Institute of Business Studies, Bangalore – BCA   
October 2020 – September 2023   
7.83 - S.G.P.A   
ST. Theresa PU College, Bangalore  - PUC   
April 2019 - May 2020 (PCMB)   
64.33%   
TRIVENI MEMORIAL HIGH SCHOOL, Bangalore - SSLC   
May 2017 - April 2018   
74.72%  
"""

print(extract_course_names(education_text))
