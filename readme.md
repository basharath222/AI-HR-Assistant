Project Title: AIML-03 AI-Based Resume Screening & Job Matching Platform
📖 Project Overview
This platform is a dual-intelligence tool designed to automate resume screening and match candidates with job roles using NLP and Generative AI. It solves the problem of manual resume screening for HR Teams and Job Portals by providing faster recruitment, bias reduction, and detailed skill-gap analysis.

The system doesn't just look for keywords; it "thinks" like an HR professional to evaluate candidate potential and provide ready-to-use interview kits.

✨ Key Features
1. Hybrid Ranking System
NLP Keyword Match %: Utilizes TF-IDF Vectorization and Cosine Similarity to measure the raw keyword overlap between the resume and the Job Description.

AI Semantic Match %: Leverages Gemini 1.5 Flash to understand the deeper context, such as project complexity, academic excellence, and qualitative experience.

2. Visual Skill-Gap Analysis
Radar (Spider) Charts: Dynamically visualizes candidate proficiency across four core dimensions: Technical, Soft-Skills, Experience, and Problem-Solving.

Color-Coded Feedback: Instantly identifies "Matched Skills" (Success) and "Missing Gaps" (Error) for fast decision-making.

3. Smart Interview Intelligence
Contextual Questions: Generates 5 personalized questions by cross-referencing the candidate's unique resume projects with the identified skill gaps.

Recruiter Checklist: Provides "Ideal Answer" bullet points and "Red Flags" to help non-technical recruiters evaluate technical responses.

🛠️ Technical Stack
Frontend: Streamlit (2026 Compliant Syntax).

Core AI: Google Gemini 1.5 Flash.

NLP Processing: Scikit-Learn (TfidfVectorizer, Cosine Similarity).

Data Handling: Pandas, JSON, PyPDF.

Resilience: Tenacity (Exponential Backoff for API stability).

📂 System Architecture
Text Extraction: Extracts plain text from uploaded PDF resumes.

Keyword Vectorization: Converts text into mathematical vectors to calculate Cosine Similarity.

Semantic Assessment: The LLM analyzes the candidate's history and identifies seniority level.

Actionable Dashboard: Results are displayed in a ranked leaderboard with detailed expanders for each candidate.

🎯 Impact
Bias Reduction: Standardized scoring ensures candidates are judged on skills and merit.

Efficiency: Reduces the time-to-hire by automating the initial screening and interview preparation phases.

Transparency: Clearly highlights what a candidate is missing, allowing HR to plan for future upskilling.