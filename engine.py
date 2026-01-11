import os
import pypdf
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import streamlit as st

load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
SCREENING_MODEL = 'gemini-2.5-flash-lite'
client = genai.Client(api_key=api_key)
def get_pdf_text(pdf_file):
    """Extracts text from PDF."""
    reader = pypdf.PdfReader(pdf_file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Change this line in engine.py:
def calculate_nlp_score(resume_text, jd_text):  # Renamed from calculate_nlp_metrics
    """ Isolated NLP analysis for a single candidate. """
    # Standard Cosine Similarity for the Keyword Match
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = round(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100, 2)
    
    return score

@retry(retry=retry_if_exception_type(errors.ClientError), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_with_genai(resume_text, jd_text):
    
    """Deep HR Analysis with structured output for clear formatting."""
    prompt = f"""
    Act as a Senior HR Partner. Analyze the Resume against the Job Description.
    Return ONLY a JSON object:
    {{
      "score": 0-100,
      "seniority": "Junior/Mid/Senior",
      "categories": {{ "Technical": 0, "Soft-Skills": 0, "Experience": 0, "Problem-Solving": 0 }},
      "matched_skills": ["List only top 5 clear matches"],
      "missing_skills": ["List only top 5 clear gaps"],
      "hr_verdict": [
          "Scannable point about technical fit",
          "Scannable point about experience relevance",
          "Final hiring recommendation"
      ]
    }}
    RESUME: {resume_text}
    JD: {jd_text}
    """
    try:
        response = client.models.generate_content(
            model=SCREENING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        return response.text
    except Exception:
        return json.dumps({"score": 0, "hr_verdict": ["Analysis Error"]})

def generate_interview_kit(candidate_name, missing_skills, jd_text, seniority, resume_text):
    """
    Generates a personalized interview kit that cross-references 
    project experience with identified skill gaps.
    """
    prompt = f"""
    Act as a Technical Interviewer. Analyze the provided Candidate Context and Job Description.
    
    Candidate: {candidate_name} ({seniority})
    Resume Details: {resume_text[:2000]}  # Include resume content for deep context
    Identified Gaps: {", ".join(missing_skills)}
    
    Generate 5 HIGHLY SPECIFIC questions:
    1. PROJECT DEEP-DIVE: Pick a major project from their resume. Ask a 'how-it-works' question that ties it to a skill in the JD.
    2. TECHNICAL VALIDATION: Ask a question about their primary programming language listed in their resume.
    3. GAP ANALYSIS: Ask how they would apply their current knowledge to learn one of the 'Missing Skills'.
    4. PROBLEM-SOLVING: A situational question based on a challenge mentioned in their resume.
    5. BEHAVIORAL: A {seniority}-level leadership or teamwork question.

    For each question, provide:
    - **Q**: [Specific & direct]
    - **Context from Resume**: [Reference the specific project/skill you are asking about]
    - **Checklist**: [2-3 technical indicators of a strong answer]
    - **Red Flag**: [1 indicator of exaggerated experience]
    """
    response = client.models.generate_content(model=SCREENING_MODEL, contents=prompt)
    return response.text

def clean_json_response(raw_response):
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, raw_response)
    return match.group(1).strip() if match else raw_response.strip()