import streamlit as st
import json
import pandas as pd
import plotly.express as px
from engine import (
    get_pdf_text, 
    analyze_with_genai, 
    clean_json_response, 
    generate_interview_kit, 
    calculate_nlp_score
)

st.set_page_config(page_title="AI HR Assistant", layout="wide")

# 1. Persistent Memory Storage
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "interview_kits" not in st.session_state:
    st.session_state.interview_kits = {}

st.title("🚀 AI HR Assistant: Career Matcher")

# 2. Sidebar Controls
with st.sidebar:
    st.header("📋 Dashboard Controls")
    st.info("Results stay in memory until cleared.")
    if st.button("Clear All Data"):
        st.session_state.analysis_results = []
        st.session_state.interview_kits = {}
        st.rerun()

# 3. Input Section
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
jd_input = st.text_area("Paste Job Description", height=150)

if st.button("Analyze Candidates", type="primary"):
    if uploaded_files and jd_input:
        results = []
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                # Extract text
                text = get_pdf_text(file)
                raw_genai = analyze_with_genai(text, jd_input)
                
                try:
                    clean_json = clean_json_response(raw_genai)
                    data = json.loads(clean_json)
                    
                    # SAVE THE TEXT HERE so it is available later
                    data.update({
                        'filename': file.name,
                        'nlp_score': calculate_nlp_score(text, jd_input),
                        'resume_content': text  # New key to store resume text
                    })
                    results.append(data)
                except Exception as e:
                    st.error(f"Error parsing {file.name}: {e}")
        
        st.session_state.analysis_results = results
        st.success("Screening Complete!")
    else:
        st.warning("Please upload resumes and provide a JD.")

# --- DASHBOARD VISUALIZATION ---
if st.session_state.analysis_results:
    # 4. Global Leaderboard (Hybrid Logic)
    st.subheader("📊 Candidate Ranking (NLP + AI Hybrid)")
    df = pd.DataFrame(st.session_state.analysis_results)
    
    # Safe column selection to avoid KeyErrors
    expected_cols = ['filename', 'seniority', 'nlp_score', 'score']
    available_cols = [col for col in expected_cols if col in df.columns]
    
    if available_cols:
        leaderboard = df[available_cols].copy()
        # Rename for HR clarity
        col_map = {
            'filename': 'Candidate', 'seniority': 'Level', 
            'nlp_score': 'NLP Keyword Match %', 'score': 'AI Semantic Match %'
        }
        leaderboard.rename(columns=col_map, inplace=True)
        st.dataframe(leaderboard.sort_values(by=leaderboard.columns[-1], ascending=False), width='stretch')

    st.divider()

    # 5. Detailed Candidate Deep-Dives
    for res in st.session_state.analysis_results:
        fname = res.get('filename', 'Unknown')
        lvl = res.get('seniority', 'N/A')
        
        with st.expander(f"👤 Assessment: {fname} ({lvl})"):
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.write(f"### 🎯 Hybrid Match: {res.get('score', 0)}%")
                st.caption(f"NLP Similarity: {res.get('nlp_score', 0)}% | Identification: {lvl}")
                
                # HR Verdict List
                st.write("**📝 HR Verdict:**")
                for point in res.get('hr_verdict', ["Review required"]):
                    st.write(f"• {point}")
                
                # Skill Formatting
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.success("**✅ Found**")
                    for s in res.get('matched_skills', []): st.write(f"- {s}")
                with sc2:
                    st.error("**⚠️ Gaps**")
                    for s in res.get('missing_skills', []): st.write(f"- {s}")
            
            with col_right:
                # Radar Chart for Visual Skill-Gap Analysis
                categories = res.get('categories', {})
                if categories:
                    df_radar = pd.DataFrame(dict(r=list(categories.values()), theta=list(categories.keys())))
                    fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
                    fig.update_traces(fill='toself', line_color='#ff4b4b')
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), margin=dict(l=40, r=40, t=20, b=20))
                    st.plotly_chart(fig, width='stretch', key=f"radar_{fname}")

            st.divider()

            # 6. Concise Interview Kit
            st.write("### 🎙️ Quick-Start Interview Script")
            if fname not in st.session_state.interview_kits:
                if st.button(f"Generate HR Kit for {fname}", key=f"btn_{fname}"):
                    with st.spinner("Analyzing projects and generating questions..."):
                        # Retrieve the stored content from the dictionary
                        stored_resume_text = res.get('resume_content', "")
                        
                        kit = generate_interview_kit(
                            fname, 
                            res.get('missing_skills', []), 
                            jd_input, 
                            lvl, 
                            stored_resume_text # Pass the retrieved text
                        )
                        st.session_state.interview_kits[fname] = kit
                        st.rerun()
                        
            if fname in st.session_state.interview_kits:
                st.markdown(st.session_state.interview_kits[fname])