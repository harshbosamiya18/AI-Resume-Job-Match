import os
import streamlit as st
from turtle import st

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.model import (
    calculate_advanced_match_score
)

from src.rag import (
    build_rag_context
)

from utils.helpers import (
    clean_ai_response,
    format_skill_list
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================
#
# The Gemini API key is stored in the project's .env file:
#
# GEMINI_API_KEY=your_api_key_here
#
# The .env file should NOT be uploaded to GitHub.
#
# load_dotenv() searches for the .env file and loads
# the environment variables into the application.
#
load_dotenv(override=True)


# ============================================================
# GEMINI MODEL CONFIGURATION
# ============================================================
#
# Keep the model name in one place so it is easy to change
# later without modifying the main AI analysis function.
#
# The environment variable GEMINI_MODEL can optionally be
# used to override the default model.
#
# Example:
#
# GEMINI_MODEL=gemini-3.6-flash
#
# If GEMINI_MODEL is not present in .env, the default
# gemini-3.6-flash model will be used.
#
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

def get_gemini_client():
    """
    Create and return the Gemini API client.

    The API key is loaded from the GEMINI_API_KEY
    environment variable.

    Returns:
        genai.Client:
            Configured Gemini API client.

    Raises:
        ValueError:
            If GEMINI_API_KEY is not available.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
        
    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Please check your .env file."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# GENERATE COMPLETE AI ANALYSIS
# ============================================================

def generate_ai_analysis(
    resume_text,
    job_text
):
    """
    Generate a complete technical career analysis.

    The complete pipeline contains:

    1. Expanded CS/IT skill matching
    2. TF-IDF similarity
    3. Semantic similarity
    4. RAG knowledge retrieval
    5. Gemini Generative AI analysis

    The RAG context is retrieved first and then inserted
    into the Gemini prompt so that the LLM can use the
    retrieved technical knowledge while generating the
    final career analysis.

    Returns:
        dict:
            Complete matching results, retrieved RAG
            context and Gemini-generated analysis.
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not isinstance(
        resume_text,
        str
    ) or not resume_text.strip():

        raise ValueError(
            "Resume text cannot be empty."
        )

    if not isinstance(
        job_text,
        str
    ) or not job_text.strip():

        raise ValueError(
            "Job description cannot be empty."
        )

    resume_text = resume_text.strip()
    job_text = job_text.strip()

    # ========================================================
    # STEP 1: CALCULATE NLP / ML MATCHING RESULTS
    # ========================================================
    #
    # This calls the existing model.py pipeline.
    #
    # The model calculates:
    #
    # - Exact Skill Match
    # - TF-IDF Similarity
    # - Semantic Similarity
    # - Final AI Match Score
    # - Matching Skills
    # - Missing Skills
    #
    results = calculate_advanced_match_score(
        resume_text,
        job_text
    )

    # ========================================================
    # EXTRACT MATCHING RESULTS SAFELY
    # ========================================================

    final_score = results.get(
        "final_score",
        0.0
    )

    skill_score = results.get(
        "skill_score",
        0.0
    )

    tfidf_score = results.get(
        "tfidf_score",
        0.0
    )

    semantic_score = results.get(
        "semantic_score",
        0.0
    )

    matching_skills = results.get(
        "matching_skills",
        []
    )

    missing_skills = results.get(
        "missing_skills",
        []
    )

    resume_skills = results.get(
        "resume_skills",
        []
    )

    job_skills = results.get(
        "job_skills",
        []
    )

    # ========================================================
    # STEP 2: RETRIEVE RELEVANT RAG KNOWLEDGE
    # ========================================================
    #
    # The RAG system searches the knowledge base and returns
    # the most relevant technical knowledge chunks.
    #
    # These retrieved chunks will later be passed directly
    # into the Gemini prompt.
    #
    # THIS IS THE RAG -> LLM CONNECTION.
    #
    rag_context = build_rag_context(
        resume_text=resume_text,
        job_text=job_text,
        missing_skills=missing_skills,
        top_k=5
    )

    if not isinstance(
        rag_context,
        str
    ) or not rag_context.strip():

        rag_context = (
            "No additional knowledge-base context "
            "was retrieved."
        )

    # ========================================================
    # STEP 3: FORMAT SKILLS FOR THE PROMPT
    # ========================================================

    matching_skills_text = format_skill_list(
        matching_skills
    )

    missing_skills_text = format_skill_list(
        missing_skills
    )

    resume_skills_text = format_skill_list(
        resume_skills
    )

    job_skills_text = format_skill_list(
        job_skills
    )

    # ========================================================
    # STEP 4: BUILD THE GENERATIVE AI PROMPT
    # ========================================================
    #
    # IMPORTANT:
    #
    # This is where the RAG -> LLM connection happens.
    #
    # The retrieved RAG knowledge is inserted into the
    # "RETRIEVED CS / IT KNOWLEDGE" section below.
    #
    # Gemini therefore receives:
    #
    # Resume
    # Job Description
    # Matching Results
    # Missing Skills
    # RAG Knowledge
    #
    # and generates the final analysis from all of them.
    #

    prompt = f"""
You are an expert technical career advisor specializing in:

- Computer Science
- Information Technology
- Software Engineering
- Data Science
- Data Analytics
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Generative AI
- Data Engineering
- Cloud Computing
- DevOps
- MLOps
- Cybersecurity
- Networking
- Web Development
- Mobile Development
- Databases
- Software Testing
- Emerging Technologies

Your task is to analyze a candidate's resume
against a specific job description.

The candidate may belong to ANY technical field.

Do not assume that the candidate is a Data Scientist,
Data Analyst, Software Engineer or any other specific
professional role unless the resume or job description
supports that conclusion.

============================================================
IMPORTANT RULES
============================================================

1. The numerical scores below were calculated by an
   NLP and Machine Learning matching system.

2. Do NOT recalculate, modify, reinterpret or contradict
   these numerical scores.

3. Do NOT invent skills, work experience, projects,
   certifications, education, achievements, companies,
   responsibilities or qualifications.

4. Only use information from:
   - Candidate resume
   - Job description
   - System matching results
   - Retrieved RAG knowledge

5. RAG knowledge is general technical information.

   Never claim that the candidate possesses a skill
   simply because that skill appears in the RAG context.

6. Matching skills are skills detected in BOTH the
   resume and the job description.

7. Missing skills are skills detected in the job
   description but not detected in the resume.

8. If no missing skills are detected, do not invent
   skill gaps.

   Instead explain that no database-detected skill gaps
   were found.

9. Resume improvement suggestions must be realistic.

   Suggest better wording, organization, emphasis and
   evidence only.

   Do not suggest false achievements.

10. Adapt the analysis to the actual technical domain
    indicated by the resume and job description.

11. Be constructive, professional, specific and concise.

12. Use Markdown formatting for readability.

============================================================
SYSTEM MATCHING RESULTS
============================================================

FINAL AI MATCH SCORE:
{final_score}%

EXACT SKILL MATCH:
{skill_score}%

TF-IDF TEXT SIMILARITY:
{tfidf_score}%

SEMANTIC SIMILARITY:
{semantic_score}%

============================================================
MATCHING SKILLS
============================================================

{matching_skills_text}

============================================================
MISSING SKILLS
============================================================

{missing_skills_text}

============================================================
DETECTED RESUME SKILLS
============================================================

{resume_skills_text}

============================================================
DETECTED JOB SKILLS
============================================================

{job_skills_text}

============================================================
RETRIEVED CS / IT KNOWLEDGE
============================================================

The following information was retrieved from the project's
technical knowledge base using the RAG retrieval system.

Use this information only as general technical background.

Do NOT treat retrieved knowledge as evidence that the
candidate has any particular skill.

{rag_context}

============================================================
CANDIDATE RESUME
============================================================

{resume_text}

============================================================
JOB DESCRIPTION
============================================================

{job_text}

============================================================
OUTPUT FORMAT
============================================================

Provide your response using EXACTLY these six sections
and keep the section numbering exactly as shown.

============================================================

## 1. MATCH EXPLANATION

Explain why the candidate received the calculated
Final AI Match Score.

Briefly discuss the relationship between:

- Exact Skill Match
- TF-IDF Similarity
- Semantic Similarity

Do not modify the numerical values.

============================================================

## 2. STRENGTHS

List the strongest qualifications, relevant technical
skills and areas of alignment.

Only mention strengths supported by the resume,
job description and detected matching results.

Use bullet points.

============================================================

## 3. SKILL GAP ANALYSIS

Analyze the detected missing skills.

For every important missing skill:

- Explain what it is.
- Explain why it may matter for this specific role.
- Explain its priority when possible.

If there are no detected missing skills, clearly state:

"No database-detected skill gaps were found."

Do not invent additional missing skills.

============================================================

## 4. RESUME IMPROVEMENTS

Give practical suggestions specifically tailored
to the provided job description.

Suggestions may include:

- Better skill organization
- More relevant project descriptions
- Clearer technical terminology
- Better evidence of existing skills
- Improved resume structure
- Highlighting relevant existing experience

Never invent metrics, projects or achievements.

============================================================

## 5. LEARNING ROADMAP

Create a prioritized learning roadmap based primarily
on the detected missing skills.

For each priority level include:

- Skill or technology
- Why it matters
- Important topics to learn
- A practical project or exercise idea

Do not recommend learning skills that are already
detected as strong matches unless useful for
reinforcement or integration.

============================================================

## 6. INTERVIEW QUESTIONS

Generate EXACTLY 5 relevant technical interview questions.

The questions must be based on:

- The job description
- Matching skills
- Missing skills

The questions should be appropriate for the actual
technical domain of the role.

Number them from 1 to 5.

============================================================
END OF INSTRUCTIONS
============================================================
"""

    # ========================================================
    # STEP 5: CREATE GEMINI CLIENT
    # ========================================================

    client = get_gemini_client()

    # ========================================================
    # STEP 6: CREATE GEMINI CHAT
    # ========================================================
    #
    # The Google Gen AI Python SDK provides the chats helper:
    #
    # client.chats.create(...)
    #
    # followed by:
    #
    # chat.send_message(...)
    #
    # This is the current supported chat pattern for the
    # Google Gen AI SDK.
    #

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1200,
                temperature=0.2
            )
        )

    except Exception as error:
        raise RuntimeError(
            "Gemini API request failed. "
            f"Please check your API key, model name, "
            f"internet connection and Gemini API access. "
            f"Original error: {error}"
        ) from error

    # ========================================================
    # STEP 8: EXTRACT AI RESPONSE
    # ========================================================

    ai_analysis = getattr(
        response,
        "text",
        ""
    )

    if ai_analysis is None:

        ai_analysis = ""

    ai_analysis = str(
        ai_analysis
    )

    # ========================================================
    # STEP 9: CLEAN AI RESPONSE
    # ========================================================

    ai_analysis = clean_ai_response(
        ai_analysis
    )

    if not ai_analysis:

        ai_analysis = (
            "The AI analysis could not be generated. "
            "Please try again."
        )

    # ========================================================
    # STEP 10: RETURN COMPLETE RESULTS
    # ========================================================
    #
    # This structure is important because app.py can use:
    #
    # output["results"]
    # output["rag_context"]
    # output["ai_analysis"]
    #
    return {
        "results": results,
        "rag_context": rag_context,
        "ai_analysis": ai_analysis
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    resume_text = """
    Data Scientist with experience in Python, SQL,
    Pandas, NumPy, Machine Learning, Scikit-learn,
    Natural Language Processing and data visualization.

    Built machine learning projects involving data
    preprocessing, feature engineering, model training,
    evaluation and text analysis.

    Familiar with Git, GitHub and Power BI.
    """

    job_text = """
    We are hiring a Machine Learning Engineer with
    strong experience in Python, Machine Learning,
    TensorFlow, Docker, Kubernetes, MLOps and NLP.

    The candidate should understand model deployment,
    containerization, machine learning pipelines and
    production ML systems.
    """

    try:

        output = generate_ai_analysis(
            resume_text=resume_text,
            job_text=job_text
        )

        results = output["results"]

        # ====================================================
        # PRINT AUTOMATED MATCH RESULTS
        # ====================================================

        print(
            "\n========== AUTOMATED AI MATCH RESULTS ==========\n"
        )

        print(
            f"Exact Skill Match: "
            f"{results.get('skill_score', 0)}%"
        )

        print(
            f"TF-IDF Similarity: "
            f"{results.get('tfidf_score', 0)}%"
        )

        print(
            f"Semantic Similarity: "
            f"{results.get('semantic_score', 0)}%"
        )

        print(
            f"\nFINAL AI MATCH SCORE: "
            f"{results.get('final_score', 0)}%"
        )

        # ====================================================
        # PRINT MATCHING SKILLS
        # ====================================================

        print(
            "\n========== MATCHING SKILLS ==========\n"
        )

        matching_skills = results.get(
            "matching_skills",
            []
        )

        if matching_skills:

            for skill in matching_skills:

                print(
                    f"✓ {skill}"
                )

        else:

            print(
                "No matching skills detected."
            )

        # ====================================================
        # PRINT MISSING SKILLS
        # ====================================================

        print(
            "\n========== MISSING SKILLS ==========\n"
        )

        missing_skills = results.get(
            "missing_skills",
            []
        )

        if missing_skills:

            for skill in missing_skills:

                print(
                    f"✗ {skill}"
                )

        else:

            print(
                "No missing skills detected."
            )

        # ====================================================
        # PRINT RAG KNOWLEDGE
        # ====================================================

        print(
            "\n========== RAG RETRIEVED KNOWLEDGE ==========\n"
        )

        print(
            output["rag_context"]
        )

        # ====================================================
        # PRINT GEMINI ANALYSIS
        # ====================================================

        print(
            "\n========== GENERATIVE AI ANALYSIS ==========\n"
        )

        print(
            output["ai_analysis"]
        )

    except Exception as error:

        print(
            "\n========== ERROR ==========\n"
        )

        print(
            str(error)
        )