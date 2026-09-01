# AI Resume & Job Match

An AI-powered resume and job matching system that combines Natural Language Processing, Machine Learning, semantic similarity, skill extraction, Retrieval-Augmented Generation (RAG), and Generative AI to analyze candidate-job compatibility.

## Project Overview

AI Resume & Job Match evaluates how well a candidate's resume matches a specific job description.

The system extracts technical skills from both the resume and job description, identifies matching and missing skills, calculates multiple similarity scores, and combines them into a final match score.

The application also uses a technical knowledge base with RAG retrieval and Google Gemini Generative AI to generate a detailed career analysis.

The final analysis includes:

- Match explanation
- Candidate strengths
- Skill gap analysis
- Resume improvement suggestions
- Personalized learning roadmap
- Technical interview questions

## Key Features

- PDF resume parsing
- TXT resume support
- NLP text preprocessing
- Technical skill extraction
- Skill alias matching
- Canonical skill normalization
- Exact skill matching
- TF-IDF similarity
- Semantic similarity
- Sentence Transformer embeddings
- Weighted resume-job match score
- RAG-based technical knowledge retrieval
- Gemini Generative AI analysis
- Skill gap identification
- Learning roadmap generation
- Interview question generation
- Streamlit web application

## Matching Methodology

The final match score combines three major signals:

| Component | Weight |
|---|---:|
| Exact Skill Match | 45% |
| TF-IDF Similarity | 20% |
| Semantic Similarity | 35% |

### Final Score

Final Match Score =
Skill Match × 0.45 +
TF-IDF Similarity × 0.20 +
Semantic Similarity × 0.35

This approach combines explicit technical skill matching with traditional NLP similarity and semantic understanding.

## NLP Pipeline

The application performs the following preprocessing steps:

1. Convert text to lowercase
2. Remove URLs
3. Remove email addresses
4. Preserve technical characters
5. Remove unnecessary characters
6. Remove extra whitespace
7. Remove stop words
8. Perform lemmatization using spaCy

## Skill Extraction

The project uses an expanded CS/IT skills database containing canonical skill names, categories, aliases, and technical terminology.

Examples of supported aliases include:

- ML → Machine Learning
- NLP → Natural Language Processing
- AI → Artificial Intelligence
- AWS → Amazon Web Services
- K8s → Kubernetes
- GenAI → Generative AI
- ReactJS → React.js
- NodeJS → Node.js

The system returns the canonical skill name regardless of which supported alias appears in the resume or job description.

## Similarity Analysis

### TF-IDF Similarity

TF-IDF is used to calculate textual similarity between the resume and job description.

### Semantic Similarity

Sentence Transformers are used to generate embeddings for the resume and job description.

The project uses:

`all-MiniLM-L6-v2`

Cosine similarity is then calculated between the generated embeddings.

## RAG System

The project includes a lightweight Retrieval-Augmented Generation pipeline.

The system retrieves relevant technical knowledge from:

`data/knowledge_base/skill_descriptions.txt`

The retrieval process uses TF-IDF and cosine similarity to identify relevant knowledge chunks.

The retrieved information is passed to Gemini as additional technical context.

The RAG information is treated only as general technical knowledge and is not considered evidence that the candidate possesses a particular skill.

## Generative AI

Google Gemini is used to generate the final career analysis.

Gemini receives:

- Resume text
- Job description
- Match scores
- Matching skills
- Missing skills
- Detected resume skills
- Detected job skills
- Retrieved RAG knowledge

The AI generates a structured analysis without inventing candidate qualifications, experience, projects, certifications, or achievements.

## Project Architecture

```text
Resume PDF/TXT
      |
      v
Resume Parser
      |
      v
Text Cleaning & NLP
      |
      +----------------------+
      |                      |
      v                      v
Skill Extraction       Similarity Analysis
      |                      |
      v                      +------------------+
Skill Matching               |                  |
      |                      v                  v
      |                 TF-IDF Similarity   Semantic Similarity
      |                      |                  |
      +----------------------+------------------+
                             |
                             v
                    Advanced Match Score
                             |
                             v
                    RAG Knowledge Retrieval
                             |
                             v
                     Gemini Generative AI
                             |
                             v
                  Final Career Analysis