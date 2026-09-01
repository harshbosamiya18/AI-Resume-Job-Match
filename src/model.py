from src.similarity import (
    calculate_skill_match,
    calculate_semantic_similarity
)

from src.feature_engineering import (
    calculate_tfidf_similarity
)


# Final score weights.
SKILL_WEIGHT = 0.45
TFIDF_WEIGHT = 0.20
SEMANTIC_WEIGHT = 0.35


def calculate_advanced_match_score(
    resume_text,
    job_text
):
    """
    Calculate the final CS/IT resume-job match score.

    Score components:

    - Exact Skill Match: 45%
    - TF-IDF Similarity: 20%
    - Semantic Similarity: 35%

    The three components are combined into one
    final AI match score.
    """

    skill_results = calculate_skill_match(
        resume_text,
        job_text
    )

    skill_score = skill_results[
        "match_percentage"
    ]

    tfidf_score = calculate_tfidf_similarity(
        resume_text,
        job_text
    )

    semantic_score = calculate_semantic_similarity(
        resume_text,
        job_text
    )

    final_score = (
        skill_score * SKILL_WEIGHT
        + tfidf_score * TFIDF_WEIGHT
        + semantic_score * SEMANTIC_WEIGHT
    )

    return {
        "final_score": round(
            final_score,
            2
        ),

        "skill_score": skill_score,

        "tfidf_score": tfidf_score,

        "semantic_score": semantic_score,

        "matching_skills": skill_results[
            "matching_skills"
        ],

        "missing_skills": skill_results[
            "missing_skills"
        ],

        "resume_skills": skill_results[
            "resume_skills"
        ],

        "job_skills": skill_results[
            "job_skills"
        ]
    }


if __name__ == "__main__":

    resume_text = """
    Data Scientist skilled in Python, SQL,
    Pandas, NumPy, Machine Learning, NLP,
    Scikit-learn and data visualization.
    """

    job_text = """
    Looking for a Machine Learning Engineer
    skilled in Python, SQL, Machine Learning,
    NLP, TensorFlow, Docker and FastAPI.
    """

    results = calculate_advanced_match_score(
        resume_text,
        job_text
    )

    print(
        "\n========== ADVANCED MATCH RESULTS =========="
    )

    print(
        f"\nExact Skill Match: "
        f"{results['skill_score']}%"
    )

    print(
        f"TF-IDF Similarity: "
        f"{results['tfidf_score']}%"
    )

    print(
        f"Semantic Similarity: "
        f"{results['semantic_score']}%"
    )

    print(
        f"\nFINAL AI MATCH SCORE: "
        f"{results['final_score']}%"
    )

    print(
        "\n---------- MATCHING SKILLS ----------"
    )

    for skill in results[
        "matching_skills"
    ]:

        print(
            f"✓ {skill}"
        )

    print(
        "\n---------- MISSING SKILLS ----------"
    )

    for skill in results[
        "missing_skills"
    ]:

        print(
            f"✗ {skill}"
        )

    print(
        "\n---------- ALL RESUME SKILLS ----------"
    )

    for skill in results[
        "resume_skills"
    ]:

        print(
            f"- {skill}"
        )

    print(
        "\n---------- JOB REQUIRED SKILLS ----------"
    )

    for skill in results[
        "job_skills"
    ]:

        print(
            f"- {skill}"
        )