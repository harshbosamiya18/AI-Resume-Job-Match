from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from src.skill_extractor import (
    extract_skills
)


# Sentence Transformer model name.
EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)


# Load the embedding model once.
embedding_model = None


def load_embedding_model():
    """
    Load the SentenceTransformer embedding model
    only when semantic similarity is required.
    """

    global embedding_model

    if embedding_model is not None:
        return embedding_model

    try:

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        return embedding_model

    except Exception as error:

        raise RuntimeError(
            "Unable to load the SentenceTransformer "
            f"model '{EMBEDDING_MODEL_NAME}'. "
            f"Error: {error}"
        ) from error


def calculate_skill_match(
    resume_text,
    job_text
):
    """
    Compare extracted CS/IT skills from the resume
    against skills required by the job description.
    """

    resume_skills_data = extract_skills(
        resume_text
    )

    job_skills_data = extract_skills(
        job_text
    )

    resume_skills = {
        item["skill"].lower(): item["skill"]
        for item in resume_skills_data
    }

    job_skills = {
        item["skill"].lower(): item["skill"]
        for item in job_skills_data
    }

    matching_skills = []
    missing_skills = []

    for skill_key, skill_name in job_skills.items():

        if skill_key in resume_skills:

            matching_skills.append(
                skill_name
            )

        else:

            missing_skills.append(
                skill_name
            )

    if not job_skills:

        match_percentage = 0.0

    else:

        match_percentage = (
            len(matching_skills)
            / len(job_skills)
        ) * 100

    return {
        "match_percentage": round(
            match_percentage,
            2
        ),

        "matching_skills": matching_skills,

        "missing_skills": missing_skills,

        "resume_skills": list(
            resume_skills.values()
        ),

        "job_skills": list(
            job_skills.values()
        )
    }


def calculate_semantic_similarity(
    resume_text,
    job_text
):
    """
    Calculate semantic similarity using
    SentenceTransformer embeddings.
    """

    if not isinstance(
        resume_text,
        str
    ):

        return 0.0

    if not isinstance(
        job_text,
        str
    ):

        return 0.0

    if not resume_text.strip() or not job_text.strip():

        return 0.0

    model = load_embedding_model()

    try:

        embeddings = model.encode(
            [
                resume_text,
                job_text
            ]
        )

        similarity_score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return round(
            float(similarity_score) * 100,
            2
        )

    except Exception as error:

        raise RuntimeError(
            "Unable to calculate semantic "
            f"similarity: {error}"
        ) from error


if __name__ == "__main__":

    resume_text = """
    Aspiring Data Scientist skilled in Python,
    SQL, Pandas, NumPy, Machine Learning,
    NLP and Scikit-learn.

    Experienced in data cleaning,
    feature engineering and data visualization.
    """

    job_text = """
    We are hiring a Machine Learning Engineer
    with strong Python and SQL skills.

    The candidate should have experience in
    Machine Learning, NLP, TensorFlow,
    feature engineering and Docker.
    """

    skill_results = calculate_skill_match(
        resume_text,
        job_text
    )

    semantic_score = calculate_semantic_similarity(
        resume_text,
        job_text
    )

    print(
        "\n========== SKILL MATCH ==========\n"
    )

    print(
        "Skill Match Score: "
        f"{skill_results['match_percentage']}%"
    )

    print(
        "\n========== SEMANTIC SIMILARITY ==========\n"
    )

    print(
        "Semantic Similarity Score: "
        f"{semantic_score}%"
    )

    print(
        "\nMatching Skills:"
    )

    for skill in skill_results[
        "matching_skills"
    ]:

        print(
            f"✓ {skill}"
        )

    print(
        "\nMissing Skills:"
    )

    for skill in skill_results[
        "missing_skills"
    ]:

        print(
            f"✗ {skill}"
        )