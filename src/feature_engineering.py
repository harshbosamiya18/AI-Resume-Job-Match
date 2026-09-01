from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from src.preprocessing import (
    preprocess_text
)


def calculate_tfidf_similarity(
    resume_text,
    job_text
):
    """
    Calculate TF-IDF cosine similarity between
    a resume and job description.
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

    processed_resume = preprocess_text(
        resume_text
    )

    processed_job = preprocess_text(
        job_text
    )

    if not processed_resume or not processed_job:
        return 0.0

    documents = [
        processed_resume,
        processed_job
    ]

    try:

        vectorizer = TfidfVectorizer()

        tfidf_matrix = (
            vectorizer.fit_transform(
                documents
            )
        )

        similarity_score = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        return round(
            float(similarity_score) * 100,
            2
        )

    except ValueError:

        return 0.0

    except Exception as error:

        print(
            f"Warning: TF-IDF similarity failed: {error}"
        )

        return 0.0


if __name__ == "__main__":

    resume_text = """
    Data Scientist skilled in Python, SQL,
    Machine Learning, NLP and Scikit-learn.
    """

    job_text = """
    We are hiring a Machine Learning Engineer
    with Python, SQL, TensorFlow, Docker and NLP.
    """

    score = calculate_tfidf_similarity(
        resume_text,
        job_text
    )

    print(
        "\n========== TF-IDF SIMILARITY ==========\n"
    )

    print(
        f"Resume-Job Similarity Score: {score}%"
    )