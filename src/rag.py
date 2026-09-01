from pathlib import Path

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)
from sklearn.metrics.pairwise import (
    cosine_similarity
)


# Get the project root directory.
# This makes the knowledge-base path work correctly
# regardless of where the Python command is executed from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "skill_descriptions.txt"
)


def load_knowledge_base(
    file_path=KNOWLEDGE_BASE_PATH
):
    """
    Load the CS/IT knowledge base and split it
    into individual knowledge chunks.

    The knowledge base is expected to contain
    separate chunks divided by blank lines.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Knowledge base path is not a file: {path}"
        )

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    return chunks


def retrieve_relevant_context(
    query,
    top_k=5,
    file_path=KNOWLEDGE_BASE_PATH
):
    """
    Retrieve the most relevant CS/IT knowledge chunks
    using TF-IDF cosine similarity.

    The query is compared against every knowledge
    chunk in the knowledge base. The highest-scoring
    chunks are returned first.
    """

    chunks = load_knowledge_base(
        file_path
    )

    if not chunks:
        return []

    if not isinstance(
        query,
        str
    ):
        return []

    query = query.strip()

    if not query:
        return []

    try:

        top_k = int(
            top_k
        )

    except (
        TypeError,
        ValueError
    ):

        top_k = 5

    if top_k < 1:
        top_k = 1

    documents = [
        query
    ] + chunks

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = (
            vectorizer.fit_transform(
                documents
            )
        )

        similarity_scores = (
            cosine_similarity(
                matrix[0:1],
                matrix[1:]
            )[0]
        )

    except ValueError:

        return []

    ranked_indices = (
        similarity_scores.argsort()[::-1]
    )

    results = []

    for index in ranked_indices[:top_k]:

        score = (
            float(
                similarity_scores[index]
            )
            * 100
        )

        results.append(
            {
                "text": chunks[index],
                "score": round(
                    score,
                    2
                )
            }
        )

    return results


def build_rag_context(
    resume_text,
    job_text,
    missing_skills,
    top_k=5
):
    """
    Build CS/IT knowledge context for the
    Generative AI analysis.

    The query combines:
    - Resume information
    - Job description
    - Missing skills

    The resulting context is retrieved from
    the knowledge base and formatted for the LLM.
    """

    if not isinstance(
        resume_text,
        str
    ):
        resume_text = ""

    if not isinstance(
        job_text,
        str
    ):
        job_text = ""

    if missing_skills is None:
        missing_skills = []

    if isinstance(
        missing_skills,
        str
    ):

        missing_skills_text = (
            missing_skills.strip()
            if missing_skills.strip()
            else "None"
        )

    else:

        missing_skills_text = (
            ", ".join(
                str(skill).strip()
                for skill in missing_skills
                if skill is not None
                and str(skill).strip()
            )
            or "None"
        )

    query = f"""
Resume:
{resume_text}

Job Description:
{job_text}

Missing Skills:
{missing_skills_text}
"""

    retrieved_chunks = (
        retrieve_relevant_context(
            query=query,
            top_k=top_k
        )
    )

    if not retrieved_chunks:

        return (
            "No relevant knowledge context "
            "retrieved."
        )

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"[Relevance: {item['score']}%]\n"
            f"{item['text']}"
        )

    return "\n\n".join(
        context_parts
    )


if __name__ == "__main__":

    resume_text = """
    Data Scientist skilled in Python, SQL,
    Machine Learning and NLP.
    """

    job_text = """
    Looking for a Machine Learning Engineer
    with Python, TensorFlow, Docker,
    Kubernetes and MLOps experience.
    """

    missing_skills = [
        "TensorFlow",
        "Docker",
        "Kubernetes",
        "MLOps"
    ]

    context = build_rag_context(
        resume_text,
        job_text,
        missing_skills
    )

    print(
        "\n========== RAG RETRIEVED KNOWLEDGE ==========\n"
    )

    print(
        context
    )