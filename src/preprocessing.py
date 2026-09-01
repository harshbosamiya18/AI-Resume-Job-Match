import re

import pandas as pd
import spacy


def load_spacy_model():
    """
    Load the spaCy English language model.

    The model is loaded only when preprocessing
    actually requires it.
    """

    try:

        return spacy.load(
            "en_core_web_sm"
        )

    except OSError as error:

        raise OSError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Please run:\n"
            "python -m spacy download en_core_web_sm"
        ) from error


nlp = None


def clean_text(text):
    """
    Clean raw resume or job description text while preserving
    useful technical characters such as +, # and .
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Preserve technical characters used in skills:
    # C++, C#, React.js, Node.js, etc.
    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def preprocess_text(text):
    """
    Perform NLP preprocessing using spaCy.
    """

    global nlp

    cleaned_text = clean_text(
        text
    )

    if not cleaned_text:
        return ""

    if nlp is None:
        nlp = load_spacy_model()

    doc = nlp(
        cleaned_text
    )

    tokens = []

    for token in doc:

        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
        ):

            lemma = token.lemma_.lower().strip()

            if lemma:
                tokens.append(
                    lemma
                )

    return " ".join(
        tokens
    )


def load_skills(
    file_path="data/raw/skills_database.csv"
):
    """
    Load the CS/IT skills database.
    """

    try:

        skills_df = pd.read_csv(
            file_path
        )

    except FileNotFoundError as error:

        raise FileNotFoundError(
            f"Skills database not found: {file_path}"
        ) from error

    except Exception as error:

        raise ValueError(
            f"Unable to read skills database: {error}"
        ) from error

    required_columns = {
        "skill",
        "category"
    }

    if not required_columns.issubset(
        skills_df.columns
    ):

        raise ValueError(
            "Skills database must contain "
            "'skill' and 'category' columns."
        )

    skills_df = skills_df.dropna(
        subset=[
            "skill",
            "category"
        ]
    ).copy()

    skills_df["skill"] = (
        skills_df["skill"]
        .astype(str)
        .str.strip()
    )

    skills_df["category"] = (
        skills_df["category"]
        .astype(str)
        .str.strip()
    )

    if "aliases" in skills_df.columns:

        skills_df["aliases"] = (
            skills_df["aliases"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    skills_df = skills_df[
        skills_df["skill"] != ""
    ]

    skills_df = skills_df[
        skills_df["category"] != ""
    ]

    return skills_df


if __name__ == "__main__":

    sample_text = """
    I am a Data Scientist skilled in Python, SQL,
    Machine Learning, Pandas, NumPy, NLP,
    Scikit-learn, TensorFlow and Docker.
    """

    print(
        "========== ORIGINAL TEXT ==========\n"
    )

    print(
        sample_text
    )

    print(
        "\n========== CLEANED TEXT ==========\n"
    )

    print(
        clean_text(
            sample_text
        )
    )

    print(
        "\n========== PREPROCESSED TEXT ==========\n"
    )

    print(
        preprocess_text(
            sample_text
        )
    )

    print(
        "\n========== SKILLS DATABASE ==========\n"
    )

    print(
        load_skills().head()
    )