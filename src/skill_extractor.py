import re

from src.preprocessing import (
    clean_text,
    load_skills
)


def normalize_text(text):
    """
    Normalize text for safe and consistent
    case-insensitive skill matching.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_skill_aliases(row):
    """
    Return the canonical skill together with
    all aliases defined in the skills database.
    """

    canonical_skill = str(
        row.get("skill", "")
    ).strip()

    aliases = []

    if canonical_skill:
        aliases.append(
            canonical_skill
        )

    aliases_value = row.get(
        "aliases",
        ""
    )

    if aliases_value is not None:

        aliases_text = str(
            aliases_value
        ).strip()

        if (
            aliases_text
            and aliases_text.lower() != "nan"
        ):

            for alias in aliases_text.split(";"):

                alias = alias.strip()

                if alias:
                    aliases.append(
                        alias
                    )

    unique_aliases = []
    seen_aliases = set()

    for alias in aliases:

        alias_key = alias.lower()

        if alias_key not in seen_aliases:

            unique_aliases.append(
                alias
            )

            seen_aliases.add(
                alias_key
            )

    return unique_aliases


def alias_exists_in_text(
    alias,
    text
):
    """
    Check whether a complete skill alias exists
    in the input text.

    Regex boundaries prevent partial matches.

    Example:
    'AI' should not match inside another word.
    """

    if not alias or not text:
        return False

    alias = str(alias).strip()

    if not alias:
        return False

    alias_lower = alias.lower()
    text_lower = text.lower()

    escaped_alias = re.escape(
        alias_lower
    )

    pattern = (
        r"(?<!\w)"
        + escaped_alias
        + r"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            text_lower,
            flags=re.IGNORECASE
        )
    )


def extract_skills(
    text,
    skills_file="data/raw/skills_database.csv"
):
    """
    Extract canonical skills from text.

    The skills database supports aliases.

    Examples:

    NLP -> Natural Language Processing
    ML -> Machine Learning
    AI -> Artificial Intelligence
    AWS -> Amazon Web Services
    GCP -> Google Cloud Platform
    K8s -> Kubernetes
    GenAI -> Generative AI

    Regardless of which alias appears in the
    resume or job description, the returned
    skill is always the canonical skill name.
    """

    if not isinstance(text, str):
        return []

    cleaned_text = clean_text(
        text
    )

    normalized_input = normalize_text(
        cleaned_text
    )

    if not normalized_input:
        return []

    skills_df = load_skills(
        skills_file
    )

    required_columns = {
        "skill",
        "category"
    }

    missing_columns = (
        required_columns
        - set(skills_df.columns)
    )

    if missing_columns:

        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Skills database is missing required "
            f"columns: {missing_text}"
        )

    found_skills = []
    found_skill_names = set()

    for _, row in skills_df.iterrows():

        canonical_skill = str(
            row["skill"]
        ).strip()

        category = str(
            row["category"]
        ).strip()

        if not canonical_skill:
            continue

        aliases = get_skill_aliases(
            row
        )

        matched_alias = None

        for alias in aliases:

            if alias_exists_in_text(
                alias=alias,
                text=normalized_input
            ):

                matched_alias = alias
                break

        if matched_alias is None:
            continue

        canonical_key = (
            canonical_skill.lower()
        )

        if canonical_key in found_skill_names:
            continue

        found_skills.append(
            {
                "skill": canonical_skill,
                "category": category,
                "matched_alias": matched_alias
            }
        )

        found_skill_names.add(
            canonical_key
        )

    return found_skills


def get_skill_names(skills):
    """
    Return only canonical skill names.
    """

    if not skills:
        return []

    return [
        item["skill"]
        for item in skills
        if isinstance(item, dict)
        and item.get("skill")
    ]


def get_matched_aliases(skills):
    """
    Return the canonical skill together with
    the alias that was actually found in text.
    """

    if not skills:
        return []

    results = []

    for item in skills:

        if not isinstance(item, dict):
            continue

        skill = item.get(
            "skill"
        )

        matched_alias = item.get(
            "matched_alias"
        )

        if skill:

            results.append(
                {
                    "skill": skill,
                    "matched_alias": (
                        matched_alias
                        if matched_alias
                        else skill
                    )
                }
            )

    return results


if __name__ == "__main__":

    sample_text = """
    I am a Machine Learning Engineer with
    experience in Python, SQL, ML, NLP,
    Generative AI and Natural Language Processing.

    I have worked with Scikit-learn, TensorFlow,
    Docker, K8s, AWS, CI/CD and MLOps.

    My development experience includes ReactJS,
    NodeJS, REST APIs, PostgreSQL and GitHub.

    I am also interested in RAG, LLMs,
    HuggingFace, LangChain and Vector Databases.
    """

    extracted_skills = extract_skills(
        sample_text
    )

    print(
        "\n========== EXTRACTED SKILLS ==========\n"
    )

    for item in extracted_skills:

        print(
            f"- {item['skill']} "
            f"({item['category']})"
        )

        print(
            f"  Matched as: "
            f"{item['matched_alias']}"
        )

    print(
        "\n========== CANONICAL SKILL NAMES ==========\n"
    )

    canonical_skills = get_skill_names(
        extracted_skills
    )

    print(
        ", ".join(canonical_skills)
    )