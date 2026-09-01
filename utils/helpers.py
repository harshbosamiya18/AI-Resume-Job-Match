import re


def format_percentage(value):
    """
    Format a numeric value as a percentage.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "0.00%"

    value = max(0.0, min(value, 100.0))

    return f"{value:.2f}%"


def format_skill_list(skills):
    """
    Convert a skill list into a comma-separated string.
    """

    if not skills:
        return "None"

    if isinstance(skills, str):
        return skills.strip() or "None"

    cleaned_skills = []

    for skill in skills:

        if skill is None:
            continue

        skill_text = str(skill).strip()

        if skill_text:
            cleaned_skills.append(skill_text)

    if not cleaned_skills:
        return "None"

    return ", ".join(cleaned_skills)


def format_skills(skills):
    """
    Format skills for display.

    This function is kept as an alias for compatibility
    with app.py imports.
    """

    return format_skill_list(skills)


def clean_whitespace(text):
    """
    Remove extra whitespace from text.
    """

    if not isinstance(text, str):
        return ""

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def validate_text_input(
    text,
    field_name="Text",
    min_length=1
):
    """
    Validate user text input and return cleaned text.
    """

    if not isinstance(text, str):
        raise ValueError(
            f"{field_name} must be text."
        )

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    try:
        min_length = int(min_length)
    except (TypeError, ValueError):
        min_length = 1

    if min_length < 1:
        min_length = 1

    if len(cleaned_text) < min_length:
        raise ValueError(
            f"{field_name} must contain at least "
            f"{min_length} characters."
        )

    return cleaned_text


def get_score_label(score):
    """
    Return a readable match label.
    """

    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Unknown"

    if score >= 85:
        return "Excellent Match"

    if score >= 70:
        return "Strong Match"

    if score >= 50:
        return "Moderate Match"

    if score >= 35:
        return "Low Match"

    return "Weak Match"


def get_score_description(score):
    """
    Return a short explanation based on the score.
    """

    try:
        score = float(score)
    except (TypeError, ValueError):
        return (
            "The match score could not be "
            "calculated correctly."
        )

    if score >= 85:
        return (
            "The candidate has a very strong alignment "
            "with the job requirements."
        )

    if score >= 70:
        return (
            "The candidate has strong alignment with "
            "the role, with some areas for improvement."
        )

    if score >= 50:
        return (
            "The candidate has a moderate match and "
            "should focus on the identified skill gaps."
        )

    if score >= 35:
        return (
            "The candidate has limited alignment with "
            "the current job requirements."
        )

    return (
        "The candidate currently has a weak match and "
        "may need significant skill development for "
        "this role."
    )


def get_score_message(score):
    """
    Return a short score message.

    This function is kept for compatibility with
    app.py imports.
    """

    return get_score_description(score)


def normalize_skill_list(skills):
    """
    Clean a skill list, remove duplicates and
    preserve the original order.
    """

    if not skills:
        return []

    if isinstance(skills, str):
        skills = skills.split(",")

    normalized_skills = []
    seen_skills = set()

    for skill in skills:

        if skill is None:
            continue

        skill_text = str(skill).strip()

        if not skill_text:
            continue

        skill_key = skill_text.lower()

        if skill_key not in seen_skills:
            normalized_skills.append(skill_text)
            seen_skills.add(skill_key)

    return normalized_skills


def safe_list(value):
    """
    Safely convert a value into a list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    if isinstance(value, str):

        if not value.strip():
            return []

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    return [value]


def count_unique_skills(skills):
    """
    Return the number of unique skills.
    """

    normalized_skills = normalize_skill_list(skills)

    return len(normalized_skills)


def calculate_skill_coverage(
    matching_skills,
    required_skills
):
    """
    Calculate skill coverage percentage.
    """

    matching_skills = normalize_skill_list(
        matching_skills
    )

    required_skills = normalize_skill_list(
        required_skills
    )

    if not required_skills:
        return 0.0

    matching_set = {
        skill.lower()
        for skill in matching_skills
    }

    required_set = {
        skill.lower()
        for skill in required_skills
    }

    matched_count = len(
        matching_set.intersection(required_set)
    )

    coverage = (
        matched_count / len(required_set)
    ) * 100

    return round(coverage, 2)


def clean_ai_response(text):
    """
    Clean the AI-generated response for display.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()