from pathlib import Path

import pdfplumber


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf"
}


def extract_text_from_txt(file_path):
    """
    Extract text from a TXT file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"TXT file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The provided TXT path is not a file: {path}"
        )

    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except OSError as error:
        raise ValueError(
            f"Unable to read TXT file: {error}"
        ) from error


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The provided PDF path is not a file: {path}"
        )

    extracted_pages = []

    try:

        with pdfplumber.open(path) as pdf:

            if not pdf.pages:
                raise ValueError(
                    "The PDF file does not contain any pages."
                )

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                try:

                    page_text = page.extract_text()

                    if page_text:

                        page_text = page_text.strip()

                        if page_text:
                            extracted_pages.append(
                                page_text
                            )

                except Exception as error:

                    print(
                        f"Warning: Could not extract "
                        f"text from page {page_number}. "
                        f"Error: {error}"
                    )

    except ValueError:
        raise

    except Exception as error:

        raise ValueError(
            f"Unable to read PDF file: {error}"
        ) from error

    return "\n\n".join(
        extracted_pages
    )


def get_file_extension(file_path):
    """
    Return the normalized file extension.
    """

    if not file_path:
        return ""

    path = Path(file_path)

    return path.suffix.lower()


def is_supported_file(file_path):
    """
    Check whether a file format is supported.
    """

    extension = get_file_extension(
        file_path
    )

    return extension in SUPPORTED_EXTENSIONS


def extract_resume_text(file_path):
    """
    Extract readable text from a supported resume file.

    Supported formats:
    - TXT
    - PDF
    """

    if not file_path:
        raise ValueError(
            "No resume file was provided."
        )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Resume file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The provided path is not a file: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        supported_formats = ", ".join(
            sorted(SUPPORTED_EXTENSIONS)
        )

        raise ValueError(
            "Unsupported file format. "
            f"Supported formats: {supported_formats}"
        )

    if extension == ".txt":

        text = extract_text_from_txt(
            path
        )

    elif extension == ".pdf":

        text = extract_text_from_pdf(
            path
        )

    else:

        text = ""

    if not isinstance(text, str):

        raise ValueError(
            "No valid text could be extracted "
            "from the resume."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "No readable text could be extracted "
            "from the resume. The PDF may be scanned "
            "or image-based."
        )

    return text


def get_supported_extensions():
    """
    Return all supported resume file extensions.
    """

    return sorted(
        SUPPORTED_EXTENSIONS
    )


if __name__ == "__main__":

    print(
        "\n========== RESUME PARSER TEST ==========\n"
    )

    print(
        "Supported file formats:"
    )

    for extension in get_supported_extensions():

        print(
            f"- {extension.upper()}"
        )

    print(
        "\nResume parser is ready."
    )