from pathlib import Path
from pypdf import PdfReader
from docx import Document


def load_txt(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    return path.read_text(
        encoding="utf-8"
    )


def load_pdf(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def load_docx(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    document = Document(str(path))

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def load_document(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".txt":
        return load_txt(file_path)

    elif extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    else:
        raise ValueError(
            f"Desteklenmeyen dosya türü: {extension}"
        )


if __name__ == "__main__":

    file_path = "data/documents/foundry_test.txt"

    text = load_document(file_path)

    print("===== BELGE OKUNDU =====")
    print(text)
    print("========================")