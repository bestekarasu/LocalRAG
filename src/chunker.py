import re


def chunk_text(text, max_chars=700, overlap_sentences=1):
    """
    Metni cümle sınırlarını koruyarak chunk'lara böler.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = []

    for sentence in sentences:

        candidate = " ".join(current + [sentence])

        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))

            # Önceki chunk'ın son cümlesini sonraki chunk'a taşı
            current = current[-overlap_sentences:]

        current.append(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


if __name__ == "__main__":
    from document_loader import load_txt

    file_path = "data/documents/foundry_test.txt"

    text = load_txt(file_path)

    chunks = chunk_text(text)

    print(f"Toplam chunk sayısı: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n===== CHUNK {i} =====")
        print(chunk)