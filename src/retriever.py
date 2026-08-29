import json
import sqlite3
import re
import numpy as np

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager
)


DB_PATH = "db/localrag.db"


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    İki vektör arasındaki cosine similarity değerini hesaplar.
    """

    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# CHUNKLARI GETİR
# ============================================================

def get_chunks():
    """
    SQLite'taki tüm chunkları getirir.
    """

    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, source, chunk, embedding
        FROM chunks
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# ============================================================
# METİN NORMALİZASYONU
# ============================================================

def normalize_text(text):
    """
    Metni karşılaştırma için normalize eder.
    """

    text = text.lower()

    text = (
        text
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOKENLARA AYIR
# ============================================================

def tokenize(text):
    """
    Metni anlamlı kelimelere ayırır.
    """

    normalized = normalize_text(
        text
    )

    return set(
        word
        for word in normalized.split()
        if len(word) >= 2
    )


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query,
    embedding_client,
    top_k=3
):
    """
    Soruya en alakalı chunkları getirir.

    Kullanılan sinyaller:

    1. Embedding similarity
    2. Kaynak adı eşleşmesi
    3. Chunk içindeki kelime eşleşmesi
    4. Soruya uygun kaynak avantajı
    """

    # --------------------------------------------------------
    # SORU EMBEDDING
    # --------------------------------------------------------

    response = (
        embedding_client
        .generate_embedding(query)
    )

    query_embedding = (
        response
        .data[0]
        .embedding
    )

    # --------------------------------------------------------
    # CHUNKLARI AL
    # --------------------------------------------------------

    rows = get_chunks()

    if not rows:
        return []

    # --------------------------------------------------------
    # SORU KELİMELERİ
    # --------------------------------------------------------

    query_lower = normalize_text(
        query
    )

    query_tokens = tokenize(
        query
    )

    # --------------------------------------------------------
    # KAYNAKLARI BUL
    # --------------------------------------------------------

    sources = sorted(
        set(
            row[1]
            for row in rows
        )
    )

    # --------------------------------------------------------
    # SORUNUN BELGE ADIYLA EŞLEŞMESİ
    # --------------------------------------------------------

    source_matches = {}

    for source in sources:

        source_normalized = normalize_text(
            Path_stem(source)
        )

        source_tokens = tokenize(
            source_normalized
        )

        matches = (
            query_tokens
            .intersection(
                source_tokens
            )
        )

        source_matches[source] = matches

    # --------------------------------------------------------
    # EN GÜÇLÜ KAYNAĞI BUL
    # --------------------------------------------------------

    preferred_source = None
    best_source_match_count = 0

    for source, matches in source_matches.items():

        if len(matches) > best_source_match_count:

            best_source_match_count = (
                len(matches)
            )

            preferred_source = source

    # --------------------------------------------------------
    # GENEL KONU KELİMELERİ
    # --------------------------------------------------------

    keyword_groups = {

        "cv.pdf": [
            "cv",
            "ozgecmis",
            "egitim",
            "education",
            "universite",
            "university",
            "degree",
            "bsc",
            "mezuniyet",
            "staj",
            "internship",
            "deneyim",
            "experience",
            "beceri",
            "skills",
            "proje",
            "projects"
        ],

        "ticket.docx": [
            "ticket",
            "sistem",
            "system",
            "chatbot",
            "faq",
            "ogrenci",
            "student",
            "ogretmen",
            "teacher",
            "talep",
            "destek",
            "support",
            "akademik",
            "academic"
        ],

        "foundry_test.txt": [
            "foundry",
            "embedding",
            "rag",
            "sqlite",
            "yerel",
            "local",
            "vektor",
            "vector",
            "model"
        ]
    }

    # --------------------------------------------------------
    # SONUÇLAR
    # --------------------------------------------------------

    results = []

    for row in rows:

        chunk_id = row[0]
        source = row[1]
        chunk = row[2]
        embedding_json = row[3]

        # --------------------------------------------
        # EMBEDDING SIMILARITY
        # --------------------------------------------

        embedding = json.loads(
            embedding_json
        )

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        adjusted_score = similarity

        # --------------------------------------------
        # KAYNAK ADI EŞLEŞMESİ
        # --------------------------------------------

        source_match_count = len(
            source_matches.get(
                source,
                set()
            )
        )

        if source_match_count > 0:

            # Her eşleşen belge adı kelimesi
            # güçlü avantaj sağlar.

            adjusted_score += min(
                source_match_count * 0.30,
                0.60
            )

        # --------------------------------------------
        # TAM KAYNAK AVANTAJI
        # --------------------------------------------

        if (
            preferred_source is not None
            and source == preferred_source
        ):

            adjusted_score += 0.20

        # --------------------------------------------
        # CHUNK KELİME EŞLEŞMESİ
        # --------------------------------------------

        chunk_tokens = tokenize(
            chunk
        )

        chunk_matches = (
            query_tokens
            .intersection(
                chunk_tokens
            )
        )

        keyword_hits = len(
            chunk_matches
        )

        adjusted_score += min(
            keyword_hits * 0.03,
            0.15
        )

        # --------------------------------------------
        # ÖZEL KAYNAK KELİME GRUBU
        # --------------------------------------------

        source_keywords = (
            keyword_groups.get(
                source,
                []
            )
        )

        source_keyword_hits = sum(
            1
            for keyword in source_keywords
            if keyword in query_lower
        )

        if source_keyword_hits > 0:

            adjusted_score += min(
                source_keyword_hits * 0.05,
                0.20
            )

        # --------------------------------------------
        # SONUÇ
        # --------------------------------------------

        results.append(
            {
                "id": chunk_id,
                "source": source,
                "chunk": chunk,
                "score": adjusted_score,
                "original_score": similarity,
                "source_matches": list(
                    source_matches.get(
                        source,
                        set()
                    )
                ),
                "keyword_hits": keyword_hits
            }
        )

    # --------------------------------------------------------
    # SIRALA
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# DOSYA ADI STEM
# ============================================================

def Path_stem(filename):
    """
    Dosya uzantısını kaldırır.

    Örnek:

    Help Meet Projesi.pdf
    →
    Help Meet Projesi
    """

    if "." in filename:

        return filename.rsplit(
            ".",
            1
        )[0]

    return filename


# ============================================================
# TERMINAL TESTİ İÇİN SEARCH
# ============================================================

def search(
    query,
    top_k=3
):
    """
    Terminal üzerinden retrieval testi.
    """

    print(
        "1. Foundry Local başlatılıyor..."
    )

    config = Configuration(
        app_name="LocalRAG"
    )

    try:

        FoundryLocalManager.initialize(
            config
        )

    except Exception:

        pass

    manager = (
        FoundryLocalManager.instance
    )

    print(
        "2. Embedding modeli alınıyor..."
    )

    model = (
        manager
        .catalog
        .get_model(
            "qwen3-embedding-0.6b"
        )
    )

    if not model.is_loaded:

        model.load()

    client = (
        model
        .get_embedding_client()
    )

    print(
        "3. Soru embedding'e dönüştürülüyor..."
    )

    results = retrieve(
        query,
        client,
        top_k=top_k
    )

    print(
        "4. SQLite'taki chunklar okunuyor..."
    )

    if model.is_loaded:

        model.unload()

    return results


# ============================================================
# TERMINAL TESTİ
# ============================================================

if __name__ == "__main__":

    query = "Help Meet Projesi nedir?"

    print(
        f"\nSORU: {query}\n"
    )

    results = search(
        query,
        top_k=10
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"===== SONUÇ {i} ====="
        )

        print(
            f"Kaynak: "
            f"{result['source']}"
        )

        print(
            f"Benzerlik: "
            f"{result['score']:.4f}"
        )

        print(
            f"Orijinal similarity: "
            f"{result['original_score']:.4f}"
        )

        print(
            f"Kaynak eşleşmeleri: "
            f"{result['source_matches']}"
        )

        print(
            f"Kelime eşleşmeleri: "
            f"{result['keyword_hits']}"
        )

        print(
            result["chunk"]
        )

        print()