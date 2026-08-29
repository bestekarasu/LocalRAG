from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from document_loader import load_document
from chunker import chunk_text
from database import create_database, insert_chunk


DOCUMENTS_DIR = Path("data/documents")


print("1. Foundry Local başlatılıyor...")

config = Configuration(app_name="LocalRAG")

FoundryLocalManager.initialize(config)

manager = FoundryLocalManager.instance

print("2. Manager hazır.")

print("3. Embedding modeli alınıyor...")

model = manager.catalog.get_model("qwen3-embedding-0.6b")

print("4. Embedding modeli indiriliyor/kontrol ediliyor...")

model.download(
    lambda progress: print(
        f"\rİndirme: {progress:.1f}%",
        end="",
        flush=True
    )
)

print("\n5. Embedding modeli yükleniyor...")

model.load()

print("6. Embedding modeli hazır.")

embedding_client = model.get_embedding_client()


# Desteklenen belgeleri bul
supported_extensions = {".txt", ".pdf", ".docx"}

documents = [
    file
    for file in DOCUMENTS_DIR.iterdir()
    if file.is_file() and file.suffix.lower() in supported_extensions
]

print(f"7. Bulunan belge sayısı: {len(documents)}")

if not documents:
    print("data/documents klasöründe desteklenen belge bulunamadı.")
    model.unload()
    raise SystemExit


print("8. SQLite veritabanı hazırlanıyor...")

create_database()


total_chunks = 0


for document_path in documents:

    print("\n==============================")
    print(f"BELGE: {document_path.name}")
    print("==============================")

    print("9. Belge okunuyor...")

    text = load_document(document_path)

    print(
        f"Belge okundu. Karakter sayısı: {len(text)}"
    )

    print("10. Belge chunk'lara ayrılıyor...")

    chunks = chunk_text(text)

    print(
        f"Toplam chunk sayısı: {len(chunks)}"
    )

    print("11. Chunk embeddingleri oluşturuluyor...")

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"   Chunk {index}/{len(chunks)} işleniyor..."
        )

        response = embedding_client.generate_embedding(chunk)

        embedding = response.data[0].embedding

        insert_chunk(
            source=document_path.name,
            chunk=chunk,
            embedding=embedding
        )

        total_chunks += 1


print("\n12. Tüm belgeler işlendi.")

print(f"Toplam chunk sayısı: {total_chunks}")


model.unload()

print("13. Embedding modeli bellekten çıkarıldı.")

print("\n==============================")
print("INDEXLEME BAŞARILI!")
print("==============================")