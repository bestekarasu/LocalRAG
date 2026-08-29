from foundry_local_sdk import Configuration, FoundryLocalManager

print("1. Foundry Local başlatılıyor...")

config = Configuration(app_name="LocalRAG")
FoundryLocalManager.initialize(config)

manager = FoundryLocalManager.instance

print("2. Manager hazır.")

model = manager.catalog.get_model("qwen3-embedding-0.6b")

print(f"3. Embedding modeli bulundu: {model.alias}")

print("4. Embedding modeli indiriliyor/kontrol ediliyor...")

model.download(
    lambda progress: print(
        f"\rİndirme: {progress:.1f}%",
        end="",
        flush=True
    )
)

print("\n5. Embedding modeli hazır.")

print("6. Embedding modeli yükleniyor...")
model.load()

print("7. Embedding modeli yüklendi.")

print("8. Embedding client oluşturuluyor...")
client = model.get_embedding_client()

print("9. Embedding client hazır.")

text = "Microsoft Foundry Local yapay zeka modellerini yerel olarak çalıştırmayı sağlar."

print("10. Metin embedding'e dönüştürülüyor...")

response = client.generate_embedding(text)

embedding = response.data[0].embedding

print("11. Embedding oluşturuldu.")
print(f"Embedding tipi: {type(embedding)}")
print(f"Embedding boyutu: {len(embedding)}")
print(f"İlk 10 değer: {embedding[:10]}")

model.unload()

print("12. Model bellekten çıkarıldı.")
print("EMBEDDING TEST BAŞARILI!")