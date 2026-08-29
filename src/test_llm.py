from foundry_local_sdk import Configuration, FoundryLocalManager

print("1. Foundry Local başlatılıyor...")

config = Configuration(app_name="LocalRAG")
FoundryLocalManager.initialize(config)

manager = FoundryLocalManager.instance

print("2. Foundry Local Manager hazır.")

model = manager.catalog.get_model("qwen3-8b")

print(f"3. Model bulundu: {model.alias}")

print("4. Model kontrol ediliyor / indiriliyor...")

model.download(
    lambda progress: print(
        f"\rİndirme: {progress:.1f}%",
        end="",
        flush=True
    )
)

print("\n5. Model indirildi/hazır.")

print("6. Qwen3-8B yükleniyor...")
model.load()

print("7. Model başarıyla yüklendi.")

client = model.get_chat_client()

messages = [
    {
        "role": "user",
        "content": "Merhaba! Kendini kısaca tanıt."
    }
]

print("8. Modele soru gönderiliyor...")

response = client.complete_chat(messages)

print("\n===== MODEL CEVABI =====")
print(response.choices[0].message.content)
print("========================")

model.unload()

print("\n9. Model bellekten çıkarıldı.")
print("TEST BAŞARILI!")