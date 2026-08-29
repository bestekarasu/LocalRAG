import streamlit as st

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager
)


st.set_page_config(
    page_title="Foundry Local Test"
)

st.title("🧪 Foundry Local - LLM Test")

if st.button("Qwen3-4B Test Et"):

    try:

        st.write("1. Foundry Local başlatılıyor...")

        config = Configuration(
            app_name="LocalRAGTest"
        )

        try:
            FoundryLocalManager.initialize(config)
        except Exception:
            pass

        manager = FoundryLocalManager.instance

        st.write("2. Manager hazır.")

        st.write("3. Qwen3-4B modeli alınıyor...")

        model = manager.catalog.get_model(
            "qwen3-4b"
        )

        st.write(
            f"Model bulundu. "
            f"Loaded: {model.is_loaded}"
        )

        if not model.is_loaded:

            st.write("4. Model yükleniyor...")

            model.load()

        st.write("5. Model hazır.")

        client = model.get_chat_client()

        client.settings.temperature = 0.0
        client.settings.max_tokens = 64

        messages = [
            {
                "role": "user",
                "content": "Sadece Merhaba yaz."
            }
        ]

        st.write(
            "6. complete_chat çağrılıyor..."
        )

        response = client.complete_chat(
            messages
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if "<think>" in answer:

            if "</think>" in answer:

                answer = (
                    answer
                    .split("</think>", 1)[1]
                    .strip()
                )

            else:

                answer = (
                    answer
                    .replace("<think>", "")
                    .strip()
                )

        st.success("LLM çalışıyor!")

        st.subheader("Cevap")

        st.write(answer)

    except Exception as e:

        st.error("LLM testi başarısız.")

        st.code(
            repr(e)
        )