import streamlit as st

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager
)


st.title("Foundry Local - LLM Test")


if st.button("Qwen3-4B Test Et"):

    try:

        st.write("1. Foundry Local başlatılıyor...")

        config = Configuration(
            app_name="LocalRAGTest"
        )

        try:

            FoundryLocalManager.initialize(
                config
            )

        except Exception:

            pass


        manager = FoundryLocalManager.instance

        st.write("2. Manager hazır.")


        model = manager.catalog.get_model(
            "qwen3-4b"
        )

        st.write("3. Model bulundu.")


        if not model.is_loaded:

            st.write(
                "4. Model yükleniyor..."
            )

            model.load()


        st.write(
            "5. Model hazır."
        )


        client = model.get_chat_client()


        client.settings.temperature = 0.0
        client.settings.max_tokens = 256


        st.write(
            "6. Qwen3-4B'ye istek gönderiliyor..."
        )


        response = client.complete_chat(

            [
                {
                    "role": "user",
                    "content": "Sadece Merhaba yaz."
                }
            ]

        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        st.subheader("Cevap")

        st.write(answer)


    except Exception as e:

        st.error(
            "Hata oluştu:"
        )

        st.code(
            str(e)
        )