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

        # =====================================================
        # 1. FOUNDRY LOCAL
        # =====================================================

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


        # =====================================================
        # 2. MODEL
        # =====================================================

        st.write("3. Qwen3-4B modeli alınıyor...")

        model = manager.catalog.get_model(
            "qwen3-4b"
        )

        st.write(
            f"Model: qwen3-4b"
        )

        st.write(
            f"Cached: {model.is_cached}"
        )

        st.write(
            f"Loaded: {model.is_loaded}"
        )


        # =====================================================
        # 3. MODELİ İNDİR
        # =====================================================

        if not model.is_cached:

            st.write(
                "4. Model SDK cache'ine indiriliyor..."
            )

            model.download(
                lambda _progress: None
            )

            st.success(
                "Model indirildi."
            )

        else:

            st.write(
                "4. Model zaten SDK cache'inde."
            )


        # =====================================================
        # 4. MODELİ YÜKLE
        # =====================================================

        if not model.is_loaded:

            st.write(
                "5. Model yükleniyor..."
            )

            model.load()

        st.success(
            "Qwen3-4B modeli hazır."
        )


        # =====================================================
        # 5. CHAT CLIENT
        # =====================================================

        client = model.get_chat_client()

        client.settings.temperature = 0.0
        client.settings.max_tokens = 64


        # =====================================================
        # 6. TEST SORUSU
        # =====================================================

        messages = [
            {
                "role": "user",
                "content": "Sadece Merhaba yaz."
            }
        ]


        st.write(
            "6. complete_chat çağrılıyor..."
        )


        # Streaming YOK
        response = client.complete_chat(
            messages
        )


        # =====================================================
        # 7. CEVAP
        # =====================================================

        answer = (
            response
            .choices[0]
            .message
            .content
        )


        # =====================================================
        # 8. THINK TEMİZLE
        # =====================================================

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


        # =====================================================
        # 9. SONUÇ
        # =====================================================

        st.success(
            "LLM başarıyla çalıştı!"
        )

        st.subheader(
            "Cevap"
        )

        st.write(
            answer
        )


    except Exception as e:

        st.error(
            "LLM testi başarısız."
        )

        st.code(
            repr(e)
        )
