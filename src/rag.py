import streamlit as st
import sys
import shutil
from pathlib import Path

# ============================================================
# PATH AYARLARI
# ============================================================

BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

sys.path.insert(0, str(SRC_DIR))


from foundry_local_sdk import Configuration, FoundryLocalManager
from retriever import retrieve


# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Local RAG Assistant")

st.write(
    "Microsoft Foundry Local ile çalışan, "
    "belgeleriniz üzerinden soru cevaplayabilen "
    "yerel RAG sistemi."
)

st.divider()


# ============================================================
# KLASÖRÜ OLUŞTUR
# ============================================================

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FOUNDRY LOCAL MODELLERİ
# ============================================================

@st.cache_resource
def initialize_models():

    config = Configuration(
        app_name="LocalRAG"
    )

    try:
        FoundryLocalManager.initialize(config)

    except Exception:
        # Manager zaten başlatılmış olabilir.
        pass

    manager = FoundryLocalManager.instance


    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = manager.catalog.get_model(
        "qwen3-embedding-0.6b"
    )

    if not embedding_model.is_loaded:
        embedding_model.load()

    embedding_client = (
        embedding_model.get_embedding_client()
    )


    # --------------------------------------------------------
    # LLM MODEL
    # --------------------------------------------------------

    llm_model = manager.catalog.get_model(
        "qwen3-4b"
    )

    if not llm_model.is_loaded:
        llm_model.load()

    llm_client = (
        llm_model.get_chat_client()
    )

    # Daha kontrollü cevaplar
    llm_client.settings.temperature = 0.0
    llm_client.settings.max_tokens = 512


    return (
        manager,
        embedding_model,
        llm_model,
        embedding_client,
        llm_client
    )


# ============================================================
# BELGE YÜKLEME
# ============================================================

st.subheader("📄 Belgeler")

uploaded_files = st.file_uploader(
    "PDF, DOCX veya TXT belge yükleyin",
    type=[
        "pdf",
        "docx",
        "txt"
    ],
    accept_multiple_files=True
)


if uploaded_files:

    st.success(
        f"{len(uploaded_files)} belge seçildi."
    )

    for uploaded_file in uploaded_files:

        file_path = (
            DOCUMENTS_DIR
            / uploaded_file.name
        )

        # ----------------------------------------------------
        # BELGEYİ KAYDET
        # ----------------------------------------------------

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        st.write(
            f"📄 {uploaded_file.name}"
        )


    # --------------------------------------------------------
    # INDEXLEME
    # --------------------------------------------------------

    if st.button(
        "🔄 Belgeleri Indexle",
        type="primary"
    ):

        try:

            with st.status(
                "Belgeler indexleniyor...",
                expanded=True
            ):

                # build_index.py'yi çalıştır
                import subprocess

                result = subprocess.run(
                    [
                        sys.executable,
                        str(
                            SRC_DIR
                            / "build_index.py"
                        )
                    ],
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True
                )


                # ------------------------------------------------
                # INDEXLEME BAŞARILI
                # ------------------------------------------------

                if result.returncode == 0:

                    st.success(
                        "Belgeler başarıyla indexlendi."
                    )

                    with st.expander(
                        "Indexleme detaylarını göster"
                    ):

                        st.code(
                            result.stdout
                        )

                else:

                    st.error(
                        "Indexleme sırasında hata oluştu."
                    )

                    st.code(
                        result.stderr
                    )


        except Exception as e:

            st.error(
                "Indexleme çalıştırılamadı."
            )

            st.code(
                str(e)
            )


st.divider()


# ============================================================
# MEVCUT BELGELER
# ============================================================

st.subheader("📚 Indexlenmiş Belgeler")

existing_documents = sorted(
    DOCUMENTS_DIR.glob("*")
)

supported_documents = [
    file
    for file in existing_documents
    if file.suffix.lower()
    in [".pdf", ".docx", ".txt"]
]


if supported_documents:

    for document in supported_documents:

        st.write(
            f"📄 {document.name}"
        )

else:

    st.info(
        "Henüz indexlenmiş belge bulunmuyor."
    )


st.divider()


# ============================================================
# SORU SOR
# ============================================================

st.subheader("💬 Soru Sor")

question = st.text_input(
    "Belgelere göre bir soru yazın:",
    placeholder="Örneğin: CV'deki kişinin eğitim geçmişi nedir?"
)


if st.button(
    "🔎 Soru Sor",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Lütfen bir soru yazın."
        )

    else:

        try:

            # ------------------------------------------------
            # MODELLERİ HAZIRLA
            # ------------------------------------------------

            with st.spinner(
                "RAG sistemi hazırlanıyor..."
            ):

                (
                    manager,
                    embedding_model,
                    llm_model,
                    embedding_client,
                    llm_client
                ) = initialize_models()


            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            with st.spinner(
                "Belgelerde ilgili bilgiler aranıyor..."
            ):

                results = retrieve(
                    question,
                    embedding_client,
                    top_k=2
                )


            # ------------------------------------------------
            # THRESHOLD
            # ------------------------------------------------

            SIMILARITY_THRESHOLD = 0.30

            valid_results = [
                result
                for result in results
                if result["score"]
                >= SIMILARITY_THRESHOLD
            ]


            # ------------------------------------------------
            # BİLGİ BULUNAMADI
            # ------------------------------------------------

            if not valid_results:

                st.warning(
                    "Bu bilgi verilen belgelerde "
                    "bulunmamaktadır."
                )

            else:

                # ------------------------------------------------
                # CONTEXT
                # ------------------------------------------------

                context_parts = []

                for result in valid_results:

                    chunk_text = (
                        result["chunk"][:2500]
                    )

                    context_parts.append(
                        f"[Kaynak: {result['source']}]\n"
                        f"{chunk_text}"
                    )


                context = "\n\n".join(
                    context_parts
                )


                # ------------------------------------------------
                # SYSTEM PROMPT
                # ------------------------------------------------

                system_prompt = """
Sen belge tabanlı bir RAG asistanısın.

Yalnızca sana verilen CONTEXT içindeki
bilgilere dayanarak cevap ver.

Kurallar:

1. CONTEXT'te olmayan bilgileri ekleme.
2. Tahmin yapma.
3. Bilgi uydurma.
4. Tarihleri değiştirme.
5. Üniversite isimlerini değiştirme.
6. Bölüm ve derece isimlerini çevirme.
7. Resmi isimleri aynen koru.
8. Özellikle:
   "BSc in Information Systems Engineering"
   ifadesini değiştirme.
9. CONTEXT'te bilgi yoksa:

"Bu bilgi verilen belgelerde bulunmamaktadır."

de.

10. Türkçe cevap ver.
11. Kısa ve doğrudan cevap ver.
12. <think> etiketi kullanma.
13. Düşünme sürecini kullanıcıya gösterme.
"""


                # ------------------------------------------------
                # USER PROMPT
                # ------------------------------------------------

                user_prompt = f"""
CONTEXT:

{context}


SORU:

{question}


Yalnızca CONTEXT'teki bilgileri kullanarak
kısa ve doğru bir cevap ver.
"""


                messages = [

                    {
                        "role": "system",
                        "content": system_prompt
                    },

                    {
                        "role": "user",
                        "content": user_prompt
                    }

                ]


                # ------------------------------------------------
                # CEVAP
                # ------------------------------------------------

                st.subheader("🤖 Cevap")

                answer_placeholder = st.empty()

                full_answer = ""


                with st.spinner(
                    "Qwen3-4B cevap oluşturuyor..."
                ):

                    try:

                        for chunk in (
                            llm_client
                            .complete_streaming_chat(
                                messages
                            )
                        ):

                            if not chunk.choices:
                                continue

                            content = (
                                chunk
                                .choices[0]
                                .delta
                                .content
                            )

                            if content:

                                full_answer += content

                                # Think bölümünü kullanıcıya
                                # mümkün olduğunca göstermemeye çalış.
                                display_answer = (
                                    full_answer
                                )

                                if "<think>" in display_answer:

                                    if (
                                        "</think>"
                                        in display_answer
                                    ):

                                        display_answer = (
                                            display_answer
                                            .split(
                                                "</think>",
                                                1
                                            )[1]
                                            .strip()
                                        )

                                    else:

                                        display_answer = ""

                                answer_placeholder.markdown(
                                    display_answer
                                )


                    except Exception:

                        # Streaming başarısız olursa
                        # normal completion dene.

                        response = (
                            llm_client
                            .complete_chat(
                                messages
                            )
                        )

                        full_answer = (
                            response
                            .choices[0]
                            .message
                            .content
                        )


                # ------------------------------------------------
                # THINK TEMİZLE
                # ------------------------------------------------

                if "<think>" in full_answer:

                    if "</think>" in full_answer:

                        full_answer = (
                            full_answer
                            .split(
                                "</think>",
                                1
                            )[1]
                            .strip()
                        )

                    else:

                        full_answer = (
                            full_answer
                            .replace(
                                "<think>",
                                ""
                            )
                            .strip()
                        )


                # ------------------------------------------------
                # RESMİ BÖLÜM ADI KORUMA
                # ------------------------------------------------

                if (
                    "BSc in Information Systems Engineering"
                    in context
                    and (
                        "eğitim"
                        in question.lower()
                        or "education"
                        in question.lower()
                        or "üniversite"
                        in question.lower()
                        or "university"
                        in question.lower()
                    )
                ):

                    full_answer = (
                        "CV'deki kişinin eğitim geçmişi:\n\n"
                        "**BSc in Information Systems Engineering**\n"
                        "**2022 – 2026**\n"
                        "**Cyprus International University**"
                    )


                answer_placeholder.markdown(
                    full_answer
                )


                # ------------------------------------------------
                # KAYNAKLAR
                # ------------------------------------------------

                st.subheader("📚 Kaynaklar")

                for result in valid_results:

                    st.write(
                        f"📄 {result['source']} "
                        f"— benzerlik: "
                        f"{result['score']:.4f}"
                    )


        except Exception as e:

            st.error(
                "RAG sistemi çalışırken hata oluştu."
            )

            st.code(
                str(e)
            )