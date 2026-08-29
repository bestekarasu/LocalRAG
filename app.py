import sys
from pathlib import Path

import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager

PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src"
DOCUMENTS_DIR = PROJECT_DIR / "data" / "documents"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chunker import chunk_text
from database import delete_source, get_chunk_count, get_source_names, insert_chunk
from document_loader import load_document
from retriever import retrieve

st.set_page_config(page_title="Local RAG", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp { background: #0b1020; }
[data-testid="stSidebar"] { background: #11182c; border-right: 1px solid #25304a; }
.hero { padding: 1.1rem 0 .5rem; }
.eyebrow { color: #8ba4ff; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.hero h1 { margin: .2rem 0; font-size: 2.25rem; letter-spacing: -.04em; }
.muted { color: #aab5cf; }
.metric-card { background: #121a30; border: 1px solid #293653; border-radius: 14px; padding: 14px 16px; }
.source-card { background: #121a30; border: 1px solid #293653; border-radius: 10px; padding: 10px 12px; margin: 7px 0; }
.status-dot { color: #4ade80; font-size: .82rem; }
div[data-testid="stChatMessage"] { border: 1px solid #25304a; border-radius: 14px; padding: .35rem .7rem; }
.stButton > button { border-radius: 9px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_manager():
    FoundryLocalManager.initialize(Configuration(app_name="LocalRAG"))
    return FoundryLocalManager.instance


def load_embedding_model(manager):
    model = manager.catalog.get_model("qwen3-embedding-0.6b")
    if not model.is_cached:
        model.download(lambda _progress: None)
    if not model.is_loaded:
        model.load()
    return model, model.get_embedding_client()


def load_chat_model(manager):
    model = manager.catalog.get_model("qwen3-4b")
    if not model.is_cached:
        model.download(lambda _progress: None)
    if not model.is_loaded:
        model.load()
    client = model.get_chat_client()
    client.settings.temperature = 0.0
    client.settings.max_tokens = 256
    return model, client


def release_model(model):
    try:
        if model is not None and model.is_loaded:
            model.unload()
    except Exception:
        pass


def clean_answer(answer):
    if "<think>" not in answer:
        return answer.strip()
    if "</think>" in answer:
        return answer.split("</think>", 1)[1].strip()
    return answer.replace("<think>", "").strip()


def index_files(files):
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    indexed = []
    try:
        manager = get_manager()
        model, client = load_embedding_model(manager)
        for uploaded_file in files:
            destination = DOCUMENTS_DIR / Path(uploaded_file.name).name
            destination.write_bytes(uploaded_file.getvalue())
            text = load_document(destination)
            chunks = chunk_text(text)
            if not chunks:
                raise ValueError(f"{uploaded_file.name} içinde okunabilir metin bulunamadı.")
            delete_source(destination.name)
            for chunk in chunks:
                response = client.generate_embedding(chunk)
                insert_chunk(destination.name, chunk, response.data[0].embedding)
            indexed.append((destination.name, len(chunks)))
    finally:
        release_model(model)
    return indexed


def answer_question(question):
    embedding_model = None
    chat_model = None
    try:
        manager = get_manager()
        embedding_model, embedding_client = load_embedding_model(manager)
        results = retrieve(question, embedding_client, top_k=3)
        release_model(embedding_model)
        embedding_model = None

        sources = [item for item in results if item["score"] >= 0.18]
        if not sources:
            return "Bu bilgi eklenmiş belgelerde bulunamadı.", []

        context = "\n\n".join(
            f"[Kaynak: {item['source']}]\n{item['chunk'][:1600]}"
            for item in sources
        )
        messages = [
            {"role": "system", "content": (
                "Yerel belgeler için bir soru-cevap asistanısın. Yalnızca verilen bağlama dayan. "
                "Yanıt bulunmuyorsa bunu açıkça söyle. Türkçe, kısa ve doğrudan cevap ver; "
                "düşünme sürecini gösterme."
            )},
            {"role": "user", "content": f"BAĞLAM:\n{context}\n\nSORU:\n{question}"},
        ]
        chat_model, chat_client = load_chat_model(manager)
        response = chat_client.complete_chat(messages)
        answer = clean_answer(response.choices[0].message.content or "")
        return answer or "Bu bilgi eklenmiş belgelerde bulunamadı.", sources
    finally:
        release_model(embedding_model)
        release_model(chat_model)


if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### Belge merkezi")
    st.caption("PDF, DOCX ve TXT dosyalarını ekleyip tek seferde indeksleyin.")
    uploads = st.file_uploader("Belgeleri seçin", type=["pdf", "docx", "txt"], accept_multiple_files=True, label_visibility="collapsed")
    if st.button("Belgeleri indekse ekle", use_container_width=True, disabled=not uploads):
        with st.spinner("Belgeler hazırlanıyor ve indeksleniyor..."):
            try:
                indexed_files = index_files(uploads)
                total = sum(count for _, count in indexed_files)
                st.success(f"{len(indexed_files)} belge, {total} parça olarak eklendi.")
            except Exception as error:
                st.error("Belgeler indekslenemedi.")
                st.code(str(error))

    st.divider()
    st.markdown("#### İndeks durumu")
    try:
        document_names = get_source_names()
        chunk_count = get_chunk_count()
    except Exception:
        document_names, chunk_count = [], 0
    st.metric("İndekslenen belge", len(document_names))
    st.metric("Bilgi parçası", chunk_count)
    if document_names:
        with st.expander("İndekslenen belgeler", expanded=False):
            for name in document_names:
                st.write(f"• {name}")

    st.divider()
    if st.button("Sohbeti temizle", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Yerel bilgi asistanı</div>
  <h1>Belgelerinizle net cevaplar bulun.</h1>
  <p class="muted">Sorular, yalnızca indekslediğiniz belgelerin içeriğine göre yanıtlanır.</p>
</div>
""", unsafe_allow_html=True)

left, middle, right = st.columns(3)
with left:
    st.markdown(f"<div class='metric-card'><span class='muted'>Belgeler</span><br><b>{len(document_names)}</b></div>", unsafe_allow_html=True)
with middle:
    st.markdown(f"<div class='metric-card'><span class='muted'>Bilgi parçaları</span><br><b>{chunk_count}</b></div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='metric-card'><span class='status-dot'>● Sistem hazır</span><br><span class='muted'>Qwen3-4B ile yerel çalışma</span></div>", unsafe_allow_html=True)

st.divider()
if not document_names:
    st.info("Başlamak için soldaki belge merkezinden en az bir belge ekleyin.")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
        if item["sources"]:
            with st.expander("Kullanılan kaynaklar"):
                for source in item["sources"]:
                    st.markdown(
                        f"<div class='source-card'><b>{source['source']}</b><br>"
                        f"<span class='muted'>Uygunluk skoru: {source['score']:.2f}</span></div>",
                        unsafe_allow_html=True,
                    )

question = st.chat_input("Belgelere dayalı bir soru sorun")
if question:
    if not document_names:
        st.warning("Önce en az bir belgeyi indekse ekleyin.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Belgeler taranıyor, yanıt hazırlanıyor..."):
                try:
                    answer, sources = answer_question(question)
                    st.markdown(answer)
                    if sources:
                        with st.expander("Kullanılan kaynaklar"):
                            for source in sources:
                                st.markdown(
                                    f"<div class='source-card'><b>{source['source']}</b><br>"
                                    f"<span class='muted'>Uygunluk skoru: {source['score']:.2f}</span></div>",
                                    unsafe_allow_html=True,
                                )
                    st.session_state.history.append({"question": question, "answer": answer, "sources": sources})
                except Exception as error:
                    st.error("Yanıt hazırlanırken bir sorun oluştu.")
                    with st.expander("Teknik ayrıntı"):
                        st.code(str(error))
