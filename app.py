import streamlit as st
from ingest import extract_text_from_pdf, chunk_text, create_vector_store
from rag import generate_answer

st.title("📄 GenAI Enterprise Knowledge Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    vectorstore = create_vector_store(chunks)

    st.success("Document processed!")

    query = st.text_input("Ask a question")

    if query:
        docs = vectorstore.similarity_search(query, k=3)
        answer = generate_answer(query, docs)

        st.write("### Answer")
        st.write(answer)