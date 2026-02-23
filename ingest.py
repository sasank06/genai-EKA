from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return chunks

load_dotenv()

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("faiss_index")

    print("Vector store saved locally.")

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    create_vector_store(chunks)

    # print("Extracted text preview:\n")
    # print(text[:1000])
    # print(f"Total chunks created: {len(chunks)}")
    # print("\nFirst chunk preview:\n")
    # print(chunks[0])