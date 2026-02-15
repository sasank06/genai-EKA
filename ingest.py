from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)


    print("Extracted text preview:\n")
    print(text[:1000])
    print(f"Total chunks created: {len(chunks)}")
    print("\nFirst chunk preview:\n")
    print(chunks[0])