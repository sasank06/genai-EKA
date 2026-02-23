from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ===============================
# Load Embedding Model (once)
# ===============================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===============================
# Load FLAN-T5 Model (once)
# ===============================
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# ===============================
# Retrieval Function
# ===============================
def retrieve_chunks(query):
    vectorstore = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # Step 1: semantic search
    docs = vectorstore.similarity_search(query, k=5)

    # Step 2: simple keyword boost
    if "name" in query.lower():
        docs = sorted(
            docs,
            key=lambda d: "@" in d.page_content,  # header chunk usually contains email
            reverse=True
        )

    return docs[:3]


# ===============================
# Generation Function
# ===============================
def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are extracting information from a document.

If the question asks for a person's name,
look for a full uppercase name at the top of the document.

Answer ONLY with the exact phrase found in the context.
If not found, say: I don't know.


Context:
{context}

Question: {query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=False,
    num_beams=4,
    early_stopping=True
)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


# ===============================
# Main Runner
# ===============================
# if __name__ == "__main__":
#     query = input("Enter your question: ")

#     docs = retrieve_chunks(query)
#     for i, doc in enumerate(docs):
#         print(f"\n--- Chunk {i+1} ---\n")
#         print(doc.page_content)
#     answer = generate_answer(query, docs)

#     print("\nGenerated Answer:\n")
#     print(answer)