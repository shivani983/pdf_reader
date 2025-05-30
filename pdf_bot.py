import pdfplumber
import numpy as np
import requests
import streamlit as st
import chromadb
from chromadb.config import Settings


EURI_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI5Y2ViOTI0MS00MDllLTQ3N2ItOTZhNy1mOTFmOTcyYjEyMWMiLCJlbWFpbCI6InNoaXZhbml2aXJhbmc5ODNAZ21haWwuY29tIiwiaWF0IjoxNzQ0ODE1NDY2LCJleHAiOjE3NzYzNTE0NjZ9.1VE4DJgC6wBDk_bMmt5k_W7SeEl_NbF4iTTw5NWHeRk"
EURI_CHAT_URL = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
EURI_EMBED_URL = "https://api.euron.one/api/v1/euri/alpha/embeddings"


conversation_memory = []
def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_euri_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    res = requests.post(EURI_EMBED_URL, headers=headers, json=payload)

    # 👇 Debugging: Print the full API response if it fails
    if res.status_code != 200:
        st.error("Failed to get embeddings from EURI API.")
        st.json(res.json())  # Show the error in Streamlit
        raise Exception(f"EURI API error: {res.status_code}")

    response_json = res.json()

    if "data" not in response_json:
        st.error("EURI API response missing 'data'. Full response:")
        st.json(response_json)
        raise KeyError("'data' not found in EURI embedding response")

    return [d["embedding"] for d in response_json["data"]]

def build_vector_store(chunks):
    import chromadb
    client = chromadb.Client(Settings(
    chroma_db_impl="duckdb",
    persist_directory=".chromadb"
))

    # ✅ This avoids the "already exists" error
    collection = client.get_or_create_collection(name="pdf_chunks")

    embeddings = get_euri_embeddings(chunks)

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[f"chunk_{i}"]
        )

    return collection


def retrieve_context(question, collection, top_k=3):
    q_embed = get_euri_embeddings([question])[0]
    results = collection.query(query_embeddings=[q_embed], n_results=top_k)
    return "\n\n".join(results["documents"][0])

def ask_euri_with_context(question, context, memory=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions from the given PDF."}
    ]
    if memory:
        messages.extend(memory)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": 0.3
    }

    res = requests.post(EURI_CHAT_URL, headers=headers, json=payload)
    reply = res.json()['choices'][0]['message']['content']
    memory.append({"role": "user", "content": question})
    memory.append({"role": "assistant", "content": reply})
    return reply

# Streamlit UI
st.title("PDF READER with ChromaDB")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_question = st.text_input("Ask a question about the document:")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    full_text = extract_text_from_pdf("temp.pdf")
    chunks = split_text(full_text)
    collection = build_vector_store(chunks)

    st.success("PDF loaded and vector store created.")

    if user_question:
        context = retrieve_context(user_question, collection)
        response = ask_euri_with_context(user_question, context, conversation_memory)
        st.markdown("### Answer")
        st.write(response)