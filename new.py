import streamlit as st
import os
import requests
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY", "K5QN2Uig3Xd1ihGAVkCyrGN5MIWyBHjp")
API_URL = "https://api.mistral.ai/v1/chat/completions"

# Define file paths
FILE_PATH = os.path.join(os.path.dirname(__file__), "output.txt")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.pkl")

# Load text file
def load_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
        return None

# Initialize FAISS
def initialize_vectorstore():
    if "vectorstore" not in st.session_state:
        # Initialize embeddings first
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if os.path.exists(FAISS_INDEX_PATH):
            with open(FAISS_INDEX_PATH, "rb") as f:
                st.session_state.vectorstore = pickle.load(f)
            st.success("FAISS index loaded from disk.")
        else:
            text = load_text_file(FILE_PATH)
            if not text:
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            doc_splits = text_splitter.create_documents([text])

            vectorstore = FAISS.from_documents(doc_splits, st.session_state.embeddings)

            with open(FAISS_INDEX_PATH, "wb") as f:
                pickle.dump(vectorstore, f)

            st.session_state.vectorstore = vectorstore
            st.session_state.docs = doc_splits
            st.success("Text document loaded into FAISS!")

# Query Mistral
def query_mistral_with_context(user_prompt):
    if "vectorstore" not in st.session_state:
        return "No document has been loaded yet."

    if user_prompt not in st.session_state:
        st.session_state[user_prompt] = st.session_state.embeddings.embed_query(user_prompt)

    query_embedding = st.session_state[user_prompt]
    st.session_state.vectorstore.index.nprobe = 10  
    similar_docs = st.session_state.vectorstore.similarity_search_by_vector(query_embedding, k=3)

    context = "\n\n".join([doc.page_content[:500] for doc in similar_docs])  

    if not context:
        return "I couldn't find relevant information."

    prompt = f"""
    You are an AI assistant that answers questions **only** using the provided context.
    If the answer is not in the context, reply: "I don't know based on the given document."

    Context:
    {context}

    Question: {user_prompt}
    """

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": "open-mistral-7b", "messages": [{"role": "user", "content": prompt}]}

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("choices")[0].get("message").get("content")
    else:
        st.error(f"Error querying Mistral API: {response.text}")
        return None

# Initialize FAISS
initialize_vectorstore()

# Streamlit UI
st.title("Text Document Query Assistant")
prompt = st.text_input("Enter your query:")

if st.button("Query Mistral"):
    if prompt:
        with st.spinner("Querying..."):
            response = query_mistral_with_context(prompt)
            if response:
                st.success("Query completed!")
                st.write(response)
    else:
        st.warning("Please enter a query.")
