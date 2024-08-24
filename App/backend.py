from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader # For individual PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import io
import ollama
from bs4 import BeautifulSoup as bs

# Function to load and split a single PDF document from a BytesIO object
def load_and_split_documents(pdf_file):
    pdf_path = "/tmp/temp.pdf"  # Temporary file path
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        raise ValueError("No documents were found in the provided PDF.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    if not splits:
        raise ValueError("Document splitting resulted in empty chunks.")
    
    return splits

def load_and_split_websites(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    return splits

# Function to initialize vectorstore
def initialize_vectorstore(documents, model="nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=model)
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, ids=doc_ids)
    return vectorstore

# Function to query the vectorstore
def query_vectorstore(retriever, query, top=10):
    retrieved_docs = retriever.invoke(query, top_k=top)
    context = ' '.join([doc.page_content for doc in retrieved_docs])  # Use `doc.page_content`
    return context

# Function to respond to a query
def respond_to_query(lst_messages, retriever, model="llama3", use_knowledge=False):
    q = lst_messages[-1]["content"]
    context = query_vectorstore(retriever, q)
    
    if use_knowledge:
        prompt = f"Give the most accurate answer using your knowledge and the following additional information: \n{context}"
    else:
        prompt = f"Give the most accurate answer using only the following information: \n{context}"
    
    response = ollama.chat(model=model, messages=[{"role": "system", "content": prompt}] + lst_messages, stream=True)
    full_response = ""
    for res in response:
        chunk = res["message"]["content"]
        full_response += chunk
    return full_response