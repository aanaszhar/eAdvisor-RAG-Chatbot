import os
import sys
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import uuid

# --- 1. Configuration ---
DATA_PATH = "data"
FAISS_INDEX_PATH = "faiss_index.faiss"
CHILD_DOCS_STORE_PATH = "child_docs.pkl"   # Stores the small chunks
PARENT_DOCS_STORE_PATH = "parent_docs.pkl" # Stores the large chunks

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Parent-Child Chunking Setup ---
PARENT_CHUNK_SIZE = 1200
PARENT_CHUNK_OVERLAP = 250
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

def load_and_chunk_docs():
    """
    Loads PDFs and creates Parent and Child chunks.
    Uses .get_text("text") to be compatible with all pymupdf versions.
    """
    print(f"Loading documents from {DATA_PATH}...")
    parent_docs = [] # List of large "Parent" chunks (for context)
    child_docs = []  # List of small "Child" chunks (for searching)
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.pdf'):
            filepath = os.path.join(DATA_PATH, filename)
            doc_fitz = fitz.open(filepath)
            print(f"  - Processing {filename}...")
            
            for page_num, page in enumerate(doc_fitz):
                page_text = page.get_text("text") 
                if not page_text.strip():
                    continue

                # 1. Create large "Parent" chunks from the page
                parent_chunks_on_page = parent_splitter.split_text(page_text)
                
                for parent_chunk_text in parent_chunks_on_page:
                    parent_id = str(uuid.uuid4())
                    parent_metadata = {
                        "source": filename, 
                        "page": page_num + 1,
                        "parent_id": parent_id
                    }
                    parent_docs.append({
                        "text": parent_chunk_text,
                        "metadata": parent_metadata
                    })
                    
                    # 2. Create small "Child" chunks from that Parent chunk
                    child_chunks_of_parent = child_splitter.split_text(parent_chunk_text)
                    
                    for child_chunk_text in child_chunks_of_parent:
                        child_docs.append({
                            "text": child_chunk_text,
                            "metadata": parent_metadata # Child stores its parent's info
                        })
            
            doc_fitz.close()
            
    print(f"Created {len(parent_docs)} Parent chunks.")
    print(f"Created {len(child_docs)} Child chunks.")
    return parent_docs, child_docs

def create_index(parent_docs, child_docs):
    if not child_docs:
        print("No child documents to index. Exiting.")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts_for_embedding = [doc["text"] for doc in child_docs]
    
    print(f"Embedding {len(texts_for_embedding)} child chunks...")
    embeddings = model.encode(
        texts_for_embedding,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    
    # Create FAISS index from CHILD embeddings
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # --- Save all three components ---
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Create a dictionary for fast lookup of parent docs by ID
    parent_doc_store = {doc["metadata"]["parent_id"]: doc for doc in parent_docs}
    with open(PARENT_DOCS_STORE_PATH, 'wb') as f:
        pickle.dump(parent_doc_store, f)
        
    # Save the child docs list (the index file refers to this list's order)
    with open(CHILD_DOCS_STORE_PATH, 'wb') as f:
        pickle.dump(child_docs, f)
        
    print(f"Successfully saved FAISS index to {FAISS_INDEX_PATH}")
    print(f"Successfully saved Parent doc store to {PARENT_DOCS_STORE_PATH}")
    print(f"Successfully saved Child doc store to {CHILD_DOCS_STORE_PATH}")

if __name__ == "__main__":
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(CHILD_DOCS_STORE_PATH):
        os.remove(CHILD_DOCS_STORE_PATH)
    if os.path.exists(PARENT_DOCS_STORE_PATH):
        os.remove(PARENT_DOCS_STORE_PATH)
        
    parent_docs, child_docs = load_and_chunk_docs()
    create_index(parent_docs, child_docs)