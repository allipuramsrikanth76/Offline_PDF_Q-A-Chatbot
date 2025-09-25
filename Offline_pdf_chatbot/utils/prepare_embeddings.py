import os

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer 
from tqdm import tqdm


from config import PERSIST_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from utils.pdf_loader import load_pdf_text


#pdf convert into docs
def pdfs_to_documents(pdf_paths):
    docs = []

    for p in pdf_paths:
        text = load_pdf_text(p)
        meta = {"source": os.path.basename(p)}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

#chunking Its purpose is to split large documents into smaller, overlapping text chunks
def chunk_documents(documents):
    spliter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    new_docs = []
    for d in documents:
        splits = spliter.split_text(d.page_content)
        for i, s in enumerate(splits):
            new_docs.append(Document(page_content=s, metadata={**d.metadata, "chunk": i}))
    return new_docs


def build_faiss_index(documents, persist_path = PERSIST_PATH, model_name = EMBEDDING_MODEL):
    Path(persist_path).parent.mkdir(parents=True, exist_ok=True)

    #langchain HuggingFaceEmbedding will wrap sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    #build / persist FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(persist_path)
    print(f"Saved FAISS index to {persist_path}")

    return vectorstore



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--pdfs", nargs="+", help="List of PDF paths to ingest", required=True)
    parser.add_argument("--persist", default=PERSIST_PATH)
    args = parser.parse_args()


    docs = pdfs_to_documents(args.pdfs)
    print(f"Loaded {len(docs)} PDF documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    build_faiss_index(chunks, persist_path=args.persist)









