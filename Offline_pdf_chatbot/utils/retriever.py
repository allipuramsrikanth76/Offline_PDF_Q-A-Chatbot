from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from config import PERSIST_PATH, TOP_K, OLLAMA_MODEL, EMBEDDING_MODEL

def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        PERSIST_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

def get_qa_chain(retriever):
    llm = OllamaLLM(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

if __name__=="__main__":
    retriever = load_retriever()
    qa = get_qa_chain(retriever)
    try:
        response = qa.invoke({"query": "What is the main contribution of the documents?"})
        print(response["result"])
    except Exception as e:
        print(f"QA failed: {e}")
