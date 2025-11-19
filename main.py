"""
AmbedkarGPT - KalpIT AI Intern Assignment
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_vectordb(speech_path="speech.txt", persist_dir="db"):
    if not os.path.exists(speech_path):
        raise FileNotFoundError(f"{speech_path} not found. Place speech.txt in the same folder.")
    loader = TextLoader(speech_path, encoding="utf-8")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\\n")
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    return vectordb

def build_rag(vectordb, model_name="phi3:mini"):
    retriever = vectordb.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are AmbedkarGPT. Answer using ONLY the context provided below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )
    llm = Ollama(model=model_name)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def chat_loop(rag_chain):
    print("AmbedkarGPT — ask questions about the speech. Type 'exit' to quit.")
    while True:
        q = input("\nYour question: ")
        if q.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        try:
            a = rag_chain.invoke(q)
            print("\nAnswer:\n", a)
        except Exception as e:
            print("Error:", e)

def main():
    print("Building vector store (this may take a minute the first run)...")
    vectordb = build_vectordb()
    vectordb.persist()
    print("Vector store ready. Building RAG pipeline...")
    rag_chain = build_rag(vectordb, model_name="phi3:mini")
    chat_loop(rag_chain)

if __name__ == "__main__":
    main()
