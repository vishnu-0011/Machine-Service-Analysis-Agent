from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("machine_service.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Instantiate Qwen3:4B LLM from Ollama for RAG
llm = OllamaLLM(model="qwen3:4b")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        # Use Problem_Type and Service_Status as content, others as metadata
        page_content = f"{row['Problem_Type']} - {row['Service_Status']}"
        metadata = {
            "cost": row["Cost"],
            "hours": row["Hours"],
            "date": row["Date"],
            "machine_id": row["Machine_ID"]
        }
        document = Document(
            page_content=page_content,
            metadata=metadata,
            id=str(i)
        )
        print(f"Adding document: {page_content} | Metadata: {metadata}")
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    batch_size = 5000  # ChromaDB max is 5461, so use 5000 for safety
    total_docs = len(documents)
    print(f"Adding {total_docs} documents to the vector store in batches...")
    for start in range(0, total_docs, batch_size):
        end = min(start + batch_size, total_docs)
        print(f"Adding documents {start} to {end-1}")
        vector_store.add_documents(documents=documents[start:end], ids=ids[start:end])
    print("All documents added successfully.")
    

retriever = vector_store.as_retriever(search_kwargs={"k": 100})

# Example: Retrieval-Augmented Generation (RAG) with Qwen3:4B
from langchain_core.runnables import RunnableSequence

# Compose a simple RAG pipeline: retrieve docs, then generate answer
rag_chain = RunnableSequence(
    retriever | (lambda docs: "\n".join([d.page_content for d in docs])) | llm
)

# Example usage:
# question = "What do users say about the service quality?"
# answer = rag_chain.invoke(question)
# print("RAG answer:", answer)