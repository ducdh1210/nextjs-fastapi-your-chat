from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.vectorstores import Chroma
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate

import bs4
from langchain import hub

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
load_dotenv()

# load vector store
os.environ["PGVECTOR_CONNECTION_STRING"] = (
    f"""postgresql+psycopg://{os.getenv("PGVECTOR_USER")}:{os.getenv("PGVECTOR_PWD")}@{os.getenv("PGVECTOR_HOST")}:{os.getenv("PGVECTOR_PORT")}/{os.getenv("PGVECTOR_DB")}"""
)

vector_store = PGVector(
    embeddings=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL")),
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_CONNECTION_STRING"),
    use_jsonb=True,
)
retriever = vector_store.as_retriever()
print(retriever.invoke("who is John Mearsheimer?"))


class UrlModel(BaseModel):
    url: str


@app.post("/api/scrape")
async def get_vectorstore_from_url(item: UrlModel):
    url = item.url
    global vector_store
    logger.info(f"Received request to scrape URL: {url}")

    try:
        # Log the received URL for debugging
        logger.info(f"Received URL: {url}")

        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        vector_store = PGVector.from_documents(
            documents=document_chunks,
            embedding=OpenAIEmbeddings(),
            collection_name=os.getenv("PGVECTOR_COLLECTION"),
            distance_strategy=DistanceStrategy.COSINE,
            connection=os.environ["PGVECTOR_CONNECTION_STRING"],
            pre_delete_collection=True,
        )
        logger.info("Vector store initialized")

        return {"message": "Vector store initialized"}
    except Exception as e:
        logger.error(f"Error in get_vectorstore_from_url: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if vector_store is None:
        raise HTTPException(status_code=404, detail="Vector store not found")

    try:
        retriever = vector_store.as_retriever()
        print(retriever.invoke("who is John Mearsheimer?"))

        prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question} 

        Context: {context} 

        Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt)
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # response = rag_chain.invoke({"question": request.message})
        response = rag_chain.invoke("who is John Mearsheimer?")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
