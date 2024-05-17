from operator import itemgetter
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.vectorstores import Chroma
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

from api.chain import runnable

# from langchain.chains import create_history_aware_retriever, create_retrieval_chain


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
load_dotenv()


class UrlModel(BaseModel):
    url: str


class ChatRequest(BaseModel):
    message: str


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


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response = runnable.invoke({"question": request.message})
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
