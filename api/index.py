import json
from operator import itemgetter
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.vectorstores import Chroma
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import DistanceStrategy

import os
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

from sse_starlette import EventSourceResponse

from api.chain import runnable


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


@app.post("/api/streaming", response_class=EventSourceResponse)
async def streaming(
    request: ChatRequest,
):
    try:
        return EventSourceResponse(
            generate_response(request=request),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_response(request: ChatRequest):
    async for event in runnable.astream_events(
        {"question": request.message},
        version="v1",
    ):
        kind = event.get("event")
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            yield json.dumps({"chunk": content, "message_type_id": "streaming"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
