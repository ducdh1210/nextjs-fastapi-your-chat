from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os
from dotenv import load_dotenv
from operator import itemgetter


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

prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""

prompt = ChatPromptTemplate.from_template(prompt)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    | StrOutputParser()
)
# rag_chain = prompt | llm | StrOutputParser()

rag_chain_with_source = RunnableParallel(
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
).assign(answer=rag_chain)

runnable = rag_chain_with_source
