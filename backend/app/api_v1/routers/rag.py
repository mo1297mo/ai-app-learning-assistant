from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from config import settings
from data_models.schemas import UserQuery
from typing import List, Dict, AsyncGenerator
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

import tempfile
import structlog
import json
import requests
import asyncio
import httpx
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI

client = OpenAI(api_key=settings.openai_api_key)

logger = structlog.get_logger()

qa_router = APIRouter()

# Embedding model initialization
model_name = settings.embedding_settings.em_model_name
embedding_dimensions = settings.embedding_settings.embedding_dimensions
model_kwargs = settings.embedding_settings.em_model_kwargs
encode_kwargs = settings.embedding_settings.em_encode_kwargs
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# Callback handlers
stream_callback_handler = AsyncIteratorCallbackHandler()

# Models
chat_model_llama2 = ChatOllama(
    model="llama2",
    verbose=True,
    callback_manager=CallbackManager([stream_callback_handler]),
)

# QA prompt template
template = """[INST] <<SYS>> Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. <</SYS>>
{context}
Question: {question}
Helpful Answer:[/INST]"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Qdrant initialization
def initialize_qdrant(host: str, api_key: str, prefer_grpc: bool):
    qdrant_client = QdrantClient(host=host, api_key=api_key, prefer_grpc=prefer_grpc, https=False)

    def create_collection(collection_name: str):
        try:
            qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} already exists.")
        except Exception:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimensions, distance=Distance.COSINE),
            )
            logger.info(f"Collection {collection_name} is successfully created.")

    create_collection(settings.qdrant_collection_name)
    return qdrant_client

qdrant_client = initialize_qdrant(host=settings.qdrant_host, api_key=settings.qdrant_api_key, prefer_grpc=False)
qdrant_vectordb = Qdrant(qdrant_client, settings.qdrant_collection_name, embedding_model)

def get_qa_chain(model_choice: str):
    if model_choice == "llama2":
        return load_qa_chain(llm=chat_model_llama2, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
    elif model_choice == "gpt":
        return None  # We will handle GPT separately
    else:
        raise ValueError("Unsupported model choice. Choose 'llama2' or 'gpt'.")

# Function to create a Google Calendar event
def create_google_calendar_event(summary, description, start_time, end_time):
    creds = Credentials(
        token=None,
        refresh_token=settings.google_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scopes=["https://www.googleapis.com/auth/calendar.events"]
    )
    service = build("calendar", "v3", credentials=creds)
    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_time, "timeZone": "Europe/Berlin"},
        "end": {"dateTime": end_time, "timeZone": "Europe/Berlin"},
    }
    event = service.events().insert(calendarId="primary", body=event).execute()
    return event

# Function to handle assistant's response for creating a calendar event
async def handle_calendar_event_request(filled_prompt: str):
    assistant_prompt = f"The user wants to create a calendar event. Extract the event details from the following input in JSON format with keys: summary, description, start_time, end_time {filled_prompt}"

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "Extract the event details from the user's input in JSON format with keys: summary, description, start_time, end_time."
            },
            {
                "role": "user",
                "content": assistant_prompt
            }
        ],
        temperature=0.5,
    )

    event_details = response.choices[0].message.content.strip()
    logger.info(f"Event details received from assistant: {event_details}")
    # Assuming the response is in JSON format with keys: summary, description, start_time, end_time
    try:
        event_data = json.loads(event_details)

        summary = event_data.get("summary")
        description = event_data.get("description")
        start_time = parse_date(event_data.get("start_time")).isoformat()
        end_time = parse_date(event_data.get("end_time")).isoformat()

        logger.info(f"Parsed event data: {event_data}")

        create_google_calendar_event(summary, description, start_time, end_time)
        return {"status": "Event created successfully"}
    except json.JSONDecodeError:
        return {"status": "Failed to parse event details"}
    except Exception as e:
        return {"status": f"Failed to create event: {str(e)}"}

@qa_router.post('/upload')
async def upload_file(request: Request, file: UploadFile):
    filename = file.filename
    content = await file.read()

    # Create a temporary file to store the PDF content
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=settings.text_splitter['chunk_size'],
            chunk_overlap=settings.text_splitter['chunk_overlap']
        )
        documents = text_splitter.split_documents(data)
        logger.info(documents[1])
        logger.info(f"Number of Chunks: {len(documents)}")

        # Insert the text chunks into the vector database
        insert_into_vectordb(documents, filename)

    return {"filename": filename, "status": "success"}

@qa_router.post("/ask")
async def query_index(request: Request, input_query: UserQuery):
    question = input_query.question
    model_choice = input_query.model_choice
    qa_chain = get_qa_chain(model_choice)

    relevant_docs = qdrant_vectordb.similarity_search(question, k=1)
    logger.info(f"======{relevant_docs}")
    context = relevant_docs[0].page_content
    filled_prompt = QA_CHAIN_PROMPT.format(question=question, context=context)

    if "deadline" in question.lower() or "calendar" in question.lower():
        event_response = await handle_calendar_event_request(filled_prompt)
        if "status" in event_response and "successfully" in event_response["status"]:
            response_message = "Event created successfully"
        else:
            response_message = event_response.get("status", "Failed to create event")
        return JSONResponse({"answer": response_message})

    if model_choice == "gpt":
        async def stream_response_generator():
            max_retries = 3
            retries = 0
            while retries < max_retries:
                try:
                    async with httpx.AsyncClient() as client:
                        async with client.stream(
                            "POST",
                            "https://api.openai.com/v1/chat/completions",
                            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                            json={
                                "model": "gpt-3.5-turbo",
                                "messages": [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": filled_prompt}
                                ],
                                "stream": True
                            }
                        ) as response:
                            async for chunk in response.aiter_bytes():
                                decoded_chunk = chunk.decode("utf-8")
                                for line in decoded_chunk.splitlines():
                                    if line.startswith("data: "):
                                        data = line[len("data: "):]
                                        if data.strip() == "[DONE]":
                                            return
                                        else:
                                            try:
                                                json_data = json.loads(data)
                                                if "choices" in json_data:
                                                    choice = json_data["choices"][0]
                                                    if "delta" in choice and "content" in choice["delta"]:
                                                        yield choice["delta"]["content"]
                                            except json.JSONDecodeError:
                                                continue
                                await asyncio.sleep(0.1)  # Adjust sleep interval
                    return  # Successfully finished streaming
                except httpx.StreamClosed:
                    retries += 1
                    logger.error(f"Stream was closed unexpectedly. Retry {retries}/{max_retries}.")

            yield "Stream closed unexpectedly after multiple retries."

        return StreamingResponse(
            stream_response_generator(),
            media_type="text/event-stream"
        )
    else:
        response = await qa_chain.acall({"input_documents": relevant_docs, "question": question})
        logger.info(response)  # Log the response to inspect its structure
        return JSONResponse({"answer": response.get('output_text', 'No answer found')})


@qa_router.post("/ask1")
async def query_index_another_approach(request: Request, input_query: UserQuery):
    """This is another function for streaming the response back to the client."""
    question = input_query.question
    model_choice = input_query.model_choice
    qa_chain = get_qa_chain(model_choice)

    relevant_docs = qdrant_vectordb.similarity_search(question, k=1)
    logger.info(f"======{relevant_docs}")

    stream_callback_handler = AsyncIteratorCallbackHandler()
    gen = create_generator(relevant_docs, question, qa_chain, stream_callback_handler)
    return StreamingResponse(gen, media_type="text/event-stream")

async def run_call(relevant_docs, question: str, qa_chain, stream_callback_handler: AsyncIteratorCallbackHandler):
    qa_chain.callbacks = [stream_callback_handler]
    response = await qa_chain.acall({"input_documents": relevant_docs, "question": question})
    return response

async def create_generator(relevant_docs, question: str, qa_chain,
                           stream_callback_handler: AsyncIteratorCallbackHandler):
    run = asyncio.create_task(run_call(relevant_docs, question, qa_chain, stream_callback_handler))

    async for token in stream_callback_handler.aiter():
        logger.info(token)
        yield token

    await run

def insert_into_vectordb(documents: List[Document], filename: str):
    for document in documents:
        document.metadata["source"] = filename
    qdrant_vectordb.add_documents(documents)
