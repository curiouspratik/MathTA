import os
import shutil
import sys
from typing import Dict
from llama_index.llms.ollama import Ollama
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# This is not used but required by llama-index and must be set FIRST
import logging
os.environ["OPENAI_API_KEY"] = "sk-..."

import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

# Check if the code is running in a Docker container
if os.path.exists("/.dockerenv"):
    OLLAMA_URL = "http://host.docker.internal:11434/"  # Docker-compatible URL
else:
    OLLAMA_URL = "http://localhost:11434/"  # Default URL

def is_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get(OLLAMA_URL, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Check if Ollama is running BEFORE starting FastAPI
if not is_ollama_running():
    print("❌ Ollama server is not running. Start Ollama first.")
    sys.exit(1)  # Exit before FastAPI starts
else:
    print("✅ Ollama server is running. Starting FastAPI...")
    # FastAPI app initialization
    app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI is running with Ollama available"}


def get_available_models():
    """Retrieve and print available models from Ollama."""
    try:
        response = requests.get(OLLAMA_URL+"/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models] if models else []
    except requests.exceptions.RequestException:
        return []


# Directory to store uploaded files
UPLOAD_DIRECTORY = "uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload a file."""
    try:
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"filename": file.filename, "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading the file: {str(e)}")

@app.get("/check-files/")
async def check_uploaded_files():
    """Endpoint to check if there are any uploaded files."""
    try:
        files = os.listdir(UPLOAD_DIRECTORY)
        if not files:
            return {"message": "No files uploaded"}
        return {"uploaded_files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while checking files: {str(e)}")


# List of LLM models
llm_models = ["mistral"] # Default model
current_llm = "mistral"
@app.get("/available-llms/")
async def get_available_llms():
    """Endpoint to get the list of available LLM models."""
    try:
        llm_models = get_available_models() # Fetch the list of available models
        if not llm_models:
            return {"message": "No LLM models available"}
        return {"available_llms": llm_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching LLMs: {str(e)}")
    
@app.post("/change-llm/")
async def change_llm(model_name: str):
    """Endpoint to change the current LLM model."""
    global current_llm
    try:
        llm_models = get_available_models()  # Fetch the list of available models
        if model_name not in llm_models:
            raise HTTPException(status_code=400, detail="Invalid LLM model name")
        current_llm = model_name
        return {"message": f"Current LLM model changed to {current_llm}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while changing LLM model: {str(e)}")


# List of embedding models
embedding_models = ["mixedbread-ai/mxbai-embed-large-v1", "Salesforce/SFR-Embedding-Mistral"]
current_embedding = "mixedbread-ai/mxbai-embed-large-v1"
@app.get("/available-embeddings/")
async def get_available_embeddings():
    """Endpoint to get the list of available embedding models."""
    try:
        if not embedding_models:
            return {"message": "No embedding models available"}
        return {"available_embeddings": embedding_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching embeddings: {str(e)}")

@app.post("/change-embedding/")
async def change_embedding(model_name: str):
    """Endpoint to change the current embedding model."""
    global current_embedding
    try:
        if model_name not in embedding_models:
            raise HTTPException(status_code=400, detail="Invalid embedding model name")
        current_embedding = model_name
        return {"message": f"Current embedding model changed to {current_embedding}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while changing embedding model: {str(e)}")


# Logger setup
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def set_settings():
    """Initialize LLM and embedding settings."""
    global current_embedding
    global current_llm
    global OLLAMA_URL
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=current_embedding, 
        device="cpu"
    )
    Settings.llm = Ollama(
        model=current_llm, 
        base_url=OLLAMA_URL, 
        request_timeout=360
    )
    return Settings

@app.get("/api/math-query/", response_model=Dict[str, str])
async def get_math_query(prompt: str = " "):
    """Endpoint to retrieve a math query answer using RAG."""
    try:
        set_settings()
        # Load documents
        docs = SimpleDirectoryReader(input_dir=UPLOAD_DIRECTORY).load_data()

        # Build index and query engine
        index = VectorStoreIndex.from_documents(docs)
        
        
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="refine",
            streaming=True
        )

        # Generate response
        response = []
        for text in query_engine.query(prompt).response_gen:
            response.append(text)

        response = " ".join(response)
        return {"response": response}

    except Exception as e:
        logger.error(f"Error processing math query: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the query.")



@app.on_event("shutdown")
def cleanup_uploaded_files():
    """Clean up the uploaded files directory on application shutdown."""
    try:
        if os.path.exists(UPLOAD_DIRECTORY):
            shutil.rmtree(UPLOAD_DIRECTORY)  # Remove all files and the directory
            os.makedirs(UPLOAD_DIRECTORY)  # Recreate the directory for future use
    except Exception as e:
        print(f"An error occurred during cleanup: {str(e)}")
