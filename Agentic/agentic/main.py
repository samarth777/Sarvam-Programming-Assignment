from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from groq import Groq
from utils.schema import NCERTTextbookAssistant, QueryResult

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables
assistant = None
groq_client = None

class Source(BaseModel):
    content: str

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

class AudioResponse(BaseModel):
    answer: str
    transcribed: str

def initialize_assistant():
    global assistant
    if assistant is None:
        data_folder = "./data"
        index_folder = "./index"
        if os.path.exists(index_folder):
            assistant = NCERTTextbookAssistant(
                data_path=data_folder,
                index_path=index_folder
            )
            print("Loaded existing index.")
        else:
            assistant = NCERTTextbookAssistant(
                data_path=data_folder,
                index_path=index_folder
            )
            print("Created and saved new index.")

def initialize_groq():
    global groq_client
    if groq_client is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        groq_client = Groq(api_key=groq_api_key)

@app.on_event("startup")
async def startup_event():
    initialize_assistant()
    initialize_groq()

@app.get("/health_check", response_model=ChatResponse)
async def health_check():
    return ChatResponse(
        answer="NCERT Textbook Assistant is working!",
    )

@app.post("/query", response_model=ChatResponse)
async def query(request: QueryRequest):
    if not assistant:
        raise HTTPException(status_code=500, detail="Assistant not initialized")
    try:
        result = assistant.query(request.query)
        return ChatResponse(
            answer=result.answer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/audio", response_model=AudioResponse)
async def audio_query(audio: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized")
    try:
        audio_content = await audio.read()
        translation = groq_client.audio.translations.create(
            file=("recording.wav", audio_content),
            model="whisper-large-v3",
            prompt="Transcribe NCERT textbook related questions",
            response_format="json",
            temperature=0.0,
        )
        print(translation.text)
        query_text = translation.text
        result = assistant.query(query_text)
        return AudioResponse(
            answer=result.answer,
            transcribed=query_text,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing audio query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)