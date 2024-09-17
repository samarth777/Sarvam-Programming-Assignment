# Sarvam Programming Assignment
Sarvam Programming Assignment - Samarth P

## Tech Stack

### Orchestration Framework, LLM and Embedding Model
- LlamaIndex
- Groq Llama-3.1-70b
- Jina AI Embeddings
- Whisper Large 3 (for voice input)

### Backend
- FastAPI

### Frontend
- Next.js
- Tailwind CSS
- Framer Motion

### Development Tools
- Poetry

## Project Structure

The project is divided into three main components:

1. **RAG System:**
   - `RAG/main.py`: FastAPI backend for the RAG system
   - `RAG/streamlit_main.py`: Streamlit interface for the RAG system
   - `RAG/utils/schema.py`: Contains the RAGPipeline class implemented using LlamaIndex

2. **Agentic System:**
   - `Agentic/main.py`: FastAPI backend for the Agentic system
   - `Agentic/streamlit_main.py`: Streamlit interface for the Agentic system
   - `Agentic/utils/agent.py`: Contains the ReActAgent class for decision-making

3. **Frontend (React):**
   - `page.tsx`: Main React component for the chatbot interface

## Implementation Details

### Part 1: Building a RAG System

The RAG (Retrieval-Augmented Generation) system is implemented in the `RAG` folder. Key features include:
- Document loading and indexing using LlamaIndex
- Vector storage for efficient retrieval
- Query processing using Groq LLM and Jina AI Embeddings
- Voice input processing using Whisper Large 3

The system can be queried through a FastAPI endpoint and accessed through the frontend or Streamlit interface.

### Part 2: Building an Intelligent Agent

The agent is implemented in the `Agentic` folder:
- Uses LlamaIndex's ReActAgent for decision-making
- Incorporates multiple tools:
  - `ncert_textbook_search`: For searching the NCERT textbook database
  - `get_subject_overview`: Provides an overview of a subject for a specific grade
  - `suggest_practice_questions`: Suggests practice questions for a given topic and grade
- Voice input processing using Whisper Large 3

The agent can decide when to use the vector database and when to use other tools based on the query.

### Part 3: Voice Integration

Both the RAG and Agentic systems support voice-based input:
- Utilizes Whisper Large 3 for accurate speech-to-text conversion
- Enables users to interact with the chatbot using voice commands or queries
- The frontend includes audio recording functionality to capture voice input

## Setup Instructions

### Backend Setup

1. Ensure you have Python 3.11+ installed.
2. Install Poetry:
   ```
   pip install poetry
   ```
3. Clone the repository and navigate to the project directory.
4. Install dependencies:
   ```
   poetry install
   ```
5. Set up environment variables:
   - Create a `.env` file in both the `RAG` and `Agentic` folders with your API keys:
     ```
     GROQ_API_KEY=your_groq_api_key
     JINA_API_KEY=your_jina_api_key
     ```

### Frontend Setup

1. Ensure you have Node.js and npm installed.
2. Navigate to the frontend directory.
3. Install dependencies:
   ```
   npm install
   ```

## Running the Application

### Backend

You can run either the RAG system or the Agentic system:

1. For the RAG system:
   ```
   cd RAG
   uvicorn main:app --reload
   ```

2. For the Agentic system:
   ```
   cd Agentic
   uvicorn main:app --reload
   ```

### Streamlit Interface

To run the Streamlit interface for either system:
```
streamlit run streamlit_main.py
```

### Frontend

1. Start the development server:
   ```
   npm run dev
   ```
2. Open a web browser and go to `http://localhost:3000` (or the port specified in your console).

## Using the Chatbot

1. Type your query in the input field or use the microphone button for voice input.
2. For voice input, speak clearly into your microphone. Whisper Large 3 will transcribe your speech to text.
3. The chatbot will process your query (text or transcribed speech) and provide a response based on the NCERT textbook content.
4. You can ask questions about various subjects, request overviews, or ask for practice questions using either text or voice.

## Testing

You can test different components of the system:
- Use the FastAPI backend with the Next.js frontend for a full-fledged experience, including voice input.
- Use the Streamlit interface for quick, standalone testing of either the RAG or Agentic system, which also supports voice input.

