from typing import List
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

class Document(BaseModel):
    content: str = Field(..., description="The content of the document")
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the document"
    )

class QueryResult(BaseModel):
    answer: str = Field(..., description="The answer to the query")
    source_nodes: List[str] = Field(
        ..., description="The source nodes used to generate the answer"
    )

class RAGPipeline:
    def __init__(
        self,
        document_path: str,
    ):
        self.document_path = document_path
        self.configure_settings()
        self.index = None

    def configure_settings(self):
        # TODO: Add parameters for changing Embedding model and LLM
        Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        Settings.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )

    def load_and_index_documents(self):
        documents = SimpleDirectoryReader(self.document_path).load_data()
        if not documents:
            raise ValueError("No documents loaded. Check the document path or file.")
        self.index = VectorStoreIndex.from_documents(documents)

    def query(self, query: str) -> QueryResult:
        if not self.index:
            raise ValueError(
                "Index not created. Call load_and_index_documents() first."
            )
        query_engine = self.index.as_query_engine(similarity_top_k=7)
        response = query_engine.query(query)
        return QueryResult(
            answer=response.response,
            source_nodes=[node.node.text for node in response.source_nodes],
        )

    def save_index(self, path: str):
        if not self.index:
            raise ValueError(
                "Index not created. Call load_and_index_documents() first."
            )
        self.index.storage_context.persist(persist_dir=path)

    @classmethod
    def load_index(cls, path: str):
        pipeline = cls("")
        storage_context = StorageContext.from_defaults(persist_dir=path)
        index = load_index_from_storage(storage_context)
        pipeline.index = index
        return pipeline