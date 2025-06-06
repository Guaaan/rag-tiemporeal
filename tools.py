import json
import random
import re
import chainlit as cl
from datetime import datetime, timedelta
import uuid
from azure.core.credentials import AzureKeyCredential
from azure.identity import get_bearer_token_provider
from azure.search.documents.aio import SearchClient  # Nota el .aio para async
from azure.search.documents.models import VectorizableTextQuery
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import logging
from typing import Any

# Configurar el nivel de registro
logging.basicConfig(level=logging.DEBUG)

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configuración de búsqueda
SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", None)
IDENTIFIER_FIELD = "chunk_id"
CONTENT_FIELD = "chunk"
TITLE_FIELD = "title"
EMBEDDING_FIELD = "embedding"  # Ajustar si usas búsqueda vectorial
TITLE_FIELD = "title"  # Ajustar según tu índice
USE_VECTOR_SEARCH = os.getenv("USE_VECTOR_SEARCH", "false").lower() == "true"

# Patrón para validar identificadores de fuentes
KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

required_env_vars = ["AZURE_SEARCH_ENDPOINT", "INDEX_NAME", "AZURE_SEARCH_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        print(f"Falta la variable de entorno: {var}")

# Inicializar cliente de búsqueda
search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.environ["INDEX_NAME"],
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
)

# Definición de herramientas
search_tool_def = {
    "name": "search_knowledge_base",
    "description": "Busca en tu base de conocimientos en español información sobre ITAM (curriculum, clientes, proveedores, Contactos Emergencias, Direccion y telefono de contacto). El archivo ITAM.md te ayudará a responder preguntas generales relacionadas a ITAM.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Consulta de búsqueda"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

grounding_tool_def = {
    "name": "report_grounding",
    "description": "Reporta el uso de fuentes de la base de conocimientos como parte de una respuesta (citar la fuente). Las fuentes aparecen entre corchetes antes de cada pasaje. Siempre usa esta herramienta para citar fuentes cuando respondas con información de la base de conocimientos.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Lista de nombres de fuentes usadas en la última respuesta"
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}

async def search_knowledge_base_handler(query: str) -> str:
    logging.debug(f"Búsqueda iniciada con query: {query}")

    try:
        vector_queries = []
        if USE_VECTOR_SEARCH:
            vector_queries.append(VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=50,
                fields=EMBEDDING_FIELD
            ))

        search_results = await search_client.search(
            search_text=query,
            query_type="semantic" if SEMANTIC_CONFIG else "simple",
            semantic_configuration_name=SEMANTIC_CONFIG,
            top=5,
            vector_queries=vector_queries,
            select="chunk_id, chunk, title"
        )

        results = []
        async for result in search_results:
            results.append(f"[{result['chunk_id']}] - {result['title']}: {result['chunk']}\n-----")

        return "\n".join(results) if results else "No se encontraron resultados."

    except Exception as e:
        logging.error(f"Error en la búsqueda: {str(e)}")
        return "Error en la búsqueda."
      
async def report_grounding_handler(params: dict) -> dict:
    """Manejador para reportar fuentes usadas"""
    sources = [s for s in params["sources"] if KEY_PATTERN.match(s)]
    source_list = " OR ".join(sources)
    logging.debug(f"Citando fuentes: {source_list}")
    
    try:
        search_results = search_client.search(
            search_text=source_list,
            search_fields=[IDENTIFIER_FIELD],
            select=[IDENTIFIER_FIELD, TITLE_FIELD, CONTENT_FIELD],
            top=len(sources),
            query_type="full"
        )
        
        docs = []
        async for result in search_results:
            docs.append({
                "source_id": result[IDENTIFIER_FIELD],
                "title": result[TITLE_FIELD],
                "content": result[CONTENT_FIELD]
            })
        
        return {"sources": docs}
    
    except Exception as e:
        logging.error(f"Error al citar fuentes: {str(e)}")
        return {"sources": []}

# Lista de herramientas
tools = [
    (search_tool_def, search_knowledge_base_handler),
    (grounding_tool_def, report_grounding_handler),
]