import json
import random
import re
import chainlit as cl
from datetime import datetime, timedelta
import uuid
from azure.core.credentials import AzureKeyCredential
from azure.identity import get_bearer_token_provider
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import logging
from typing import Any
import base64
import requests
from typing import Dict

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configuración de Net2Phone
NET2PHONE_CLIENT_ID = os.getenv("NET2PHONE_CLIENT_ID")
NET2PHONE_CLIENT_SECRET = os.getenv("NET2PHONE_CLIENT_SECRET")
NET2PHONE_API_KEY = os.getenv("NET2PHONE_API_KEY")
NET2PHONE_API_BASE_URL = os.getenv("NET2PHONE_API_BASE_URL", "https://api.voipnow.com")

# Configuración de búsqueda
SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", None)
IDENTIFIER_FIELD = "chunk_id"
CONTENT_FIELD = "chunk"
TITLE_FIELD = "title"
EMBEDDING_FIELD = "embedding"
USE_VECTOR_SEARCH = os.getenv("USE_VECTOR_SEARCH", "false").lower() == "true"

# Patrón para validar identificadores de fuentes
KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

# Validar variables de entorno requeridas
required_env_vars = [
    "AZURE_SEARCH_ENDPOINT", 
    "INDEX_NAME", 
    "AZURE_SEARCH_KEY",
    "NET2PHONE_CLIENT_ID",
    "NET2PHONE_CLIENT_SECRET",
    "NET2PHONE_API_KEY"
]

for var in required_env_vars:
    if not os.getenv(var):
        logging.error(f"Falta la variable de entorno: {var}")

# Inicializar cliente de búsqueda
search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.environ["INDEX_NAME"],
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
)

# Configurar el nivel de registro
logging.basicConfig(level=logging.DEBUG)

async def get_net2phone_token() -> str:
    """Obtiene un token de acceso OAuth2 para Net2Phone API."""
    auth_string = f"{NET2PHONE_CLIENT_ID}:{NET2PHONE_CLIENT_SECRET}"
    auth_bytes = auth_string.encode("utf-8")
    auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "client_credentials",
        "scope": "api"
    }

    try:
        response = requests.post(
            f"{NET2PHONE_API_BASE_URL}/oauth/token",
            headers=headers,
            data=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        logging.error(f"Error en autenticación Net2Phone: {str(e)}")
        raise Exception(f"No se pudo obtener token: {str(e)}")

async def make_phone_call_handler(params: dict) -> dict:
    """Versión corregida del manejador de llamadas"""
    # Extraer parámetros correctamente
    phone_number = params.get("phone_number")
    if not phone_number:
        return {"status": "error", "message": "Número de teléfono no proporcionado"}
    
    contact_name = params.get("contact_name", "Contacto de emergencia")
    
    try:
        # Resto de la implementación permanece igual
        access_token = await get_net2phone_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-API-Key": NET2PHONE_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "destination": phone_number,
            "caller_id": "ITAMBot",
            "call_type": "regular",
            "custom_data": {
                "contact_name": contact_name,
                "reason": "Emergencia"
            }
        }
        
        response = requests.post(
            f"{NET2PHONE_API_BASE_URL}/v3/calls",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 201:
            return {
                "status": "success",
                "message": f"Llamada iniciada a {contact_name} ({phone_number})",
                "call_id": response.json().get("call_id")
            }
        else:
            return {
                "status": "error",
                "message": f"Error al llamar: {response.text}",
                "status_code": response.status_code
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error en la API: {str(e)}"
        }

# Definición de herramientas
search_tool_def = {
    "name": "search_knowledge_base",
    "description": "Busca en la base de conocimientos sobre ITAM (empleados, contactos de emergencia, etc.).",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Consulta de búsqueda"}
        },
        "required": ["query"]
    }
}

grounding_tool_def = {
    "name": "report_grounding",
    "description": "Cita fuentes usadas en la respuesta.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Fuentes usadas"
            }
        },
        "required": ["sources"]
    }
}

call_tool_def = {
    "name": "make_phone_call",
    "description": "Realiza llamadas a contactos de emergencia",
    "parameters": {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Número en formato internacional (+56912345678), si hay espacios en blanco, se eliminarán automáticamente",
            },
            "contact_name": {
                "type": "string",
                "description": "Nombre del contacto (opcional)"
            }
        },
        "required": ["phone_number"]
    }
}

async def search_knowledge_base_handler(query: str) -> str:
    """Busca en la base de conocimientos de Azure."""
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
            results.append(f"[{result['chunk_id']}] {result['title']}: {result['chunk']}\n-----")
        
        return "\n".join(results) if results else "No hay resultados"

    except Exception as e:
        logging.error(f"Error en búsqueda: {str(e)}")
        return "Error al buscar información"

async def report_grounding_handler(params: dict) -> dict:
    """Manejador para citar fuentes."""
    try:
        sources = [s for s in params["sources"] if KEY_PATTERN.match(s)]
        if not sources:
            return {"sources": []}

        search_results = search_client.search(
            search_text=" OR ".join(sources),
            search_fields=[IDENTIFIER_FIELD],
            select=[IDENTIFIER_FIELD, TITLE_FIELD, CONTENT_FIELD],
            top=len(sources)
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

# Lista final de herramientas
tools = [
    (search_tool_def, search_knowledge_base_handler),
    (grounding_tool_def, report_grounding_handler),
    (call_tool_def, make_phone_call_handler)  # ¡Herramienta de llamada agregada!
]