import os
import logging
import requests
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from typing import Dict, Any, List
import re
import time
import json

# Cargar variables de entorno
load_dotenv()

# Configuración de Net2Phone
NET2PHONE_CLIENT_ID = os.getenv("NET2PHONE_CLIENT_ID")
NET2PHONE_API_BASE_URL = os.getenv("NET2PHONE_API_BASE_URL", "https://api.n2p.io")

# Configuración de búsqueda en Azure Search
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME")
SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")
USE_VECTOR_SEARCH = os.getenv("USE_VECTOR_SEARCH", "false").lower() == "true"

IDENTIFIER_FIELD = "chunk_id"   # ID único
CONTENT_FIELD = "chunk"         # texto del documento
TITLE_FIELD = "title"           # título del chunk

EMBEDDING_FIELD = "text_vector" # campo vectorial

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patrón para validar identificadores seguros
KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

# Inicializar cliente de búsqueda
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# ------------------- 📞 Net2Phone -------------------


import time
import json

# Archivo temporal para caché persistente del token Net2Phone
TOKEN_CACHE_FILE = '/tmp/n2p_token.json'

def save_token_to_file(token, expiry):
    try:
        with open(TOKEN_CACHE_FILE, 'w') as f:
            json.dump({'token': token, 'expiry': expiry}, f)
    except Exception as e:
        logger.warning(f"No se pudo guardar el token en disco: {e}")

def load_token_from_file():
    if not os.path.exists(TOKEN_CACHE_FILE):
        return None, 0
    try:
        with open(TOKEN_CACHE_FILE, 'r') as f:
            data = json.load(f)
            return data.get('token'), data.get('expiry', 0)
    except Exception as e:
        logger.warning(f"No se pudo leer el token en disco: {e}")
        return None, 0


# Variables globales para caché en memoria
_n2p_token = None
_n2p_token_expiry = 0

async def get_net2phone_token_password() -> str:
    """
    Obtiene y cachea el token OAuth2 usando grant_type=password para Net2Phone v1.
    Reutiliza el token hasta que expire.
    """
    global _n2p_token, _n2p_token_expiry

    # 1. Intenta usar el token en memoria
    if _n2p_token and time.time() < _n2p_token_expiry:
        return _n2p_token

    # 2. Intenta cargar el token desde archivo
    token_file, expiry_file = load_token_from_file()
    if token_file and time.time() < expiry_file:
        _n2p_token = token_file
        _n2p_token_expiry = expiry_file
        return _n2p_token

    # 3. Solicita un nuevo token
    url = f"{NET2PHONE_API_BASE_URL}/v1/oauth/token/"
    data = {
        "client_id": NET2PHONE_CLIENT_ID,
        "username": os.getenv("NET2PHONE_USERNAME"),
        "password": os.getenv("NET2PHONE_PASSWORD"),
        "grant_type": "password"
    }

    try:
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        _n2p_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        _n2p_token_expiry = time.time() + int(expires_in) - 60  # 1 min de margen
        # Guarda el token y expiración en archivo
        save_token_to_file(_n2p_token, _n2p_token_expiry)
        return _n2p_token
    except Exception as e:
        logger.error(f"Error autenticando Net2Phone: {e}")
        raise Exception("No se pudo obtener el token Net2Phone")

async def make_phone_call_handler(phone_number: str, contact_name: str = "Contacto de emergencia") -> Dict[str, Any]:
    """
    Realiza una llamada a un número usando Net2Phone.
    """
    from_number = "234802005"
    try:
        # Normaliza el número chileno: si empieza con +56, reemplaza por 9
        phone = phone_number.replace(" ", "")
        if phone.startswith("+56"):
            # Elimina +56 y toma los siguientes 8 dígitos (sin el 9 inicial)
            phone = "9" + phone[3:]

        token = await get_net2phone_token_password()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.integrate.v1.9.0+json"
        }

        payload = {
            "to": phone,
            "from": from_number
        }

        resp = requests.post(f"{NET2PHONE_API_BASE_URL}/v1/calls/", headers=headers, json=payload, timeout=10)

        if resp.status_code in (200, 201):
            print(f"📞 Llamada iniciada a {contact_name} ({phone})")
            return {
                "status": "success",
                "message": f"Llamada iniciada a {contact_name} ({phone})",
                "details": resp.json()
            }
        else:
            print(f"❌ Error al realizar la llamada: {resp.status_code} - {resp.text}")
            return {
                "status": "error",
                "message": resp.text,
                "code": resp.status_code
            }

    except Exception as e:
        logger.error(f"Error realizando la llamada: {e}")
        return {"status": "error", "message": str(e)}

# ------------------- 🔍 Búsqueda -------------------

async def search_knowledge_base_handler(query: str) -> str:
    """
    Busca información en la base de conocimientos de Azure.
    """
    try:
        vector_queries = []
        if USE_VECTOR_SEARCH:
            vector_queries.append(VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=50,
                fields=EMBEDDING_FIELD
            ))

        results = []
        if USE_VECTOR_SEARCH:
            # Solo vector search, nunca semantic
            search_results = await search_client.search(
                search_text=query,
                query_type="simple",
                top=5,
                vector_queries=vector_queries,
                select=f"{IDENTIFIER_FIELD},{TITLE_FIELD},{CONTENT_FIELD},parent_id"
            )
        else:
            # Solo semantic si está habilitado
            search_results = await search_client.search(
                search_text=query,
                query_type="semantic" if SEMANTIC_CONFIG else "simple",
                semantic_configuration_name=SEMANTIC_CONFIG if SEMANTIC_CONFIG else None,
                top=5,
                select=f"{IDENTIFIER_FIELD},{TITLE_FIELD},{CONTENT_FIELD},parent_id"
            )

        async for r in search_results:
            results.append(f"[{r[IDENTIFIER_FIELD]}] {r[TITLE_FIELD]}: {r[CONTENT_FIELD]}")

        return "\n---\n".join(results) if results else "No se encontraron resultados"

    except Exception as e:
        logger.error(f"Error en la búsqueda: {e}")
        return "Error buscando información"

# ------------------- 📑 Grounding -------------------

async def report_grounding_handler(sources: List[str]) -> Dict[str, Any]:
    """
    Devuelve detalles de las fuentes citadas.
    """
    try:
        valid_sources = [s for s in sources if KEY_PATTERN.match(s)]
        if not valid_sources:
            return {"sources": []}

        search_results = search_client.search(
            search_text=" OR ".join(valid_sources),
            search_fields=[IDENTIFIER_FIELD],
            select=[IDENTIFIER_FIELD, TITLE_FIELD, CONTENT_FIELD],
            top=len(valid_sources)
        )

        docs = []
        async for r in search_results:
            docs.append({
                "source_id": r[IDENTIFIER_FIELD],
                "title": r[TITLE_FIELD],
                "content": r[CONTENT_FIELD]
            })

        return {"sources": docs}

    except Exception as e:
        logger.error(f"Error en grounding: {e}")
        return {"sources": []}

# ------------------- 🧰 Definición de herramientas -------------------

search_tool_def = {
    "name": "search_knowledge_base",
    "description": "Busca información en la base de conocimientos sobre ITAM.",
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
    "description": "Devuelve las fuentes usadas en la respuesta.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Identificadores de las fuentes"
            }
        },
        "required": ["sources"]
    }
}

call_tool_def = {
    "name": "make_phone_call",
    "description": "Realiza una llamada al contacto de emergencia.",
    "parameters": {
        "type": "object",
        "properties": {
            "phone_number": {
                "type": "string",
                "description": "Número en formato internacional (+56...) deberá ."
            },
            "contact_name": {
                "type": "string",
                "description": "Nombre del contacto (opcional)"
            }
        },
        "required": ["phone_number"]
    }
}

# Lista final de herramientas
tools = [
    (search_tool_def, search_knowledge_base_handler),
    (grounding_tool_def, report_grounding_handler),
    (call_tool_def, make_phone_call_handler)
]
