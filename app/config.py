import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
USAR_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() in ("1", "TRUE")
MODELO = os.getenv("MODELO", "gemini-2.5-flash")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "gemini-embedding-001")
DIMENSIONES = int(os.getenv("DIMENSIONES", "768"))

DOCS_DIR = Path(os.getenv("DOCS_DIR", RAIZ / "docs"))
CACHE_INDICE = Path(os.getenv("CACHE_INDICE", RAIZ / "data" / "indice.json"))

TAMANO_FRAGMENTO = 900
SOLAPAMIENTO = 150
RESULTADOS = 4
MAX_SUBIDA_BYTES = 5 * 1024 * 1024
CONSULTAS_POR_HORA = int(os.getenv("CONSULTAS_POR_HORA", "20"))
