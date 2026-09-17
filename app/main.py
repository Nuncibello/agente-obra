"""
API del asistente de obra.

  GET  /              interfaz web mínima para probarlo
  GET  /salud         estado del servicio y del índice
  GET  /documentos    qué documentos están indexados
  POST /documentos    sube un PDF, .md o .txt y lo suma al índice
  POST /preguntar     {"pregunta": "...", "sesion": "..."} -> respuesta del agente con fuentes
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from collections import defaultdict, deque

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config
from .rag import Indice

log = logging.getLogger("agente_obra")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Pregunta(BaseModel):
    pregunta: str = Field(min_length=3, max_length=1000)
    sesion: str | None = Field(default=None, max_length=64)


class Respuesta(BaseModel):
    respuesta: str
    fuentes: list[str]
    busquedas: list[str]
    sesion: str
    ms: int


def crear_app(indice: Indice | None = None, agente=None) -> FastAPI:
    estado: dict = {"indice": indice, "agente": agente, "listo": asyncio.Event()}

    @asynccontextmanager
    async def ciclo(app: FastAPI):
        if estado["indice"] is None:
            from google import genai

            from . import agente as modulo_agente
            from .rag import embedder_gemini

            if not config.GOOGLE_API_KEY:
                raise RuntimeError("Falta GOOGLE_API_KEY")
            cliente = genai.Client(api_key=config.GOOGLE_API_KEY, vertexai=config.USAR_VERTEX)
            estado["indice"] = Indice(embedder_gemini(cliente))
            estado["agente"] = modulo_agente.crear(estado["indice"])

        async def indexar():
            t = time.perf_counter()
            try:
                n = await estado["indice"].cargar_carpeta(config.DOCS_DIR, config.CACHE_INDICE)
                log.info("índice listo: %s fragmentos en %.1fs", n, time.perf_counter() - t)
            except Exception:
                log.exception("no se pudo indexar la documentación")
            finally:
                estado["listo"].set()

        tarea = asyncio.create_task(indexar())  # el servicio arranca ya; el índice se arma en paralelo
        yield
        tarea.cancel()

    app = FastAPI(title="Asistente de obra", version="1.0.0", lifespan=ciclo)
    consultas: dict[str, deque] = defaultdict(deque)

    def limitar(request: Request, maximo: int = config.CONSULTAS_POR_HORA):
        """Demo pública: tope de consultas por IP por hora, para que nadie agote la cuota del modelo."""
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "")).split(",")[0].strip()
        ahora, cola = time.time(), consultas[ip]
        while cola and ahora - cola[0] > 3600:
            cola.popleft()
        if len(cola) >= maximo:
            raise HTTPException(429, "Llegaste al límite de consultas de la demo por esta hora")
        cola.append(ahora)

    @app.get("/salud")
    async def salud():
        i = estado["indice"]
        return {"ok": True, "indice_listo": estado["listo"].is_set(),
                "documentos": len(i.documentos()) if i else 0, "fragmentos": len(i.fragmentos) if i else 0,
                "modelo": config.MODELO}

    @app.get("/documentos")
    async def documentos():
        return estado["indice"].documentos()

    @app.post("/documentos", status_code=201)
    async def subir(request: Request, archivo: UploadFile = File(...)):
        limitar(request, maximo=5)
        nombre = Path(archivo.filename or "documento").name
        datos = await archivo.read(config.MAX_SUBIDA_BYTES + 1)
        if len(datos) > config.MAX_SUBIDA_BYTES:
            raise HTTPException(413, "El archivo supera 5 MB")
        if nombre.lower().endswith(".pdf"):
            texto = await asyncio.to_thread(_texto_pdf, datos)
        elif nombre.lower().endswith((".md", ".txt")):
            texto = datos.decode("utf-8", errors="replace")
        else:
            raise HTTPException(415, "Formatos aceptados: PDF, .md y .txt")
        if not texto.strip():
            raise HTTPException(422, "No encontré texto en el archivo")
        await asyncio.wait_for(estado["listo"].wait(), timeout=120)
        n = await estado["indice"].agregar([(nombre, texto)])
        return {"documento": nombre, "fragmentos_nuevos": n}

    @app.post("/preguntar", response_model=Respuesta)
    async def preguntar(p: Pregunta, request: Request):
        limitar(request)
        try:
            await asyncio.wait_for(estado["listo"].wait(), timeout=120)
        except asyncio.TimeoutError:
            raise HTTPException(503, "El índice todavía se está armando, probá en unos segundos")
        t = time.perf_counter()
        try:
            r = await asyncio.wait_for(estado["agente"].preguntar(p.pregunta, p.sesion), timeout=90)
        except asyncio.TimeoutError:
            raise HTTPException(504, "El modelo tardó demasiado en responder")
        except Exception as e:
            log.exception("falló la consulta")
            raise HTTPException(502, f"No pude consultar al modelo: {type(e).__name__}")
        return Respuesta(**r, ms=int((time.perf_counter() - t) * 1000))

    @app.get("/", response_class=HTMLResponse)
    async def inicio():
        return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

    return app


def _texto_pdf(datos: bytes) -> str:
    from pypdf import PdfReader

    return "\n".join((pagina.extract_text() or "") for pagina in PdfReader(io.BytesIO(datos)).pages)


app = crear_app()
