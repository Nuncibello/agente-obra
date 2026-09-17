"""
RAG liviano: fragmenta documentos, genera embeddings con Gemini y busca por similitud coseno.

El índice vive en memoria (numpy) y se cachea en disco por hash de contenido, así un reinicio no
vuelve a pagar embeddings de documentos que no cambiaron. Para una base grande el paso siguiente
sería pgvector o Vertex AI Vector Search; para decenas de documentos, esto es más simple y rápido.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Sequence

import numpy as np

from . import config

Embedder = Callable[[Sequence[str], str], Awaitable[list[list[float]]]]


@dataclass
class Fragmento:
    id: str
    fuente: str
    titulo: str
    texto: str


@dataclass
class Indice:
    embedder: Embedder
    fragmentos: list[Fragmento] = field(default_factory=list)
    _vectores: dict[str, list[float]] = field(default_factory=dict)
    _matriz: np.ndarray | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ------------------------------------------------------------ carga

    async def cargar_carpeta(self, carpeta: Path, cache: Path | None = None) -> int:
        documentos = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(carpeta.glob("*.md"))]
        return await self.agregar(documentos, cache)

    async def agregar(self, documentos: Sequence[tuple[str, str]], cache: Path | None = None) -> int:
        nuevos = [f for nombre, texto in documentos for f in fragmentar(nombre, texto)]
        if not nuevos:
            return 0

        guardados = _leer_cache(cache)
        faltan = [f for f in nuevos if f.id not in guardados]
        if faltan:
            vectores = await self._embeber_en_lotes([f.texto for f in faltan], "RETRIEVAL_DOCUMENT")
            guardados.update({f.id: v for f, v in zip(faltan, vectores)})
            _escribir_cache(cache, guardados)

        async with self._lock:
            existentes = {f.id for f in self.fragmentos}
            sumar = [f for f in nuevos if f.id not in existentes]
            self.fragmentos.extend(sumar)
            self._vectores.update({f.id: guardados[f.id] for f in sumar})
            self._matriz = _normalizar(np.array([self._vectores[f.id] for f in self.fragmentos], dtype=np.float32))
        return len(sumar)

    async def _embeber_en_lotes(self, textos: list[str], tarea: str, lote: int = 90) -> list[list[float]]:
        partes = [textos[i:i + lote] for i in range(0, len(textos), lote)]
        limite = asyncio.Semaphore(3)

        async def uno(parte):
            async with limite:
                return await self.embedder(parte, tarea)

        resultados = await asyncio.gather(*(uno(p) for p in partes))
        return [v for r in resultados for v in r]

    # ------------------------------------------------------------ búsqueda

    async def buscar(self, consulta: str, k: int = config.RESULTADOS) -> list[dict]:
        if not self.fragmentos or self._matriz is None:
            return []
        (vector,) = await self.embedder([consulta], "RETRIEVAL_QUERY")
        q = _normalizar(np.array([vector], dtype=np.float32))[0]
        puntajes = self._matriz @ q
        mejores = np.argsort(-puntajes)[:k]
        return [{"fuente": self.fragmentos[i].fuente, "titulo": self.fragmentos[i].titulo,
                 "texto": self.fragmentos[i].texto, "similitud": round(float(puntajes[i]), 4)} for i in mejores]

    def documentos(self) -> list[dict]:
        conteo: dict[str, dict] = {}
        for f in self.fragmentos:
            d = conteo.setdefault(f.fuente, {"fuente": f.fuente, "titulo": f.titulo, "fragmentos": 0})
            d["fragmentos"] += 1
        return sorted(conteo.values(), key=lambda d: d["fuente"])


# ---------------------------------------------------------------- utilidades

def fragmentar(nombre: str, texto: str, tamano: int = config.TAMANO_FRAGMENTO,
               solape: int = config.SOLAPAMIENTO) -> list[Fragmento]:
    texto = re.sub(r"[ \t]+", " ", texto).strip()
    if not texto:
        return []
    titulo = next((l.lstrip("# ").strip() for l in texto.splitlines() if l.strip()), nombre)
    fragmentos, inicio = [], 0
    while inicio < len(texto):
        fin = min(len(texto), inicio + tamano)
        if fin < len(texto):  # cortar en un salto de línea o espacio para no partir palabras
            corte = max(texto.rfind("\n", inicio + tamano // 2, fin), texto.rfind(" ", inicio + tamano // 2, fin))
            fin = corte if corte > inicio else fin
        pedazo = texto[inicio:fin].strip()
        if pedazo:
            h = hashlib.sha1(f"{nombre}|{pedazo}".encode()).hexdigest()[:16]
            fragmentos.append(Fragmento(id=h, fuente=nombre, titulo=titulo, texto=pedazo))
        if fin >= len(texto):
            break
        inicio = max(fin - solape, inicio + 1)
    return fragmentos


def _normalizar(m: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(normas == 0, 1, normas)


def _leer_cache(cache: Path | None) -> dict[str, list[float]]:
    if cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _escribir_cache(cache: Path | None, datos: dict) -> None:
    if not cache:
        return
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(datos), encoding="utf-8")
    except OSError:
        pass  # en un contenedor de solo lectura seguimos sin cache


def embedder_gemini(cliente) -> Embedder:
    """Embeddings con la API async de google-genai."""
    from google.genai import types

    async def embeber(textos: Sequence[str], tarea: str) -> list[list[float]]:
        r = await cliente.aio.models.embed_content(
            model=config.MODELO_EMBEDDINGS, contents=list(textos),
            config=types.EmbedContentConfig(task_type=tarea, output_dimensionality=config.DIMENSIONES))
        return [e.values for e in r.embeddings]

    return embeber
