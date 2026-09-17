"""
Pruebas sin red: embeddings falsos y deterministas, y un agente simulado para la API.
"""

import asyncio
import hashlib

import httpx
import numpy as np
import pytest

from app.main import crear_app
from app.rag import Indice, fragmentar


async def embedder_falso(textos, tarea):
    """Bolsa de palabras hasheada: textos con palabras en común quedan cerca."""
    vectores = []
    for t in textos:
        v = np.zeros(64, dtype=np.float32)
        for palabra in t.lower().split():
            v[int(hashlib.md5(palabra.encode()).hexdigest(), 16) % 64] += 1
        vectores.append(v.tolist())
    return vectores


def test_fragmentar_respeta_tamano_y_solapamiento():
    texto = "# Ficha\n" + " ".join(f"palabra{i}" for i in range(600))
    partes = fragmentar("ficha.md", texto, tamano=300, solape=60)
    assert len(partes) > 5
    assert all(len(p.texto) <= 300 for p in partes)
    assert all(p.titulo == "Ficha" for p in partes)
    assert len({p.id for p in partes}) == len(partes)


async def test_busqueda_devuelve_el_documento_relevante(tmp_path):
    (tmp_path / "piscina.md").write_text("# Piscina\nLuminaria sumergible IP68 para piscinas", encoding="utf-8")
    (tmp_path / "proyector.md").write_text("# Proyector\nProyector de fachada 50W exterior", encoding="utf-8")
    indice = Indice(embedder_falso)
    await indice.cargar_carpeta(tmp_path)
    r = await indice.buscar("luminaria sumergible piscinas", k=1)
    assert r[0]["fuente"] == "piscina.md"


async def test_cache_evita_volver_a_embeber(tmp_path):
    (tmp_path / "a.md").write_text("# A\ntexto de prueba", encoding="utf-8")
    cache = tmp_path / "cache.json"
    llamadas = []

    async def contador(textos, tarea):
        llamadas.append(len(textos))
        return await embedder_falso(textos, tarea)

    await Indice(contador).cargar_carpeta(tmp_path, cache)
    await Indice(contador).cargar_carpeta(tmp_path, cache)
    assert llamadas == [1]


class AgenteFalso:
    async def preguntar(self, pregunta, sesion=None):
        return {"respuesta": f"eco: {pregunta}", "fuentes": ["a.md"], "busquedas": [pregunta], "sesion": sesion or "s1"}


@pytest.fixture
async def cliente(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# A\nDocumento de prueba", encoding="utf-8")
    monkeypatch.setattr("app.config.DOCS_DIR", tmp_path)
    monkeypatch.setattr("app.config.CACHE_INDICE", tmp_path / "cache.json")
    app = crear_app(indice=Indice(embedder_falso), agente=AgenteFalso())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            for _ in range(50):  # el índice se arma en segundo plano al arrancar
                if (await c.get("/salud")).json()["indice_listo"]:
                    break
                await asyncio.sleep(0.02)
            yield c


async def test_preguntar(cliente):
    r = await cliente.post("/preguntar", json={"pregunta": "¿Qué IP tiene?"})
    assert r.status_code == 200
    j = r.json()
    assert j["respuesta"] == "eco: ¿Qué IP tiene?" and j["fuentes"] == ["a.md"] and j["ms"] >= 0


async def test_validacion_de_entrada(cliente):
    assert (await cliente.post("/preguntar", json={"pregunta": "a"})).status_code == 422


async def test_subir_documento_y_listarlo(cliente):
    r = await cliente.post("/documentos", files={"archivo": ("nuevo.md", b"# Nuevo\nTexto nuevo", "text/markdown")})
    assert r.status_code == 201 and r.json()["fragmentos_nuevos"] == 1
    nombres = [d["fuente"] for d in (await cliente.get("/documentos")).json()]
    assert "nuevo.md" in nombres


async def test_rechaza_formatos_no_soportados(cliente):
    r = await cliente.post("/documentos", files={"archivo": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415


async def test_salud(cliente):
    j = (await cliente.get("/salud")).json()
    assert j["ok"] and j["indice_listo"] and j["documentos"] == 1
